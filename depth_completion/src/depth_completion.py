import os, time, sys, random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
sys.path.insert(0, os.getcwd())
import datasets
from utils.src import data_utils, eval_utils
from utils.src.log_utils import log
from erosion2d import Erosion2d
from depth_completion_model import DepthCompletionModel
from utils.src.transforms import Transforms
from PIL import Image


def set_random_seed(seed):
    '''Repeat initialization, augmentation, and stochastic keep-mask sampling.'''
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    '''Match Python/NumPy worker RNGs to the seed assigned by the DataLoader.'''
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def train(train_images_path,
          train_sparse_depth_path,
          train_intrinsics_path,
          train_ground_truth_path,
          val_image_path,
          val_sparse_depth_path,
          val_intrinsics_path,
          val_ground_truth_path,
          # Batch settings
          n_batch,
          n_height,
          n_width,
          n_train_sample_limit,
          # Input settings
          input_channels_image,
          input_channels_depth,
          normalized_image_range,
          # Depth network settings
          model_name,
          network_modules,
          min_predict_depth,
          max_predict_depth,
          n_iteration,
          window_size,
          n_self_attention,
          n_shift,
          use_outlier_rejection,
          # Loss function settings
          supervision_type,
          w_losses,
          w_weight_decay_depth,
          w_weight_decay_pose,
          # Training settings
          learning_rates,
          learning_schedule,
          n_step_per_gradient_accumulation,
          augmentation_probabilities,
          augmentation_schedule,
          # Photometric data augmentations
          augmentation_random_brightness,
          augmentation_random_contrast,
          augmentation_random_gamma,
          augmentation_random_hue,
          augmentation_random_saturation,
          augmentation_random_gaussian_blur_kernel_size,
          augmentation_random_gaussian_blur_sigma_range,
          augmentation_random_noise_type,
          augmentation_random_noise_spread,
          # Geometric data augmentations
          augmentation_padding_mode,
          augmentation_random_crop_type,
          augmentation_random_crop_to_shape,
          augmentation_random_flip_type,
          augmentation_random_rotate_max,
          augmentation_random_crop_and_pad,
          augmentation_random_resize_to_shape,
          augmentation_random_resize_and_pad,
          augmentation_random_resize_and_crop,
          # Occlusion data augmentations
          augmentation_random_remove_patch_percent_range_image,
          augmentation_random_remove_patch_size_image,
          augmentation_random_remove_patch_percent_range_depth,
          augmentation_random_remove_patch_size_depth,
          # Evaluation settings
          min_evaluate_depth,
          max_evaluate_depth,
          # Checkpoint settings
          checkpoint_path,
          n_step_per_checkpoint,
          n_step_per_summary,
          n_image_per_summary,
          start_step_validation,
          restore_paths,
          # Hardware settings
          device='cuda',
          n_thread=8,
          debug_depth=False,
          debug_interval=500,
          debug_max_samples=3,
          detach_scaleless_depth=True,
          supervise_outlier_rejection=False,
          rejection_keep_tolerance=(0.1, 0.02),
          rejection_reject_tolerance=(0.3, 0.05),
          rejection_supervision_levels=None,
          n_propagation_iterations=(2, 4, 6, 8, 12),
          n_scale_refinement=2,
          rejection_loss_type='cross_entropy',
          encoder_type='conv',
          rejection_keep_weight=2.0,
          w_rejection_l1=None,
          use_sparse_depth_features=False,
          seed=0,
          fixed_geometry_stage='scale',
          geometry_restore_path=None):

    set_random_seed(seed)
    is_fixed_geometry = 'fixed_geometry_scale_propagation' in model_name
    if is_fixed_geometry:
        # This independent baseline deliberately has no rejection, raw sparse
        # feature fusion, or unconstrained scale-refinement head.
        use_outlier_rejection = False
        use_sparse_depth_features = False
        supervise_outlier_rejection = False
        detach_scaleless_depth = True
        n_scale_refinement = 0
        if fixed_geometry_stage == 'scale' and not geometry_restore_path and not restore_paths:
            raise ValueError('Scale training requires --geometry_restore_path or --restore_paths; '
                             'pretrain with --fixed_geometry_stage geometry first.')
    # Separate loader generators keep shuffling independent of model size and
    # prevent validation iterator creation from advancing training's RNG stream.
    train_generator = torch.Generator().manual_seed(seed)
    validation_generator = torch.Generator().manual_seed(seed + 1)

    # Select device to run on
    if device == 'cuda' or device == 'gpu':
        device = torch.device('cuda')
    elif device == 'cpu':
        device = torch.device('cpu')
    else:
        raise ValueError('Unsupported device: {}'.format(device))

    if rejection_supervision_levels is None:
        rejection_supervision_levels = list(range(5))
    rejection_loss_active = supervise_outlier_rejection and use_outlier_rejection and \
        'scale_propagation' in model_name and supervision_type == 'supervised'
    rejection_loss_active = rejection_loss_active and rejection_loss_type != 'none'
    rejection_supervision_active = rejection_loss_active and rejection_loss_type == 'cross_entropy'
    rejection_sparsity_active = rejection_loss_active and rejection_loss_type == 'l1'
    # CE keeps its existing weight. L1 optionally has a separate bash override
    # because it now penalizes keeping anchors, rather than rejecting them.
    rejection_loss_weight = w_losses.get('w_rejection', 0.1)
    if rejection_loss_type == 'l1' and w_rejection_l1 is not None:
        rejection_loss_weight = w_rejection_l1

    # Set up checkpoint and event paths
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    checkpoint_dirpath = os.path.join(checkpoint_path, 'checkpoints-{}')
    log_path = os.path.join(checkpoint_path, 'results.txt')
    event_path = os.path.join(checkpoint_path, 'events')

    best_results = {
        'step': -1,
        'mae': np.infty,
        'rmse': np.infty,
        'imae': np.infty,
        'irmse': np.infty
    }

    '''
    Load input paths and set up dataloaders
    '''
    # Read paths for training
    train_images_paths = data_utils.read_paths(train_images_path)
    train_sparse_depth_paths = data_utils.read_paths(train_sparse_depth_path)

    n_train_sample = len(train_images_paths)

    if train_intrinsics_path is not None:
        train_intrinsics_paths = data_utils.read_paths(train_intrinsics_path)
    else:
        train_intrinsics_paths = [None] * n_train_sample

    if train_ground_truth_path is not None:
        train_ground_truth_paths = data_utils.read_paths(train_ground_truth_path)
    else:
        train_ground_truth_paths = [None] * n_train_sample

    # Make sure number of paths match number of training sample
    input_paths = [
        train_images_paths,
        train_sparse_depth_paths,
        train_intrinsics_paths,
        train_ground_truth_paths,
    ]

    for paths in input_paths:
        assert len(paths) == n_train_sample

    if n_train_sample_limit > 0:
        train_images_paths = train_images_paths[:n_train_sample_limit]
        train_sparse_depth_paths = train_sparse_depth_paths[:n_train_sample_limit]
        train_intrinsics_paths = train_intrinsics_paths[:n_train_sample_limit]
        train_ground_truth_paths = train_ground_truth_paths[:n_train_sample_limit]
        n_train_sample = len(train_images_paths)

    # Set up training dataloader
    n_train_step = \
        np.ceil(
            learning_schedule[-1] *
            np.floor(n_train_sample / n_batch).astype(np.int32) /
            n_step_per_gradient_accumulation).astype(np.int32)

    if supervision_type == 'unsupervised':
        train_dataloader = torch.utils.data.DataLoader(
            datasets.DepthCompletionMonocularTrainingDataset(
                images_paths=train_images_paths,
                sparse_depth_paths=train_sparse_depth_paths,
                intrinsics_paths=train_intrinsics_paths,
                random_crop_shape=(n_height, n_width),
                random_crop_type=augmentation_random_crop_type),
            batch_size=n_batch,
            shuffle=True,
            num_workers=n_thread,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=train_generator)
    elif supervision_type == 'supervised':
        train_dataloader = torch.utils.data.DataLoader(
            datasets.DepthCompletionSupervisedTrainingDataset(
                image_paths=train_images_paths,
                sparse_depth_paths=train_sparse_depth_paths,
                intrinsics_paths=train_intrinsics_paths,
                ground_truth_paths=train_ground_truth_paths,
                random_crop_shape=(n_height, n_width),
                random_crop_type=augmentation_random_crop_type),
            batch_size=n_batch,
            shuffle=True,
            num_workers=n_thread,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=train_generator)
    else:
        raise ValueError('Unsupported supervision type: {}'.format(supervision_type))

    # Set up normalization
    train_transforms_normalization = Transforms(
        normalized_image_range=normalized_image_range)

    # Set up data augmentations
    train_transforms_geometric = Transforms(
        random_flip_type=augmentation_random_flip_type,
        random_rotate_max=augmentation_random_rotate_max,
        random_crop_and_pad=augmentation_random_crop_and_pad,
        random_resize_to_shape=augmentation_random_resize_to_shape,
        random_resize_and_pad=augmentation_random_resize_and_pad,
        random_resize_and_crop=augmentation_random_resize_and_crop)

    train_transforms_crop_to_shape = Transforms(
        random_crop_to_shape=augmentation_random_crop_to_shape)

    train_transforms_photometric = Transforms(
        random_brightness=augmentation_random_brightness,
        random_contrast=augmentation_random_contrast,
        random_gamma=augmentation_random_gamma,
        random_hue=augmentation_random_hue,
        random_saturation=augmentation_random_saturation,
        random_gaussian_blur_kernel_size=augmentation_random_gaussian_blur_kernel_size,
        random_gaussian_blur_sigma_range=augmentation_random_gaussian_blur_sigma_range,
        random_noise_type=augmentation_random_noise_type,
        random_noise_spread=augmentation_random_noise_spread,
        random_remove_patch_percent_range=augmentation_random_remove_patch_percent_range_image,
        random_remove_patch_size=augmentation_random_remove_patch_size_image)

    train_transforms_point_cloud = Transforms(
        random_remove_patch_percent_range=augmentation_random_remove_patch_percent_range_depth,
        random_remove_patch_size=augmentation_random_remove_patch_size_depth)

    # Load validation data if it is available
    is_available_validation = \
        val_image_path is not None and \
        val_sparse_depth_path is not None and \
        val_intrinsics_path is not None and \
        val_ground_truth_path is not None

    if is_available_validation:
        val_image_paths = data_utils.read_paths(val_image_path)
        val_sparse_depth_paths = data_utils.read_paths(val_sparse_depth_path)
        val_intrinsics_paths = data_utils.read_paths(val_intrinsics_path)
        val_ground_truth_paths = data_utils.read_paths(val_ground_truth_path)

        n_val_sample = len(val_image_paths)

        for paths in [val_sparse_depth_paths, val_intrinsics_paths, val_ground_truth_paths]:
            assert len(paths) == n_val_sample

        val_dataloader = torch.utils.data.DataLoader(
            datasets.DepthCompletionInferenceDataset(
                image_paths=val_image_paths,
                sparse_depth_paths=val_sparse_depth_paths,
                intrinsics_paths=val_intrinsics_paths,
                ground_truth_paths=val_ground_truth_paths,
                load_image_triplets=False,
                random_crop_shape=(n_height, n_width)
                if 'partition_attention' in model_name and
                'ablation' in model_name else None,
                random_crop_type=['none']),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=validation_generator)

        val_transforms = Transforms(
            normalized_image_range=normalized_image_range)

    '''
    Set up the model
    '''
    # Build depth completion network
    depth_completion_model = DepthCompletionModel(
        model_name=model_name,
        network_modules=network_modules,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        n_iteration=n_iteration,
        window_size=window_size,
        n_self_attention=n_self_attention,
        n_shift=n_shift,
        use_outlier_rejection=use_outlier_rejection,
        device=device,
        detach_scaleless_depth=detach_scaleless_depth,
        n_propagation_iterations=n_propagation_iterations,
        n_scale_refinement=n_scale_refinement,
        encoder_type=encoder_type,
        use_sparse_depth_features=use_sparse_depth_features,
        fixed_geometry_stage=fixed_geometry_stage,
        geometry_restore_path=geometry_restore_path)

    parameters_depth_model = depth_completion_model.parameters_depth()

    if 'unsupervised' in supervision_type:
        parameters_pose_model = depth_completion_model.parameters_pose()
    else:
        parameters_pose_model = []

    depth_completion_model.train()

    depth_debug = None
    if debug_depth:
        from depth_completion_debug import DepthCompletionDebug
        depth_debug = DepthCompletionDebug(
            depth_completion_model,
            checkpoint_path,
            interval=debug_interval,
            max_samples=debug_max_samples)
        depth_debug.provenance(
            settings=dict(
                model_name=model_name, supervision_type=supervision_type,
                fixed_geometry_stage=fixed_geometry_stage if is_fixed_geometry else None,
                geometry_restore_path=geometry_restore_path,
                geometry_reference='one RGB-only full-resolution Z resized without renormalization'
                if is_fixed_geometry else 'independently updated geometry at each level',
                encoder_type=encoder_type,
                use_sparse_depth_features=use_sparse_depth_features,
                encoder_input='rgb_sparse' if use_sparse_depth_features else 'rgb_only',
                decoder_fusion_uses_sparse_depth=use_sparse_depth_features,
                seed=seed, data_order_seed=seed, validation_loader_seed=seed + 1,
                sparse_scale_coordinates='current_resolution_grid_pixel',
                detach_scaleless_depth=detach_scaleless_depth,
                supervise_outlier_rejection=supervise_outlier_rejection,
                rejection_supervision_active=rejection_supervision_active,
                rejection_loss_active=rejection_loss_active,
                rejection_loss_type=rejection_loss_type,
                rejection_supervision_levels=list(rejection_supervision_levels),
                rejection_keep_weight=rejection_keep_weight,
                rejection_resolution_weights=[1.0 / (4 ** r) for r in range(5)],
                rejection_input='normalized coarse features, current-grid depth, sparse depth, y/x cell offsets',
                n_propagation_iterations=list(n_propagation_iterations),
                n_scale_refinement=n_scale_refinement,
                w_si=w_losses.get('w_si', 1.0),
                w_log_l1=w_losses.get('w_log_l1', 1.0),
                rejection_keep_tolerance=rejection_keep_tolerance,
                rejection_reject_tolerance=rejection_reject_tolerance,
                w_rejection=w_losses.get('w_rejection', 0.1),
                w_rejection_l1=w_rejection_l1,
                rejection_loss_weight=rejection_loss_weight,
                l1_regularized_mask='keep',
                w_losses=dict(w_losses), learning_rates=learning_rates,
                learning_schedule=learning_schedule, restore_paths=restore_paths,
                accumulation=n_step_per_gradient_accumulation,
                train_shape=[n_height, n_width], batch_size=n_batch,
                min_evaluate_depth=min_evaluate_depth, max_evaluate_depth=max_evaluate_depth,
                normalized_image_range=normalized_image_range,
                debug_interval=debug_interval, debug_max_samples=debug_max_samples),
            datasets=dict(train=train_dataloader.dataset,
                          validation=val_dataloader.dataset if is_available_validation else None))

    # Set up tensorboard summary writers
    train_summary_writer = SummaryWriter(event_path + '-train')
    val_summary_writer = SummaryWriter(event_path + '-val')

    '''
    Log input paths
    '''
    log('Training input paths:', log_path)
    train_input_paths = [
        train_images_path,
        train_sparse_depth_path,
        train_intrinsics_path,
        train_ground_truth_path
    ]

    for path in train_input_paths:
        if path is not None:
            log(path, log_path)
    log('', log_path)

    log('Validation input paths:', log_path)
    val_input_paths = [
        val_image_path,
        val_sparse_depth_path,
        val_intrinsics_path,
        val_ground_truth_path
    ]
    for path in val_input_paths:
        if path is not None:
            log(path, log_path)
    log('', log_path)

    log_input_settings(
        log_path,
        # Batch settings
        n_batch=n_batch,
        n_height=n_height,
        n_width=n_width,
        # Input settings
        input_channels_image=input_channels_image,
        input_channels_depth=input_channels_depth,
        normalized_image_range=normalized_image_range)

    log_network_settings(
        log_path,
        # Depth network settings
        model_name=model_name,
        network_modules=network_modules,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        use_outlier_rejection=use_outlier_rejection,
        # Weight settings
        parameters_depth_model=parameters_depth_model,
        parameters_pose_model=parameters_pose_model)

    if 'scale_propagation' in model_name:
        if is_fixed_geometry:
            log('fixed_geometry_stage={}  geometry_restore_path={}'.format(
                fixed_geometry_stage, geometry_restore_path), log_path)
            log('geometry stage: full-resolution SI only; scale stage: multiscale log L1 only', log_path)
        log('seed={}  use_sparse_depth_features={}'.format(
            seed, use_sparse_depth_features), log_path)
        log('detach_scaleless_depth={}'.format(detach_scaleless_depth), log_path)
        log('encoder_type={}'.format(encoder_type), log_path)
        log('w_supervised={}  w_si={}  w_log_l1={}'.format(
            w_losses.get('w_supervised', 1.0),
            w_losses.get('w_si', 1.0), w_losses.get('w_log_l1', 1.0)), log_path)
        log('n_propagation_iterations={}  n_scale_refinement={}'.format(
            n_propagation_iterations, n_scale_refinement), log_path)
        log('supervise_outlier_rejection={}  effective={}  w_rejection={}'.format(
            supervise_outlier_rejection, rejection_loss_active,
            w_losses.get('w_rejection', 0.1)), log_path)
        log('rejection_loss_type={}'.format(rejection_loss_type), log_path)
        log('rejection_loss_weight={}  l1_regularized_mask=keep'.format(
            rejection_loss_weight), log_path)
        log('rejection_keep_weight={}  rejection_resolution_weights={}'.format(
            rejection_keep_weight, [1.0 / (4 ** r) for r in range(5)]), log_path)
        if rejection_loss_active:
            log('rejection_supervision_levels={} (0=full, 4=bottom)'.format(
                rejection_supervision_levels), log_path)
            log('rejection_keep_tolerance={}  rejection_reject_tolerance={}'.format(
                rejection_keep_tolerance, rejection_reject_tolerance), log_path)

    log_training_settings(
        log_path,
        # Training settings
        n_batch=n_batch,
        n_train_sample=n_train_sample,
        n_train_step=n_train_step,
        learning_rates=learning_rates,
        learning_schedule=learning_schedule,
        n_step_per_gradient_accumulation=n_step_per_gradient_accumulation,
        # Augmentation settings
        augmentation_probabilities=augmentation_probabilities,
        augmentation_schedule=augmentation_schedule,
        # Photometric data augmentations
        augmentation_random_brightness=augmentation_random_brightness,
        augmentation_random_contrast=augmentation_random_contrast,
        augmentation_random_gamma=augmentation_random_gamma,
        augmentation_random_hue=augmentation_random_hue,
        augmentation_random_saturation=augmentation_random_saturation,
        augmentation_random_gaussian_blur_kernel_size=augmentation_random_gaussian_blur_kernel_size,
        augmentation_random_gaussian_blur_sigma_range=augmentation_random_gaussian_blur_sigma_range,
        augmentation_random_noise_type=augmentation_random_noise_type,
        augmentation_random_noise_spread=augmentation_random_noise_spread,
        # Geometric data augmentations
        augmentation_padding_mode=augmentation_padding_mode,
        augmentation_random_crop_type=augmentation_random_crop_type,
        augmentation_random_crop_to_shape=augmentation_random_crop_to_shape,
        augmentation_random_flip_type=augmentation_random_flip_type,
        augmentation_random_rotate_max=augmentation_random_rotate_max,
        augmentation_random_crop_and_pad=augmentation_random_crop_and_pad,
        augmentation_random_resize_to_shape=augmentation_random_resize_to_shape,
        augmentation_random_resize_and_pad=augmentation_random_resize_and_pad,
        augmentation_random_resize_and_crop=augmentation_random_resize_and_crop,
        # Occlusion data augmentations
        augmentation_random_remove_patch_percent_range_image=augmentation_random_remove_patch_percent_range_image,
        augmentation_random_remove_patch_size_image=augmentation_random_remove_patch_size_image,
        augmentation_random_remove_patch_percent_range_depth=augmentation_random_remove_patch_percent_range_depth,
        augmentation_random_remove_patch_size_depth=augmentation_random_remove_patch_size_depth)

    log_loss_func_settings(
        log_path,
        # Loss function settings
        supervision_type=supervision_type,
        w_losses=w_losses,
        w_weight_decay_depth=w_weight_decay_depth,
        w_weight_decay_pose=w_weight_decay_pose)

    log_evaluation_settings(
        log_path,
        min_evaluate_depth=min_evaluate_depth,
        max_evaluate_depth=max_evaluate_depth)

    log_system_settings(
        log_path,
        # Checkpoint settings
        checkpoint_path=checkpoint_path,
        n_step_per_checkpoint=n_step_per_checkpoint,
        summary_event_path=event_path,
        n_step_per_summary=n_step_per_summary,
        n_image_per_summary=n_image_per_summary,
        start_step_validation=start_step_validation,
        restore_paths=restore_paths,
        # Hardware settings
        device=device,
        n_thread=n_thread)

    '''
    Train model
    '''
    # Initialize optimizer with starting learning rate
    learning_schedule_pos = 0

    learning_rate = learning_rates[0]

    augmentation_schedule_pos = 0
    augmentation_probability = augmentation_probabilities[0]

    optimizer_depth = torch.optim.Adam([
        {
            'params' : parameters_depth_model,
            'weight_decay' : w_weight_decay_depth
        }],
        lr=learning_rate)

    if 'unsupervised' in supervision_type:
        optimizer_pose = torch.optim.Adam([
            {
                'params' : parameters_pose_model,
                'weight_decay' : w_weight_decay_pose
            }],
            lr=learning_rate)
    else:
        optimizer_pose = None

    # Split along batch across multiple GPUs
    if torch.cuda.device_count() > 1:
        depth_completion_model.data_parallel()

    # Start training
    train_step = 0
    micro_step = 0
    last_validation_step = -1

    if len(restore_paths) > 0:
        try:
            train_step, optimizer_depth, optimizer_pose = depth_completion_model.restore_model(
                restore_paths,
                optimizer_depth=optimizer_depth,
                optimizer_pose=optimizer_pose)
        except Exception:
            print('Failed to restore optimizer for depth network: Ignoring...')
            train_step, _ = depth_completion_model.restore_model(
                restore_paths)

        for g in optimizer_depth.param_groups:
            g['lr'] = learning_rate

        n_train_step = n_train_step + train_step

    time_start = time.time()

    optimizer_depth.zero_grad()
    if 'unsupervised' in supervision_type:
        optimizer_pose.zero_grad()

    # Define padding for input image, sparse depth, validity map
    padding_modes = [augmentation_padding_mode, 'constant', 'constant']

    # Map interpolation mode names to enums
    interpolation_modes = \
        train_transforms_geometric.map_interpolation_mode_names_to_enums(['nearest'])

    erosion2d = Erosion2d(device=device)

    log('Begin training...', log_path)
    for epoch in range(1, learning_schedule[-1] + 1):

        # Set learning rate schedule
        if epoch > learning_schedule[learning_schedule_pos]:
            learning_schedule_pos = learning_schedule_pos + 1
            learning_rate = learning_rates[learning_schedule_pos]

            # Update optimizer learning rates for depth network
            for g in optimizer_depth.param_groups:
                g['lr'] = learning_rate

            if 'unsupervised' in supervision_type:
                # Update optimizer learning rates for pose network
                for g in optimizer_pose.param_groups:
                    g['lr'] = learning_rate

        # Set augmentation schedule
        if -1 not in augmentation_schedule and epoch > augmentation_schedule[augmentation_schedule_pos]:
            augmentation_schedule_pos = augmentation_schedule_pos + 1
            augmentation_probability = augmentation_probabilities[augmentation_schedule_pos]

        for inputs in train_dataloader:

            micro_step = micro_step + 1
            is_optimizer_step = \
                micro_step % n_step_per_gradient_accumulation == 0
            next_train_step = train_step + 1
            is_summary_step = \
                is_optimizer_step and \
                next_train_step % n_step_per_summary == 0

            is_debug_step = depth_debug is not None and is_optimizer_step and \
                (next_train_step == 1 or next_train_step % depth_debug.interval == 0)
            if depth_debug is not None:
                depth_debug.begin('train', next_train_step, active=is_debug_step)

            # Fetch data
            inputs = [
                in_.to(device) for in_ in inputs
            ]

            if supervision_type == 'supervised':
                image0, \
                    sparse_depth0, \
                    intrinsics, \
                    ground_truth0 = inputs

                image1 = image0.detach().clone()
                image2 = image0.detach().clone()
            elif supervision_type == 'unsupervised':
                image0, \
                    image1, \
                    image2, \
                    sparse_depth0, \
                    intrinsics = inputs

                ground_truth0 = None
            else:
                raise ValueError('Unsupported supervision type: {}'.format(supervision_type))

            # Validity map is where sparse depth is available
            if is_debug_step:
                depth_debug.inputs('before_augmentation', image0, sparse_depth0, ground_truth0)

            validity_map0 = torch.where(
                sparse_depth0 > 0,
                torch.ones_like(sparse_depth0),
                sparse_depth0)

            # Perform geometric augmentation i.e. crop, flip, etc. on the input image
            crop_inputs = [image0, image1, image2, sparse_depth0, validity_map0]
            if rejection_supervision_active:
                crop_inputs.append(ground_truth0)
            cropped, [intrinsics], _ = train_transforms_crop_to_shape.transform(
                images_arr=crop_inputs,
                intrinsics_arr=[intrinsics],
                interpolation_modes=interpolation_modes,
                random_transform_probability=augmentation_probability)

            image0, image1, image2, sparse_depth0, validity_map0 = cropped[:5]
            if rejection_supervision_active:
                ground_truth0 = cropped[5]
            geometric_inputs = [image0, sparse_depth0, validity_map0]
            if rejection_supervision_active:
                # GT follows the exact sampled transform with nearest sampling
                # and zero padding. It is used only for labels, never inference.
                geometric_inputs.append(ground_truth0)
            geometric_outputs, \
                [input_intrinsics], \
                transform_performed_geometric = train_transforms_geometric.transform(
                    images_arr=geometric_inputs,
                    intrinsics_arr=[intrinsics],
                    padding_modes=padding_modes,
                    random_transform_probability=augmentation_probability)
            input_image0, input_sparse_depth0, input_validity_map0 = geometric_outputs[:3]

            do_erode = 'random_resize_to_shape' in transform_performed_geometric or \
                'random_resize_and_crop' in transform_performed_geometric

            if do_erode:

                if 'random_resize_and_crop' in transform_performed_geometric:
                    do_resize, _, _, _, _ = transform_performed_geometric['random_resize_and_crop']
                else:
                    do_resize = torch.ones(n_batch, device=device).bool()

                input_sparse_depth0_arr = []
                input_validity_map0_arr = []

                for batch_index in range(n_batch):
                    z = torch.unsqueeze(
                        input_sparse_depth0[batch_index, ...], dim=0)
                    v = torch.unsqueeze(
                        input_validity_map0[batch_index, ...], dim=0)

                    if do_resize[batch_index]:
                        v = erosion2d(v)
                        z = z * v

                        v = torch.where(
                            z > 0,
                            torch.ones_like(z),
                            torch.zeros_like(z))

                    input_sparse_depth0_arr.append(z)
                    input_validity_map0_arr.append(v)

                input_sparse_depth0 = torch.cat(input_sparse_depth0_arr, dim=0)
                input_validity_map0 = torch.cat(input_validity_map0_arr, dim=0)

            # Perform point removal from sparse depth
            [input_sparse_depth0], _ = train_transforms_point_cloud.transform(
                images_arr=[input_sparse_depth0],
                random_transform_probability=augmentation_probability)

            # Perform photometric augmentation i.e. masking, brightness, contrast, etc. on the input image
            [input_image0], _ = train_transforms_photometric.transform(
                images_arr=[input_image0],
                random_transform_probability=augmentation_probability)

            # Normalize all images
            [image0, image1, image2, input_image0], _ = \
                train_transforms_normalization.transform(
                    images_arr=[image0, image1, image2, input_image0])

            # Forward through the network
            # Inputs: augmented image, augmented sparse depth map, original (but aligned) validity map
            if depth_debug is not None:
                depth_debug.record_training_batch(input_image0.shape[0], epoch, micro_step)
            if is_debug_step:
                depth_debug.inputs('network_input', input_image0, input_sparse_depth0)
                depth_debug.sparse_levels(input_sparse_depth0)
                depth_debug.emit('augmentation', operations=list(transform_performed_geometric),
                                 epoch=epoch, micro_step=micro_step,
                                 accumulation=n_step_per_gradient_accumulation)

            output_depth0 = depth_completion_model.forward_depth(
                image=input_image0,
                sparse_depth=input_sparse_depth0,
                validity_map=input_validity_map0,
                intrinsics=input_intrinsics,
                return_all_outputs=True,
                return_rejection_outputs=rejection_supervision_active,
                return_rejection_masks=rejection_sparsity_active)
            if rejection_loss_active:
                output_depth0, rejection_outputs = output_depth0
                # Match selected sparse pixels to GT in network coordinates,
                # before reverse_transform changes the depth-map coordinates.
                rejection_target = geometric_outputs[3] if rejection_supervision_active else None
                rejection_loss, rejection_info = depth_completion_model.compute_loss_rejection(
                    rejection_target, input_sparse_depth0, rejection_outputs,
                    rejection_keep_tolerance, rejection_reject_tolerance,
                    rejection_supervision_levels, rejection_loss_type, rejection_keep_weight)
                if is_debug_step and rejection_supervision_active:
                    depth_debug.rejection_training(
                        rejection_outputs, input_sparse_depth0, rejection_target)
                if is_debug_step and rejection_sparsity_active:
                    depth_debug.emit('rejection_sparsity',
                                     levels=list(rejection_supervision_levels),
                                     loss=float(rejection_loss.detach()),
                                     loss_weight=rejection_loss_weight,
                                     regularized_mask='keep',
                                     fractions={k: float(v) for k, v in rejection_info.items()
                                                if k.startswith('rejection_kept_fraction_')},
                                     rejected_fractions={k: float(v) for k, v in rejection_info.items()
                                                        if k.startswith('rejection_rejected_fraction_')},
                                     mask_source='actual straight-through masks used for scale injection')

            if is_summary_step:
                output_depth0_initial = output_depth0[0].detach().clone()

            if 'unsupervised' in supervision_type:
                pose0to1 = depth_completion_model.forward_pose(image0, image1)
                pose0to2 = depth_completion_model.forward_pose(image0, image2)
            else:
                pose0to1 = None
                pose0to2 = None

            output_depth0 = train_transforms_geometric.reverse_transform(
                images_arr=output_depth0,
                transform_performed=transform_performed_geometric,
                padding_modes=[padding_modes[0]])

            # Compute loss function
            w_losses['epoch'] = epoch

            loss, loss_info = depth_completion_model.compute_loss(
                image0=image0,
                image1=image1,
                image2=image2,
                output_depth0=output_depth0,
                sparse_depth0=sparse_depth0,
                validity_map0=validity_map0,
                ground_truth0=ground_truth0,
                intrinsics=intrinsics,
                pose0to1=pose0to1,
                pose0to2=pose0to2,
                supervision_type=supervision_type,
                w_losses=w_losses)

            if rejection_loss_active:
                weighted_rejection = rejection_loss_weight * rejection_loss
                loss = loss + weighted_rejection
                loss_info.update(rejection_info)
                loss_info['loss_rejection_weighted'] = weighted_rejection
                loss_info['loss'] = loss

            if is_debug_step:
                depth_debug.training_prediction(
                    output_depth0[0], ground_truth0, sparse_depth0, loss_info)
                depth_debug.gradient_objectives(loss_info)

            # Compute gradient and backpropagate

            parameter = next(
                p for p in depth_completion_model.parameters_depth()
                if p.requires_grad)

            accumulated_loss = loss / n_step_per_gradient_accumulation
            accumulated_loss.backward()

            if is_optimizer_step:
                if is_debug_step:
                    depth_debug.before_step(optimizer_depth)
                optimizer_depth.step()
                if is_debug_step:
                    depth_debug.after_step()

                if 'unsupervised' in supervision_type:
                    optimizer_pose.step()

                optimizer_depth.zero_grad()

                if 'unsupervised' in supervision_type:
                    optimizer_pose.zero_grad()

                train_step = next_train_step

            if is_summary_step:

                if 'unsupervised' in supervision_type:
                    image1to0 = loss_info.pop('image1to0')
                    image2to0 = loss_info.pop('image2to0')
                else:
                    image1to0 = image0
                    image2to0 = image0

                # Log summary
                depth_completion_model.log_summary(
                    summary_writer=train_summary_writer,
                    tag='train_inputs',
                    step=train_step,
                    image0=input_image0,
                    output_depth0=output_depth0_initial.detach().clone(),
                    sparse_depth0=input_sparse_depth0,
                    validity_map0=input_validity_map0,
                    n_image_per_summary=min(n_batch, n_image_per_summary))

                depth_completion_model.log_summary(
                    summary_writer=train_summary_writer,
                    tag='train',
                    step=train_step,
                    image0=image0,
                    image1to0=image1to0.detach().clone(),
                    image2to0=image2to0.detach().clone(),
                    output_depth0=output_depth0[0].detach().clone(),
                    sparse_depth0=sparse_depth0,
                    validity_map0=validity_map0,
                    ground_truth0=ground_truth0,
                    pose0to1=pose0to1,
                    pose0to2=pose0to2,
                    scalars=loss_info,
                    n_image_per_summary=min(n_batch, n_image_per_summary),
                    scale_outputs=depth_completion_model.scale_outputs(output_depth0))

            # Log results and save checkpoints
            if is_optimizer_step and \
                    (train_step % n_step_per_checkpoint) == 0:
                time_elapse = (time.time() - time_start) / 3600
                time_remain = (n_train_step - train_step) * time_elapse / train_step

                log('Step={:6}/{}  Loss={:.5f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
                    train_step, n_train_step, loss.item(), time_elapse, time_remain),
                    log_path)

                if train_step >= start_step_validation and is_available_validation:
                    # Switch to validation mode
                    depth_completion_model.eval()

                    with torch.no_grad():
                        # Perform validation
                        best_results = validate(
                            depth_model=depth_completion_model,
                            dataloader=val_dataloader,
                            transforms=val_transforms,
                            step=train_step,
                            best_results=best_results,
                            min_evaluate_depth=min_evaluate_depth,
                            max_evaluate_depth=max_evaluate_depth,
                            device=device,
                            summary_writer=val_summary_writer,
                            n_image_per_summary=n_image_per_summary,
                            log_path=log_path,
                            depth_debug=depth_debug)
                        last_validation_step = train_step

                    # Switch back to training
                    depth_completion_model.train()

                # Save checkpoints
                depth_completion_model.save_model(
                    checkpoint_dirpath.format(train_step),
                    train_step,
                    optimizer_depth,
                    optimizer_pose)

    n_remaining_accumulation = \
        micro_step % n_step_per_gradient_accumulation

    if n_remaining_accumulation != 0:
        # The losses in the final incomplete group were divided by the full
        # accumulation count. Restore their average over the actual count.
        gradient_scale = \
            n_step_per_gradient_accumulation / n_remaining_accumulation
        for parameter in depth_completion_model.parameters_depth():
            if parameter.grad is not None:
                parameter.grad.mul_(gradient_scale)

        if 'unsupervised' in supervision_type:
            for parameter in depth_completion_model.parameters_pose():
                if parameter.grad is not None:
                    parameter.grad.mul_(gradient_scale)

        optimizer_depth.step()

        if 'unsupervised' in supervision_type:
            optimizer_pose.step()

        optimizer_depth.zero_grad()

        if 'unsupervised' in supervision_type:
            optimizer_pose.zero_grad()

        train_step = train_step + 1

    # Perform validation for final step
    if is_available_validation and last_validation_step != train_step:
        depth_completion_model.eval()

        with torch.no_grad():
            best_results = validate(
                depth_model=depth_completion_model,
                dataloader=val_dataloader,
                transforms=val_transforms,
                step=train_step,
                best_results=best_results,
                min_evaluate_depth=min_evaluate_depth,
                max_evaluate_depth=max_evaluate_depth,
                device=device,
                summary_writer=val_summary_writer,
                n_image_per_summary=n_image_per_summary,
                log_path=log_path,
                depth_debug=depth_debug)

    # Save checkpoints
    depth_completion_model.save_model(
        checkpoint_dirpath.format(train_step),
        train_step,
        optimizer_depth,
        optimizer_pose)

    if depth_debug is not None:
        depth_debug.close()

def validate(depth_model,
             dataloader,
             transforms,
             step,
             best_results,
             min_evaluate_depth,
             max_evaluate_depth,
             device,
             summary_writer,
             n_image_per_summary=4,
             n_interval_per_summary=250,
             log_path=None,
             depth_debug=None):

    n_sample = len(dataloader)
    mae = np.zeros(n_sample)
    rmse = np.zeros(n_sample)
    imae = np.zeros(n_sample)
    irmse = np.zeros(n_sample)

    image_summary = []
    output_depth_summary = []
    sparse_depth_summary = []
    validity_map_summary = []
    ground_truth_summary = []
    scale_output_summary = dict(
        scaleless_depths=[[] for _ in range(5)],
        scale_maps=[[] for _ in range(5)])

    if depth_debug is not None:
        depth_debug.start_validation(step)

    for idx, inputs in enumerate(dataloader):

        if depth_debug is not None:
            depth_debug.begin('validation', step, sample=idx,
                              active=idx in depth_debug.fixed_indices)

        # Move inputs to device
        inputs = [
            in_.to(device) for in_ in inputs
        ]

        image, sparse_depth, intrinsics, ground_truth = inputs

        with torch.no_grad():
            # Validity map is where sparse depth is available
            validity_map = torch.where(
                sparse_depth > 0,
                torch.ones_like(sparse_depth),
                sparse_depth)

            [image], _ = transforms.transform(
                images_arr=[image],
                random_transform_probability=0.0)

            # Collect explicit auxiliary maps only for displayed scale-model
            # samples; all other validation calls keep the single-depth API.
            collect_scale = (
                summary_writer is not None and idx % n_interval_per_summary == 0 and
                'scale_propagation' in depth_model.model_name)
            # Forward through network
            output_depth = depth_model.forward_depth(
                image=image,
                sparse_depth=sparse_depth,
                validity_map=validity_map,
                intrinsics=intrinsics,
                return_all_outputs=collect_scale)
            if collect_scale:
                maps = depth_model.scale_outputs(output_depth)
                for name, levels in maps.items():
                    for collected, value in zip(scale_output_summary[name], levels):
                        collected.append(value.detach())
                output_depth = output_depth[0]

        if depth_debug is not None:
            if depth_debug.active:
                depth_debug.inputs('validation_input', image, sparse_depth, ground_truth)
                depth_debug.sparse_levels(sparse_depth)
            depth_debug.validation_sample(
                idx, image, sparse_depth, ground_truth, output_depth,
                min_evaluate_depth, max_evaluate_depth, dataloader.dataset)

        if (idx % n_interval_per_summary) == 0 and summary_writer is not None:
            image_summary.append(image)
            output_depth_summary.append(output_depth)
            sparse_depth_summary.append(sparse_depth)
            validity_map_summary.append(validity_map)
            ground_truth_summary.append(ground_truth)

        # Convert to numpy to validate
        output_depth = np.squeeze(output_depth.cpu().numpy())
        ground_truth = np.squeeze(ground_truth.cpu().numpy())

        validity_map = np.where(ground_truth > 0, 1, 0)

        # Select valid regions to evaluate
        validity_mask = np.where(validity_map > 0, 1, 0)
        min_max_mask = np.logical_and(
            ground_truth > min_evaluate_depth,
            ground_truth < max_evaluate_depth)
        mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)

        output_depth = output_depth[mask]
        ground_truth = ground_truth[mask]

        # Compute validation metrics
        mae[idx] = eval_utils.mean_abs_err(1000.0 * output_depth, 1000.0 * ground_truth)
        rmse[idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth, 1000.0 * ground_truth)
        imae[idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth, 0.001 * ground_truth)
        irmse[idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth, 0.001 * ground_truth)

    if depth_debug is not None:
        depth_debug.finish_validation()

    # Compute mean metrics
    mae   = np.mean(mae)
    rmse  = np.mean(rmse)
    imae  = np.mean(imae)
    irmse = np.mean(irmse)

    # Log to tensorboard
    if summary_writer is not None:
        depth_model.log_summary(
            summary_writer=summary_writer,
            tag='eval',
            step=step,
            image0=torch.cat(image_summary, dim=0),
            output_depth0=torch.cat(output_depth_summary, dim=0),
            sparse_depth0=torch.cat(sparse_depth_summary, dim=0),
            validity_map0=torch.cat(validity_map_summary, dim=0),
            ground_truth0=torch.cat(ground_truth_summary, dim=0),
            scalars={'mae' : mae, 'rmse' : rmse, 'imae' : imae, 'irmse': irmse},
            n_image_per_summary=n_image_per_summary,
            scale_outputs={
                name: [torch.cat(values, dim=0) for values in levels]
                for name, levels in scale_output_summary.items()
            } if scale_output_summary['scale_maps'][0] else None)

    # Print validation results to console
    log('Validation results:', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        step, mae, rmse, imae, irmse),
        log_path)

    n_improve = 0
    if np.round(mae, 2) <= np.round(best_results['mae'], 2):
        n_improve = n_improve + 1
    if np.round(rmse, 2) <= np.round(best_results['rmse'], 2):
        n_improve = n_improve + 1
    if np.round(imae, 2) <= np.round(best_results['imae'], 2):
        n_improve = n_improve + 1
    if np.round(irmse, 2) <= np.round(best_results['irmse'], 2):
        n_improve = n_improve + 1

    if n_improve > 2:
        best_results['step'] = step
        best_results['mae'] = mae
        best_results['rmse'] = rmse
        best_results['imae'] = imae
        best_results['irmse'] = irmse

    log('Best results:', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        best_results['step'],
        best_results['mae'],
        best_results['rmse'],
        best_results['imae'],
        best_results['irmse']), log_path)

    return best_results


def run(image_path,
        sparse_depth_path,
        intrinsics_path,
        ground_truth_path,
        # Restore path settings
        restore_paths,
        # Input settings
        input_channels_image,
        input_channels_depth,
        normalized_image_range,
        # Depth network settings
        model_name,
        network_modules,
        min_predict_depth,
        max_predict_depth,
        n_iteration,
        window_size,
        n_self_attention,
        n_shift,
        use_outlier_rejection,
        # Evaluation settings
        min_evaluate_depth,
        max_evaluate_depth,
        # Output settings
        output_path,
        save_outputs,
        keep_input_filenames,
        # Hardware settings
        device,
        encoder_type='conv',
        use_sparse_depth_features=False,
        n_scale_refinement=2,
        fixed_geometry_stage='scale',
        fixed_geometry_mode='hierarchical',
        geometry_restore_path=None,
        n_propagation_iterations=(2, 4, 6, 8, 12)):

    if 'fixed_geometry_scale_propagation' in model_name:
        use_outlier_rejection = False
        use_sparse_depth_features = False
        n_scale_refinement = 0

    # Select device to run on
    if device == 'cuda' or device == 'gpu':
        device = torch.device('cuda')
    elif device == 'cpu':
        device = torch.device('cpu')
    else:
        raise ValueError('Unsupported device: {}'.format(device))

    '''
    Set up output paths
    '''
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    log_path = os.path.join(output_path, 'results.txt')
    output_dirpath = os.path.join(output_path, 'outputs')

    if save_outputs:
        # Create output directories
        image_dirpath = os.path.join(output_dirpath, 'image')
        output_depth_dirpath = os.path.join(output_dirpath, 'output_depth')
        sparse_depth_dirpath = os.path.join(output_dirpath, 'sparse_depth')
        ground_truth_dirpath = os.path.join(output_dirpath, 'ground_truth')

        dirpaths = [
            output_dirpath,
            image_dirpath,
            output_depth_dirpath,
            sparse_depth_dirpath,
            ground_truth_dirpath
        ]

        for dirpath in dirpaths:
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)

    '''
    Load input paths and set up dataloader
    '''
    image_paths = data_utils.read_paths(image_path)
    sparse_depth_paths = data_utils.read_paths(sparse_depth_path)
    intrinsics_paths = data_utils.read_paths(intrinsics_path)

    is_available_ground_truth = False

    if ground_truth_path is not None and ground_truth_path != '':
        is_available_ground_truth = True
        ground_truth_paths = data_utils.read_paths(ground_truth_path)
    else:
        ground_truth_paths = None

    n_sample = len(image_paths)

    input_paths = [
        image_paths,
        sparse_depth_paths,
        intrinsics_paths
    ]

    if is_available_ground_truth:
        input_paths.append(ground_truth_paths)

    for paths in input_paths:
        assert n_sample == len(paths)

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(
        datasets.DepthCompletionInferenceDataset(
            image_paths=image_paths,
            sparse_depth_paths=sparse_depth_paths,
            intrinsics_paths=intrinsics_paths,
            ground_truth_paths=ground_truth_paths,
            load_image_triplets=False),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False)

    # Initialize transforms to normalize image
    transforms = Transforms(
        normalized_image_range=normalized_image_range)

    '''
    Set up the model
    '''
    # Build depth completion network
    depth_model = DepthCompletionModel(
        model_name=model_name,
        network_modules=network_modules,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        n_iteration=n_iteration,
        window_size=window_size,
        n_self_attention=n_self_attention,
        n_shift=n_shift,
        use_outlier_rejection=use_outlier_rejection,
        device=device,
        encoder_type=encoder_type,
        use_sparse_depth_features=use_sparse_depth_features,
        n_scale_refinement=n_scale_refinement,
        fixed_geometry_stage=fixed_geometry_stage,
        fixed_geometry_mode=fixed_geometry_mode,
        geometry_restore_path=geometry_restore_path,
        n_propagation_iterations=n_propagation_iterations)

    # Restore model and set to evaluation mode
    depth_model.restore_model(restore_paths)
    depth_model.eval()

    parameters_depth_model = depth_model.parameters_depth()

    '''
    Log input paths
    '''
    log('Input paths:', log_path)
    input_paths = [
        image_path,
        sparse_depth_path,
        intrinsics_path,
    ]

    if is_available_ground_truth:
        input_paths.append(ground_truth_path)

    for path in input_paths:
        log(path, log_path)
    log('', log_path)

    '''
    Log all settings
    '''
    log_input_settings(
        log_path,
        # Input settings
        input_channels_image=input_channels_image,
        input_channels_depth=input_channels_depth,
        normalized_image_range=normalized_image_range)

    log_network_settings(
        log_path,
        # Depth network settings
        model_name=model_name,
        network_modules=network_modules,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        use_outlier_rejection=use_outlier_rejection,
        # Weight settings
        parameters_depth_model=parameters_depth_model)

    if 'fixed_geometry_scale_propagation' in model_name:
        log('fixed_geometry_stage={}  fixed_geometry_mode={}  n_propagation_iterations={}'.format(
            fixed_geometry_stage, fixed_geometry_mode, n_propagation_iterations), log_path)

    log_evaluation_settings(
        log_path,
        min_evaluate_depth=min_evaluate_depth,
        max_evaluate_depth=max_evaluate_depth)

    log_system_settings(
        log_path,
        # Checkpoint settings
        checkpoint_path=output_path,
        restore_paths=restore_paths,
        # Hardware settings
        device=device,
        n_thread=1)

    '''
    Run model
    '''
    # Set up metrics in case groundtruth is available
    mae = np.zeros(n_sample)
    rmse = np.zeros(n_sample)
    imae = np.zeros(n_sample)
    irmse = np.zeros(n_sample)

    time_elapse = 0.0

    for idx, inputs in enumerate(dataloader):

        # Move inputs to device
        inputs = [
            in_.to(device) for in_ in inputs
        ]

        if dataloader.dataset.is_available_ground_truth:
            image, sparse_depth, intrinsics, ground_truth = inputs
        else:
            image, sparse_depth, intrinsics = inputs

        time_start = time.time()

        with torch.no_grad():
            # Validity map is where sparse depth is available
            validity_map = torch.where(
                sparse_depth > 0,
                torch.ones_like(sparse_depth),
                sparse_depth)

            [image], _ = transforms.transform(
                images_arr=[image],
                random_transform_probability=0.0)

            # Forward through network
            output_depth = depth_model.forward_depth(
                image=image,
                sparse_depth=sparse_depth,
                validity_map=validity_map,
                intrinsics=intrinsics,
                return_all_outputs=False)

        time_elapse = time_elapse + (time.time() - time_start)

        # Convert to numpy
        output_depth = np.squeeze(output_depth.detach().cpu().numpy())

        # Save to output
        if save_outputs:
            image = np.transpose(np.squeeze(image.cpu().numpy()), (1, 2, 0))
            sparse_depth = np.squeeze(sparse_depth.cpu().numpy())

            if keep_input_filenames:
                filename = os.path.splitext(os.path.basename(image_paths[idx]))[0] + '.png'
            else:
                filename = '{:010d}.png'.format(idx)

            image_path = os.path.join(image_dirpath, filename)
            image = (255 * image).astype(np.uint8)
            Image.fromarray(image).save(image_path)

            output_depth_path = os.path.join(output_depth_dirpath, filename)
            data_utils.save_depth(output_depth, output_depth_path)

            sparse_depth_path = os.path.join(sparse_depth_dirpath, filename)
            data_utils.save_depth(sparse_depth, sparse_depth_path)

        if is_available_ground_truth:

            ground_truth = np.squeeze(ground_truth.cpu().numpy())
            validity_map = np.where(ground_truth > 0, 1, 0)

            if save_outputs:
                ground_truth_path = os.path.join(ground_truth_dirpath, filename)
                data_utils.save_depth(ground_truth, ground_truth_path)

            validity_mask = np.where(validity_map > 0, 1, 0)
            min_max_mask = np.logical_and(
                ground_truth > min_evaluate_depth,
                ground_truth < max_evaluate_depth)
            mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)

            output_depth = output_depth[mask]
            ground_truth = ground_truth[mask]

            mae[idx] = eval_utils.mean_abs_err(1000.0 * output_depth, 1000.0 * ground_truth)
            rmse[idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth, 1000.0 * ground_truth)
            imae[idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth, 0.001 * ground_truth)
            irmse[idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth, 0.001 * ground_truth)

    # Compute total time elapse in ms
    time_elapse = time_elapse * 1000.0

    if is_available_ground_truth:
        mae_mean   = np.mean(mae)
        rmse_mean  = np.mean(rmse)
        imae_mean  = np.mean(imae)
        irmse_mean = np.mean(irmse)

        mae_std = np.std(mae)
        rmse_std = np.std(rmse)
        imae_std = np.std(imae)
        irmse_std = np.std(irmse)

        # Print evaluation results to console and file
        log('Evaluation results:', log_path)
        log('{:>8}  {:>8}  {:>8}  {:>8}'.format(
            'MAE', 'RMSE', 'iMAE', 'iRMSE'),
            log_path)
        log('{:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
            mae_mean, rmse_mean, imae_mean, irmse_mean),
            log_path)

        log('{:>8}  {:>8}  {:>8}  {:>8}'.format(
            '+/-', '+/-', '+/-', '+/-'),
            log_path)
        log('{:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
            mae_std, rmse_std, imae_std, irmse_std),
            log_path)

    # Log run time
    log('Total time: {:.2f} ms  Average time per sample: {:.2f} ms'.format(
        time_elapse, time_elapse / float(n_sample)))


'''
Helper functions for logging
'''
def log_input_settings(log_path,
                       input_channels_image,
                       input_channels_depth,
                       normalized_image_range,
                       n_batch=None,
                       n_height=None,
                       n_width=None):

    batch_settings_text = ''
    batch_settings_vars = []

    if n_batch is not None:
        batch_settings_text = batch_settings_text + 'n_batch={}'
        batch_settings_vars.append(n_batch)

    batch_settings_text = \
        batch_settings_text + '  ' if len(batch_settings_text) > 0 else batch_settings_text

    if n_height is not None:
        batch_settings_text = batch_settings_text + 'n_height={}'
        batch_settings_vars.append(n_height)

    batch_settings_text = \
        batch_settings_text + '  ' if len(batch_settings_text) > 0 else batch_settings_text

    if n_width is not None:
        batch_settings_text = batch_settings_text + 'n_width={}'
        batch_settings_vars.append(n_width)

    log('Input settings:', log_path)

    if len(batch_settings_vars) > 0:
        log(batch_settings_text.format(*batch_settings_vars),
            log_path)

    log('input_channels_image={}  input_channels_depth={}'.format(
        input_channels_image, input_channels_depth),
        log_path)
    log('normalized_image_range={}'.format(normalized_image_range),
        log_path)

    log('', log_path)

def log_network_settings(log_path,
                         # Depth network settings
                         model_name,
                         network_modules,
                         min_predict_depth,
                         max_predict_depth,
                         use_outlier_rejection=True,
                         # Pose network settings
                         encoder_type_pose=None,
                         rotation_parameterization_pose=None,
                         # Weight settings
                         parameters_depth_model=[],
                         parameters_pose_model=[]):

    # Computer number of parameters
    n_parameter_depth = sum(p.numel() for p in parameters_depth_model)
    n_parameter_pose = sum(p.numel() for p in parameters_pose_model)

    n_parameter = n_parameter_depth + n_parameter_pose

    n_parameter_text = 'n_parameter={}'.format(n_parameter)
    n_parameter_vars = []

    if n_parameter_depth > 0 :
        n_parameter_text = \
            n_parameter_text + '  ' if len(n_parameter_text) > 0 else n_parameter_text

        n_parameter_text = n_parameter_text + 'n_parameter_depth={}'
        n_parameter_vars.append(n_parameter_depth)

    if n_parameter_pose > 0 :
        n_parameter_text = \
            n_parameter_text + '  ' if len(n_parameter_text) > 0 else n_parameter_text

        n_parameter_text = n_parameter_text + 'n_parameter_pose={}'
        n_parameter_vars.append(n_parameter_pose)

    log('Depth network settings:', log_path)
    log('model_name={}'.format(model_name),
        log_path)
    log('network_modules={}'.format(network_modules),
        log_path)
    log('min_predict_depth={:.2f}  max_predict_depth={:.2f}'.format(
        min_predict_depth, max_predict_depth),
        log_path)
    log('use_outlier_rejection={}'.format(use_outlier_rejection),
        log_path)
    log('', log_path)

    if encoder_type_pose is not None and rotation_parameterization_pose is not None:
        log('Pose network settings:', log_path)
        log('encoder_type_pose={}'.format(encoder_type_pose),
            log_path)
        log('rotation_parameterization_pose={}'.format(
            rotation_parameterization_pose),
            log_path)
        log('', log_path)

    log('Weight settings:', log_path)
    log(n_parameter_text.format(*n_parameter_vars),
        log_path)
    log('', log_path)

def log_training_settings(log_path,
                          # Training settings
                          n_batch,
                          n_train_sample,
                          n_train_step,
                          learning_rates,
                          learning_schedule,
                          n_step_per_gradient_accumulation,
                          # Augmentation settings
                          augmentation_probabilities,
                          augmentation_schedule,
                          # Photometric data augmentations
                          augmentation_random_brightness,
                          augmentation_random_contrast,
                          augmentation_random_gamma,
                          augmentation_random_hue,
                          augmentation_random_saturation,
                          augmentation_random_gaussian_blur_kernel_size,
                          augmentation_random_gaussian_blur_sigma_range,
                          augmentation_random_noise_type,
                          augmentation_random_noise_spread,
                          # Geometric data augmentations
                          augmentation_padding_mode,
                          augmentation_random_crop_type,
                          augmentation_random_crop_to_shape,
                          augmentation_random_flip_type,
                          augmentation_random_rotate_max,
                          augmentation_random_crop_and_pad,
                          augmentation_random_resize_to_shape,
                          augmentation_random_resize_and_pad,
                          augmentation_random_resize_and_crop,
                          # Occlusion data augmentations
                          augmentation_random_remove_patch_percent_range_image,
                          augmentation_random_remove_patch_size_image,
                          augmentation_random_remove_patch_percent_range_depth,
                          augmentation_random_remove_patch_size_depth):

    log('Training settings:', log_path)
    log('n_sample={}  n_epoch={}  n_step={}'.format(
        n_train_sample, learning_schedule[-1], n_train_step),
        log_path)
    log('learning_schedule=[%s]' %
        ', '.join('{}-{} : {}'.format(
            ls * (n_train_sample // n_batch), le * (n_train_sample // n_batch), v)
            for ls, le, v in zip([0] + learning_schedule[:-1], learning_schedule, learning_rates)),
        log_path)
    log('n_step_per_gradient_accumulation={}'.format(
        n_step_per_gradient_accumulation), log_path)
    log('', log_path)

    log('Augmentation settings:', log_path)
    log('augmentation_schedule=[%s]' %
        ', '.join('{}-{} : {}'.format(
            ls * (n_train_sample // n_batch), le * (n_train_sample // n_batch), v)
            for ls, le, v in zip([0] + augmentation_schedule[:-1], augmentation_schedule, augmentation_probabilities)),
        log_path)
    log('Photometric data augmentations:', log_path)
    log('augmentation_random_brightness={}'.format(augmentation_random_brightness),
        log_path)
    log('augmentation_random_contrast={}'.format(augmentation_random_contrast),
        log_path)
    log('augmentation_random_gamma={}'.format(augmentation_random_gamma),
        log_path)
    log('augmentation_random_hue={}'.format(augmentation_random_hue),
        log_path)
    log('augmentation_random_saturation={}'.format(augmentation_random_saturation),
        log_path)
    log('augmentation_random_gaussian_blur_kernel_size={}  augmentation_random_gaussian_blur_sigma_range={}'.format(
        augmentation_random_gaussian_blur_kernel_size, augmentation_random_gaussian_blur_sigma_range),
        log_path)
    log('augmentation_random_noise_type={}  augmentation_random_noise_spread={}'.format(
        augmentation_random_noise_type, augmentation_random_noise_spread),
        log_path)

    log('Geometric data augmentations:', log_path)
    log('augmentation_padding_mode={}'.format(augmentation_padding_mode),
        log_path)
    log('augmentation_random_crop_type={}'.format(augmentation_random_crop_type),
        log_path)
    log('augmentation_random_crop_to_shape={}'.format(augmentation_random_crop_to_shape),
        log_path)
    log('augmentation_random_flip_type={}'.format(augmentation_random_flip_type),
        log_path)
    log('augmentation_random_rotate_max={}'.format(augmentation_random_rotate_max),
        log_path)
    log('augmentation_random_crop_and_pad={}'.format(augmentation_random_crop_and_pad),
        log_path)
    log('augmentation_random_resize_to_shape={}'.format(augmentation_random_resize_to_shape),
        log_path)
    log('augmentation_random_resize_and_pad={}'.format(augmentation_random_resize_and_pad),
        log_path)
    log('augmentation_random_resize_and_crop={}'.format(augmentation_random_resize_and_crop),
        log_path)

    log('Occlusion data augmentations:', log_path)
    log('augmentation_random_remove_patch_percent_range_image={}  augmentation_random_remove_patch_size_image={}'.format(
        augmentation_random_remove_patch_percent_range_image, augmentation_random_remove_patch_size_image),
        log_path)
    log('augmentation_random_remove_patch_percent_range_depth={}  augmentation_random_remove_patch_size_depth={}'.format(
        augmentation_random_remove_patch_percent_range_depth, augmentation_random_remove_patch_size_depth),
        log_path)
    log('', log_path)

def log_loss_func_settings(log_path,
                           # Loss function settings
                           supervision_type,
                           w_losses,
                           w_weight_decay_depth,
                           w_weight_decay_pose):

    w_losses_text = ''
    for idx, (key, value) in enumerate(w_losses.items()):

        if idx > 0 and idx % 3 == 0:
            w_losses_text = w_losses_text + '\n'

        w_losses_text = w_losses_text + '{}={:.1e}'.format(key, value) + '  '

    log('Loss function settings:', log_path)
    log('supervision_type={}'.format(supervision_type), log_path)
    log(w_losses_text, log_path)
    log('w_weight_decay_depth={:.1e}  w_weight_decay_pose={:.1e}'.format(
        w_weight_decay_depth, w_weight_decay_pose),
        log_path)
    log('', log_path)

def log_evaluation_settings(log_path,
                            min_evaluate_depth,
                            max_evaluate_depth):

    log('Evaluation settings:', log_path)
    log('min_evaluate_depth={:.2f}  max_evaluate_depth={:.2f}'.format(
        min_evaluate_depth, max_evaluate_depth),
        log_path)
    log('', log_path)

def log_system_settings(log_path,
                        # Checkpoint settings
                        checkpoint_path,
                        n_step_per_checkpoint=None,
                        summary_event_path=None,
                        n_step_per_summary=None,
                        n_image_per_summary=None,
                        start_step_validation=None,
                        restore_paths=None,
                        # Hardware settings
                        device=torch.device('cuda'),
                        n_thread=8):

    log('Checkpoint settings:', log_path)

    if checkpoint_path is not None:
        log('checkpoint_path={}'.format(checkpoint_path), log_path)

        if n_step_per_checkpoint is not None:
            log('n_step_per_checkpoint={}'.format(n_step_per_checkpoint), log_path)

        if start_step_validation is not None:
            log('start_step_validation={}'.format(start_step_validation), log_path)

        log('', log_path)

        summary_settings_text = ''
        summary_settings_vars = []

    if summary_event_path is not None:
        log('Tensorboard settings:', log_path)
        log('event_path={}'.format(summary_event_path), log_path)

    if n_step_per_summary is not None:
        summary_settings_text = summary_settings_text + 'n_step_per_summary={}'
        summary_settings_vars.append(n_step_per_summary)

        summary_settings_text = \
            summary_settings_text + '  ' if len(summary_settings_text) > 0 else summary_settings_text

    if n_image_per_summary is not None:
        summary_settings_text = summary_settings_text + 'n_image_per_summary={}'
        summary_settings_vars.append(n_image_per_summary)

        summary_settings_text = \
            summary_settings_text + '  ' if len(summary_settings_text) > 0 else summary_settings_text

    if len(summary_settings_text) > 0:
        log(summary_settings_text.format(*summary_settings_vars), log_path)

    if restore_paths is not None and restore_paths != '':
        log('restore_paths={}'.format(restore_paths),
            log_path)

    log('', log_path)

    log('Hardware settings:', log_path)
    log('device={}'.format(device.type), log_path)
    log('n_thread={}'.format(n_thread), log_path)
    log('', log_path)
