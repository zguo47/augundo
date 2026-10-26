import argparse
import torch
from depth_completion import train


class ParseStrFloatKeyValueAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for each in values:
            try:
                key, value = each.split("=")
                getattr(namespace, self.dest)[key] = float(value)
            except ValueError as ex:
                message = "\nTraceback: {}".format(ex)
                message += "\nError on '{}' || It should be 'key=value'".format(each)
                raise argparse.ArgumentError(self, str(message))


def parse_boolean(value):
    return value.lower() in ['true', '1', 'yes']


parser = argparse.ArgumentParser()

# Training and validation input filepaths
parser.add_argument('--train_images_path',
    type=str, required=True, help='Path to list of training image triplets paths')
parser.add_argument('--train_sparse_depth_path',
    type=str, required=True, help='Path to list of training sparse depth paths')
parser.add_argument('--train_intrinsics_path',
    type=str, default=None, help='Path to list of training camera intrinsics paths')
parser.add_argument('--train_ground_truth_path',
    type=str, default=None, help='Path to list of training ground_truth paths')
parser.add_argument('--val_image_path',
    type=str, default=None, help='Path to list of validation image paths')
parser.add_argument('--val_sparse_depth_path',
    type=str, default=None, help='Path to list of validation sparse depth paths')
parser.add_argument('--val_intrinsics_path',
    type=str, default=None, help='Path to list of validation camera intrinsics paths')
parser.add_argument('--val_ground_truth_path',
    type=str, default=None, help='Path to list of validation ground truth depth paths')

# Batch parameters
parser.add_argument('--n_batch',
    type=int, default=8, help='Number of samples per batch')
parser.add_argument('--n_height',
    type=int, default=480, help='Height of of sample')
parser.add_argument('--n_width',
    type=int, default=640, help='Width of each sample')
parser.add_argument('--n_train_sample_limit',
    type=int, default=-1, help='Maximum number of training samples to use; disabled if -1')

# Input settings
parser.add_argument('--input_channels_image',
    type=int, default=3, help='Number of input image channels')
parser.add_argument('--input_channels_depth',
    type=int, default=2, help='Number of input depth channels')
parser.add_argument('--normalized_image_range',
    nargs='+', type=float, default=[0, 1], help='Range of image intensities after normalization')

# Depth network settings
parser.add_argument('--model_name',
    type=str, default='kbnet_void', help='Depth completion model name')
parser.add_argument('--network_modules',
    nargs='+', type=str, default=[], help='modules to build for networks')
parser.add_argument('--min_predict_depth',
    type=float, default=0.10, help='Minimum value of predicted depth')
parser.add_argument('--max_predict_depth',
    type=float, default=10.00, help='Maximum value of predicted depth')
parser.add_argument('--n_iteration',
    type=int, default=1, help='Number of fine-to-coarse/coarse-to-fine iterations')
parser.add_argument('--window_size',
    type=int, default=16, help='Full-resolution attention window size')
parser.add_argument('--n_self_attention',
    type=int, default=2, help='Number of consecutive self-attention blocks')
parser.add_argument('--n_shift',
    type=int, default=4, help='Number of regular/shifted attention pairs')
parser.add_argument('--encoder_type',
    choices=['conv', 'conv_attention'], default='conv',
    help='Scale encoder: current 3/3/3/3/3 conv or 3/6/6/4+2SA/4+2SA')
parser.add_argument('--use_sparse_depth_features',
    type=parse_boolean, default=False,
    help='Concatenate raw sparse depth to the scale encoder input and decoder fusion')
parser.add_argument('--use_outlier_rejection',
    type=parse_boolean, default=True, help='Whether to reject sparse depth outliers')
parser.add_argument('--detach_scaleless_depth',
    type=parse_boolean, default=True, help='Detach scaleless depth in scale-propagation metric losses')
parser.add_argument('--supervise_outlier_rejection',
    type=parse_boolean, default=False, help='Enable the selected auxiliary rejection loss (cross_entropy or l1)')
parser.add_argument('--rejection_loss_type',
    choices=['cross_entropy', 'l1', 'none'], default='cross_entropy',
    help='Auxiliary rejection objective: CE labels or L1 kept-anchor sparsity; supervise_outlier_rejection enables it')
parser.add_argument('--w_rejection_l1',
    type=float, default=None,
    help='L1 kept-anchor penalty weight; defaults to w_losses w_rejection (0.1)')
parser.add_argument('--rejection_supervision_levels',
    nargs='+', type=int, choices=range(5), default=None,
    help='Rejection-loss resolutions: 0=full, ..., 4=bottom; defaults to all, weighted by 1/4**r')
parser.add_argument('--rejection_keep_weight',
    type=float, default=2.0,
    help='Relative CE weight for good measurements versus bad; 1 restores equal-class weighting')
parser.add_argument('--n_propagation_iterations',
    nargs=5, type=int, default=[2, 4, 6, 8, 12],
    help='Scale propagation exchanges at bottom, 1/8, 1/4, 1/2, full resolution; 0 disables exchanges')
parser.add_argument('--n_scale_refinement',
    type=int, default=2,
    help='Number of weight-shared scale refinement passes at every level; 0 disables refinement')
parser.add_argument('--rejection_keep_tolerance',
    nargs=2, type=float, default=[0.1, 0.02], metavar=('METERS', 'FRACTION'),
    help='Label keep when error <= max(METERS, FRACTION * GT)')
parser.add_argument('--rejection_reject_tolerance',
    nargs=2, type=float, default=[0.3, 0.05], metavar=('METERS', 'FRACTION'),
    help='Label reject when error > max(METERS, FRACTION * GT); ignore the gap')

# Training settings
parser.add_argument('--seed',
    type=int, default=0, help='Fixed random seed for initialization, sampling, and augmentation')
parser.add_argument('--learning_rates',
    nargs='+', type=float, default=[1e-4, 5e-5], help='Space delimited list of learning rates')
parser.add_argument('--learning_schedule',
    nargs='+', type=int, default=[5, 10], help='Space delimited list to change learning rate')
parser.add_argument('--n_step_per_gradient_accumulation',
    type=int, default=1, help='Number of batches to accumulate before updating weights')
parser.add_argument('--debug_depth',
    type=parse_boolean, default=False, help='Log depth, scale, gradient and validation diagnostics')
parser.add_argument('--debug_interval',
    type=int, default=500, help='Optimizer-step interval for detailed training diagnostics')
parser.add_argument('--debug_max_samples',
    type=int, default=3, help='Maximum saved worst/failing validation examples per category')

# Augmentation settings
parser.add_argument('--augmentation_probabilities',
    nargs='+', type=float, default=[1.00], help='Probabilities to use data augmentation. Note: there is small chance that no augmentation take place even when we enter augmentation pipeline.')
parser.add_argument('--augmentation_schedule',
    nargs='+', type=int, default=[-1], help='If not -1, then space delimited list to change augmentation probability')

# Photometric data augmentations
parser.add_argument('--augmentation_random_brightness',
    nargs='+', type=float, default=[-1, -1], help='Range of brightness adjustments for augmentation, if does not contain -1, apply random brightness')
parser.add_argument('--augmentation_random_contrast',
    nargs='+', type=float, default=[-1, -1], help='Range of contrast adjustments for augmentation, if does not contain -1, apply random contrast')
parser.add_argument('--augmentation_random_gamma',
    nargs='+', type=float, default=[-1, 1], help='Range of gamma adjustments for augmentation, if does not contain -1, apply random gamma')
parser.add_argument('--augmentation_random_hue',
    nargs='+', type=float, default=[-1, -1], help='Range of hue adjustments for augmentation, if does not contain -1, apply random hue')
parser.add_argument('--augmentation_random_saturation',
    nargs='+', type=float, default=[-1, -1], help='Range of saturation adjustments for augmentation, if does not contain -1, apply random saturation')
parser.add_argument('--augmentation_random_gaussian_blur_kernel_size',
    nargs='+', type=int, default=[-1, -1], help='List of kernel sizes to be used for gaussian blur')
parser.add_argument('--augmentation_random_gaussian_blur_sigma_range',
    nargs='+', type=float, default=[-1, -1], help='Min and max standard deviation for gaussian blur')
parser.add_argument('--augmentation_random_noise_type',
    type=str, default='none', help='Random noise to add: gaussian, uniform')
parser.add_argument('--augmentation_random_noise_spread',
    type=float, default=-1, help='If gaussian noise, then standard deviation; if uniform, then min-max range')

# Geometric data augmentations
parser.add_argument('--augmentation_padding_mode',
    type=str, default='constant', help='Padding used: constant, edge, reflect or symmetric.')
parser.add_argument('--augmentation_random_crop_type',
    nargs='+', type=str, default=['none'], help='Random crop type for data augmentation: horizontal, vertical, anchored, bottom')
parser.add_argument('--augmentation_random_crop_to_shape',
    nargs='+', type=int, default=[-1, -1], help='Random crop to : horizontal, vertical, anchored, bottom')
parser.add_argument('--augmentation_random_flip_type',
    nargs='+', type=str, default=['none'], help='Random flip type for data augmentation: horizontal, vertical')
parser.add_argument('--augmentation_random_rotate_max',
    type=float, default=-1, help='Max angle for random rotation, disabled if -1')
parser.add_argument('--augmentation_random_crop_and_pad',
    nargs='+', type=float, default=[-1, -1], help='If set to positive numbers, then treat as min and max percentage to crop and pad')
parser.add_argument('--augmentation_random_resize_to_shape',
    nargs='+', type=float, default=[-1, -1], help='If set to positive numbers, then treat as min and max percentage to resize')
parser.add_argument('--augmentation_random_resize_and_pad',
    nargs='+', type=float, default=[-1, -1], help='If set to positive numbers, then treat as min and max percentage to resize and pad (or crop if max size is larger than 1)')
parser.add_argument('--augmentation_random_resize_and_crop',
    nargs='+', type=float, default=[-1, -1], help='If set to positive numbers, then treat as min and max percentage to resize and crop ')

# Occlusion data augmentations
parser.add_argument('--augmentation_random_remove_patch_percent_range_image',
    nargs='+', type=float, default=[-1, -1], help='If not -1, randomly remove patches covering percentage of image as augmentation')
parser.add_argument('--augmentation_random_remove_patch_size_image',
    nargs='+', type=int, default=[-1, -1], help='If not -1, patch size for random remove patch augmentation for image')
parser.add_argument('--augmentation_random_remove_patch_percent_range_depth',
    nargs='+', type=float, default=[-1, -1], help='If not -1, randomly remove patches covering percentage of depth map as augmentation')
parser.add_argument('--augmentation_random_remove_patch_size_depth',
    nargs='+', type=int, default=[-1, -1], help='If not -1, patch size for random remove patch augmentation for depth map')

# Loss function settings
parser.add_argument('--supervision_type',
    type=str, default='unsupervised', help='Supervision type for training')
parser.add_argument('--w_losses',
    nargs='+', type=str, action=ParseStrFloatKeyValueAction, help='Weight of each loss term as key-value pairs: w_color=0.90 w_smoothness=2.00; scale propagation also accepts w_si=1.0 w_log_l1=1.0')
parser.add_argument('--w_weight_decay_depth',
    type=float, default=0.00, help='Weight of weight decay regularization for depth')
parser.add_argument('--w_weight_decay_pose',
    type=float, default=0.00, help='Weight of weight decay regularization for pose')

# Evaluation settings
parser.add_argument('--min_evaluate_depth',
    type=float, default=0.20, help='Minimum value of depth to evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, default=5.00, help='Maximum value of depth to evaluate')

# Checkpoint settings
parser.add_argument('--checkpoint_path',
    type=str, required=True, help='Path to save checkpoints')
parser.add_argument('--n_step_per_checkpoint',
    type=int, default=1000, help='Number of iterations for each checkpoint')
parser.add_argument('--n_step_per_summary',
    type=int, default=1000, help='Number of iterations before logging summary')
parser.add_argument('--n_image_per_summary',
    type=int, default=4, help='Number of samples to include in visual display summary')
parser.add_argument('--start_step_validation',
    type=int, default=5000, help='Number of steps before starting validation')
parser.add_argument('--restore_paths',
    nargs='+', type=str, default=[], help='Path to restore depth or pose model from checkpoint')

# Hardware settings
parser.add_argument('--device',
    type=str, default='gpu', help='Device to use: gpu, cpu')
parser.add_argument('--n_thread',
    type=int, default=8, help='Number of threads for fetching')


args = parser.parse_args()

if __name__ == '__main__':

    # Depth network settings
    args.model_name = args.model_name.lower()

    # Loss settings
    args.supervision_type = args.supervision_type.lower()

    # Training settings
    assert len(args.learning_rates) == len(args.learning_schedule)
    assert args.n_step_per_gradient_accumulation > 0
    assert all(0 <= keep <= reject for keep, reject in zip(
        args.rejection_keep_tolerance, args.rejection_reject_tolerance))

    args.augmentation_random_crop_type = [
        crop_type.lower() for crop_type in args.augmentation_random_crop_type
    ]

    args.augmentation_random_flip_type = [
        flip_type.lower() for flip_type in args.augmentation_random_flip_type
    ]

    # Hardware settings
    args.device = args.device.lower()
    if args.device not in ['cpu', 'gpu', 'cuda']:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.device = 'cuda' if args.device == 'gpu' else args.device

    train(train_images_path=args.train_images_path,
          train_sparse_depth_path=args.train_sparse_depth_path,
          train_intrinsics_path=args.train_intrinsics_path,
          train_ground_truth_path=args.train_ground_truth_path,
          val_image_path=args.val_image_path,
          val_sparse_depth_path=args.val_sparse_depth_path,
          val_intrinsics_path=args.val_intrinsics_path,
          val_ground_truth_path=args.val_ground_truth_path,
          # Batch settings
          n_batch=args.n_batch,
          n_height=args.n_height,
          n_width=args.n_width,
          n_train_sample_limit=args.n_train_sample_limit,
          # Input settings
          input_channels_image=args.input_channels_image,
          input_channels_depth=args.input_channels_depth,
          normalized_image_range=args.normalized_image_range,
          # Depth network settings
          model_name=args.model_name,
          network_modules=args.network_modules,
          min_predict_depth=args.min_predict_depth,
          max_predict_depth=args.max_predict_depth,
          n_iteration=args.n_iteration,
          window_size=args.window_size,
          n_self_attention=args.n_self_attention,
          n_shift=args.n_shift,
          use_outlier_rejection=args.use_outlier_rejection,
          # Loss function settings
          supervision_type=args.supervision_type,
          w_losses=args.w_losses,
          w_weight_decay_depth=args.w_weight_decay_depth,
          w_weight_decay_pose=args.w_weight_decay_pose,
          # Training settings
          learning_rates=args.learning_rates,
          learning_schedule=args.learning_schedule,
          n_step_per_gradient_accumulation=args.n_step_per_gradient_accumulation,
          # Augmentation setting
          augmentation_probabilities=args.augmentation_probabilities,
          augmentation_schedule=args.augmentation_schedule,
          # Photometric data augmentations
          augmentation_random_brightness=args.augmentation_random_brightness,
          augmentation_random_contrast=args.augmentation_random_contrast,
          augmentation_random_gamma=args.augmentation_random_gamma,
          augmentation_random_hue=args.augmentation_random_hue,
          augmentation_random_saturation=args.augmentation_random_saturation,
          augmentation_random_gaussian_blur_kernel_size=args.augmentation_random_gaussian_blur_kernel_size,
          augmentation_random_gaussian_blur_sigma_range=args.augmentation_random_gaussian_blur_sigma_range,
          augmentation_random_noise_type=args.augmentation_random_noise_type,
          augmentation_random_noise_spread=args.augmentation_random_noise_spread,
          # Geometric data augmentations
          augmentation_padding_mode=args.augmentation_padding_mode,
          augmentation_random_crop_type=args.augmentation_random_crop_type,
          augmentation_random_crop_to_shape=args.augmentation_random_crop_to_shape,
          augmentation_random_flip_type=args.augmentation_random_flip_type,
          augmentation_random_rotate_max=args.augmentation_random_rotate_max,
          augmentation_random_crop_and_pad=args.augmentation_random_crop_and_pad,
          augmentation_random_resize_to_shape=args.augmentation_random_resize_to_shape,
          augmentation_random_resize_and_pad=args.augmentation_random_resize_and_pad,
          augmentation_random_resize_and_crop=args.augmentation_random_resize_and_crop,
          # Occlusion data augmentations
          augmentation_random_remove_patch_percent_range_image=args.augmentation_random_remove_patch_percent_range_image,
          augmentation_random_remove_patch_size_image=args.augmentation_random_remove_patch_size_image,
          augmentation_random_remove_patch_percent_range_depth=args.augmentation_random_remove_patch_percent_range_depth,
          augmentation_random_remove_patch_size_depth=args.augmentation_random_remove_patch_size_depth,
          # Evaluation settings
          min_evaluate_depth=args.min_evaluate_depth,
          max_evaluate_depth=args.max_evaluate_depth,
          # Checkpoint settings
          checkpoint_path=args.checkpoint_path,
          n_step_per_checkpoint=args.n_step_per_checkpoint,
          n_step_per_summary=args.n_step_per_summary,
          n_image_per_summary=args.n_image_per_summary,
          start_step_validation=args.start_step_validation,
          restore_paths=args.restore_paths,
          # Hardware settings
          device=args.device,
          n_thread=args.n_thread,
          debug_depth=args.debug_depth,
          debug_interval=args.debug_interval,
          debug_max_samples=args.debug_max_samples,
          detach_scaleless_depth=args.detach_scaleless_depth,
          supervise_outlier_rejection=args.supervise_outlier_rejection,
          rejection_loss_type=args.rejection_loss_type,
          w_rejection_l1=args.w_rejection_l1,
          encoder_type=args.encoder_type,
          use_sparse_depth_features=args.use_sparse_depth_features,
          seed=args.seed,
          rejection_supervision_levels=args.rejection_supervision_levels,
          rejection_keep_weight=args.rejection_keep_weight,
          n_propagation_iterations=args.n_propagation_iterations,
          n_scale_refinement=args.n_scale_refinement,
          rejection_keep_tolerance=args.rejection_keep_tolerance,
          rejection_reject_tolerance=args.rejection_reject_tolerance)
