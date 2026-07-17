#!/bin/bash

export CUDA_VISIBLE_DEVICES=${GPU:-0}

python depth_completion/src/train_depth_completion.py \
--train_images_path training/kitti/supervised/kitti_train_image.txt \
--train_sparse_depth_path training/kitti/supervised/kitti_train_sparse_depth.txt \
--train_intrinsics_path training/kitti/supervised/kitti_train_intrinsics.txt \
--train_ground_truth_path training/kitti/supervised/kitti_train_ground_truth.txt \
--val_image_path validation/kitti/kitti_val_image.txt \
--val_sparse_depth_path validation/kitti/kitti_val_sparse_depth.txt \
--val_intrinsics_path validation/kitti/kitti_val_intrinsics.txt \
--val_ground_truth_path validation/kitti/kitti_val_ground_truth.txt \
--n_batch 8 \
--n_height 320 \
--n_width 768 \
--model_name kbnet_kitti \
--network_modules depth \
--input_channels_image 3 \
--input_channels_depth 2 \
--normalized_image_range 0 1 \
--min_predict_depth 1.5 \
--max_predict_depth 100.0 \
--learning_rates ${LR:-1e-3} \
--learning_schedule 100 \
--augmentation_probabilities 1.0 \
--augmentation_schedule -1 \
--augmentation_random_brightness 0.50 1.50 \
--augmentation_random_contrast 0.50 1.50 \
--augmentation_random_gamma -1 -1 \
--augmentation_random_hue -0.1 0.1 \
--augmentation_random_saturation 0.50 1.50 \
--augmentation_random_noise_type none \
--augmentation_random_noise_spread -1 \
--augmentation_padding_mode edge \
--augmentation_random_crop_type horizontal bottom anchored \
--augmentation_random_crop_to_shape -1 -1 -1 -1 \
--augmentation_random_flip_type horizontal \
--augmentation_random_rotate_max -1 \
--augmentation_random_crop_and_pad 0.90 1.00 \
--augmentation_random_resize_and_pad -1 -1 \
--augmentation_random_resize_and_crop -1 -1 \
--augmentation_random_resize_to_shape -1 -1 \
--augmentation_random_remove_patch_percent_range_image 1e-3 5e-3 \
--augmentation_random_remove_patch_size_image 5 5 \
--augmentation_random_remove_patch_percent_range_depth 0.60 0.70 \
--augmentation_random_remove_patch_size_depth 1 1 \
--supervision_type supervised \
--w_losses w_supervise=1.0 \
--w_weight_decay_depth 0.00 \
--w_weight_decay_pose 0.00 \
--min_evaluate_depth 0.0 \
--max_evaluate_depth 100.0 \
--n_step_per_summary 5000 \
--n_image_per_summary 8 \
--n_step_per_checkpoint 5000 \
--start_step_validation 100000 \
--checkpoint_path \
    trained_models/depth_completion/kbnet/kitti/supervised_lr_${LR:-1e-3} \
--restore_paths \
    trained_models/depth_completion/kbnet/kitti/supervised_lr_${LR:-1e-3}/checkpoints-5000/kbnet-5000.pth \
--device gpu \
--n_thread 8
