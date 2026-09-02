#!/bin/bash

export CUDA_VISIBLE_DEVICES=${GPU:-0}

# RGB and sparse depth use parallel five-level partition-attention branches.
python depth_completion/src/train_depth_completion.py \
--train_images_path training/void/supervised/void_train_image_1500.txt \
--train_sparse_depth_path training/void/supervised/void_train_sparse_depth_1500.txt \
--train_intrinsics_path training/void/supervised/void_train_intrinsics_1500.txt \
--train_ground_truth_path training/void/supervised/void_train_ground_truth_1500.txt \
--val_image_path testing/void/void_test_image_1500.txt \
--val_sparse_depth_path testing/void/void_test_sparse_depth_1500.txt \
--val_intrinsics_path testing/void/void_test_intrinsics_1500.txt \
--val_ground_truth_path testing/void/void_test_ground_truth_1500.txt \
--n_batch 2 \
--n_height 480 \
--n_width 640 \
--model_name partition_attention_void \
--input_channels_image 3 \
--input_channels_depth 2 \
--normalized_image_range 0 1 \
--min_predict_depth 0.1 \
--max_predict_depth 8.0 \
--learning_rates ${LR:-1e-4} \
--learning_schedule 40 \
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
--augmentation_random_crop_type horizontal vertical \
--augmentation_random_crop_to_shape -1 -1 -1 -1 \
--augmentation_random_flip_type horizontal vertical \
--augmentation_random_rotate_max -1 \
--augmentation_random_crop_and_pad -1 -1 \
--augmentation_random_resize_and_pad -1 -1 \
--augmentation_random_resize_and_crop -1 -1 \
--augmentation_random_resize_to_shape -1 -1 \
--augmentation_random_remove_patch_percent_range_image -1 -1 \
--augmentation_random_remove_patch_size_image -1 -1 \
--augmentation_random_remove_patch_percent_range_depth -1 -1 \
--augmentation_random_remove_patch_size_depth -1 -1 \
--supervision_type supervised \
--w_losses w_supervised=1.0 \
--w_weight_decay_depth 0.00 \
--w_weight_decay_pose 0.00 \
--min_evaluate_depth 0.2 \
--max_evaluate_depth 5.0 \
--n_step_per_summary 500 \
--n_image_per_summary 4 \
--n_step_per_checkpoint 1000 \
--start_step_validation 1000 \
--checkpoint_path \
    trained_models/depth_completion/partition_attention/void1500/supervised_parallel \
--device gpu \
--n_thread 8
