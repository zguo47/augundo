#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${CHECKPOINT:?Set CHECKPOINT to the trained model .pth file}"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"${PYTHON:-python}" depth_completion/src/diagnose_scale_propagation.py \
    --suite separation --checkpoint "$CHECKPOINT" \
    --separation_protocol extended --separation_levels 0 1 2 3 4 \
    --output "${OUTPUT:-diagnose_outputs/separation_kitti_$(date +%Y%m%d_%H%M%S)}" \
    --images validation/kitti/kitti_val_image.txt \
    --sparse_depths validation/kitti/kitti_val_sparse_depth.txt \
    --ground_truths validation/kitti/kitti_val_ground_truth.txt \
    --model "${MODEL:-adaptation}" --encoder_type "${ENCODER:-conv_attention}" \
    --min_predict_depth 1.5 --max_predict_depth 100.0 --max_evaluate_depth 100.0 \
    --propagation_anchor_strength "${ANCHOR_STRENGTH:-2}" \
    --n_propagation_iterations 1 2 3 4 6 --n_scale_refinement 1 \
    --sample_count 100 --seeds 42 43 44 \
    "$@"
