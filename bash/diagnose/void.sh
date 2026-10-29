#!/bin/bash
set -euo pipefail

# Use MODEL=original for the original scale-propagation checkpoint family.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${CHECKPOINT:?Set CHECKPOINT to the trained model .pth file}"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"${PYTHON:-python}" depth_completion/src/diagnose_scale_propagation.py \
    --checkpoint "$CHECKPOINT" \
    --output "${OUTPUT:-diagnose_outputs/void_$(date +%Y%m%d_%H%M%S)}" \
    --images testing/void/void_test_image_1500.txt \
    --sparse_depths testing/void/void_test_sparse_depth_1500.txt \
    --ground_truths testing/void/void_test_ground_truth_1500.txt \
    --model "${MODEL:-adaptation}" \
    --encoder_type "${ENCODER:-conv_attention}" \
    --min_predict_depth 0.1 \
    --max_predict_depth 8.0 \
    --max_evaluate_depth 8.0 \
    --n_propagation_iterations 1 2 3 4 6 \
    --n_scale_refinement 1 \
    --indices 0 200 400 \
    "$@"
