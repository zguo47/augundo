# Shared-geometry scale-propagation baseline

This is a separate pipeline, selected with
`--model_name fixed_geometry_scale_propagation_kitti` or
`--model_name fixed_geometry_scale_propagation_void`.
The existing `scale_propagation_*` pipeline is unchanged.

## One-run joint training (default)

An RGB-only five-resolution U-Net predicts one positive full-resolution
geometry `Z`, normalized once to spatial mean one. Scale propagation uses
resized copies of this same final prediction at every level. Geometry and
scale are trained together in one run; no pretrained checkpoint is required.
The historical `fixed_geometry_*` filenames mean a shared geometry reference
within each forward pass, not frozen geometry across optimizer steps.

From the repository root:

```bash
bash bash/depth_completion/train/fixed_geometry_scale_propagation/train_fixed_geometry_scale_propagation-kitti-supervised.sh
```

The VOID script in the same directory ends in `-void1500-supervised.sh`.
Use `ENCODER_TYPE=conv_attention` to select the existing larger encoder;
otherwise the existing all-convolution encoder is used. Match that selection
in evaluation. `DEBUG=true` enables the existing diagnostics.

For an existing custom training script, use:

```text
--model_name fixed_geometry_scale_propagation_kitti
--fixed_geometry_stage joint
--use_outlier_rejection false
--n_scale_refinement 0
--w_losses w_supervised=1.0 w_si=1.0 w_log_l1=1.0
```

Both objectives are active in joint mode: full-resolution SI for `Z`, and
multiscale log L1 for metric depth. They are summed in a single backward and
optimizer step. With `--detach_scaleless_depth true` (default), metric loss
cannot update geometry through either `Z` OR the RGB guidance features; SI
still trains the entire RGB U-Net each step. Setting the flag to `false`
also permits metric gradients into geometry through both routes.

Sparse depth and scale maps never enter the geometry U-Net. Geometry is not
updated or renormalized independently at decoder levels. Its parameters do
change between training steps, so joint training does not guarantee the same
optimization stability as a separately pretrained and frozen geometry model.
RGB decoder features guide the separate scale decoder.
The initial scale is the mean of full-resolution positive sparse-to-geometry
ratios (one scalar per image). With no measurements it falls back to one.

The scale map is bilinearly upsampled and refined by the existing sigmoid,
outgoing-neighbor propagation. Coarse sparse depth uses the existing
valid-measurement selector. Ratios use geometry at the current grid cell.
Injection happens once per level, without anchor restoration inside or after
the loop. There is no rejection head and no standalone learned scale update.
The final depth is `clamp(Z * M, min_depth, max_depth)`.

The scale loss is the existing **unclamped** log L1 metric loss, summed over
all five resolutions with weights `1 / 4**r`, with `r=0` at full resolution.
The joint objective is `w_supervised * (w_si * SI_r0 +
w_log_l1 * sum_r(4**(-r) * log_L1_r))`. The common 16-map output layout preserves
TensorBoard geometry/scale views and the debug interface.

Scripts accept additional CLI arguments at the end, for example:

```bash
bash bash/depth_completion/train/fixed_geometry_scale_propagation/train_fixed_geometry_scale_propagation-kitti-supervised.sh \
    --n_propagation_iterations 1 2 3 4 6
```

This is intentionally a stability ablation, not the full proposed redesign:
there is no log-scale solver, new confidence module, or new geometry loss.
Coarse sampling and limited propagation reach remain possible limitations.

## Optional separate-stage ablations

`STAGE=geometry` / `--fixed_geometry_stage geometry` trains only full-resolution
SI geometry. Displayed metric predictions then use sparse global alignment
for previews, not a metric training loss. `STAGE=scale` freezes a pretrained
geometry network and trains only scale propagation with log L1.

Only that optional scale-only mode requires `GEOMETRY_CHECKPOINT=...` /
`--geometry_restore_path ...` (or a full resume checkpoint). A geometry restore
loads geometry weights only and starts a new optimizer/step. To resume the
same training mode, use `RESUME_CHECKPOINT=...` / `--restore_paths ...` instead.

## Global-scale-only comparison

Run the same checkpoint with propagation enabled and bypassed:

```bash
CHECKPOINT=/absolute/path/scale_checkpoint.pth MODE=hierarchical \
bash bash/depth_completion/run/fixed_geometry_scale_propagation/run_fixed_geometry_scale_propagation-kitti.sh

CHECKPOINT=/absolute/path/scale_checkpoint.pth MODE=global \
bash bash/depth_completion/run/fixed_geometry_scale_propagation/run_fixed_geometry_scale_propagation-kitti.sh
```

Use the parallel VOID evaluation script for VOID. Pass the same
`--n_propagation_iterations` used in scale training. The global control skips
all learned propagation and returns `Z` multiplied by the same initial global
scale. Neither mode uses GT to compute a prediction. Compare the two on the
same checkpoint, evaluation split, and depth range.
