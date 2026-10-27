# Fixed-geometry scale-propagation baseline

This is a separate pipeline, selected with
`--model_name fixed_geometry_scale_propagation_kitti` or
`--model_name fixed_geometry_scale_propagation_void`.
The existing `scale_propagation_*` pipeline is unchanged.

## Geometry stage

An RGB-only five-resolution U-Net predicts one positive full-resolution
geometry `Z`, normalized once to spatial mean one. Only the existing SI loss
at full resolution trains this stage. Sparse depth is never an input to the
geometry encoder or decoder. It is used for global metric alignment of the
displayed predictions, so validation depth metrics here are not SI metrics.

From the repository root:

```bash
STAGE=geometry bash bash/depth_completion/train/fixed_geometry_scale_propagation/train_fixed_geometry_scale_propagation-kitti-supervised.sh
```

The VOID script in the same directory ends in `-void1500-supervised.sh`.
Use `ENCODER_TYPE=conv_attention` to select the existing larger encoder;
otherwise the existing all-convolution encoder is used. Match that selection
in both stages and evaluation. `DEBUG=true` enables the existing diagnostics.

## Scale stage

Load a checkpoint produced by the geometry stage:

```bash
STAGE=scale GEOMETRY_CHECKPOINT=/absolute/path/fixed-geometry-scale-propagation-20000.pth \
bash bash/depth_completion/train/fixed_geometry_scale_propagation/train_fixed_geometry_scale_propagation-kitti-supervised.sh
```

`--geometry_restore_path` loads **only geometry**, starts a new optimizer, and
starts the scale experiment at step zero. `RESUME_CHECKPOINT=/absolute/path/...`
instead passes `--restore_paths` to resume the same stage and its optimizer.
Do not resume geometry-stage optimizer state as a scale-stage optimizer.

During scale training the whole geometry network is frozen and kept in eval
mode. Each scale level receives a resized copy of the same final `Z`; those
copies are not renormalized. RGB decoder features guide the scale decoder.
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
`w_supervised` and `w_log_l1` control scale-stage loss; `w_supervised` and
`w_si` control geometry-stage loss. Inactive losses are reported for monitoring
but do not train either network. The common 16-map output layout preserves
TensorBoard geometry/scale views and the debug interface.

Scripts accept additional CLI arguments at the end, for example:

```bash
STAGE=scale GEOMETRY_CHECKPOINT=/absolute/path/geometry.pth \
bash bash/depth_completion/train/fixed_geometry_scale_propagation/train_fixed_geometry_scale_propagation-kitti-supervised.sh \
    --n_propagation_iterations 1 2 3 4 6
```

This is intentionally a stability ablation, not the full proposed redesign:
there is no log-scale solver, new confidence module, or new geometry loss.
Coarse sampling and limited propagation reach remain possible limitations.

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
