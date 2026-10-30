# Geometry supervision experiments

Implemented for both `scale_propagation_*` and `fixed_geometry_scale_propagation_*`.
No inference architecture, propagation, rejection, detachment or existing loss
default is changed. Existing training bash files are unchanged.

## Training: change only --w_losses and checkpoint_path

Use separate runs from the same initialization, with matched data, augmentation,
seed, batch size and optimizer-update count. Do not reuse an output directory.
For a clean replacement experiment use these three configurations:

```bash
# Existing SI baseline
--w_losses w_supervised=1.0 w_si=1.0 w_log_l1=1.0 w_affine=0.0 w_scale_aligned_l1=0.0 \

# Supplied affine loss, retaining the fitted z-offset
--w_losses w_supervised=1.0 w_si=0.0 w_log_l1=1.0 w_affine=1.0 w_scale_aligned_l1=0.0 \

# Scale-aligned L1, no offset
--w_losses w_supervised=1.0 w_si=0.0 w_log_l1=1.0 w_affine=0.0 w_scale_aligned_l1=1.0 \
```

Both new losses default to zero. They can also supplement SI (leave `w_si=1`),
or be enabled together, but separate runs are easier to interpret. Weight 1 is
an initial experiment, NOT an assertion of equal gradient strengths. Inspect
the weighted losses and debug gradient-objective norms before comparing weights.

For every r=0..4, geometry is upsampled to GT size as before. The total is:

`w_supervised * sum_r 4**(-r) * (w_si*SI + w_affine*Affine + w_scale_aligned_l1*AlignedL1 + w_log_l1*existing_metric_loss)`.

Any configured rejection loss is added exactly as before. GT masking and depth
range clamping for training are unchanged. The existing metric loss called
`log_l1` remains `log(1 + abs(prediction - GT))`; this experiment does not change it.

- **Affine:** converts depth to calibrated XYZ using each image's intrinsics;
  preserves the supplied loss's shared XYZ scale and z-only offset, 64x64
  nearest-valid alignment samples, truncated alignment objective (threshold 1),
  inverse-GT-depth weights and full-image spatial normalization. Uses the pasted
  defaults beta=0, sparsity_aware=False, improve_weighting=False. The original
  alignment solver is adapted for torch 1.10 and streamed to bound allocations.
  Camera intrinsics must correspond to the images/GT. Missing GT is never filled.
  This is NOT a simple two-parameter depth least-squares loss. The affine fit
  is substantially more expensive than SI/L1, especially at five levels.
- **Aligned L1:** fit one positive scalar `a` per image minimizing
  `mean_valid(abs(a*Z - GT))`. The exact scalar is the weighted median of GT/Z
  with weights Z. There is no offset. Loss is in metres; unlike log losses,
  large absolute errors have greater influence. Both fitted alignments retain
  gradients through the selected solution.

Neither fitted transform is used at inference. Affine supervision can tolerate
z-offset errors that are harmful to scale propagation, which is why the tests
also measure scale-only alignment and local depth relationships.

## Checkpoint diagnosis

Run the same launcher/indices/seeds for every checkpoint. All checkpoints are
scored using ALL geometry metrics; the weight flags only specify the training
objective being logged. Match all model options to the training run. Defaults
are adaptation model, conv_attention encoder, detached geometry, soft-anchor
strength 2, iterations 1 2 3 4 6, one refinement, RGB-only features.

```bash
# SI baseline
CHECKPOINT=/absolute/path/baseline.pth GPU=0 bash bash/diagnose/geometry_kitti.sh

# Affine experiment
CHECKPOINT=/absolute/path/affine.pth GPU=0 bash bash/diagnose/geometry_kitti.sh \
    --w_si 0 --w_affine 1

# Aligned-L1 experiment
CHECKPOINT=/absolute/path/aligned_l1.pth GPU=0 bash bash/diagnose/geometry_kitti.sh \
    --w_si 0 --w_scale_aligned_l1 1
```

Use `geometry_void.sh` for VOID. Set `MODEL=original` for the original scale
propagation wrapper. Extra CLI options override launcher defaults, e.g.
`--indices 0 499 999 --seeds 42` for a first smoke test or `--intrinsics_path ...`
for a different aligned camera list. Exact affine fitting on 100 images x three
seeds x five levels is costly; start with the smoke test.

Send back the generated ZIP. It contains source/checkpoint/data provenance,
`report.md`, `geometry_summary.json`, per-image/per-level `diagnostics.jsonl`,
and aligned depth maps for the first three images. Results include:

- Final unmodified metric depth MAE/RMSE and inverse errors.
- Each level's SI, scale-aligned L1, affine objective, fitted scale/z-offset,
  and nonpositive affine-aligned depth fraction.
- Scale-only aligned geometry MAE/RMSE, far-depth (>40m) errors for KITTI,
  and a separate affine-aligned z-depth error.
- Local log-depth-ratio errors between two valid GT pixels at distances 1,2,4,8.

Success means improved **scale-only geometry/local ratios AND actual final
depth**, not just a smaller affine objective. A lower affine error with large
offsets but unchanged scale-only errors indicates tolerated distortion rather
than better geometry. These are checkpoint comparisons, not proof of causality
from a single run; keep training conditions matched. Existing separation
oracle tests remain available for follow-up localization.

Training TensorBoard logs expose new `loss_affine[_weighted]`,
`loss_scale_aligned_l1[_weighted]`, per-level loss/fit values. Debug mode also
captures the new sources and their separate parameter-gradient norms/conflicts.
