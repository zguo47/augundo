# Geometry-aware scale adaptation experiment

This replaces the former fixed-geometry baseline under the existing model names
`fixed_geometry_scale_propagation_kitti` and
`fixed_geometry_scale_propagation_void`. The original `scale_propagation_*`
implementation is unchanged.

The encoder and joint decoder follow the original model. Every finer level:
1. Bilinearly upsamples geometry, scale, and decoder features.
2. Fuses features and the encoder skip, then updates and normalizes geometry.
3. Adapts scale using features, old/new geometry, inherited scale, current metric
   depth, and log geometry change.
4. Predicts rejection and propagation weights using the adapted metric depth.
5. Injects accepted ratios once and propagates.
6. Refines the propagated scale with a separate head using features, detached
   geometry, current scale, accepted ratios, keep mask, and anchor discrepancy.
   No anchor restoration occurs.

The bottom level keeps the original initialization, propagation, and refinement;
there is no previous geometry to adapt to there. Adaptation uses KBNet convolutions
and starts at identity. The separate post-propagation head also starts at
identity. Neither head imposes inverse geometry compensation.

Use a fresh checkpoint directory with:
```bash
--model_name fixed_geometry_scale_propagation_kitti \
--fixed_geometry_stage joint \
--detach_scaleless_depth true \
--n_scale_refinement 1 \
--w_losses w_supervised=1.0 w_si=1.0 w_log_l1=1.0 \
```
Keep the remaining training settings matched to the original comparison run.
The existing rejection and sparse-feature flags are now honored.
`--n_scale_refinement 1` performs one adaptation and one post-propagation
refinement per finer level, and the original one refinement at the bottom.
Setting it to 0 disables both heads and bottom refinement.
No bash scripts were changed.

Geometry in metric products and sparse-ratio denominators remains detached when
using the configuration above. Shared features still receive metric gradients,
as in the original model. SI supervises the actual five geometry predictions,
weighted by 1/4^r; metric log L1 and rejection losses retain original conventions.
The old separate geometry/scale stages, global-only mode, auxiliary heads, and
geometry-only checkpoint initialization no longer apply.

Debug stage records distinguish `after_geometry_update`,
`scale_adaptation_1`, `after_scale_adaptation`, `after_propagation`, and
`refinement_1`. Training records distinguish `scale_adaptation` from
`scale_refinement`. Previous checkpoints lack the new post-propagation heads;
train fresh in a new checkpoint directory for this comparison.
