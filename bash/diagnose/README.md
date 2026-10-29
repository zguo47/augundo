# Isolated scale-propagation diagnosis

Run from an existing checkpoint, with no training or model changes. Defaults
match the recent KITTI joint adaptation + post-refinement experiment. Match
the encoder, model family, iterations, refinement count, input-feature flag,
detach flag, depth bounds, and loss settings to the run you are diagnosing.
Checkpoint parameters load strictly. Runtime settings are not all stored in a
checkpoint, so strict loading alone cannot detect incorrect iteration counts.

```bash
CHECKPOINT=/path/to/model-5000.pth GPU=0 bash bash/diagnose/kitti.sh
CHECKPOINT=/path/to/model-5000.pth GPU=0 bash bash/diagnose/void.sh
```

The default model is `adaptation`: the current implementation with the historical
`fixed_geometry_scale_propagation_*` name. For the original model:

```bash
CHECKPOINT=/path/to/model.pth MODEL=original ENCODER=conv \
  bash bash/diagnose/void.sh --n_propagation_iterations 1 2 3 4 6
```

Arguments after the script override its defaults. Examples:

```bash
# A short first pass: one image, one point from each diagnostic group per level.
CHECKPOINT=/path/to/model.pth bash bash/diagnose/kitti.sh \
  --indices 0 --points_per_group 1

# Reproduce a run with explicit rejection supervision only at full resolution.
CHECKPOINT=/path/to/model.pth bash bash/diagnose/kitti.sh \
  --rejection_loss cross_entropy --rejection_levels 0 --w_rejection 0.1

# For L1, w_rejection here is the training script's w_rejection_l1 value.
CHECKPOINT=/path/to/model.pth bash bash/diagnose/kitti.sh \
  --rejection_loss l1 --w_rejection 0.002
```

Additional options: `--w_si`, `--w_log_l1`, `--w_supervised`,
`--detach_scaleless_depth true|false`, `--use_sparse_depth_features true|false`,
`--n_scale_refinement`, `--seed`, `--device cpu`, `--skip_gradients`.
`PYTHON` selects a Python executable, `OUTPUT` selects a **new** output directory.
Rejection is enabled because this runner diagnoses that head. Do not interpret
an untrained rejection head from a checkpoint trained with rejection disabled.

## What runs

1. One unaugmented full-image forward and gradient probe. No optimizer exists.
   The loss is the actual wrapper's multiscale SI/log L1 and optional rejection
   loss, evaluated on these inputs, not an augmented training minibatch.
2. Replay with exactly the same rejection masks; abort if outputs disagree.
3. At each level, select up to two points each from GT-good, GT-bad, and
   GT-known near-boundary groups. Groups may overlap. Unknown/ambiguous GT
   points are excluded. Force each point kept/rejected, holding all other
   masks fixed, and measure objective and final SSE differences.
4. At each level independently: bypass neighbor mixing (retain injection),
   remove known GT-disagreeing anchors, suppress observed cross-GT-boundary
   propagation edges, bypass post-refinement, and (where present) bypass
   pre-propagation scale adaptation. Other levels stay active.
5. Save per-stage SSE added/removed, final error groups, rejection confusion
   counts, keep probabilities, gate/logit gradients, and immediate incoming
   neighbor traces for ten worst final pixels.

Default cost is up to **86 forwards per image**, including a gradient pass.
Batch size is one; attention backward can still require substantial memory.
Use `--skip_gradients` if necessary; the decisive keep/reject counterfactuals
still run. `--points_per_group 0` skips point tests, not stage interventions.
Repeat with other indices/seeds to establish whether results generalize.

## Share these results

The runner prints the ZIP path when finished. Send that ZIP back for diagnosis:

- `report.md`: readable baseline and intervention summary with interpretation.
- `diagnostics.jsonl`: detailed measurements, flushed after each test.
- `sample_*_maps.npz`: native stage predictions, masks, probabilities, labels,
  ratio gaps and gradients (no full activation/attention dumps).
- `provenance.json`, `sources/`: exact settings, checkpoint hash, data-list
  hashes, source snapshots and versions.

The ZIP excludes checkpoint weights, RGB images and full ground-truth arrays.
It includes dataset paths, some GT values in worst-pixel traces, and predicted
scene geometry; review it before sharing outside your project.

## Interpretation limits

All errors are meters (not the usual displayed millimeters); inverse errors
are 1/km. Evaluation uses positive finite GT within the supplied bounds, no
additional spatial crop. Stages use **unclamped** native Z*M resized to GT;
final metrics use the model's final clamped output. The final worst-1% mask is
held constant across stages and interventions. SSE changes need not match MAE.

The GT rejection labels use existing tolerances and original selected sparse
measurement coordinates. Coarse oracle-edge GT uses nearest sampling; holes
remain unknown and their edges are not suppressed. These oracles are partial
tests, not perfect surface labels. A small effect does not prove a mechanism
is harmless everywhere. Immediate neighbor traces do not recover multi-hop
source ancestry. Frozen later masks isolate controlled effects, not the total
effect of changing all downstream rejection decisions.

The runner does not automatically prescribe an architecture change: compare
actual discrete effects, local gradient direction, and probability saturation
before concluding why a head failed to learn. No new training is required.
