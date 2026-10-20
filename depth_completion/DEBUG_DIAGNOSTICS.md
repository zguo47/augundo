# Scale-propagation diagnostics (schema 2)

Enable the existing bash arguments:

```bash
--debug_depth true \
--debug_interval 100 \
--debug_max_samples 3 \
```

`--debug_depth false` disables the diagnostic object and all its hooks. The
model, loss, normal validation metrics, and checkpoint format are unchanged.
This mode works with the current scale-propagation model. Future changes to the
fusion/refinement input layout must also update the explicitly documented hook
channel slices and their tests.

## Where to look

Files are written under `<checkpoint_path>/debug/`:

| File/event | Meaning |
| --- | --- |
| `diagnostics.jsonl` | Append-only records with schema, run ID, optimizer step, phase, sample, elapsed time, images and microbatches seen since this process started. |
| `validation_latest.json` | Dataset-wide pixel-weighted error buckets, including depth, sparse proximity and GT boundaries. These differ from averaging per-image metrics in the standard training log. |
| `stages_latest.json` | Per-stage accuracy for fixed and current worst samples. |
| `fixed_<index>.npz` | Latest native intermediate maps for N deterministic, evenly spaced validation indices, recorded in the manifest. |
| `worst_<rank>.npz`, `worst_<rank>_stages.npz` | Latest worst-case outputs and a separate eval replay containing intermediate maps. Worst ranks may change between checkpoints. |
| `sources/<run_id>/` | Exact source snapshots, hashes in the manifest, and startup RNG state. Detach sites are read from actual source, not a manually maintained loss label. |
| `stage_accuracy` | Before/after upsampling, geometry update, anchor-message generation and each scale-refinement pass. Includes improvement/worsening fractions, aligned geometry, scale log error and update multipliers. |
| `fixed_error_tracks` | Errors at the previous checkpoint's 50 worst pixel coordinates in that fixed image. Paths and shape must agree. Tracking is within a process; a restart begins a new run. |
| `gradient_objectives` | Separate SI/metric gradient norms and cosine similarity by parameter group, without accumulating these probes into `.grad`. Null cosine means at least one norm is zero. |
| `anchor_message_ablation` | Extra eval forward with Q/support set to zero at all scales. Direct sparse feedback and the initial global scale are still present. This is message-path influence, not removing all sparse information. |

## Reading the measurements correctly

- All stage metric errors resize the **native product** Z*M to full resolution
  and use the same raw GT/evaluation mask. Stage outputs are not clamped, so the
  final validation result can differ. `factorization_product` separately
  evaluates resized Z times resized M and explicitly labels that distinction.
- Geometry alignment fits a single scalar using mean log GT/Z on valid pixels.
  It is diagnostic only and is never used for prediction or training.
- Scale error compares M with GT/Z. It is not an independent ground-truth scale
  label: its definition depends on the model's current geometry.
- The current model does not change M inside anchor-message propagation.
  `after_anchor_messages` therefore has the same metric map as the preceding
  stage; Q/support affect M through subsequent learned refinement.
- Accepted-anchor distances are Euclidean full-resolution pixel distances to
  the selected measurements' original coordinates. No accepted anchors means
  infinite distance. Support is bilinearly resized only to group errors; neither
  support nor proximity guarantees correct surface association.
- Errors near accepted measurements disagreeing with GT are an association,
  not proof that those measurements caused the error.
- GT boundaries require a valid adjacent pair with a jump exceeding both 0.1m
  and 5%, dilated by two pixels. Missing GT alone is not an edge.
- Tensor arrays preserve float32 maps and integer selected coordinates. NPZ
  names use `<level>__<stage>__<field>`; features are not saved to avoid large
  C-channel dumps. Copy the latest NPZ files elsewhere if historical arrays
  are needed; scalar records remain append-only.

## Sampling and overhead

Ordinary training diagnostics run on the existing debug interval and only on
optimizer-step microbatches. Separate-loss gradients run at step 1 and every
10 × debug interval, before normal backward. They describe that microbatch,
whereas the normal gradient log includes accumulated gradients.

Every validation saves detailed arrays for the N fixed images and does
up to N additional eval forwards for the current worst images. Message ablations
run on fixed images at the first validation, then at steps divisible by 10 ×
debug interval. These probes add compute and disk I/O but do not update the
model or replace official metrics. `debug_max_samples=0` disables detailed
validation snapshots/replays. No full-dataset training was used to benchmark
the overhead.

Startup RNG state is not a full training-resume snapshot: worker state, sampler
position and optimizer state still require the normal checkpoint pipeline.
Recorded image counts are since this process started, not inferred lifetime
counts after resume. An invoked optimizer step with no parameter changes is
reported explicitly; it is not automatically labeled a skipped AMP step.

## Optional supervised rejection

The scale-propagation training CLI accepts `--supervise_outlier_rejection true`
(default false). It also requires supervised training and `--use_outlier_rejection
true`; startup logs record the requested and effective settings. Inference
never receives GT. No model parameters or checkpoint tensor shapes are added.

Keep labels use error <= max(0.1 meters, 2% GT); reject labels use error >
max(0.3 meters, 5% GT). Missing/nonfinite values and the ambiguous gap are ignored.
Override with `--rejection_keep_tolerance 0.1 0.02` and
`--rejection_reject_tolerance 0.3 0.05` (absolute meters, relative fraction).
Keep tolerances must be no larger than reject tolerances.

GT follows the input's exact crop/flip/resize using nearest interpolation and
zero padding. Each native-level logit is labeled using GT gathered at the
selected measurement's coordinate in this augmented input. Logits are not
upsampled or reverse-transformed for classification; only the existing depth
maps follow the existing reverse-transform path. Depth prediction never takes
GT as an input. Labels compare raw GT, not the depth-loss clamped target.

At each level, CE is averaged separately over keep and reject classes; available
class means are averaged. Levels have equal weight. The total objective adds
`w_rejection * mean(level CE)` to the existing depth loss, with default
`w_rejection=0.1`. Override using the existing argument
`--w_losses w_supervised=1.0 w_rejection=0.1`.

TensorBoard logs `train_loss_rejection`, `train_loss_rejection_weighted`, five
`train_loss_rejection_r*` terms and class counts. In diagnostics, the new
`rejection_supervision` records include class counts, confusion, reject
precision/recall, keep recall, and keep probabilities on good/bad points. Null
precision/recall means the corresponding denominator is empty, not perfect or
zero performance. Reports use the same tolerance helper as training. They are
also available as baseline diagnostics when classification supervision is off.
Fixed/worst stage snapshots also store native rejection logits and target labels
(0 keep, 1 reject, -1 ignore) beside selected measurement indices.
Validation summary aggregates the fixed/worst subset, deduplicating each image
and level; this is not a dataset-wide rejection score. Separate-objective
gradient probes include weighted rejection gradients and their alignment with
SI/metric gradients when the extra loss is active.
