# Does hierarchical rejection help? A--E checkpoint experiment

This suite changes **only keep decisions**, without training, changing model
weights, or touching normal validation. It is independent of the existing
single-point/stage diagnostic suite, which remains available unchanged.

```bash
CHECKPOINT=/path/to/model.pth GPU=1 bash bash/diagnose/hierarchy_kitti.sh
CHECKPOINT=/path/to/model.pth GPU=1 bash bash/diagnose/hierarchy_void.sh
```

Defaults: current adaptation model, conv_attention encoder, detached geometry,
no raw sparse features, iterations 1/2/3/4/6, one refinement pass, no explicit
rejection loss, and the dataset's usual depth bounds. **Match these to the
checkpoint's training settings.** MODEL=original selects the original family.
All existing diagnostic model/loss arguments can override script defaults.

The scripts choose 100 uniformly spaced validation indices, without looking
at model errors, and repeat with seeds 42/43/44. These are exploratory samples,
not necessarily a representative sample of every scene. There are 9 forward
passes per image per seed (eight policies plus replay), no backward pass.

```bash
# Quick check using the previous three images and one seed:
CHECKPOINT=/path/to/model.pth bash bash/diagnose/hierarchy_kitti.sh \
    --indices 0 499 999 --seeds 42

# Full validation set with the default three seeds:
CHECKPOINT=/path/to/model.pth bash bash/diagnose/hierarchy_kitti.sh --sample_count 0

# Save no spatial maps (JSON results are still complete):
CHECKPOINT=/path/to/model.pth bash bash/diagnose/hierarchy_kitti.sh --hierarchy_map_samples 0
```

Explicit indices override sample_count. OUTPUT chooses a new output directory,
PYTHON chooses the interpreter. Assign variables on the **same command** as
the script or `export` them first. No retraining or optimizer state is needed.

## Policies

| Policy | r1--r4 | r0 |
|---|---|---|
| A_keep_all | Keep all available measurements | Keep all |
| B_r0_only | Keep all | Baseline learned mask |
| C_learned | Baseline learned masks | Baseline learned mask |
| D_oracle_r1 through D_oracle_r4 | GT-guided mask at that one level; keep all elsewhere | Baseline learned mask |
| E_oracle_hierarchy | GT-guided masks at all four coarse levels | Baseline learned mask |

For each image/seed, first run C normally and capture the sampled masks.
Replay C to verify deterministic output. B--E use **exactly that same r0
mask**, even though their internal features change. All model computations
other than mask decisions recompute normally. This isolates the controlled
effect of coarse rejection; it is not deployment with re-evaluated r0 decisions.

The oracle compares each selected sparse value against GT at its original
measurement coordinate, using existing labels:

- Good: absolute difference <= max(0.1 meters, 2% of GT).
- Bad: absolute difference > max(0.3 meters, 5% of GT).
- Otherwise ambiguous; missing/nonfinite GT is unknown.

It rejects known bad measurements, **restores known good measurements**, and
keeps ambiguous/unknown measurements. Missing measurements always remain zero.
Labels use raw, coordinate-matched GT, not depth-clamped or pooled GT. This is
an imperfect GT-consistency oracle, not a guaranteed optimal subset of anchors.

## Output to send back

Send the generated ZIP in `diagnose_outputs/`. It contains:

- `report.md`: pooled final metrics and paired A--E comparisons.
- `hierarchy_summary.json`: aggregates, per-seed metrics, depth/boundary groups,
  paired comparisons, and image-bootstrap intervals.
- `diagnostics.jsonl`: every image/seed/policy's losses, final metrics, rejection
  counts, stage metrics, stage SSE changes versus B, and SSE added/removed.
- Spatial NPZ files for the first three selected images, first seed only:
  all policy predictions/masks, labels, selected original indices, and error
  changes relative to B. They contain scene geometry and GT-derived information.
- Configuration, checkpoint/data hashes, and exact source snapshots.

Primary comparisons: **C vs B** (current hierarchy), **E vs B** (corrected
hierarchy), **D vs B** (which level), and **E vs each D** (does combining help).
B vs A tests the reused learned r0 mask. Improvements from different D tests
must not be summed; levels interact.

MAE/RMSE are shown in millimeters in report.md, meters in JSON. Inverse errors
are 1/km. Pooled results average errors across images and seeds, weighted by
evaluated pixels. Paired intervals resample **images** after averaging seeds,
not pixels or seeds as independent observations. Their statistic is mean
per-image RMSE difference, not the difference of pooled RMSE. Very small
subsets cannot provide reliable population confidence intervals.

No additional spatial crop is applied. Final metrics use positive finite GT
within evaluation bounds. Stage metrics use unclamped native Z*M upsampled to
GT. The baseline C worst-1% mask is fixed across policies within each seed.

These are GT-assisted controlled tests, not deployable evaluation scores and
not proof that a rejection head can learn the decisions. If the oracle helps,
the next experiment is matched rejection-head training and held-out evaluation.
Neither that training nor GT-guided production inference is added here.
