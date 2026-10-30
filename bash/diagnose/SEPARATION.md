# Measurement / relative geometry / connection diagnosis

Run from the repository root with the checkpoint's architecture, iteration
counts, refinement count and anchor strength. The baseline uses that actual
strength and its own sampled gates, unlike the previous lambda=0 anchor sweep.

```bash
CHECKPOINT=/path/to/model.pth GPU=1 ANCHOR_STRENGTH=2 bash bash/diagnose/separation_kitti.sh
CHECKPOINT=/path/to/model.pth GPU=1 ANCHOR_STRENGTH=0 bash bash/diagnose/separation_void.sh
```

For a quick check append `--indices 0 499 999 --seeds 42` on KITTI, or
`--indices 0 399 799 --seeds 42` on VOID. Defaults are 100 images and 3 seeds,
The extended launchers now test all five levels independently. There are 20
policies per level: 97 forwards per image/seed (one native baseline, one exact
replay, and 19 interventions at each of five levels). This is substantially
more expensive than the original 10-forward r0 suite.
No optimizer updates occur. Training scripts and model equations are untouched.

Start with a small smoke test:

```bash
CHECKPOINT=/path/to/model.pth GPU=1 ANCHOR_STRENGTH=2 \
bash bash/diagnose/separation_kitti.sh --indices 0 499 999 --seeds 42 --hierarchy_map_samples 1
```

Append `--separation_levels 0` to test only full resolution (21 forwards per
image/seed). To reproduce the previous protocol, append
`--separation_protocol original --separation_levels 0`.

Each trial changes **one level's propagation**, not all levels together.
All upstream inputs are checked unchanged; downstream stages recompute with
the baseline rejection decisions frozen. Full GT is used unchanged for final
evaluation. Coarse geometry tests sample the nearest original GT pixel at each
align-corners grid location; they never average or fill GT. Unknown samples
remain unknown. These are sparse point oracles, not dense truth for a cell.
Measurement correction separately gathers GT at the selected measurement's
original coordinate. Coarse cells containing multiple surfaces remain a
limitation, even in these tests.

- A replaces only accepted GT-inconsistent measurements with their measured GT
  inside propagation. Keep decisions and point coverage stay fixed. Unknown,
  ambiguous and already-good measurements stay as they were. This tests bad
  values without confounding removal of support.
- G substitutes GT-relative geometry at valid GT pixels **inside propagation**.
  Incoming scale is reparameterized to preserve its original metric depth:
  `M_oracle = Z * M / Z_oracle`. Ratios become `S / Z_oracle`; propagation uses
  the same learned affinities. The propagated product is converted back into
  the original geometry's scale coordinates for subsequent refinement.
  Thus G changes the relative geometry used to transfer scales, rather than
  simply replacing the incoming prediction by GT. G still exposes GT shape
  and is an oracle diagnostic, not a proposed trainable implementation.
- C blocks an edge only when both endpoints have GT and their depth difference
  exceeds `max(0.3m, 5% * min(depths))`. Its outgoing weight is removed from
  both numerator and denominator. Self and unknown edges remain. This is a
  depth-discontinuity proxy, not perfect surface segmentation.

All eight combinations (baseline/A/G/C/AG/AC/GC/AGC) test interactions. The
additional `injection_only` control removes neighbor mixing at the selected level while
preserving initial accepted-point injection. Rejection gates are frozen from
the native-strength baseline. Local replay holds geometry/features/weights
fixed and must reproduce the model's immediately-after-propagation result.
Downstream refinement recomputes normally, using ORIGINAL geometry and raw
measurements; A therefore deliberately does not correct its measurement inputs.

## Additional controls

- `G_injection_only` versus `injection_only`: geometry correction without
  neighbor mixing, retaining initial measurement injection. The converted-back
  depths should match up to roundoff. Compare G versus baseline separately to
  establish that the gain needs exchange.
- `G_bypass` versus `bypass`: also remove initial injection. This checks that
  coordinate reparameterization alone cannot improve predictions. Local and
  final maximum differences are recorded in `separation_null_control` rows.
- `G_anchor`: correct geometry only at accepted anchors using GT at their
  selected original coordinates. `G_nonanchor`: correct it only elsewhere,
  where grid-point GT exists. Both use the same alignment as full G at r0;
  their effects need not add. At coarse levels measurement and grid positions
  differ, so these are NOT an exact partition of the full-G intervention.
- `matched`, `matched_G`, `matched_C`, `matched_GC`: restrict exchange to the
  identical graph of known-GT endpoint pairs, retaining self connections.
  Geometry changes only on pixels incident to that graph. Compare these with
  `matched`, not just the unrestricted baseline. All policies also report
  metrics on the common `observable_graph` region. Restricting the graph itself
  is a confound versus production; G and C are still different-strength oracles.
- `E` versus baseline: replace soft anchoring with exact restoration after
  **every** exchange. `AE` versus A repeats that comparison after correcting
  accepted bad measurements. Together they separate remaining dilution from
  reinforcing bad measurements. Anchor agreement immediately after propagation
  is reported even without GT; final refinement can undo exactness. No training
  or production restoration is introduced.

## Checkpoint replication

Run the identical command twice, changing only CHECKPOINT and OUTPUT, once for
the early checkpoint and once for a later checkpoint from the same training
run. Keep architecture, iterations, anchor strength, indices and seeds fixed.
Send both ZIPs. Different training runs/configurations do not isolate time.
The checkpoint step/hash and configuration are captured in each ZIP; no
checkpoint is automatically selected and no additional training is launched.

Read `report.md` and `separation_summary.json`: compare improvement immediately
after propagation with final-output improvement. JSONL adds stage metrics,
common baseline worst-1% pixels, accepted-good/bad anchors, non-anchor pixels,
far-depth pixels, eligible edge coverage and blocked weight mass. Maps for the
first 3 images allow pixel-level inspection. Comparisons bootstrap images after
seed averaging. Source snapshots, checkpoint hash and arguments are in the ZIP.
The summary has separate `levels.r0` through `levels.r4` entries; do not pool
different intervention levels together. Legacy top-level policies/comparisons
still refer only to r0. Empty coarse GT groups are reported as unavailable,
not zero error. Maps include grid GT and original-coordinate anchor GT.

A gain implicates accepted bad values; G gain demonstrates sensitivity to
relative geometry; C gain implicates the tested cross-discontinuity connections.
Factor gains are not additive and are not percentages of all error explained.
Limited GT and imperfect C labels mean a null result cannot rule out a cause.
If local improvement disappears after refinement, inspect that downstream head.
Single-level tests do not establish combined hierarchy effects or how a
retrained architecture will perform. Share the generated ZIP.
