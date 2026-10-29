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
10 forwards per image/seed (native baseline, exact replay, 8 interventions).
No optimizer updates occur. Training scripts and model equations are untouched.

The test targets **r0 propagation** to avoid inventing dense coarse GT from
sparse KITTI labels. All upstream stages are identical. GT is never filled,
averaged or resized; unavailable/unevaluated pixels remain uncorrected.

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
additional `injection_only` control removes neighbor mixing at r0 while
preserving initial accepted-point injection. Rejection gates are frozen from
the native-strength baseline. Local replay holds geometry/features/weights
fixed and must reproduce the model's immediately-after-propagation result.
Downstream refinement recomputes normally, using ORIGINAL geometry and raw
measurements; A therefore deliberately does not correct its measurement inputs.

Read `report.md` and `separation_summary.json`: compare improvement immediately
after propagation with final-output improvement. JSONL adds stage metrics,
common baseline worst-1% pixels, accepted-good/bad anchors, non-anchor pixels,
far-depth pixels, eligible edge coverage and blocked weight mass. Maps for the
first 3 images allow pixel-level inspection. Comparisons bootstrap images after
seed averaging. Source snapshots, checkpoint hash and arguments are in the ZIP.

A gain implicates accepted bad values; G gain demonstrates sensitivity to
relative geometry; C gain implicates the tested cross-discontinuity connections.
Factor gains are not additive and are not percentages of all error explained.
Limited GT and imperfect C labels mean a null result cannot rule out a cause.
If local improvement disappears after refinement, inspect that downstream head.
The experiment does not test the origin of errors inherited from coarse levels
or establish how a retrained architecture will perform. Share the generated ZIP.
