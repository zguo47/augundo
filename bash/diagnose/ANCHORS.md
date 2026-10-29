# Persistent accepted-anchor experiment

Production scale-propagation models now use `--propagation_anchor_strength 2.0`.
Use `0` for the previous update. This is a finite source weight, not hard
restoration. Refinement, rejection, geometry detachment, and iteration counts
are unchanged. No parameters were added, so existing checkpoints load directly.
The strength is a runtime setting: pass the same flag to training and inference;
it is not recovered from old checkpoint state dictionaries. The value 2 is a
starting ablation setting, not a validated optimum. Existing training scripts
were not edited.

The existing one-time injection is preserved for a controlled comparison.
It still replaces the inherited scale at accepted pixels. The new per-iteration
soft constraint does not hard-restore anchors: gradients through incoming
predictions, weights and the keep decision remain available, although stronger
constraints can attenuate them. The final refinement head is unchanged and may
still undo a propagation improvement; the stage diagnostics expose this.

Run from the repository root, matching the architecture and iterations used
to train the checkpoint:

```bash
CHECKPOINT=/path/to/model.pth GPU=1 bash bash/diagnose/anchors_kitti.sh
CHECKPOINT=/path/to/model.pth GPU=1 bash bash/diagnose/anchors_void.sh
```

Quick targeted check (includes the previously analyzed sample 499):

```bash
CHECKPOINT=/path/to/model.pth GPU=1 bash bash/diagnose/anchors_kitti.sh \
    --indices 0 499 999 --seeds 42 --anchor_strengths 2
```

Full defaults: 100 uniformly spaced images, three seeds, strengths 1/2/4.
Each seed/image runs an old-update baseline, its frozen-mask replay, and twelve
interventions: r0-only/all-level, GT-good-only/all-accepted, each strength.
Thus the full sweep is considerably slower than ordinary validation.

The oracle variant changes only persistent tether eligibility. It does not
change initial sparse injection or keep/reject decisions. Labels use GT at
the selected measurement's original location. Bad and unknown measurements
are still initially injected if the frozen learned mask accepted them.
All downstream features and heads recompute normally in full-model trials.

Outputs: report.md, anchor_summary.json (pooled metrics and image-bootstrap
comparisons), diagnostics.jsonl (stage/anchor errors and per-iteration local
replays), maps for the first three images/first seed, and provenance/source
snapshots. Local replays hold all block inputs fixed and use the production
function. Stage metrics reveal whether refinement undoes propagation gains.
The sibling ZIP is the artifact to share for analysis. No model weights change.

Success means not just better matching at anchors: check final evaluated RMSE,
common baseline worst-1% SSE, errors away from anchors, and bad-anchor damage.
An oracle-only gain is not evidence that the deployable accepted-anchor variant
works. A checkpoint intervention does not establish retrained performance.
