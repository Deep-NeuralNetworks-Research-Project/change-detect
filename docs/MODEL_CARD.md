# MODEL_CARD template (fill one per shipped model)

> This is a **human-in-the-loop reviewer aid**. It reduces search time. It is **not a production decision tool** and must not be used to autonomously accept or reject edits.

## Model
- Name / config key:
- Commit:
- Seeds:

## Intended use
Reviewer aid for paired RGB frames (reference vs edited) at the same semantic time. Binary mask of *meaningful* edits; nuisance (lighting, codec, viewpoint, …) should be suppressed.

## Out of scope
Autonomous publication, legal/forensic decisions, real-time moderation without a human.

## Training data + licences
- Public: SYSU-CD, LEVIR-CD, PCD/TSUNAMI — cite the dataset papers; follow their licences.
- Video set: authorised footage only; no redistribution of frames (see `DATA_CARD.md` / `ETHICS.md` in the team repo).

## Eval data
- Splits: scene/source-disjoint (never frame-level).
- π (changed-pixel ratio) per split:
- Threshold (val-only; F1-optimal or precision-constrained):

## Metrics (cross-seed)
Report mean ± std **and** median [min, max] over ≥3 seeds, plus bootstrap 95% CIs over the test set. Do not report p-values at n=3.

| metric | pooling | seed 0 | seed 1 | seed 2 | mean ± std | bootstrap 95% CI |
|---|---|---|---|---|---|---|
| F1 | aggregate |  |  |  |  |  |
| IoU | aggregate |  |  |  |  |  |
| mF1 | per-image, TN-only excluded |  |  |  |  |  |
| Boundary IoU |  |  |  |  |  |  |
| ECE (foreground, equal-mass) |  |  |  |  |  |  |
| AURC (pair-level SDC) |  |  |  |  |  |  |

## Efficiency
- GPU model (logged every run):
- Params total / trainable (shared encoder counted once):
- Pair-input FLOPs (tool + convention):
- Latency median + IQR, AMP/TF32:
- Peak memory reserved / allocated:

## Known failure modes
- Nuisance conditions with largest F1 drop:
- Small objects / thin structures (Boundary IoU):
- Calibration under shift (T fitted on …):

## Contact
P5 metrics owner — see CODEOWNERS.
