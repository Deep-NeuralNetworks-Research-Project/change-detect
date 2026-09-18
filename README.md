# Pair-Order Consistent & Uncertainty-Aware Semantic Change Detection

University of Moratuwa, CSE — five-person semester project
(Ekanayake, Lelwala, Pabasara, Bulagala, Rajapaksha).

Given two RGB frames `I1` and `I2`, predict a binary mask of *meaningful
edits* while suppressing nuisance (camera motion, lighting, codec, blur).
The system is a human-in-the-loop reviewer aid: calibrated confidence and
deferral are first-class, not extras.

This directory is the **team monorepo**. It merges the three public slices
from [Deep-NeuralNetworks-Research-Project](https://github.com/Deep-NeuralNetworks-Research-Project):

| Slice | Role | Landed as |
|---|---|---|
| [`baseline-infa-bula`](https://github.com/Deep-NeuralNetworks-Research-Project/baseline-infa-bula) | P2 infra + P1 data | trainer, Hydra, datasets, FC-Siam-Diff, RGB/SSIM |
| [`modeling-encoders-Weenuka`](https://github.com/Deep-NeuralNetworks-Research-Project/modeling-encoders-Weenuka) | P3 encoders / fusion / decoder | ResNet-18, EfficientNet, abs-diff & signed fusion, U-Net decoder |
| [`metrics-integration-kusal`](https://github.com/Deep-NeuralNetworks-Research-Project/metrics-integration-kusal) | P5 metrics / paper | segmentation, boundary, calibration, corruptions, ACM skeleton |

P4 (alignment, heads, pair-order loss, `proposed.py`) had no public slice;
those paths are scaffolded here so the frozen layout in
`research/06-engineering-delivery.md` is complete.

Frozen contracts: `docs/architecture.md` and `CLAUDE.md`. Do not break them
in a quiet edit.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # or the CUDA wheel
pip install -e ".[dev]"
pytest -m "not gpu and not slow" -q
```

## Commands

```bash
# Train (Hydra)
python -m cdlib.cli.train data=levir_cd model=siamese_resnet18
python -m cdlib.cli.train +experiment=ablation_no_alignment
python -m cdlib.cli.train -m train.optimizer.lr=1e-3,3e-4,1e-4 model=fc_siam_diff
python -m cdlib.cli.train resume_from=auto

# Evaluate / benchmark (runnable without a checkpoint)
python -m cdlib.cli.evaluate +metrics=all --img-size 32
python -m cdlib.cli.benchmark --model dummy --img-size 64 --device cpu
python -m cdlib.cli.export_masks exp_id=demo --overlays best,median,worst

# Dry-construct every experiment config (CI)
python -m cdlib.cli.validate_configs
```

## Layout

```
src/cdlib/
  data/          P1  loaders, splits, pair augs
  engine/        P2  contract-blind trainer
  cli/           P2  train; P5  evaluate / benchmark / export_masks
  models/
    encoders/    P3  resnet18, efficientnet_b0/b2
    fusion/      P3  absdiff, signed_fusion
    decoders/    P3  unet_decoder
    baselines/   P2  rgb_ssim, fc_siam_diff; P3  siamese_resnet18
    alignment/   P4  identity, bounded
    heads/       P4  confidence, directional
    proposed.py  P4  composes the above
    build.py     P2  build_model / build_dataset / build_loss / build_optimizer
  losses/        P2  bce_dice; P4  pair_order, calibration
  metrics/       P5  segmentation, boundary, region, calibration, swap, …
configs/         Hydra groups (data, model, loss, train, experiment, metrics)
paper/           ACM acmart skeleton (P5) + P3 experiments draft
research/        six briefs — read these before guessing
team-claude/     per-role CLAUDE.local.md checklists
docs/member-tracks/  later paper-completion tracks (Members 1–5)
```

## Model names (`model.name`)

| Key | What it is |
|---|---|
| `rgb_ssim` | Non-DNN floor |
| `fc_siam_diff` | TorchGeo FC-Siam-Diff (not Daudt's original trunk — say so in the paper) |
| `siamese_resnet18` | P3 stronger baseline: ResNet-18 + abs-diff + U-Net |
| `proposed` / `proposed_effnet` | Encoder + signed fusion + alignment + decoder + heads |

## Non-negotiable rules

1. Aggregate (corpus) F1, not per-image mean.
2. Scene/source-disjoint splits; never split at frame level.
3. LEVIR-CD: 256×256 non-overlapping crops → 7,120 / 1,024 / 2,048.
4. Geometric aug shared across the pair; photometric aug independent per frame.
5. Siamese BatchNorm: concat `[I1;I2]` on the batch dim, one forward, split.
6. Checkpoint every 10–15 minutes of wall clock, including RNG state.
7. Notebooks: env bootstrap + one CLI call + a plot. Nothing else.
8. ≥3 seeds for principal models. No significance tests at n=3.
9. Never commit footage, `.env`, or W&B keys.

## Ownership

See `.github/CODEOWNERS`. Adding a component is **one new file + one registry
line**. Do not edit `trainer.py` or `build.py` to add a feature.
