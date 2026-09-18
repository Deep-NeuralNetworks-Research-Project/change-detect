# My role: P2 — Baseline & Infrastructure Lead

> Copy this to `CLAUDE.local.md` at the repo root. It layers on top of the shared `CLAUDE.md`.

**I own:** `src/cdlib/engine/**`, `src/cdlib/cli/**` (except `benchmark.py` → P5), `src/cdlib/utils/**`, `configs/**`, `.github/workflows/**`, `notebooks/**`, `src/cdlib/models/build.py`, `src/cdlib/models/baselines/{rgb_ssim,fc_siam_diff}.py`, `src/cdlib/losses/bce_dice.py`
**CODEOWNERS:** `/src/cdlib/engine/ /src/cdlib/cli/ /configs/ /.github/ /src/cdlib/models/build.py /src/cdlib/models/baselines/rgb_ssim.py /src/cdlib/models/baselines/fc_siam_diff.py /src/cdlib/losses/bce_dice.py @me`
(P5 adds a *later*, more-specific line for `/src/cdlib/cli/benchmark.py` — later lines win in CODEOWNERS, so order matters in that file.)
**My week-1 work makes everyone else's work parallelisable.** Front-load the scaffolding; the registry pattern is what stops five people colliding.

## What I must not break

All four frozen contracts — I'm the one who consumes them. `trainer.py` backprops `out["loss"]` and auto-logs every `loss/*` key without ever inspecting what's inside. If I find myself special-casing a particular loss or model in the trainer, the contract is wrong and we fix the contract, not the trainer.

## Gotchas

**Siamese BatchNorm.** Concatenate `[I1;I2]` along the **batch** dim into one forward pass, then split. Two sequential forwards give BN half the effective batch and mix running statistics from both time-points asymmetrically. This is a real accuracy bug, not a style preference.

**Use TorchGeo's `FCSiamDiff`, don't hand-roll the original.** It wraps `segmentation_models_pytorch` and exposes `encoder_name`, so one code path serves the mandatory baseline, P3's ResNet-18 baseline, and P4's proposed model. Caveat for the paper: it is *not* architecturally identical to Daudt's 4-stage 16/32/64/128 net — note the substitution rather than claiming it is "the" FC-Siam-Diff.

**The original FC-Siam-Diff paper gives no training recipe.** No optimiser, LR, batch size, epochs, or patch size. Budget real tuning time; don't treat a bad first run as an architecture result.

**Baselines must be genuinely tuned.** *A Change Detection Reality Check* (arXiv:2402.06994) shows a well-tuned plain U-Net matching ChangeFormer. A strawman baseline is worse than no baseline — it makes our contribution look fake to a reviewer who knows this paper. Give the ResNet-18 baseline the same LR sweep the proposed model gets.

**Checkpoint on wall-clock, not epochs.** Every 10–15 min: model, optimiser, scheduler, epoch, global_step, RNG states (torch/numpy/python), best_metric, cfg_hash. `resume_from=auto` picks newest by mtime and restores RNG so augmentation continues deterministically. Retain last-2 + best. Colab preempts; this is certain.

**Kaggle `/kaggle/working` is ephemeral.** W&B or Drive is the durable store. Never treat the working dir as persistence.

**CI budget is hard.** Under 2–3 min if the repo is private (2,000 free Linux min/month). CPU-only, synthetic tensors, batch=2, H=W=32.

---

## Checklist

### Week 1 — scaffolding
- [ ] Read `research/06-engineering-delivery.md` §1–5 and `research/01` §1–2
- [ ] Repo skeleton per the layout in `research/06` §1; `pyproject.toml`, `pip install -e .`
- [ ] `utils/registry.py` — generic `Registry` class; then the five `*_REGISTRY` instances
- [ ] `models/build.py` with the four builders: `build_model / build_dataset / build_loss / build_optimizer`
- [ ] Write the five frozen contracts into `docs/architecture.md` — day one, before anyone codes against them
- [ ] Hydra + OmegaConf config tree; light `configs/schema.py` dataclasses for CI typo-catching only
- [ ] `.pre-commit-config.yaml`: ruff, black, nbstripout, check-yaml, + custom hook grepping for hardcoded `/content/` or `/kaggle/` outside `notebooks/`
- [ ] `utils/seed.py` — seeds python/numpy/torch/cuda, `torch.use_deterministic_algorithms(True)` where feasible
- [ ] CODEOWNERS + branch protection (1 non-author approval, no direct push to `main`)

### Week 2 — engine and CI
- [ ] `engine/trainer.py` — contract-blind: backprop `out["loss"]`, auto-log `loss/*`
- [ ] `utils/checkpoint.py` — 10–15 min wall-clock cadence, full RNG state, `resume_from=auto`, keep last-2 + best
- [ ] Auto-capture at run start: `config.yaml`, `commit.txt` (+ dirty diff), `command.txt`, `env.txt` (pip freeze + CUDA/cuDNN/GPU model), `checkpoint_sha256.txt`
- [ ] `ci.yml`: ruff + black --check → `pytest -m "not gpu"` → Hydra-compose every `configs/experiment/*.yaml` and dry-construct
- [ ] `tests/test_registries.py` (no dup keys, all importable), `tests/test_shapes.py` (parametrized over every registry key), `tests/test_overfit_one_batch.py`, `tests/test_determinism.py`, `tests/test_config_validate.py` (backs the CI config job)
- [ ] Colab + Kaggle driver notebooks — bootstrap cell, one CLI call, one plotting cell. Nothing else
- [ ] Drive symlinks for `results/` and `checkpoints/`; W&B key via Colab Secrets / Kaggle Add-ons
- [ ] `requirements-lock-colab.txt` and `requirements-lock-kaggle.txt` (base images differ — two files, pinned)

### Week 3 — baselines
- [ ] `models/baselines/rgb_ssim.py` — non-DNN floor, ~4h of work. Threshold `SSIM(I1,I2) < T` and `|I1−I2| > T`. Expect it to fail hard on exactly our nuisance list; that failure *is* the "why learning" narrative
- [ ] `models/baselines/fc_siam_diff.py` via TorchGeo `FCSiamDiff`, `encoder_name` swappable
- [ ] Verify the batch-dim-concat BN pattern is actually what runs
- [ ] LR sweep: `python -m cdlib.cli.train -m train.lr=1e-3,3e-4,1e-4 model=fc_siam_diff`
- [ ] Log baselines to the shared benchmark table with ≥3 seeds

### Weeks 4+ — keep the machine running
- [ ] Loss weighting wired from P1's per-dataset changed-pixel ratios, recomputed per training stage
- [ ] **Talk to P5 before finalising the loss**: Dice hurts calibration and we promise calibration results. Decide together on temperature scaling vs a calibration-aware term
- [ ] Named experiment configs so ablations are one-line: `ablation_no_alignment`, `ablation_no_confidence_gate`, `ablation_no_pairorder` (P4 defines what each overrides)
- [ ] Watch CI minutes; make the repo public if course policy allows (unlimited CI)
- [ ] Keep `docs/architecture.md` current — bus-factor insurance

## Commands I'll use

```bash
pip install -e . && pre-commit install && nbstripout --install
pytest -m "not gpu" -q
python -m cdlib.cli.train +experiment=baseline_fcsiamdiff_sysu
python -m cdlib.cli.train -m train.lr=1e-3,3e-4,1e-4 model=fc_siam_diff,siamese_resnet18
python -m cdlib.cli.train resume_from=auto        # after a Colab preemption
```

## Hand-offs

| To | What | When |
|---|---|---|
| everyone | registries, builders, trainer, `docs/architecture.md` | end of week 1 |
| everyone | green CI + working Colab/Kaggle drivers | end of week 2 |
| P5 | multirun sweep support (`-m`) for ablation orchestration | **end of week 2**, with Hydra — P5 expects it from week 2 |
| P5 | baseline numbers with ≥3 seeds for the benchmark table | week 4 |

**I depend on:** P1's loaders (end of week 2) before any real training run; **P5's Dice-vs-calibration decision (week 4)** before I finalise the loss config.
