# Research Dossier — Pair-Order Consistent and Uncertainty-Aware Semantic Change Detection

Six research briefs compiled for the proposal in `../proposal.pdf` (University of Moratuwa, CSE, semester-scope; free Colab/Kaggle compute). ~244 KB total, all citations web-verified at time of writing (2026-09-06).

| # | Brief | Covers |
|---|---|---|
| 1 | [Baselines & architectures](01-baselines-architectures.md) | FC-Siam-Diff exact architecture, Siamese ResNet-18, EfficientNet encoders, BIT/ChangeFormer feasibility, fusion strategies, LEVIR/SYSU leaderboards, reproducibility gotchas |
| 2 | [Datasets & data pipeline](02-datasets-data-pipeline.md) | SYSU-CD, LEVIR-CD(+), WHU-CD, PCD, ChangeSim, CLEVR-Change; licensing, splits, changed-pixel ratios; video-substitute datasets; frame correspondence; annotation protocol |
| 3 | [Alignment & pair-order consistency](03-alignment-pair-order.md) | Bounded change-aware alignment, the collapse danger, PyTorch design, symmetry/order-invariance theory, directional SCD, **novelty assessment + ablation table** |
| 4 | [Uncertainty & calibration](04-uncertainty-calibration.md) | MC Dropout/ensembles/TTA/evidential/SWAG, ECE under imbalance, risk–coverage & selective prediction, distribution shift, conformal prediction, loss–calibration interactions |
| 5 | [Evaluation & efficiency](05-evaluation-efficiency.md) | Boundary IoU/HD95, aggregate-vs-per-image F1, connected-component false-alert rates, nuisance corruption suite, FLOPs/latency/memory measurement, seed statistics, tracking |
| 6 | [Engineering & delivery](06-engineering-delivery.md) | Repo layout, frozen interfaces, Hydra configs, Colab/Kaggle workflow, CI, risk register, critical path, ethics/licensing, writing toolchain |

---

## The seven findings that should change the proposal

**1. There is a January 2026 paper that may pre-empt contribution (B).**
Dong et al., *"Exchange Is All You Need for Remote Sensing Change Detection,"* arXiv:2601.07805, formalises bi-temporal order-invariance as an orthogonal permutation operator. Not yet peer-reviewed, but close enough in spirit that **someone must read it in full in week 1** and flag it to the supervisor. This is a week-1 risk check, not a week-10 surprise. (Brief 3 §7)

**2. Signed fusion makes order-consistency a real constraint, not a freebie — and that is the strongest version of the argument.**
With signed difference fusion, swapping order gives `h(b−a) = h(−(a−b))`, which equals `h(a−b)` only if `h` is even. Nothing in standard training enforces that. Weight sharing alone buys symmetry only for symmetric fusion (`|a−b|`, `a+b`, `max`). Write this out formally — it is the crux fact reviewers will look for, and it is exactly why the loss is needed. (Brief 3 §4.1)

**3. The planned BCE+Dice loss actively undermines the calibration deliverable.**
Dice loss is documented to cause overconfidence (Mehrtash et al., IEEE TMI 2020; Yeung et al., DSC++). Focal loss, by contrast, *improves* calibration (Mukhoti et al., NeurIPS 2020). The proposal promises both BCE+Dice *and* ECE/Brier/risk–coverage results — these pull against each other. Fixes: post-hoc temperature scaling fitted on shift-representative data, or a calibration-aware auxiliary term. (Brief 4 §6)

**4. Naive pixel-ECE will make your calibration look great and mean nothing.**
At 2–5% changed pixels, pooled all-pixel ECE is dominated by easy background. Use class-wise / foreground-restricted ECE plus equal-mass ("adaptive") binning. Same structural issue as reporting overall accuracy on an imbalanced task. (Brief 4 §2)

**5. A well-tuned plain U-Net matches ChangeFormer on LEVIR-CD.**
*A Change Detection Reality Check* (Corley et al., arXiv:2402.06994): plain U-Net ResNet-50 scores F1 90.38, U-Net-SiamDiff 90.46, versus ChangeFormer 91.11 and TinyCD 91.05 — no architectural novelty, just a modern recipe. On corrected WHU-CD splits, simple U-Nets **beat** BIT and ChangeFormer. Two consequences: give your Siamese ResNet-18 baseline a genuinely tuned recipe, and expect that a naively-trained BIT may underperform it for recipe reasons — anticipate this in the write-up so it does not read as a bug. (Brief 1 §7)

**6. EfficientNet's FLOP advantage is not a speed advantage.**
Depthwise separable convs have low arithmetic intensity; combined with unfused SiLU, B0/B2 can be **slower in wall-clock than ResNet-18 on T4/P100 despite ~4× fewer FLOPs**. Default to ResNet-18 (also simpler for the Siamese-BN issue) unless you benchmark step time first. (Brief 1 §3)

**7. Two datasets will bite you.**
PCD's original host is dead; only TSUNAMI is downloadable and **GSV is not hosted at all** — archive on first access, keep GSV off the critical path. WHU-CD has **no official split** and its commonly circulated version has **~85% train/test leakage** — pick a split, document it, and never cite literature baselines as directly comparable. (Briefs 1 §7, 2 §4–5)

---

## Cross-cutting decisions to lock in week 1

- **Metric protocol:** aggregate (corpus) F1, not per-image averaged. State it explicitly. Report changed-pixel ratio alongside. (Briefs 1 §7, 5 §2)
- **Crop convention:** LEVIR-CD 256×256 non-overlapping (7,120 / 1,024 / 2,048) or your numbers compare to nothing. (Brief 1 §7)
- **Splits:** scene/source-disjoint, never frame-level, with temporal buffers at cut points. (Brief 2 §8)
- **Baseline implementation:** TorchGeo `FCSiamDiff` with swappable `encoder_name` — one code path serves the mandatory baseline, the stronger baseline, and the proposed model. (Brief 1 §1)
- **Siamese BN:** concatenate `[I1;I2]` along the batch dim into one forward pass. (Brief 1 §2)
- **Frozen interfaces + registries** before parallel work starts. (Brief 6 §1)

## Effort picture

Individual briefs estimate 80–200 person-hours each for their own scope; these overlap heavily. The integrated bottom-up plan in **Brief 6** is the one to use: **≈448 person-hours**, ~7–8 hrs/person/week for 5 people over 12–13 weeks. It front-loads infra/registry work in weeks 1–2 precisely so the rest parallelises.

## Provenance

Briefs 1, 2, 5, 6 and 3, 4 were produced by separate research agents with live web search across two batches (the first batch lost three agents to a rate limit; all were re-run). Brief 6 was recovered from the session transcript after context compaction. Where an agent could not verify a claim it is marked inline as inferred or unconfirmed — those markers are deliberate, keep them until checked.
