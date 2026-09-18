# My role: P5 — Metrics, Robustness & Paper Integration

> Copy this to `CLAUDE.local.md` at the repo root. It layers on top of the shared `CLAUDE.md`.

**I own:** `src/cdlib/metrics/**`, `src/cdlib/cli/benchmark.py`, ablation orchestration, `paper/main.tex`, `references.bib`, figure standards
**CODEOWNERS:** `/src/cdlib/metrics/ /paper/ /src/cdlib/cli/benchmark.py @me`
(This line must sit **after** P2's `/src/cdlib/cli/` line in the CODEOWNERS file — later, more-specific lines win.)
**Everything the project claims passes through me.** If a metric is defined wrong, every table in the paper is wrong — and several of these metrics have a wrong-but-plausible version that looks fine. My unit tests against hand-computed values are the only thing standing between us and a silently invalid result.

## What I must not break

The metric contract: `reset() / update(outputs, batch) / compute() -> dict[str,float]`. Consider literally subclassing `torchmetrics.Metric` — free DDP-safety, and it forces the contract.

## Decisions I make once, for everyone

**Aggregate (corpus) F1, not per-image averaged.** Sum TP/FP/FN across the whole test set, then compute F1 once. Per-image averaging gives a nearly-empty tile the same weight as a heavily-changed one; under our imbalance the two diverge substantially. This is documented as *the* reason CD numbers don't reconcile across papers (Corley et al., arXiv:2402.06994). Implement both, report aggregate, state it explicitly in the paper.

**Report the changed-pixel ratio alongside every result.** Without it, F1 is uninterpretable and overall accuracy is meaningless (98% accuracy at 2% change is a model that predicts nothing).

**One decision threshold, fixed on validation scenes only.** Never tuned on test. Record whether it's F1-optimal or precision-constrained.

## Gotchas

**Nuisance corruption must be applied to ONE frame of the pair only.** That is what makes it a nuisance test rather than a global image transform. Corrupting both frames identically tests nothing. Run both directions (T1 clean/T2 corrupted, and the reverse) — asymmetry there is itself a finding, and it interacts with P4's order-consistency claim.

**Naive pixel-ECE is dominated by easy background** at 2–5% changed pixels and will look fantastic while meaning nothing. Use class-wise / foreground-restricted ECE with equal-mass (adaptive) binning, not uniform-width bins.

**Dice loss causes overconfidence** (Mehrtash et al., IEEE TMI 2020; Yeung et al. DSC++), and Focal *improves* calibration (Mukhoti et al., NeurIPS 2020). Our planned BCE+Dice therefore fights our calibration deliverable. **Raise this with P2 and P4 by week 4** — the fix is temperature scaling fitted on shift-representative data (not clean validation), or a calibration-aware auxiliary term.

**Temperature scaling must be fitted on shift-representative data.** Ovadia et al. (NeurIPS 2019): calibration degrades under distribution shift, and a temperature fitted on clean validation doesn't transfer. We train on aerial/street and test on video — this is exactly that situation.

**FLOPs for a two-input model.** fvcore, ptflops, thop and `torch.utils.flop_counter` disagree, and MACs vs FLOPs is a factor of 2. Pick one tool, state which, state the convention, and pass both inputs.

**Latency measurement.** Warm-up iterations excluded, `torch.cuda.synchronize()`, CUDA events, repeats, median + IQR, fixed batch size, AMP/TF32 disclosed. **Log the GPU model every run** — Colab's hardware varies run to run, so an unlogged latency number is meaningless.

**Peak memory:** `torch.cuda.max_memory_allocated()` ≠ reserved ≠ what `nvidia-smi` shows. Pick one, say which.

**At n=3 seeds, don't run significance tests.** They aren't meaningful. Report mean ± std or median + range, plus bootstrap CIs over the test set.

---

## Checklist

### Week 1–2 — foundations (front-load; everything downstream depends on these)
- [ ] Read `research/05-evaluation-efficiency.md` fully, and `research/04-uncertainty-calibration.md` §2, §3, §6
- [ ] `metrics/segmentation.py` — P/R/F1/IoU with **both** pooling protocols, changed-pixel ratio reporting
- [ ] `tests/test_metrics_handcomputed.py` — hand-built 4×4 and 5×5 mask pairs with calculator-verified values. **Write these before trusting any number**
- [ ] W&B project set up; `utils/reproducibility.py` seeding utility (with P2)
- [ ] **`cli/benchmark.py` minimal version by end of week 2** — warm-up-excluded latency (median + IQR), `torch.cuda.synchronize()`, CUDA events, GPU model logged. **P3 is blocked on this for their week-3 encoder benchmark**
- [ ] Agree `nuisance_label` taxonomy with P1 so my stratification bins match their labels
- [ ] Overleaf project + `acmart` (`\documentclass[sigconf]{acmart}`), section files, `references.bib` skeleton with `author_year_firstword` keys

### Week 2–3 — the corruption suite (fiddliest thing I own)
- [ ] `metrics/robustness.py` + corruption suite, all applied to **one frame only**:
  - [ ] brightness / gamma / colour grading (LUT or ASC-CDL)
  - [ ] Gaussian blur, motion blur
  - [ ] JPEG compression
  - [ ] **H.264/HEVC round-trip via ffmpeg at a given CRF** — encode the still as a short static clip so the encoder produces realistic I/P-frame structure, then extract the middle frame back
  - [ ] synthetic shadows (Albumentations `RandomShadow`)
  - [ ] small random homography / translation for viewpoint jitter
  - [ ] occlusion (`CoarseDropout` / paste)
- [ ] Severity levels calibrated visually (ImageNet-C style), documented
- [ ] Run both corruption directions (T1 clean/T2 corrupted and reverse)
- [ ] Coordinate with P1 in **week 2, before I start building**: their nuisance mining is for *training negatives*, mine is for *evaluation*. One implementation, two callers

### Week 3–4 — boundary, region, calibration, efficiency
- [ ] `metrics/` Boundary IoU (Cheng et al. CVPR 2021, `dilation_ratio=0.02`), HD95 + ASD via MONAI; handle empty-mask edge cases
- [ ] Connected-component metrics: `cv2.connectedComponentsWithStats`, min-region-size filtering, per-image false-alert rate (a pair is "alerted" if any predicted region exceeds size *k*), object-level matched-component F1 at an IoU threshold. Include a min-size-vs-metric sweep, not one fixed *k*
- [ ] `metrics/calibration.py`: ECE (**foreground-restricted + class-wise**, equal-mass bins), Brier, NLL, reliability diagrams
- [ ] Risk–coverage: selective risk, coverage, AURC, E-AURC. **Per-pair rejection matters most** — we need an image-level deferral policy for the human-in-the-loop story
- [ ] *(moved up — see week 2)* full `cli/benchmark.py` polish: params (trainable vs total), pair-input FLOPs, peak memory
- [ ] `metrics/registry.py` entries; shape tests

### Week 4 — raise the loss/calibration conflict
- [ ] Bring the Dice-overconfidence finding to P2 and P4 with the citations
- [ ] Agree the fix: post-hoc temperature scaling on shift-representative data, and/or calibration-aware auxiliary term
- [ ] Implement temperature scaling; verify it's fitted on the right data

### Weeks 5–9 — orchestration and analysis
- [ ] `stats.py`: bootstrap CIs over the test set, ablation table generator
- [ ] Orchestrate ablations across seeds with P2's multirun support
- [ ] Nuisance-stratified evaluation: results by condition × severity × change-size bin, as tables and heatmaps
- [ ] Run P4's swap-consistency metric across **every** model, including ones not trained with the loss (B7)
- [ ] Full evaluation: all baselines × ≥3 seeds × corruption suite × efficiency. **Largest wall-clock risk in the project** — budget Colab/Kaggle quota deliberately
- [ ] Qualitative overlays for best/median/worst pairs

### Weeks 8–13 — paper integration (I own `main.tex`)
- [ ] Enforce figure standards: vector PDF; **colour-blind-safe overlays, no red-green**. TP blue, FP vermillion/orange, FN magenta/purple (Okabe-Ito); diverging blue↔orange for confidence maps. Validate with a CVD simulator
- [ ] Weekly review of section drafts — not just at the end
- [ ] `.bib` hygiene: no unused/duplicate entries, DOIs present
- [ ] Draft abstract, intro, results, discussion, conclusion
- [ ] Page limit and format compliance
- [ ] `MODEL_CARD.md` per shipped model: intended use, training data + licences, eval data, metrics with cross-seed CIs, known failure modes, explicit out-of-scope statement (**not a production decision tool**)

## Commands I'll use

```bash
pytest tests/test_metrics_handcomputed.py -v
python -m cdlib.cli.evaluate exp_id=<id> +metrics=all      # metrics group is plural everywhere
python -m cdlib.cli.evaluate +robustness=full          # stratified by condition × severity
python -m cdlib.cli.benchmark model=rgb_ssim,fc_siam_diff,siamese_resnet18,proposed_effnet
python -m cdlib.cli.export_masks exp_id=<id> --overlays best,median,worst
```

## Hand-offs

| To | What | When |
|---|---|---|
| P1 | `nuisance_label` taxonomy agreement | week 2 |
| P2, P4 | the Dice-vs-calibration conflict + recommended fix | **week 4** |
| P3 | `cli/benchmark.py` — they are blocked on it | **end of week 2** |
| everyone | benchmark table format + figure standards | week 3 |
| team | weekly section review | weeks 8–13 |

**I depend on:** P2's trainer + multirun (end of week 2), P1's `nuisance_label` and corruption-suite coordination (week 2), P4's swap-consistency metric (week 5), everyone's results for the tables.

**I own the Dice-vs-calibration decision** (week 4) — I raise it, P2 and P4 implement against it.
