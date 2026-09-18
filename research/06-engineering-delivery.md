# Software Architecture, Collaboration & Delivery Brief
**Project:** Pair-Order Consistent and Uncertainty-Aware Semantic Change Detection from Paired RGB Frames — University of Moratuwa, CSE, 5-person semester project (Ekanayake, Lelwala, Pabasara, Bulagala, Rajapaksha)

Grounded in `proposal.pdf` (RGB/SSIM, FC-Siam-Diff, Siamese ResNet-18, EfficientNet-B0/B2-based proposed model with signed+absolute fusion, bounded alignment, confidence head, pair-order consistency; SYSU-CD/LEVIR-CD/PCD + authorised video set; 3-seed principal runs; P/R/F1/IoU + boundary + robustness + calibration + efficiency metrics).

---

## 1. Repository layout

```
change-detect/
├── pyproject.toml                 # single pip-installable package: pip install -e .
├── requirements-lock-colab.txt    # pip freeze, per-runtime (Colab base image differs from Kaggle)
├── requirements-lock-kaggle.txt
├── .pre-commit-config.yaml
├── .github/workflows/{ci.yml, config-validate.yml}
├── configs/
│   ├── config.yaml                 # root Hydra defaults list
│   ├── data/{sysu_cd,levir_cd,pcd,video_set}.yaml
│   ├── model/{rgb_ssim,fc_siam_diff,siamese_resnet18,proposed_effnet}.yaml
│   ├── loss/{bce_dice,bce_dice_pairorder}.yaml
│   ├── train/default.yaml
│   └── experiment/                 # one file = one ablation, one-line diff (see §2)
│       ├── baseline_fcsiamdiff_sysu.yaml
│       ├── ablation_no_alignment.yaml
│       ├── ablation_no_confidence_gate.yaml
│       └── ablation_no_pairorder.yaml
├── src/cdlib/
│   ├── data/{datasets/{sysu_cd,levir_cd,pcd,video_pairs}.py, transforms.py, splits.py, registry.py}
│   ├── models/
│   │   ├── encoders/{resnet.py, efficientnet.py, registry.py}     # ENCODER_REGISTRY
│   │   ├── fusion/{absdiff.py, signed_fusion.py, registry.py}     # FUSION_REGISTRY
│   │   ├── alignment/{bounded_alignment.py, identity.py, registry.py}  # ALIGNMENT_REGISTRY
│   │   ├── decoders/{unet_decoder.py, registry.py}
│   │   ├── heads/{mask_head.py, confidence_head.py, directional_head.py}
│   │   ├── baselines/{rgb_ssim.py, fc_siam_diff.py, siamese_resnet18.py}
│   │   ├── proposed.py             # composes encoder+fusion+alignment+decoder+heads
│   │   └── build.py                # build_model(cfg) -> nn.Module  (single entrypoint)
│   ├── losses/{bce_dice.py, pair_order_consistency.py, calibration.py, registry.py}  # LOSS_REGISTRY
│   ├── metrics/{segmentation.py, robustness.py, calibration.py, efficiency.py, registry.py}
│   ├── engine/{trainer.py, evaluator.py, callbacks.py}
│   ├── cli/{train.py, evaluate.py, export_masks.py, benchmark.py}
│   └── utils/{seed.py, checkpoint.py, logging.py, registry.py}   # generic Registry class
├── notebooks/{colab_train_driver.ipynb, kaggle_train_driver.ipynb}   # thin drivers only
├── scripts/{download_sysu_cd.sh, download_levir_cd.sh, prepare_video_set.py, make_overlays.py}
├── tests/{test_shapes.py, test_overfit_one_batch.py, test_determinism.py,
│          test_metrics_handcomputed.py, test_registries.py, test_config_validate.py}
├── results/exp_<id>/{config.yaml, seed_0/, seed_1/, seed_2/, env.txt, commit.txt, command.txt, checkpoint_sha256.txt}
└── docs/{MODEL_CARD.md, DATA_CARD.md, ETHICS.md, architecture.md}
```

**Why this works for 5 people:** every pluggable component (encoder, fusion, alignment, decoder, loss, metric) lives in its own file plus one line in a `*_REGISTRY`. Adding a component means writing a new file and registering a key — never editing `trainer.py`, `build.py`, or another person's component file. PRs collide almost never because the diff surface per task is one new file + one registry line.

### Frozen interfaces (lock these in week 1–2, before parallel work starts)

**Dataset `__getitem__` contract** (every dataset in `data/datasets/*` must return exactly this):
```python
def __getitem__(self, idx: int) -> dict:
    return {
        "img1": Tensor[C,H,W] float32 [0,1],
        "img2": Tensor[C,H,W] float32 [0,1],
        "mask": Tensor[1,H,W] float32 {0,1,-1},   # -1 = ignore region
        "nuisance_label": Tensor[] int64,          # 0=clean, 1..k=nuisance type, -1=unknown
        "meta": {"source_video": str, "scene_id": str, "frame_idx": tuple[int,int],
                  "pair_id": str, "dataset": str},
    }
```

**Model `forward` contract** (every baseline + the proposed model, in `build_model` output):
```python
def forward(self, img1: Tensor, img2: Tensor) -> dict:
    return {
        "logits": Tensor[B,1,H,W],                 # change-mask logits (pre-sigmoid)
        "confidence": Tensor[B,1,H,W] | None,       # confidence-head output
        "aux": {                                     # experimental heads only ADD keys here
            "directional_logits": Tensor[B,2,H,W] | None,  # appeared/disappeared, when available
            "alignment_offset": Tensor[B,2,H,W] | None,
        },
    }
```

**Loss contract:**
```python
def compute(self, outputs: dict, batch: dict) -> dict:
    return {"loss": Tensor[], "loss/bce": Tensor[], "loss/dice": Tensor[], "loss/pairorder": Tensor[]}
```
`trainer.py` backprops `out["loss"]` and auto-logs every `loss/*` key to W&B — it never needs to know what's inside a given loss.

**Metric contract** (torchmetrics-style; consider literally subclassing `torchmetrics.Metric` — free DDP-safety even on single-GPU Colab):
```python
class Metric:
    def reset(self) -> None: ...
    def update(self, outputs: dict, batch: dict) -> None: ...
    def compute(self) -> dict[str, float]: ...
```

**Registry builders** (only 4 entrypoints the CLI ever calls):
```python
def build_model(cfg) -> nn.Module
def build_dataset(cfg, split: str) -> Dataset
def build_loss(cfg) -> Loss
def build_optimizer(cfg, params) -> Optimizer
```

Write these five contracts into `docs/architecture.md` on day one and enforce with `tests/test_shapes.py` (parametrized over every registry key) + `tests/test_registries.py`. Nobody renames or removes a key without a team-wide PR review.

---

## 2. Config management — recommendation: **Hydra + OmegaConf, YAML-first**

Rejected: plain argparse (one big file → constant merge conflicts, no composition, doesn't scale to 5 people); fully-typed dataclass-everywhere OmegaConf structured configs (too much boilerplate for a semester deadline, slows prototyping).

Recommended: Hydra's config-group composition + CLI overrides, with a *light* structured-config layer (a handful of dataclasses in `configs/schema.py` — `TrainConfig`, `DataConfig`, `ModelConfig`) used only for typo-catching validation in CI, not for every leaf.

Why Hydra specifically:
- **One-line ablation diffs**, exactly as required: `python -m cdlib.cli.train +experiment=ablation_no_alignment` where the whole ablation is
```yaml
# configs/experiment/ablation_no_alignment.yaml
defaults: [override /model: proposed_effnet, _self_]
model:
  alignment: identity   # swap ALIGNMENT_REGISTRY entry; everything else unchanged
```
- **Multirun sweeps** reproduce the proposal's own tuning grid for free: `python -m cdlib.cli.train -m train.lr=1e-3,3e-4,1e-4 model=fc_siam_diff,siamese_resnet18`.
- **Auto-saved resolved config per run** feeds directly into the `results/<exp_id>/config.yaml` reproducibility artefact (§5) — no manual bookkeeping.
- CLI overrides (`data=sysu_cd model.encoder=efficientnet_b0 seed=1`) mean nobody edits a shared config file to run a variant — eliminates the most common config merge conflict.

---

## 3. Colab / Kaggle workflow

- **Bootstrap cell** (identical in both notebooks): `!git clone <repo> && cd change-detect && pip install -e . -q`.
- **Drive** (Colab): mount once, symlink `results/` and `checkpoints/` to `/content/drive/MyDrive/cd-project/...` so state survives runtime resets.
- **Kaggle**: attach datasets as read-only `/kaggle/input/...`; `/kaggle/working` is ephemeral unless you "Save Version" or push to a Kaggle Dataset/Drive/W&B — treat W&B (or Drive) as the durable store, not `/kaggle/working`.
- **Secrets**: Colab Secrets pane (`google.colab.userdata`) / Kaggle Add-ons → Secrets for `WANDB_API_KEY`; never commit `.env` or `.netrc`; gitignore `wandb/`.
- **Anti-drift rule**: notebooks contain only (a) env bootstrap, (b) one call into `cdlib.cli.train` / `!python -m cdlib.cli.train ...`, (c) a results-plotting cell calling `cdlib.utils` plot helpers. Zero inline model/loss/data logic — everything importable and unit-tested. Enforce with `nbstripout --install` (pre-commit) which strips outputs/execution counts before every commit — this alone eliminates most notebook diff noise; escalate to `nbdime` only for genuine structural conflicts.
- **Checkpoint/resume for free-tier timeouts** (Colab ≈12h/90min-idle, Kaggle 12h/30min-idle): `trainer.py` writes every 10–15 min of wall clock (not just per-epoch, given preemption risk): `{model_state, optimizer_state, scheduler_state, epoch, global_step, rng_states(torch/numpy/python), best_metric, cfg_hash}` to the Drive-mounted path. `resume_from=auto` picks the newest checkpoint by mtime and restores RNG state so shuffling/augmentation continues deterministically. Retain last-2 + best to bound Drive storage.

---

## 4. Git workflow, CI, and testing

**Branches:** `type/scope-desc` — `feat/encoder-efficientnet`, `fix/dice-loss-nan`, `exp/ablation-alignment`. One branch per registry component keeps parallel work file-disjoint by construction.

**Review:** CODEOWNERS mapping directories to the responsible person (`/src/cdlib/data/ @P1`, `/src/cdlib/models/alignment/ @P4`, …) + branch protection requiring 1 non-author approval before merge to `main`. Never push to `main` directly; `main` is always CI-green.

**CI (`ci.yml`, must run in <2–3 min — keep this hard budget if the repo is private, since free tier is 2,000 Linux min/month; unlimited if public):**
1. `ruff` + `black --check`.
2. `pytest -m "not gpu"` — CPU-only, tiny synthetic tensors (batch=2, H=W=32).
3. Config validation job — Hydra-compose every `configs/experiment/*.yaml` and dry-construct via `build_model`/`build_dataset` (no training) to catch broken configs before merge.

**pre-commit hooks:** ruff, black, nbstripout, trailing-whitespace/end-of-file-fixer, check-yaml, plus a custom hook grepping for hardcoded `/content/` or `/kaggle/` paths outside `notebooks/`.

**Test strategy:**
- **Shape tests** — parametrized over every `ENCODER_REGISTRY`/`FUSION_REGISTRY`/`ALIGNMENT_REGISTRY`/`LOSS_REGISTRY` key, asserting the frozen forward/loss contract shapes.
- **Overfit-one-batch** — 2 synthetic pairs, ~50–100 steps, assert loss drops below a fixed fraction of its initial value; catches broken gradients/label mismatches in seconds on CPU.
- **Determinism test** — same seed + tiny config → two runs match logits within 1e-6 after N steps; validates `set_seed()` and seeded DataLoader shuffling.
- **Metric unit tests against hand-computed values** — a hand-built 4×4/5×5 mask pair with a calculator-verified P/R/F1/IoU/ECE, asserted against `metric.update()+compute()`.
- **Registry integrity test** — no duplicate keys, every entry importable, every `configs/model/*.yaml` builds against a dummy 2×2 input.

---

## 5. Reproducibility artefacts

Grounded in the **Papers-with-Code ML Code Completeness Checklist** (5 items: dependency spec, training code, evaluation code, released weights, README results table matching the paper — now part of NeurIPS's code-submission process) and the **NeurIPS reproducibility checklist** areas (claims match scope; limitations/societal impact stated; for experiments — reproduction instructions with a code link, training details incl. how hyperparameters/splits were chosen, error bars and how they were computed, compute resources per experiment, dataset license/citation/consent details, full preprocessing description).

Map directly onto `results/<exp_id>/`:
- `config.yaml` — fully resolved (Hydra-saved), every override baked in.
- `seed_{0,1,2}/` — one dir per seed for principal models (matches the proposal's ≥3-seed commitment), each with checkpoint + `metrics.json` + training log → gives error bars, not single-run numbers.
- `checkpoint_sha256.txt`, `commit.txt` (git hash + dirty diff), `command.txt` (exact CLI invocation), `env.txt` (`pip freeze` + CUDA/cuDNN/GPU model + Python version) — captured automatically by `trainer.py` at run start.
- `requirements-lock-{colab,kaggle}.txt` — pinned, not floating (base images differ between the two platforms).
- Global `set_seed()` seeds python/numpy/torch/cuda + `torch.use_deterministic_algorithms(True)` where feasible; document any non-deterministic ops kept for speed in `MODEL_CARD.md`.
- `MODEL_CARD.md` per shipped model (proposed, FC-Siam-Diff, ResNet-18): intended use, training data/licenses, eval data, metrics with cross-seed CIs, known failure modes (nuisance-shift cases), explicit out-of-scope statement (not a production decision tool), ethics notes.
- `DATA_CARD.md` per dataset (SYSU-CD, LEVIR-CD, PCD, video set): source, license, collection method, split methodology (scene-disjointness proof), known biases (aerial/street domain gap), and for the video set specifically: authorisation basis, storage location/access list, redistribution restriction, blurring status.

---

## 6. Project management

**Tracking:** GitHub Projects (Kanban, tied to issues/PRs — zero context switch from where the code lives) for code/experiment/ablation tasks; a shared Google Doc/Notion page for meeting notes, the risk register, and paper-drafting coordination (actual writing happens in Overleaf, §8). Trello adds nothing over GitHub Projects for a dev-heavy team — skip it.

**Cadence:** weekly 30–45 min sync (fixed slot) + 2×/week async standup (did/doing/blocked) in a chat channel, given unpredictable free-GPU availability; a milestone review once baselines land; a full-day integration sprint before the report deadline.

**Definition of done:**
- *Code task*: PR merged, CI green, test added/updated, registry entry documented, no new lint warnings.
- *Experiment task*: logged under `results/<exp_id>/` with the full artefact set (§5), entry added to the shared benchmark table, ≥3 seeds for principal models, overlays exported for best/median/worst-case pairs.
- *Writing task*: section drafted in Overleaf, every claim backed by a `references.bib` entry, reviewed by the integration owner, figures as vector PDF, no TODO/FIXME left.

### Risk register

| # | Risk | Likelihood | Impact | Mitigation | Owner role |
|---|---|---|---|---|---|
| 1 | Target-domain (video) annotation delay | Medium | High — blocks the post-production claim | Start authorisation + annotation-tool selection week 1; run public-dataset benchmarking fully in parallel so it isn't gating | Data lead |
| 2 | Domain mismatch: aerial/street pretrain vs. video test | High | Medium — weak transfer, disappointing final numbers | Report public-benchmark and video results separately (as proposal already scopes); fine-tune on video set, don't only test on it | Modeling lead |
| 3 | Class imbalance (unchanged pixels dominate) | High | Medium — inflated accuracy, poor F1 | Weighted BCE+Dice (already planned); stratify eval by change size | Modeling lead |
| 4 | Near-duplicate frame leakage across splits | Medium | High — silently invalid benchmark | `data/splits.py` enforces source/video/scene-level splits; unit test asserts no scene ID crosses splits | Data lead |
| 5 | Alignment module erasing real changes | Medium | High — core novelty fails silently | Bounded offset + confidence gate (as designed); with/without-alignment ablation; weekly qualitative review | Proposed-model owner |
| 6 | Free-GPU session preemption/timeout | High | Medium — wasted compute-hours | Checkpoint/resume every 10–15 min to Drive; `resume_from=auto` (§3) | Infra/MLOps lead |
| 7 | Notebook merge conflicts / "works on my notebook" drift | Medium | Medium — wasted debugging time | Registries + thin-notebook rule + nbstripout + CI (§4) | Infra/MLOps lead |
| 8 | Ethics/licensing breach on video footage (redistribution, un-blurred faces in a figure) | Low | Severe — grade/legal risk | Written permission letter, restricted storage, mandatory blur pipeline pre-export, signed-off publication-clearance log | Ethics/data-governance lead |
| 9 | Single-person bus factor on the proposed model | Medium | High — blocks ablations/writing if unavailable | Registry pattern forces documented interfaces; pair-programming + mandatory review; `docs/architecture.md` kept current | Team lead |
| 10 | Report integration crunch near deadline | High | Medium — poor-quality last-minute merge | Shared Overleaf, section outline agreed week 2, integration owner reviews weekly not just at the end, `.bib` maintained continuously | Writing/integration owner |
| 11 | GitHub Actions minutes exhausted (if repo private) | Low | Low — CI silently stops | Public repo if course policy allows (unlimited CI); else keep CI <3 min, CPU-only | Infra/MLOps lead |
| 12 | Compute budget insufficient for 3 seeds × 4 models × ablations × 3 datasets | Medium | High — can't meet the seed commitment | Prioritise seeds for principal models only ("where time permits", per proposal); iterate ablations fast on PCD (200 pairs) before scaling to SYSU-CD (20k pairs) | Team lead / Modeling lead |

---

### Critical-path analysis

```
data pipeline → baseline training (RGB/SSIM, FC-Siam-Diff, ResNet-18)
             → proposed model (encoder+fusion+alignment+confidence+pair-order)
             → ablations (fusion / alignment / confidence-gate / pair-order, nuisance-stratified)
             → writing (tables, figures, ACM paper)
```
Writing's background/related-work/method-baseline sections are **not** blocked and should start in parallel from week 1.

The video-set track (authorisation → annotation → fine-tune → final test) is deliberately **kept off the true critical path**: the proposal itself provides the escape hatch — *"If the authorised video test cannot be completed, the project will be reported as paired-image change detection and will make no claim about post-production performance."* Treat it as a high-value parallel stream owned by one person, gating only the strongest claim, not the deliverable.

**Suggested role split (rotate if needed):**
- **P1 – Data lead**: dataset loaders, scene-disjoint splits, augmentation, video-set governance/ethics paperwork.
- **P2 – Baseline/Infra lead**: RGB/SSIM + FC-Siam-Diff, trainer/engine, CI, Colab/Kaggle tooling.
- **P3 – Modeling A**: Siamese ResNet-18 baseline + encoder/fusion registry components.
- **P4 – Modeling B**: bounded alignment module, confidence head, pair-order consistency loss (the actual novelty — highest risk, so give it a dedicated owner, item 5/9 above).
- **P5 – Metrics/Integration lead**: metrics/calibration/efficiency measurement, ablation orchestration, paper integration owner (owns `main.tex`, `.bib`, figure standards).

This makes P1's public-dataset loaders (done ~week 2) the only true hard blocker for everyone else; the video pipeline, alignment novelty, and writing all run as parallel streams after that.

---

## 7. Ethics, licensing, and data governance (authorised video set)

Must be documented, per the proposal's own commitment ("Only authorised/licensed footage will be used; personal content will be minimized and publication examples will be cleared"):

- **Permission letter** from the rights holder, scoped explicitly: research/educational use, non-commercial, university project, retention duration, and whether redistribution of raw footage or extracted frames is permitted (default: no).
- **Restricted storage**: footage/frames on access-controlled storage (private Drive folder shared with team + supervisor only); never in the public git repo — gitignore `data/video_raw/` and any frame exports; track only checksums/manifests in git.
- **No redistribution of frames**: derived data (extracted frames, masks) inherits the same restriction unless the permission letter separates it; never upload to public dataset-sharing sites or a public GitHub release.
- **Publication clearance for figures**: any frame used as a qualitative example needs separate clearance for *publication* (stricter than internal evaluation use) — keep `docs/figure_clearance.md` mapping each published figure to a clearance record/date/approver.
- **Minimising personal content**: prefer footage without identifiable bystanders; where unavoidable, apply face blurring (simple face-detector + Gaussian blur pass) documented as a preprocessing step, mandatory before any figure leaves the private working set.
- **Ethics self-assessment**: check University of Moratuwa CSE/Faculty ethics requirements for group projects using externally-sourced video; keep a short `docs/ETHICS.md` referencing the supervisor-approved permission letter even if a full IRB isn't required.
- **DATA_CARD.md for the video set** should state: license/authorisation basis, storage location + access list, retention/deletion plan (e.g., deleted at semester end unless renewed permission is obtained), and the redistribution restriction explicitly.

**Suggested paper statement** (adapt to actual permission terms once signed):
> "All video footage used in this study was obtained with explicit authorisation from its rights holder(s) for research and educational use within this project; no raw footage or extracted frames are redistributed with this paper or in any public code/data release. Frames containing identifiable individuals were minimised, and any individual visible in figures selected for publication was anonymised via face blurring or the figure was separately cleared for publication by the rights holder. Public benchmark datasets (SYSU-CD, LEVIR-CD, PCD) are used under their respective published licenses/terms."

---

## 8. Writing/production toolchain

- **ACM acmart**: `\documentclass[sigconf]{acmart}` (current class `acmart.cls` v2.19, 2026-07-02, per ACM/Overleaf template — consistent with the proposal.pdf's own formatting). Use `[sigconf,review,anonymous]` for any double-blind review stage; confirm final variant with the supervisor/venue.
- **Overleaf**: one shared project with all 5 as collaborators; sync it to the same GitHub repo's `paper/` subfolder via Overleaf's GitHub integration so LaTeX changes go through the same PR review flow as code. Split `paper/sections/*.tex` one file per section (`abstract`, `intro`, `related-work`, `method`, `experiments`, `results`, `discussion`, `conclusion`) `\input`-ed from `main.tex` — the LaTeX analogue of the registry pattern: five people, five files, near-zero merge conflicts; the integration owner alone manages `main.tex` and cross-references.
- **Shared `.bib`**: single `references.bib`, consistent key convention (`author_year_firstword`), a CI/pre-commit check flagging unused/duplicate/missing-DOI entries before submission.
- **Figure standards**: line plots/diagrams as vector PDF (`savefig(..., format='pdf')`); change-mask overlays use a colour-blind-safe scheme — avoid red-green entirely (blue-yellow or blue-orange work for both deutan and protan CVD). Concrete recommendation for a TP/FP/FN overlay: TP in blue, FP in vermillion/orange, FN in magenta/purple (Okabe-Ito-style categorical set), validated with a CVD simulator before finalizing; for the confidence/signed-change overlay use a diverging blue↔orange scale, not red↔green.
- **Section ownership**: P1 → Datasets + related-work-on-CD-datasets; P2 → Method: baselines + implementation; P4 → Method: proposed contribution; P3 → Experiments/ablations; P5 → Results/Discussion + integration owner (also drafts abstract/intro/conclusion, owns bibliography hygiene and final page-limit/format compliance).

---

## Per-task effort estimate (person-hours)

| Category | Task | Hours |
|---|---|---|
| **Infra/setup** | Repo scaffolding + package skeleton | 8 |
| | Registry framework + `build_*` functions | 10 |
| | Hydra config system + experiment configs | 12 |
| | CI + pre-commit + nbstripout | 8 |
| | Colab/Kaggle driver notebooks + Drive/secrets | 8 |
| | Checkpoint/resume + seeding utilities | 8 |
| **Data** | SYSU-CD loader + download script | 8 |
| | LEVIR-CD loader | 5 |
| | PCD loader | 5 |
| | Scene-disjoint split implementation + tests | 10 |
| | Shared geometric + independent photometric aug | 8 |
| | Video set: permission/ethics paperwork | 6 |
| | Video set: extraction/correspondence + annotation tooling | 20 |
| | Nuisance-only hard-negative curation | 10 |
| **Baselines** | RGB/SSIM baseline | 4 |
| | FC-Siam-Diff implementation | 16 |
| | Siamese ResNet-18 implementation | 12 |
| | Baseline LR-grid runs (3 LRs × 2 models × datasets × seeds) | 30 |
| | BIT stretch goal (optional) | 16 |
| **Proposed model** | Encoder wiring (EfficientNet-B0/B2, ResNet-18) | 10 |
| | Signed + absolute fusion module | 8 |
| | Bounded alignment module (offset + confidence gate) | 20 |
| | Confidence head + calibration loss | 12 |
| | Pair-order consistency loss + directional head | 14 |
| | Integration + shape/overfit/registry tests | 10 |
| | Training + tuning | 20 |
| **Ablations/eval** | Metrics implementation + hand-computed unit tests | 16 |
| | Ablation configs + orchestration across seeds | 24 |
| | Stratified analysis (change size, nuisance severity) | 10 |
| | Qualitative overlays (colour-blind-safe) + failure cases | 10 |
| | Final video-set evaluation | 12 |
| **Writing** | ACM/Overleaf setup + shared `.bib` skeleton | 4 |
| | Section drafting (5 sections) | 40 |
| | Figures (vector export, overlay styling) | 10 |
| | Integration/editing passes (2–3 rounds) | 16 |
| | MODEL_CARD / DATA_CARD / ETHICS.md | 6 |
| | Internal review + revision | 10 |
| **PM overhead** | Weekly meetings (12 wks × 5 people × 0.5h) | 30 |
| | Task board / risk register maintenance | 6 |
| **Total** | | **≈ 448 person-hours** |

At 5 people over a ~12–13 week semester, that's roughly 7–8 hrs/person/week — plausible as a stretch-but-achievable budget; the estimate deliberately front-loads infra/registry work (weeks 1–2) because it is what makes the rest parallelizable without merge conflicts.

---

Sources: [Papers with Code — ML Code Completeness Checklist](https://medium.com/paperswithcode/ml-code-completeness-checklist-e9127b168501) · [paperswithcode/releasing-research-code](https://github.com/paperswithcode/releasing-research-code) · [NeurIPS 2026 Call For Reproducibility](https://neurips.cc/Conferences/2026/CallForReproducibility) · [Hydra tutorial — Towards Data Science](https://towardsdatascience.com/complete-tutorial-on-how-to-use-hydra-in-machine-learning-projects-1c00efcc5b9b/) · [GitHub Actions billing docs](https://docs.github.com/billing/managing-billing-for-github-actions/about-billing-for-github-actions) · [GitHub Actions free tier 2026 — CICDCalculator](https://cicdcalculator.com/github-actions-free-tier) · [ACM SIG Proceedings Template — Overleaf](https://www.overleaf.com/latex/templates/association-for-computing-machinery-acm-sig-proceedings-template/bmvfhcdnxfty) · [ACM Conference LaTeX Template (acmart) — Underleaf](https://www.underleaf.ai/templates/acm-sigconf) · [Colorblind Safe Color Schemes — NCEAS](https://www.nceas.ucsb.edu/sites/default/files/2022-06/Colorblind%20Safe%20Color%20Schemes.pdf) · [Coloring in R's Blind Spot (khroma)](https://journal.r-project.org/articles/RJ-2023-071/)