# Project: Pair-Order Consistent & Uncertainty-Aware Semantic Change Detection

University of Moratuwa, CSE — 5-person semester project. Team: Ekanayake, Lelwala, Pabasara, Bulagala, Rajapaksha.

**Task.** Given two RGB frames `I1` (reference video version) and `I2` (edited version) at the same semantic time, predict a binary mask of *meaningful edits* while suppressing nuisance: camera displacement, lighting, colour grading, blur, shadows, occlusion, codec artifacts.

**Deliverable framing.** This is a human-in-the-loop reviewer aid. It reduces search time; it does not make autonomous decisions. That is why calibrated confidence and deferral are first-class deliverables, not extras.

**Compute.** Free Colab / Kaggle GPUs (T4/P100). Assume 12h session caps and preemption. Never assume a run survives.

## Research dossier — read the right brief, don't guess

| Question about… | Read |
|---|---|
| FC-Siam-Diff internals, ResNet-18/EfficientNet encoders, BIT/ChangeFormer, leaderboards | `research/01-baselines-architectures.md` |
| Datasets, licences, splits, video correspondence, annotation | `research/02-datasets-data-pipeline.md` |
| Bounded alignment, order-consistency theory, novelty defence, ablations | `research/03-alignment-pair-order.md` |
| Uncertainty, ECE, risk–coverage, conformal, loss↔calibration | `research/04-uncertainty-calibration.md` |
| Metrics, boundary quality, corruption suite, FLOPs/latency, seed stats | `research/05-evaluation-efficiency.md` |
| Repo layout, configs, CI, Colab workflow, risk register, ethics | `research/06-engineering-delivery.md` |

If a brief marks something **inferred** or **unconfirmed**, it is not yet a fact. Keep the marker until someone verifies it.

---

## Frozen interfaces — breaking these breaks everyone

Locked in week 1. Changing a contract requires a team-wide PR review, not a quiet edit.

**Dataset `__getitem__`** — every dataset returns exactly this:
```python
{
  "img1": Tensor[C,H,W] float32 [0,1],
  "img2": Tensor[C,H,W] float32 [0,1],
  "mask": Tensor[1,H,W] float32 {0,1,-1},    # -1 = ignore
  "nuisance_label": Tensor[] int64,           # 0=clean, 1..k=nuisance type, -1=unknown
  "meta": {"source_video": str, "scene_id": str, "frame_idx": tuple[int,int],
           "pair_id": str, "dataset": str},
}
```

**Model `forward`** — every baseline and the proposed model:
```python
{
  "logits": Tensor[B,1,H,W],                  # pre-sigmoid
  "confidence": Tensor[B,1,H,W] | None,
  "aux": {                                     # experimental heads only ADD keys
    "directional_logits": Tensor[B,2,H,W] | None,
    "alignment_offset": Tensor[B,2,H,W] | None,
  },
}
```

**Loss** — `compute(outputs, batch) -> {"loss": …, "loss/bce": …, "loss/dice": …, "loss/pairorder": …}`.
`trainer.py` backprops `out["loss"]` and auto-logs every `loss/*` key. It never inspects a loss's internals.

**Metric** — torchmetrics-style `reset() / update(outputs, batch) / compute() -> dict[str,float]`.

**Builders** — the only four entrypoints the CLI calls:
`build_model(cfg)`, `build_dataset(cfg, split)`, `build_loss(cfg)`, `build_optimizer(cfg, params)`.

## The registry pattern — why PRs don't collide

Every pluggable component is **one new file + one registry line**. Adding an encoder means writing `models/encoders/foo.py` and registering a key. It never means editing `trainer.py`, `build.py`, or somebody else's component.

If Claude proposes a change that edits a shared file to add a feature, push back and ask for the registry version instead.

---

## Nine rules that are not negotiable

1. **Aggregate (corpus) F1**, not per-image averaged. Sum TP/FP/FN across the whole test set, then compute F1 once. State it explicitly in the paper. The two differ substantially under our imbalance, and mixing them is why cross-paper CD numbers don't reconcile.
2. **Scene/source-disjoint splits.** Never split at frame level. Adjacent frames are near-duplicates; frame-level splits silently invalidate the whole benchmark. `tests/` must assert no `scene_id` crosses a split.
3. **LEVIR-CD is 256×256 non-overlapping crops** → 7,120 / 1,024 / 2,048. Any other crop policy makes our numbers comparable to nothing.
4. **Geometric augmentation is shared across the pair; photometric augmentation is independent per frame.** Get this backwards and you either destroy the alignment signal or delete the nuisance-robustness signal you are trying to learn.
5. **Siamese BatchNorm**: concatenate `[I1;I2]` along the **batch** dimension into one forward pass, then split. Two sequential forwards give BN half the effective batch and mix running stats asymmetrically.
6. **Checkpoint every 10–15 minutes of wall clock**, not per epoch, including RNG state. `resume_from=auto`. Free-tier preemption is a certainty, not a risk.
7. **Notebooks contain only**: env bootstrap, one call into `cdlib.cli.*`, and a plotting cell. Zero model/loss/data logic. Enforced by `nbstripout` + CI.
8. **≥3 seeds for principal models.** Report mean ± std or median + range, plus bootstrap CIs over the test set. Do **not** run significance tests at n=3 — they are not meaningful there.
9. **Never commit** video footage, extracted frames, `.env`, or W&B keys. `data/video_raw/` is gitignored. Track checksums and manifests, never the media.

---

## Seven findings that change how we build this

These came out of the research dossier and contradict the obvious approach. Read them before disagreeing with them.

**1 — A Jan 2026 paper may pre-empt our pair-order contribution.**
Dong et al., *"Exchange Is All You Need for Remote Sensing Change Detection"*, arXiv:2601.07805, formalises bi-temporal order-invariance as an orthogonal permutation operator. Not peer-reviewed, but close enough that **P4 must read it in full in week 1** and brief the supervisor. Week-1 risk check, not a week-10 surprise.

**2 — Signed fusion makes order-consistency a real constraint, and that's our argument.**
Swapping order under signed difference fusion gives `h(b−a) = h(−(a−b))`, which equals `h(a−b)` only if `h` is even. Nothing in standard training enforces that. Weight sharing alone buys symmetry only for *symmetric* fusion (`|a−b|`, `a+b`, `max`). This is the crux fact reviewers will look for — write it out formally.

**3 — Our planned BCE+Dice loss fights our calibration deliverable.**
Dice loss is documented to cause overconfidence (Mehrtash et al., IEEE TMI 2020; Yeung et al., DSC++). Focal *improves* calibration (Mukhoti et al., NeurIPS 2020). We promise BCE+Dice *and* ECE/Brier/risk–coverage. Fix: post-hoc temperature scaling fitted on shift-representative data, or a calibration-aware auxiliary term. Do not quietly ignore this.

**4 — Naive pixel-ECE will look excellent and mean nothing.**
At 2–5% changed pixels, pooled all-pixel ECE is dominated by easy background. Use class-wise / foreground-restricted ECE with equal-mass (adaptive) binning. Same trap as reporting overall accuracy on an imbalanced task.

**5 — A well-tuned plain U-Net matches ChangeFormer on LEVIR-CD.**
*A Change Detection Reality Check* (Corley et al., arXiv:2402.06994): U-Net ResNet-50 F1 90.38, U-Net-SiamDiff 90.46 vs ChangeFormer 91.11, TinyCD 91.05 — no architectural novelty, just a modern recipe. On corrected WHU-CD splits, plain U-Nets **beat** BIT and ChangeFormer. So: give our baselines a genuinely tuned recipe, and expect a naively-trained BIT to possibly lose to our own ResNet-18. Anticipate that in the write-up so it doesn't read as a bug.

**6 — EfficientNet's FLOP advantage is not a speed advantage.**
Depthwise separable convs have low arithmetic intensity; with unfused SiLU, B0/B2 can be **slower in wall-clock than ResNet-18 on T4/P100 despite ~4× fewer FLOPs**. Default to ResNet-18 unless you have benchmarked step time on the actual GPU.

**7 — Two datasets will bite us.**
PCD's original host is dead; only TSUNAMI is downloadable and **GSV is not hosted at all** — archive on first access, keep GSV off the critical path. WHU-CD has **no official split** and the commonly circulated version has **~85% train/test leakage**.

---

## Ownership

`CODEOWNERS` (in this folder, copy to `.github/CODEOWNERS`) is generated and resolution-tested — every path resolves to exactly one owner. **Do not sort it alphabetically:** GitHub applies last-match-wins, and `/src/cdlib/cli/benchmark.py` (P5) deliberately sits below `/src/cdlib/cli/` (P2). Sorting the file silently transfers ownership.

Two deviations from the repo layout in `research/06`, both intentional:
- `tests/test_splits.py` is **added** — risk-register item 4 requires a test that no `scene_id` crosses a split, and the original layout omitted it.
- `src/cdlib/losses/` is **split**: `bce_dice.py` → P2, `pair_order_consistency.py` + `calibration.py` → P4. It is not one person's directory.

## Definition of done

- **Code task** — PR merged, CI green, test added, registry entry documented, no new lint warnings.
- **Experiment task** — logged under `results/<exp_id>/` with `config.yaml`, `commit.txt`, `command.txt`, `env.txt`, `checkpoint_sha256.txt`, per-seed dirs; entry added to the shared benchmark table; ≥3 seeds for principal models; overlays exported for best/median/worst pairs.
- **Writing task** — drafted in Overleaf, every claim has a `references.bib` entry, reviewed by P5, figures as vector PDF, no TODO left.

## Working with Claude on this repo

- Prefer editing **one component file + one registry line**. If a task seems to need a shared-file edit, say so before doing it.
- Before writing a metric, check `research/05` for the exact formula and the pitfall attached to it. Several of these metrics have a wrong-but-plausible version.
- Before writing the alignment module or the consistency loss, read `research/03` §3 and §4 — there are specific PyTorch gotchas (`align_corners`, `padding_mode`, zero-init offsets) that cause silent failure, not errors.
- When reporting results, always state which pooling protocol and which threshold produced them.
- If a run's numbers look surprisingly good, suspect leakage first (rule 2), then metric pooling (rule 1).
