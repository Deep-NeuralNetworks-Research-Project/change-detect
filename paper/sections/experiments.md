# Experiments / Ablations — draft (P3 writing assignment)

> Draft only. `/paper/` in the team repo is P5's integration territory
> (owns `main.tex`, `.bib`, cross-references per `CODEOWNERS`) — this is
> written here so it exists and can be reviewed, then handed off to be
> turned into `paper/sections/experiments.tex` and `\input`-ed from
> `main.tex`, per `research/06` §8's writing toolchain. Numbers below
> are placeholders (`TBD`) until real training runs exist; the
> *methodology* is fixed and matches what's already implemented and
> tested in this repo.

## 1. Encoder speed measurement

**Why this is here and not just an implementation detail:** the
FLOP-vs-wall-clock gap for depthwise-separable convs is a real,
citable finding (Graphcore; Nature Sci Reports 2025), not a footnote —
see `research/01-baselines-architectures.md` §3 and `CLAUDE.md`'s
finding 6.

**Protocol.** For each of `resnet18`, `efficientnet_b0`, `efficientnet_b2`:
step time via CUDA events (`torch.cuda.synchronize()`, warm-up excluded),
median + IQR over N iterations, on the actual Colab/Kaggle GPU in use
(log the GPU model — Colab's assignment varies run to run). Params and
pair-input GFLOPs (`torch.utils.flop_counter`) reported alongside, not
instead of, step time. Implementation: `scripts/benchmark_encoders.py`.

**Table (fill in from a real GPU run):**

| Encoder | Params (M) | Pair GFLOPs | Median step (ms) | IQR (ms) | Peak mem (MB) | GPU |
|---|---|---|---|---|---|---|
| ResNet-18 | 11.18 | TBD | TBD | TBD | TBD | TBD |
| EfficientNet-B0 | ~5.3 | TBD | TBD | TBD | TBD | TBD |
| EfficientNet-B2 | ~9.2 | TBD | TBD | TBD | TBD | TBD |

**Claim to make once filled in:** state plainly whether the FLOP
ranking and the wall-clock ranking agree or invert. If they invert
(the literature-predicted outcome), that inversion — not just "we used
ResNet-18" — is the reportable result, and it's what justifies
defaulting to ResNet-18 for the rest of the paper's experiments.

## 2. Fusion ablation (B6 — P4's sweep, P3's module)

**What's being compared:** `absdiff` (order-invariant, destroys
direction) vs. the four `signed_fusion` modes (`signed`,
`signed_product`, `concat`, `signed_concat`) — see
`src/cdlib/models/fusion/signed_fusion.py`'s docstring for the
swap-symmetry argument this ablation is testing empirically.

**Protocol.** Same encoder, decoder, loss, LR, seed budget across all
five fusion configs; only `model.fusion=...` changes
(`python -m cdlib.cli.train -m model.fusion=absdiff,signed,signed_product,concat,signed_concat`
once P2's CLI exists). Report aggregate F1/IoU (rule 1 in `CLAUDE.md`),
plus a directional breakdown (appeared vs. disappeared, when P4's
directional head is available) — this is exactly the axis `absdiff` is
predicted to fail on and signed fusion is predicted to recover.

**Table (fill in):**

| Fusion | Out width (×C) | Aggregate F1 | Aggregate IoU | Appeared F1 | Disappeared F1 |
|---|---|---|---|---|---|
| absdiff | 1× | TBD | TBD | TBD | TBD |
| signed | 1× | TBD | TBD | TBD | TBD |
| signed_product | 2× | TBD | TBD | TBD | TBD |
| concat | 2× | TBD | TBD | TBD | TBD |
| signed_concat | 3× | TBD | TBD | TBD | TBD |

**Claim to make once filled in:** whether `signed_product` clears
`absdiff` on the directional split specifically (the mechanism the
docstring predicts), and whether it does so more cheaply than plain
`concat` (the width/cost argument) — that pairing is the actual novelty
claim, not "our fusion has a higher F1."

## 3. Stride-32 ablation (with/without C5, dilated vs not)

**Motivation:** stride-32 is standard-finding-too-coarse for thin
change boundaries (Changer, SARAS-Net; see `CLAUDE.md`). Two escape
hatches, both implemented as ablatable flags on `ResNet18Encoder`:
`use_c5=False` (drop C5, decode from C2–C4) and `dilate_last=True`
(atrous C5, stride 16 instead of 32).

**Cost side — already measured, no training needed**
(`scripts/stride32_ablation.py`, pair input 256×256):

| Config | Params (M) | Pair GFLOPs | C5 stride |
|---|---|---|---|
| default (`use_c5=True, dilate_last=False`) | 13.13 | 10.53 | 32 |
| drop C5 (`use_c5=False`) | 11.66 | 8.08 | 16 |
| atrous C5 (`dilate_last=True`) | 13.13 | 17.43 | 16 |

Dilating costs +65% FLOPs over the default for the same C5 resolution
that dropping C5 gets for *less* than the default cost — that tradeoff
(atrous keeps more channels/context at real cost; dropping C5 is free
but loses the deepest semantic features) is itself worth stating
plainly, independent of the quality numbers below.

**Quality side — blocked on P1's loaders + P2's trainer + a GPU
session.** Once available: train all three configs, ≥3 seeds, report
aggregate F1/IoU *and* Boundary IoU / HD95 (P5's metric — this ablation
is specifically about boundary quality, not overall F1, so leading with
aggregate F1 alone would undersell or misrepresent the result).

**Table (fill in):**

| Config | Aggregate F1 | Boundary IoU | HD95 | Params (M) | Pair GFLOPs |
|---|---|---|---|---|---|
| default (stride 32) | TBD | TBD | TBD | 13.13 | 10.53 |
| drop C5 (stride 16) | TBD | TBD | TBD | 11.66 | 8.08 |
| atrous C5 (stride 16) | TBD | TBD | TBD | 13.13 | 17.43 |

## 4. Siamese ResNet-18 baseline — tuning

Per *A Change Detection Reality Check* (Corley et al., arXiv:2402.06994),
a properly-tuned plain baseline can rival BIT — treat that outcome as a
legitimate result, not a failure of the proposed model, if it happens
(`CLAUDE.md` finding 5).

**Protocol.** LR sweep (`train.lr=1e-3,3e-4,1e-4`) at the same compute
budget as the proposed model gets; ≥3 seeds on the winning LR; report
mean ± std (not a significance test at n=3, per `CLAUDE.md` rule 8).

**Table (fill in):**

| LR | Seed 0 F1 | Seed 1 F1 | Seed 2 F1 | Mean ± std |
|---|---|---|---|---|
| 1e-3 | TBD | TBD | TBD | TBD |
| 3e-4 | TBD | TBD | TBD | TBD |
| 1e-4 | TBD | TBD | TBD | TBD |

## Status

Sections 1–3's methodology and cost-side numbers are final; the
quality-side numbers in all four sections are blocked on P1 (data
loaders), P2 (trainer/CLI), and a Colab/Kaggle GPU session. Fill in as
those land — don't restructure the tables, just the TBDs.
