# My role: P4 — Modeling B (Bounded Alignment, Confidence Head, Pair-Order Loss)

> Copy this to `CLAUDE.local.md` at the repo root. It layers on top of the shared `CLAUDE.md`.

**I own:** `src/cdlib/models/alignment/**`, `src/cdlib/models/heads/**`, `src/cdlib/losses/{pair_order_consistency,calibration}.py`, `src/cdlib/models/proposed.py`
**CODEOWNERS:** `/src/cdlib/models/alignment/ /src/cdlib/models/heads/ /src/cdlib/losses/pair_order_consistency.py /src/cdlib/losses/calibration.py /src/cdlib/models/proposed.py @me`
(Not all of `/src/cdlib/losses/` — `bce_dice.py` is P2's.)
**I own the actual novelty, which means I own the highest risk in the project.** Risk register items 5 and 9 are both mine. The registry pattern and `docs/architecture.md` are my bus-factor insurance — keep interfaces documented so someone else can pick this up.

## Do this in week 1, before anything else

**Read Dong et al., "Exchange Is All You Need for Remote Sensing Change Detection", arXiv:2601.07805 (Jan 2026), in full.** It formalises bi-temporal order-invariance as an orthogonal permutation operator. It is not peer-reviewed, but it is close enough to contribution (B) that I must know exactly what it claims before we finalise our wording. Then brief the supervisor. This is a week-1 risk check — discovering it in week 10 would be a disaster.

Also read **ChangeMask** (Zheng et al., ISPRS P&RS 2022, DOI 10.1016/j.isprsjprs.2021.10.021) — the closest *established* prior work. It names "temporal symmetry" as an explicit inductive bias and builds a Time-Symmetric Transformer for the binary branch, **but it gets directional information only by classifying each frame independently and pairing results post-hoc.** It never trains a fused, signed, antisymmetric direction head. That gap is our cleanest wedge.

## The argument I have to be able to write on a whiteboard

With signed difference fusion, swapping input order gives `h(b−a) = h(−(a−b))`. That equals `h(a−b)` **only if `h` is an even function**, and nothing in standard training makes it one. Weight sharing alone buys order-invariance *only* for symmetric fusion (`|a−b|`, `a+b`, `max`).

So: our signed fusion is antisymmetric, order-consistency is therefore **not** automatic, and that is precisely why the loss is a contribution rather than a no-op. This is the crux fact a reviewer will look for. It goes in the method section as a short formal statement.

## Gotchas that fail silently

**Alignment collapse is the core danger.** An alignment module that is free to warp anything will learn to warp genuine changes into false agreement — the loss goes down and the contribution evaporates, with no error message. Bounding (`tanh(·)·max_disp`) and gating are the defences. Ablation **A6** is the probe that actually measures whether it's happening; the brief calls it the field's missing ablation. Do A6 even if time is tight.

**Gate circularity is my biggest reviewer risk.** Conditioning the alignment gate on the network's own change logits — while those logits are still training — creates a degenerate equilibrium. Derive the gate from a **decoupled** signal instead (local normalised cross-correlation, or abs-diff magnitude), not from the change head. Ablation **A4** deliberately runs the circular version to demonstrate the failure — that turns a weakness into a finding.

**`grid_sample` gotchas.** `align_corners` semantics, `padding_mode` (`border`/`reflection` have known incorrect gradients at the border — pytorch issues #23925, #24870). **Zero-init the offset conv** so the module starts at identity, and consider a lower LR on the offset branch. Get these wrong and it trains to garbage without ever erroring.

**Compute is not the constraint here.** At 64×64×128 and 32×32×256 the bounded alignment module is ~1.2 GFLOP each — trivial on a T4. Don't optimise it prematurely; correctness is the hard part.

**BCE+Dice hurts calibration** (Mehrtash TMI 2020), and my confidence head is supposed to deliver calibrated confidence. Coordinate with P5 and P2 early on the fix (temperature scaling on shift-representative data, or a calibration-aware auxiliary term). Don't discover this conflict when writing the results section.

---

## Checklist

### Week 1 — de-risk the novelty claim
- [ ] Read `research/03-alignment-pair-order.md` cover to cover, especially §2 (collapse), §3 (PyTorch), §4 (symmetry), §7 (novelty)
- [ ] **Read arXiv:2601.07805 in full**; write half a page on what it claims vs what we claim
- [ ] Read ChangeMask; confirm the "no fused antisymmetric direction head" gap holds
- [ ] Brief the supervisor on prior-art risk and our differentiation
- [ ] Draft the formal order-consistency statement (the `h` even-function argument) for the method section

### Week 3–4 — bounded alignment
- [ ] `alignment/identity.py` — the no-op, so A1 is a one-line config override
- [ ] `alignment/bounded_alignment.py`: offset head → `tanh(·) * max_disp` → change-aware gate from a **decoupled** signal → `grid_sample` warp
- [ ] Zero-init the offset conv; separate (lower) LR for the offset branch
- [ ] Expose `max_disp`, `bounded: bool`, `gated: bool`, `gate_source: {decoupled, own_logits}` as config — A2/A3/A4/A5 must each be a one-line override
- [ ] Define the named experiment configs P2 scaffolded: `ablation_no_alignment`, `ablation_no_confidence_gate`, `ablation_no_pairorder`
- [ ] Emit `alignment_offset` into `aux` per the forward contract
- [ ] Register in `ALIGNMENT_REGISTRY`; shape tests

### Week 4–5 — heads and the consistency loss
- [ ] `heads/confidence_head.py` — per-pixel confidence, emitted as `confidence` in the contract
- [ ] `heads/directional_head.py` — 2-class appeared/disappeared into `aux.directional_logits`
- [ ] `losses/pair_order_consistency.py`:
  - binary term: mask from `(I1,I2)` must equal mask from `(I2,I1)`
  - directional term: swapping order must **permute** appeared↔disappeared
  - expose `lambda_swap` so λ=0 gives ablation B1 free
- [ ] Implement the **swap-consistency error metric** — cheap, and B7 applies it to every model in the project including ones not trained with the loss
- [ ] `models/proposed.py` composing P3's encoder + fusion with my alignment + heads

### Weeks 5–8 — ablations (this is the evidence, not an afterthought)

Alignment:
- [ ] **A1** no alignment — does alignment help at all?
- [ ] **A2** unbounded offset — isolates the value of the hard bound
- [ ] **A3** bounded but ungated — isolates gating
- [ ] **A4** gate on own change logits (circular) — demonstrates the collapse risk
- [ ] **A5** full: bounded + decoupled gate — the actual claim
- [ ] **A6** genuine-change-erasure probe: synthetically insert large unambiguous objects, measure recall/IoU on those regions as `max_disp` varies. **Cheapest high-value result we have — do this even if time is short**

Consistency:
- [ ] **B1** λ_swap = 0 — is the plain signed model already order-consistent by luck? Test it, don't assume
- [ ] **B2** binary head only
- [ ] **B3** directional head only (permutation target) — likely our headline result
- [ ] **B4** full pair-order loss
- [ ] **B5** hard architectural even-function head `h(z)=g(z)+g(−z)` for the binary branch — upper-bound reference: how close does the soft loss get to a provable guarantee?
- [ ] **B6** fusion sweep (`|a−b|` vs signed vs signed+loss vs concat) — cheap, legible, demonstrates the taxonomy empirically. **Prioritise if time runs short**
- [ ] **B7** swap-consistency error computed for *every* model above, trained with the loss or not — shows this is a real previously-unmeasured failure mode
- [ ] **AB1** A5 × B4 combined — and be honest if the interaction is redundant rather than synergistic

### Ongoing
- [ ] Weekly qualitative review of alignment outputs — collapse shows up visually before it shows up in metrics
- [ ] Keep `docs/architecture.md` current for my components (bus factor)
- [ ] Draft the Method: proposed contribution section (my writing assignment), using the "prior work does X; we address Y" framing from `research/03` §7

## Commands I'll use

```bash
python -m cdlib.cli.train +experiment=ablation_no_alignment          # A1
python -m cdlib.cli.train model.alignment.bounded=false              # A2
python -m cdlib.cli.train model.alignment.gated=false                # A3
python -m cdlib.cli.train model.alignment.gate_source=own_logits     # A4
python -m cdlib.cli.train -m loss.lambda_swap=0,0.1,1.0              # B1 + sweep
python -m cdlib.cli.train -m model.fusion=absdiff,signed,signed_product,concat  # B6
python -m cdlib.cli.evaluate +metrics=swap_consistency exp_id=<any>  # B7, any model
```

## Hand-offs

| To | What | When |
|---|---|---|
| supervisor | prior-art risk briefing on arXiv:2601.07805 | **week 1** |
| team | the formal order-consistency argument | week 2 |
| P5 | swap-consistency metric so it can run across all ablations | week 5 |
| P5 | ablation results A1–A6, B1–B7, AB1 | weeks 6–9 |

**I depend on:** P2's registries (week 1), P1's loaders (week 2), P3's encoders + fusion (week 3), **P5's Dice-vs-calibration decision (week 4)** before I finalise the confidence head's loss.
