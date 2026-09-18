# Member 4 — Bulagala D.W.K.G.
## Track: Efficiency, encoder study, tuning grid, and the alignment module

**GPU need:** high for the grid and EfficientNet; low for the profiling work.
**Estimated effort:** 9–11 person-days.
**Why this track exists:** the proposal made "modest latency/memory budget" part of the claim and promised parameter count, FLOPs, peak memory and latency, plus an EfficientNet accuracy–resource comparison and a tuning grid. The paper reports none of it. Separately, the bounded alignment module was the proposal's headline contribution and was never built — and the paper's own limitations argue it interacts directly with the sign problem.

---

## Part A — Efficiency (do first; cheap and closes a promise)

- [ ] **Parameter count** per configuration. Note the paper already claims the even head costs no extra parameters — verify and report it rather than asserting it.
- [ ] **Pair-input FLOPs** (one forward pass on a 256×256 pair). Use a standard counter and state which one; FLOPs conventions differ.
- [ ] **Peak GPU memory** at train and at inference, batch 1 and batch 16.
- [ ] **Latency**, warm-up excluded, median over ≥100 runs, reported with the hardware named (RTX 4060 Laptop, 8 GB).
- [ ] **The even head's real cost.** §III-C says it costs "one extra decoder pass". Measure that: it should roughly double decoder-only latency while leaving the encoder untouched. Since the paper's headline claim is *exact invariance is free*, the word "free" needs a number attached. This is the single most important item in Part A.
- [ ] Same treatment for the SCE metric itself: it costs one extra forward pass, so quantify the audit cost for a practitioner.
- [ ] Also profile Member 2's FC-Siam-Diff, RGB/SSIM and BIT once their checkpoints exist.

Deliverable: one efficiency table, and a sentence in §V-D that replaces "costs nothing systematic" with a measured overhead.

## Part B — Tuning grid

The paper fixes 3e-4, τ=0.5, one resolution, fine-tuned encoder, for everything. That is a defensible matched-budget design and Member 1 will argue it — but the proposal promised a grid, and a reviewer will ask whether the F1 ordering across fusion operators is an artefact of one recipe.

- [ ] Run the promised grid on **two** configurations only (abs-diff and signed, the two that anchor the comparison): lr ∈ {1e-3, 3e-4, 1e-4}, frozen vs fine-tuned encoder, and two input resolutions.
- [ ] Select on validation scenes only; test stays sealed.
- [ ] **The question to answer:** does the ~2-point F1 spread between fusion operators survive per-model tuning, or does it collapse? Either answer strengthens the paper. If it collapses, the claim "fusion barely moves accuracy" gets *stronger*, not weaker.
- [ ] Also sweep the decision threshold on validation and report what a validation-selected τ would have been, alongside the deployed 0.5. This is distinct from §V-C's test-side per-ordering sweep — keep the two clearly separated in the write-up.

## Part C — EfficientNet encoder comparison

- [ ] Swap the ResNet-18 trunk for EfficientNet-B0 [11], same decoder, same recipe, abs-diff and signed fusion at minimum.
- [ ] B2 only if memory allows.
- [ ] Report on the accuracy–resource plane using Part A's metrics.
- [ ] **Report SCE too.** The prediction from §III-A is that SCE depends only on φ and h, not on the encoder — so abs-diff should hit exactly 0 on EfficientNet as well, and signed fusion should still be badly inconsistent. Confirming that across a second encoder family directly answers the limitation "one encoder, one decoder, one resolution; the symmetry argument is architecture-independent but the magnitudes are not."

## Part D — Bounded change-aware alignment module (phase 2 / stretch)

This was the proposal's main novelty and it is a paper's worth of work on its own. Treat it as stretch; land Parts A–C first.

- [ ] Implement small-offset feature alignment between the two feature maps, with the change-confidence gate so genuine appeared/moved objects are not aligned away.
- [ ] Bound the predicted offsets explicitly and report the bound.
- [ ] **The interesting question for this paper:** alignment and order-symmetry interact. A small misregistration can flip an apparent sign without flipping a magnitude — so does alignment *reduce* SCE for signed fusion, or does an asymmetric alignment module introduce a new order-dependence of its own? If your warp is direction-dependent (warping a→b is not warping b→a), you have added a second symmetry violation on top of the fusion one. Measure SCE with and without alignment and say which.
- [ ] Evaluate on PCD first — it is the only dataset with genuine viewpoint nuisance rather than registered aerial imagery, so it is where alignment should actually pay.

---

## Deliverables

1. `profile.py` and the efficiency table (all configurations + Member 2's baselines).
2. Tuning-grid results and a paragraph on whether the fusion F1 ordering is recipe-dependent.
3. EfficientNet-B0 (and B2 if possible) rows with F1 + SCE, and the cross-encoder invariance confirmation.
4. If it lands: `models/align.py`, ablation with/without alignment, and the alignment×SCE interaction result.

---

## Interfaces

**You depend on:**
- Member 3's `evaluate.py` for every accuracy number.
- Member 2's checkpoints for baseline profiling.
- Member 1's conventions for reporting format.

**Others depend on you for:**
- Member 1 needs the even-head overhead number before the "at no systematic cost" claim in the conclusion can stand as written.
- Member 3 will want your grid's validation-selected τ for the threshold discussion.

---

## Acceptance criteria

- Params, FLOPs, peak memory and latency reported for every configuration in Table I, hardware named.
- The even head's overhead measured, not asserted.
- The fusion-vs-accuracy claim tested under at least one alternative recipe.
- At least one non-ResNet encoder confirms the taxonomy, or the limitation is restated honestly.

---

## Risks

- **Part D is a trap.** It is genuinely interesting and genuinely large. If it is not working by the two-thirds mark, cut it and write the negative or partial result into the limitations. Parts A–C are what the paper actually needs.
- **8 GB caps EfficientNet-B2 and the larger resolutions.** Plan the grid around what fits; do not let one OOM configuration eat a week.
- **GPU contention.** Your grid is the most GPU-hungry item in the project. Coordinate with Member 3's queue and run overnight.
