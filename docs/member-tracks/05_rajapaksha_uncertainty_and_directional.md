# Member 5 — Rajapaksha T.N.D.W.
## Track: Uncertainty, calibration, selective prediction, and the directional head

**GPU need:** low for calibration (post-hoc on existing checkpoints), moderate for the directional head.
**Estimated effort:** 8–10 person-days.
**Why this track exists:** the title says **Uncertainty-Aware** and the paper contains no uncertainty result whatsoever — no ECE, no Brier, no risk–coverage, no abstention. The proposal also promised a directional head and a pair-order consistency loss that swaps appeared/disappeared channels; only the binary version was built. Your output decides whether Member 1 keeps the title or retitles the paper.

---

## Part A — Calibration (highest priority; cheap, unblocks the title)

All of this runs post-hoc on checkpoints that already exist. No retraining.

- [ ] **Expected calibration error** on the changed class, with the binning scheme stated, following [13] (which the paper already cites for the class-imbalance pitfall).
- [ ] **Brier score.**
- [ ] **Reliability diagrams** for a representative subset of configurations.
- [ ] **Critically: compute all of these foreground-restricted as well as pooled.** §III-B's central argument is that pooled pixel statistics are a trap under class imbalance — at π ≈ 5% the background decides the average. That argument applies to calibration at least as strongly as it does to agreement. A pooled ECE on LEVIR-CD will look excellent for a model that is badly miscalibrated where it matters. This is a direct, cheap extension of the paper's own thesis to a second metric family, and it may be the most publishable thing in your track.

### The link to SCE (this is your strongest angle)

§V-C found that the reversed ordering's F1-optimal threshold is 0.08 for signed fusion and 0.10 for concatenation, against the deployed 0.5. That is a **severe calibration shift induced purely by input ordering**. So:

- [ ] Report ECE for each ordering separately, and the ECE gap between orderings — call it Δ-ECE_swap, the natural calibration companion to ΔF1_swap.
- [ ] State the consequence plainly: for these models, calibration is not a property of the model and the data, it is a property of the model, the data, *and an arbitrary labelling convention*. That is a sharper framing than "the model is miscalibrated" and it ties the uncertainty section into the paper's spine instead of bolting it on.
- [ ] Confirm that abs-diff and the even head have Δ-ECE_swap = 0 exactly, as the identity requires. Another free correctness check.

## Part B — Selective prediction

- [ ] **Risk–coverage curves** and AURC per configuration, using max-probability (or entropy) as the confidence score.
- [ ] Define the abstention rule the proposal promised: the model defers ambiguous regions to human review. Report risk at a few fixed coverage levels (e.g. 90%, 75%, 50%).
- [ ] **SCE_prob as a label-free confidence signal.** SCE_prob is |p→ − p←| per pixel and needs no labels. Test whether it predicts error: do pixels with high swap-disagreement have higher error rates? If yes, you have a deployment-time uncertainty estimate that costs one forward pass and no annotation — a genuine contribution, and it makes the "uncertainty-aware" claim in the title real rather than cosmetic.
- [ ] Compare it against max-probability confidence on the same risk–coverage axes. Note the honest caveat: it is identically zero for symmetric fusion, so it is a signal available only for the models that need it.

## Part C — Directional head (phase 2)

The proposal's full contribution: predict *appeared* and *disappeared* separately, with a consistency loss enforcing that swapping (I1, I2) swaps the two channels while preserving the union.

- [ ] Add a 3-class or 2-channel head over the existing signed-fusion trunk.
- [ ] Implement the **permutation-target loss**: under swap, the appeared and disappeared targets exchange; the union mask is invariant. This is the correct symmetry for the directional task — equivariance, not invariance — and it is exactly what §VI-a argues signed fusion exists for.
- [ ] **Annotation problem, address early:** LEVIR-CD, SYSU-CD and PCD do not ship directional labels. Options, in order of preference: (1) derive pseudo-directional labels for LEVIR building change from the ordering plus mask morphology, clearly labelled as derived; (2) use a controlled synthetic source such as CLEVR-Change or ChangeSim, which the proposal already flagged as permissible *for a controlled directional experiment only* and with synthetic results reported separately from real ones. Do not let derived labels get quoted as if they were annotated.
- [ ] **The result the paper wants:** abs-diff *cannot* do this task — it destroys the antisymmetric part of the signal by construction. So the directional experiment is where signed fusion is necessary rather than merely preferable, and it converts §VI-a from an argument into a demonstration. Even a small, clearly-scoped result here closes the paper's biggest conceptual gap.
- [ ] Define the equivariance analogue of SCE: the fraction of predicted directional pixels whose class does *not* correctly swap under reversal. Report it.

---

## Deliverables

1. `metrics/calibration.py` — ECE, Brier, reliability diagrams, pooled and foreground-restricted. Register it as an extension to Member 3's schema, do not fork `evaluate.py`.
2. Calibration table + per-ordering ECE + Δ-ECE_swap.
3. Risk–coverage curves, AURC table, and the SCE_prob-as-confidence comparison.
4. If Part C lands: directional head, permutation-target loss, the equivariance metric, and a clearly-separated synthetic-vs-real results split.
5. A recommendation to Member 1 by the mid-point review: **keep the title or retitle.**

---

## Interfaces

**You depend on:**
- Member 3 for `evaluate.py` and its schema extension point — agree this in week 1.
- Member 1 for the foreground-restriction convention (reuse the SCE union definition exactly, including ignore-pixel masking, so the calibration numbers are comparable to the consistency numbers).

**Others depend on you for:**
- Member 1's title decision. Give them a clear go/no-go at the mid-point, not a maybe.
- The paper's answer to "you promised uncertainty and delivered none."

---

## Acceptance criteria

- ECE and Brier reported for every configuration, both pooled and foreground-restricted, with the gap between the two discussed.
- Δ-ECE_swap reported, and confirmed exactly 0 for the two architecturally guaranteed rows.
- At least one risk–coverage result with a stated abstention rule.
- Either a working directional experiment, or an explicit statement in the limitations that keeps the current honest framing.

---

## Risks

- **Part C's annotation dependency is the main threat.** Decide the label source in week 2. If neither option is workable, cut Part C early and put the effort into Parts A and B, which are cheap, certain, and sufficient to justify the title.
- **Synthetic contamination.** The proposal was explicit that synthetic results must be reported separately and must not replace the primary benchmark. Hold that line.
- **Scope overlap with Member 4.** You both have a phase-2 stretch item. If GPU time is short, Parts A and B beat everything in Part C and Part D — they are post-hoc, they close a title-level promise, and they extend the paper's own central argument.
