# Member 3 — Pabasara W.G.K.
## Track: Evaluation suite and statistical completeness

**GPU need:** moderate — mostly re-evaluating saved checkpoints (cheap), plus the missing training runs (real, but they are re-launches of existing configs).
**Estimated effort:** 8–10 person-days.
**Why this track exists:** the paper reports F1 and SCE and nothing else. The proposal promised precision, recall, IoU, boundary quality, false-positive rates per no-change pair, image-level false-alert rate, and stratification by change size. Also, Table I promises three seeds on LEVIR and five PCD folds; it delivers three folds and, in one row, one seed. You own the harness everyone else reports through.

---

## 1. `evaluate.py` — the single code path (do this first, blocks Members 2, 4, 5)

One script, one output schema, consumed by everyone. It takes a checkpoint and a split and emits a JSON with:

**Accuracy**
- changed-class precision, recall, F1, IoU (aggregate/corpus, reduced once over all valid pixels)
- boundary quality — recommend the boundary-F1 (BF) score at a tolerance of 2 px, stated explicitly
- positive-prediction rate per ordering (§III-B already relies on this; make it a first-class field)

**Order-consistency** (from the existing implementation, just refactored in)
- SCE_flip, SCE_prob, ΔF1_swap
- Spearman ρ between the forward and reversed probability maps
- per-ordering F1-optimal threshold (the §V-C control)

**Robustness**
- false-positive pixels per no-change pair
- false-positive connected regions per no-change pair (label connected components, count them)
- image-level false-alert rate: fraction of no-change pairs where any region exceeds a minimum area

**Bookkeeping**
- π on the evaluated split, under the convention Member 1 freezes
- n pixels, n ignored, n pairs, seed, checkpoint SHA-256

Ship it in week 1 even if boundary metrics land later. Everything downstream is blocked on the schema.

## 2. Recompute π properly

Member 1 needs one authoritative set of changed-pixel ratios. Compute for all three datasets, on train and test separately, valid pixels only, and hand over a small table. The paper currently quotes 4.59 / 5.1 / 4.6 / ≈5 for LEVIR and 21.3 / 23.6 / 0.236 for SYSU. Kill the ambiguity at source.

## 3. Stratification by change size

The proposal promised it and it is a genuinely interesting axis for this paper: **does the flip rate depend on the size of the changed region?** Small changes are plausibly where sign information matters most.

- [ ] Bin connected ground-truth change components into small / medium / large by area (choose thresholds from the LEVIR component-size distribution and state them).
- [ ] Report F1 and SCE_flip per bin for all configurations.
- [ ] If SCE_flip is materially higher on small components, that is a new finding worth a subsection, not just a table.

## 4. Complete the statistics

- [ ] **PCD folds 4 and 5.** Run every configuration. This converts "three folds" into the 5-fold protocol §IV already claims.
- [ ] **SYSU-CD seeds 2 and 3.** SYSU is currently single-run for every row, so the paper cannot say anything about its spread — and §V-C leans on SYSU's smaller violations being a real dataset effect rather than seed noise. That argument needs error bars.
- [ ] **λ=5 LEVIR seeds 2 and 3.** Table I's λ=5 row has no ±, breaking the caption's promise.
- [ ] Re-emit Table I with consistent n across every cell, and record n per cell in the caption or a footnote.

## 5. Nuisance suite — make it auditable

§V-E reports corruption results but the suite itself is only described in prose.

- [ ] Publish the corruption definitions and severity parameters as a config file (ImageNet-C practice [14]), covering occlusion, viewpoint, gamma, brightness, blur, compression.
- [ ] Confirm and document that corruption is applied to exactly one frame, in both directions, and that the same seeded corruption instance is used across configurations so rows are comparable.
- [ ] Emit the full severity × corruption × configuration grid as a supplementary table; the paper currently summarises it in a single paragraph.

---

## 6. Deliverables

1. `evaluate.py` + documented output schema (week 1).
2. `metrics/` — boundary F1, connected-component FP counting, per-ordering threshold sweep, stratification.
3. Recomputed π table.
4. Completed runs: PCD folds 4–5, SYSU seeds 2–3, λ=5 LEVIR seeds 2–3.
5. Expanded results tables: the main table with P/R/F1/IoU/BF, the stratified table, the full nuisance grid.
6. The GPU queue schedule (you touch it most; you run it).

---

## 7. Interfaces

**You depend on:**
- Member 1 for the conventions doc (π definition, rounding, seed reporting).
- Members 2, 4, 5 for checkpoints to evaluate.

**Others depend on you for:**
- *Everyone* needs `evaluate.py` before producing any number. This is the highest-priority item in the whole project after the conventions doc.
- Member 1 needs the recomputed π and the completed runs to close out Table I.
- Member 5 will extend your schema with calibration fields — agree the extension point with them early rather than letting them fork the script.

---

## 8. Acceptance criteria

- Every number in the paper comes out of `evaluate.py`.
- No cell in any table has an unexplained missing ±.
- PCD is genuinely 5-fold or the paper says "three of five folds" everywhere.
- Precision, recall, IoU and a boundary metric appear for every configuration, as the proposal promised.
- The nuisance suite is reproducible from a committed config.

---

## 9. Risks

- **Scope creep in the harness.** Freeze the schema early; add fields, never rename them.
- **GPU contention.** Your re-evaluations are cheap and can interleave; the seed completions are not. Give Member 2's FC-Siam-Diff the first slot since Member 1 is blocked on it, and batch your seed runs overnight.
- **Boundary metric definitions vary.** Pick one, cite it, state the tolerance. Do not let it become a second ambiguity like π.
