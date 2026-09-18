# Member 1 — Ekanayake T.N.D.S.W.
## Track: Manuscript integrity, claims audit, and reproducibility release

**GPU need:** none (except re-running the table generator).
**Estimated effort:** 6–8 person-days, spread across the whole period.
**Why this track exists:** the paper's prose and Table I currently disagree in at least eight places, the changed-pixel ratio is quoted three different ways, and the title promises two things the experiments never deliver. None of that needs new results — it needs one person who owns the numbers end to end.

---

## 1. Number audit (do this first, blocks everyone)

Every figure in the prose must be generated from the results files, not typed. Build `scripts/make_tables.py` that emits Table I **and** a `claims.json` of every number quoted in the text, then check the prose against it.

### Known discrepancies to resolve

| Location | Prose says | Table I says |
|---|---|---|
| §V-D, M5 LEVIR F1 | 0.824 ± 0.010 | 0.822 ± 0.008 |
| §V-D, abs-diff LEVIR F1 | 0.809 | 0.808 ± 0.003 |
| §V-D, signed LEVIR F1 | 0.795 | 0.798 ± 0.005 |
| §V-D, signed LEVIR SCE | 0.884 | 0.856 ± 0.057 |
| §V-D, λ=5 PCD SCE | 0.085 | 0.101 ± 0.017 |
| §V-D, λ=5 PCD SCE source | 0.935 → | 0.934 ± 0.027 |
| §V-D, LEVIR F1 delta | +0.021 | 0.798 → 0.817 = +0.019 |
| §V-D, SYSU F1 delta | −0.015 | 0.796 → 0.782 = −0.014 |
| §V-E, M6 λ=5 severity-1 SCE | 0.235 | 0.237 |

### Mean-vs-extreme confusion

§V-A quotes "52% to 89%" flip and "F1 spans 0.793 to 0.831" and "SCE range 0 to 0.97". Table I means are 0.626–0.856 and 0.798–0.822 and max 0.960. Both are probably right — the text is quoting per-seed extremes, the table quotes means. Fix by writing *"per-seed extremes; Table I reports means"* explicitly at first use, or switch the prose to means. Do not leave it for the reviewer to reconcile.

### Changed-pixel ratio (π)

Currently four different values circulate:

- §III-B: π = 4.6%
- §IV-a: π = 4.59% (LEVIR, train), 21.3% (SYSU)
- §V-B: π ≈ 5%
- §V-C: SYSU π = 0.236
- Table caption: LEVIR 5.1%, SYSU 23.6%, PCD 29.9%

**Action:** pick one convention — recommend *test split, valid pixels only* — recompute all three from the actual splits, state the convention once in §IV, and use those numbers everywhere. Member 3 can hand you the recomputed values from the eval harness.

---

## 2. Structural fixes

- [ ] **"Six configurations" vs seven table rows.** M6 appears twice (λ=0.1 and λ=5). Either call it seven rows, or label the λ sweep as a sub-row of M6 with a rule in the table.
- [ ] **Missing seeds in Table I.** The λ=5 LEVIR row has no ±, implying one seed where every other LEVIR row has three. Either mark it explicitly as n=1 or get Member 3 to run the other two seeds. The caption currently promises "mean ± std over three seeds (LEVIR-CD)", which that row does not honour.
- [ ] **Dashes.** The caption says dashes mark configurations not run, but no dashes appear in the table. Remove the sentence or restore the dashes.
- [ ] **PCD fold count.** §IV introduces the 5-fold protocol; §IV-c and the caption say three folds. Once Member 3 finishes folds 4 and 5 this resolves itself; until then say "three of five folds" everywhere.
- [ ] **PCD size.** The paper says 100 pairs, the proposal said 200. PCD is TSUNAMI (100) + GSV (100). State plainly that only the TSUNAMI subset is used and why.
- [ ] **§V-C "at comparable π".** PCD (29.9%) is comparable to SYSU (23.6%), not to LEVIR (5.1%), but the sentence claims PCD behaves like LEVIR. The logic is right; the wording reads backwards. Rewrite as: *PCD's π is close to SYSU's, yet PCD's symptom matches LEVIR's — so π does not predict the symptom.*

---

## 3. Scope and title

The title promises **Semantic** and **Uncertainty-Aware**; the paper delivers binary change detection with no calibration section at all. Two ways out, decide by the mid-point review:

1. **Retitle** to something like *"Pair-Order Consistency in Siamese Change Detection: A Label-Free Diagnostic"* and drop the uncertainty keyword. Cheap, honest, and the paper is coherent as a measurement paper.
2. **Keep the title** and integrate Member 5's calibration and directional results. Only viable if Member 5 lands ECE/Brier/risk–coverage on time.

Recommend planning for (1) and upgrading to (2) if Member 5 delivers. Do not ship a title the results do not cover.

---

## 4. The baseline-gap paragraph

The proposal cited published FC-Siam-Diff at F1 **0.871** on LEVIR-CD (ref [9]). The paper's best configuration is 0.822. The paper never addresses the gap, and a reviewer will.

Write a short paragraph in §IV or §VI explaining the matched-budget design: one recipe for all rows, 20 epochs, no per-model tuning, 256×256 crops, one 8 GB GPU. The point of the paper is the *spread across fusion operators under a fixed budget*, not the absolute ceiling. Cite Corley et al. [2] for why that is the right call. Use Member 2's reproduced FC-Siam-Diff number as the anchor once it exists.

---

## 5. Related work and references

- [ ] Verify ref [4] (SEED, arXiv:2601.07805). Check whether it has been peer reviewed since drafting and update the parenthetical.
- [ ] Add a sentence to §II-b distinguishing this work from ChangeMask and SEED in one line a skim reader will catch: *they fix it, we measure it in models that didn't.*
- [ ] Once Member 2 has BIT numbers, add BIT to §II-a as an empirically confirmed member of the unconstrained class rather than an asserted one.
- [ ] Align reference style; the paper uses IEEE, the proposal used ACM. Pick the target venue's.

---

## 6. Reproducibility release

- [ ] Public repo with the source manifest, resolved configs, command lines, environment lock, and checkpoint SHA-256s already recorded per run.
- [ ] `README.md` with a one-command reproduction of Table I from released checkpoints.
- [ ] Release `sce.py` as a standalone file — the metric is the paper's contribution and it should be copy-pasteable into someone else's repo with no dependencies beyond torch.
- [ ] Licence check on all three datasets before release; note each dataset's terms.

---

## 7. Interfaces

**You depend on:**
- Member 3 for the canonical results files, recomputed π values, and the completed PCD folds/seeds.
- Member 2 for the FC-Siam-Diff and BIT numbers.
- Member 4 for the efficiency table.
- Member 5 for the calibration section (decides the title).

**Others depend on you for:**
- The frozen conventions document (`docs/CONVENTIONS.md`): π definition, rounding, seed reporting, mean-vs-extreme rule, table format. Publish this in week 1 so nobody generates numbers in an incompatible format.
- Final integration and the submission checklist.

---

## 8. Acceptance criteria

- Zero numbers in the prose that are not produced by `make_tables.py`.
- One π convention, stated once, used everywhere.
- Table I caption describes the table that actually exists.
- Title and abstract claim only what §V demonstrates.
- A reader can reproduce Table I from the repo without asking a question.
