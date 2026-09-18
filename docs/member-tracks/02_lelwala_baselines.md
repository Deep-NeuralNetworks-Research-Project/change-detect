# Member 2 — Lelwala J.U.P.
## Track: The missing baselines

**GPU need:** high. Budget roughly 3 LEVIR runs + 3 SYSU runs + PCD folds, plus one transformer run if it fits.
**Estimated effort:** 8–10 person-days.
**Why this track exists:** the proposal called FC-Siam-Diff the *mandatory* baseline. The paper never trains it — M1 reproduces "FC-Siam-Diff's fusion behaviour, not its architecture", which the limitations section admits. RGB/SSIM (the non-DNN sanity check) and BIT are also missing. Without these the paper has no external anchor, and the 0.822-vs-published-0.871 gap has no explanation.

---

## 1. FC-Siam-Diff, the actual architecture

**Priority: highest.** This is the one item the proposal marked mandatory.

- [ ] Implement the original architecture from Daudt et al. [1]: the shallow fully-convolutional encoder with skip connections, multi-scale absolute feature differences `D^s = |E^s(I1) − E^s(I2)|`, U-Net-style decoder. Not the ResNet-18 trunk.
- [ ] Train on LEVIR-CD (3 seeds), SYSU-CD, and the PCD folds using **the shared recipe** — AdamW 3e-4, wd 0.01, cosine, batch 16, AMP, 20/15/60 epochs, BCE+Dice with the published positive weights. Do not tune it; the matched-budget argument is the whole point.
- [ ] Also run it **once** with a light tuning budget (lr from {1e-3, 3e-4, 1e-4}) so the paper can report what per-model tuning would have bought. This is what lets Member 1 write the gap paragraph honestly.
- [ ] Report F1 **and** SCE. Prediction: SCE = 0.000 exactly, because absolute-difference fusion is symmetric regardless of architecture. This is a second, independent correctness check on the taxonomy — it shows the guarantee follows from φ, not from the trunk. Say so in the write-up; it is a genuinely useful result, not just a baseline.

## 2. RGB/SSIM non-DNN sanity baseline

- [ ] Implement per-pixel RGB absolute difference and SSIM-based dissimilarity [12], thresholded.
- [ ] Sweep the threshold on validation only, fix it, evaluate on test.
- [ ] Report through Member 3's eval harness so it gets the same P/R/F1/IoU treatment.
- [ ] Note its SCE is trivially 0 (both operators are symmetric) and its F1 is presumably far below the learned models. Both facts belong in the paper: they establish the floor and they show SCE = 0 alone means nothing without F1 beside it — which is exactly the caveat §III-B already makes about the empty-union case.

## 3. BIT (stretch, but high value)

- [ ] Train BIT [6] on LEVIR-CD under the same recipe if 8 GB allows; drop to a smaller batch with gradient accumulation if needed.
- [ ] **The reason this matters:** §II-a *asserts* that transformer detectors fall in the unconstrained class because they concatenate or cross-attend. Right now that is a claim from reading their code. One measured BIT SCE number converts an assertion into evidence and materially strengthens §II and the conclusion.
- [ ] If BIT will not fit, try ChangeFormer [7] or say plainly in the limitations that the transformer class is argued but unmeasured.

## 4. Six-channel early-fusion baseline (optional)

The proposal mentioned this. Cheap to add: stack `[I1, I2]` on the channel axis into a single-tower network. It is the extreme of the unconstrained class and should show high SCE. One LEVIR run, include only if time permits.

---

## 5. Shared protocol you must follow

Take these from Member 1's `docs/CONVENTIONS.md` and do not deviate:

- τ = 0.5 fixed, never tuned on test
- ignore pixels masked before any counting, including the SCE union
- aggregate (corpus) F1, reduced once at the end
- same crops, same splits, scene-disjoint, asserted by test
- record resolved config, command line, environment, checkpoint SHA-256 per run
- report through Member 3's `evaluate.py` so every row in the paper is computed by one code path

---

## 6. Deliverables

1. `models/fc_siam_diff.py` — faithful reimplementation, with a short note on any deviation from the paper.
2. `models/ssim_baseline.py`
3. `models/bit/` (if it lands)
4. Results files for every run, in the format Member 3's harness consumes.
5. A 1-page write-up: reproduced FC-Siam-Diff vs published 0.871, what accounts for the difference (crop size, epoch budget, no tuning, single GPU), and the BIT SCE number with one sentence on what it means for §II-a.

---

## 7. Interfaces

**You depend on:**
- Member 1 for the frozen conventions (week 1).
- Member 3 for `evaluate.py` and the split manifests.

**Others depend on you for:**
- Member 1 needs your FC-Siam-Diff number to write the baseline-gap paragraph and your BIT SCE to upgrade §II-a.
- Member 3 needs your checkpoints to include in the full metric sweep.
- Member 4 will measure your models' params/FLOPs/latency — hand over clean, loadable checkpoints.

---

## 8. Acceptance criteria

- FC-Siam-Diff trained, evaluated on all three datasets, SCE confirmed at exactly 0.
- RGB/SSIM floor reported on at least LEVIR-CD.
- BIT either measured or explicitly declared out of budget in the limitations.
- Every number produced by the shared harness, not a private script.

---

## 9. Risks

- **8 GB is tight for BIT.** Mitigate with batch 4 + accumulation, AMP, 256 crops. Decide go/no-go by the mid-point review rather than burning the whole budget on it.
- **GPU contention.** One laptop 4060 serves five people. Agree a queue with Members 3 and 4 in week 1; your FC-Siam-Diff runs should get the first slot because Member 1 is blocked on them.
- **Reimplementation drift.** If your FC-Siam-Diff lands far from 0.871 even with tuning, document the exact deltas rather than quietly adjusting until it matches.
