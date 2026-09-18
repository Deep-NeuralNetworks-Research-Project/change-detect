# My role: P1 — Data Lead

> Copy this to `CLAUDE.local.md` at the repo root. It layers on top of the shared `CLAUDE.md`.

**I own:** `src/cdlib/data/**`, `scripts/download_*.sh`, `scripts/prepare_video_set.py`, `docs/DATA_CARD.md`, `docs/ETHICS.md`
**CODEOWNERS:** `/src/cdlib/data/ @me`
**I am the only hard blocker on the critical path.** Everyone's training work waits on my public-dataset loaders. Target: done by end of week 2. The video-set track is mine too, but it is deliberately *off* the critical path — treat it as a parallel stream.

## What I must not break

The dataset `__getitem__` contract in the root `CLAUDE.md`. Every loader returns the same dict — same keys, same dtypes, same `[0,1]` float range, `-1` for ignore regions. P2's trainer and P5's metrics both index into it blind.

`nuisance_label` is mine to populate. `0=clean, 1..k=nuisance type, -1=unknown`. P5's entire nuisance-stratified evaluation depends on me setting it honestly — if I don't know, it's `-1`, not `0`.

## Gotchas that will silently ruin the project if I get them wrong

**Leakage is my failure mode.** Adjacent video frames are near-duplicates. A frame-level split means the test set is in the training set and every number we report is fiction. Split by `source_video` / `scene_id`, with a temporal buffer around cut points. Write the test that asserts no `scene_id` crosses a split *before* writing the splitter.

**Augmentation asymmetry.** Geometric transforms (flip, rotate, crop, affine) must be applied **identically to both frames** — otherwise I manufacture misregistration that isn't in the data and P4's alignment module learns to fix my bug. Photometric transforms (brightness, contrast, gamma, blur, JPEG) must be applied **independently per frame** — that *is* the nuisance signal we want the model to learn to ignore. Two different code paths. Do not let a single `Compose` handle both.

**LEVIR-CD crop convention** — 256×256 non-overlapping, 7,120 / 1,024 / 2,048. Not negotiable; any other policy makes our numbers incomparable.

**Changed-pixel ratios differ 5× across our sources:** SYSU-CD 21.8%, LEVIR-CD 4.65%, our video domain likely lower still. Loss weights must be **recomputed per training stage**, not fixed once. Expose the ratio in the dataset so P2 can wire it into the loss.

**PCD is a single point of failure.** Only TSUNAMI is downloadable; GSV is not hosted anywhere. Archive TSUNAMI to our own Drive the moment I get it. Do not let GSV gate any deliverable.

**WHU-CD** has no official split and the circulating version has ~85% train/test leakage. If we touch it at all, I define the split, document it, and we never cite literature baselines as directly comparable.

**SYSU-CD label noise** is *inferred, not documented* — categories (c) pre-construction groundwork and (e) road expansion overlap our nuisance classes. Spot-check a sample before trusting the ground truth.

---

## Checklist

### Week 1 — unblock everyone, start the slow paperwork
- [ ] Read `research/02-datasets-data-pipeline.md` end to end
- [ ] Start the **video-set authorisation letter** — this has the longest lead time of anything I own. Scope it explicitly: research/educational, non-commercial, retention duration, redistribution (default: no)
- [ ] Check University of Moratuwa CSE ethics requirements for externally-sourced video
- [ ] Download SYSU-CD (BaiduYun pw `mlls`, or OneDrive) + LEVIR-CD; verify checksums; archive to team Drive
- [ ] Download PCD TSUNAMI from `sakuradaken.net/pcd_dataset.html` and **immediately mirror to team Drive**; email authors re: GSV
- [ ] Agree the frozen `__getitem__` contract with P2 and P5 — after this it doesn't change

### Week 2 — loaders and splits (hard deadline: everyone is waiting)
- [ ] `data/splits.py` — scene/source-disjoint splitter with temporal buffer
- [ ] `tests/test_splits.py` — assert no `scene_id` appears in two splits. Write this first. (Not in the original repo layout; I'm adding it — risk-register item 4 requires it)
- [ ] `data/datasets/sysu_cd.py` — official 12,000/4,000/4,000 split
- [ ] `data/datasets/levir_cd.py` — 256×256 non-overlapping crops, 7,120/1,024/2,048
- [ ] `data/datasets/pcd.py` — TSUNAMI, 5-fold CV protocol (20-pair folds)
- [ ] `data/transforms.py` — **shared geometric / independent photometric**, two separate paths
- [ ] Expose per-dataset changed-pixel ratio as a property; hand the numbers to P2 for loss weighting
- [ ] `data/registry.py` entries + `configs/data/*.yaml` for each
- [ ] Shape test passes for every registry key → **tell P2 and P3 they are unblocked**

### Weeks 3–5 — video pipeline (parallel stream)
- [ ] `scripts/prepare_video_set.py`: ffmpeg extraction + PySceneDetect cut detection
- [ ] Cross-version frame correspondence — try in this order: timecode/frame-index → **Chromaprint audio fingerprint** (robust to visual edits, usually frame-exact) → ORB/SIFT + homography fallback
- [ ] Near-duplicate dedup: `imagehash` phash first pass → CLIP ViT-B/32 + FAISS cosine (~0.95, tune)
- [ ] Nuisance-only hard-negative mining: temporally adjacent same-video pairs (guaranteed zero semantic change) + synthetic brightness/blur/JPEG/shadow perturbation. Coordinate with P5 so we don't build two corruption suites
- [ ] Validate my synthetic nuisance choices against ChangeSim's native dust/illumination axes before trusting them

### Weeks 4–7 — annotation
- [ ] Stand up CVAT self-hosted (Docker) with SAM2 integration
- [ ] Write the annotation guideline: "meaningful change" vs each nuisance category, 10–15 worked examples including edge cases (partial occlusion, shadow-only, re-grade)
- [ ] **Pilot**: 2 annotators, same 30–50 pairs, blind
- [ ] Compute agreement: mask IoU **and** pixel-level Cohen's κ. Report both — κ alone misbehaves at our prevalence
- [ ] Adjudicate, revise guideline, re-pilot on fresh 20–30 pairs until κ > 0.6–0.7
- [ ] Full annotation, ~15–25 pairs/hr SAM2-assisted once past the learning curve; every 10th pair blind re-annotated for QA
- [ ] `data/datasets/video_pairs.py` conforming to the frozen contract

### Ongoing — governance
- [ ] `docs/DATA_CARD.md` per dataset: source, licence, collection, split methodology + scene-disjointness proof, known biases; for the video set also authorisation basis, storage location, access list, retention/deletion plan, redistribution restriction
- [ ] `docs/ETHICS.md` referencing the signed permission letter
- [ ] Face-blur pass in the export path — mandatory before any frame leaves the private working set
- [ ] `docs/figure_clearance.md` — every published frame mapped to a clearance record/date/approver
- [ ] Verify `data/video_raw/` and frame exports are gitignored

### If the video set falls through
- [ ] Switch to **VideoSham** (Adobe, 352 real + 352 professionally edited pairs) — closest structural analog
- [ ] And/or **DAVIS/YTVI inpainting localisation** for object-removal masks at scale
- [ ] Tell the team immediately — the proposal's escape hatch is to report as paired-image CD with no post-production claim

## Commands I'll use

```bash
bash scripts/download_sysu_cd.sh && bash scripts/download_levir_cd.sh
pytest tests/test_splits.py tests/test_shapes.py -v
python -m cdlib.cli.train data=sysu_cd model=fc_siam_diff train.fast_dev_run=true   # smoke test my loader
python scripts/prepare_video_set.py --ref ref.mp4 --edit edit.mp4 --out data/video_pairs/
```

## Hand-offs

| To | What | When |
|---|---|---|
| P2 | working loaders + changed-pixel ratios for loss weighting | end of week 2 |
| P3, P4 | same — they can't train without them | end of week 2 |
| P5 | `nuisance_label` taxonomy so stratified eval bins match my labels | week 2 |
| P5 | coordinate on corruption suite — mine is for training negatives, theirs is for eval | **week 2** (before P5 starts building it in week 2–3) |
