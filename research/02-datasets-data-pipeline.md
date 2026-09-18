# Research Brief 2: Datasets, Licensing, Splits & Data Engineering

*Agent-researched with live web verification, 2026-09-05. Anything not independently verifiable is flagged as such rather than presented as fact.*

## 1. Dataset comparison table

| Dataset | Pairs | Resolution / GSD | Domain | Changed-px ratio | Official split | License | URL |
|---|---|---|---|---|---|---|---|
| **SYSU-CD** | 20,000 | 256x256, 0.5 m | Aerial, Hong Kong 2007-2014 | **21.8%** | 12,000 / 4,000 / 4,000 (6:2:2) | No LICENSE file — treat as research-only | [liumency/SYSU-CD](https://github.com/liumency/SYSU-CD) |
| **LEVIR-CD** | 637 -> 7,120/1,024/2,048 crops | 1024x1024, 0.5 m/px | Google Earth buildings, 5-14 yr span | **4.65%** | 445/64/128 full pairs (7:1:2) | Academic non-commercial + Google Earth ToS | [justchenhao.github.io/LEVIR](https://justchenhao.github.io/LEVIR/) |
| **LEVIR-CD+** | 985 (637 train + 348 test) | 1024x1024, 0.5 m/px | LEVIR-CD extension | not published separately | 637 / 348 | **Unconfirmed** ("unknown" on HF mirror) — verify with authors | [HF: blanchon/LEVIR_CDPlus](https://huggingface.co/datasets/blanchon/LEVIR_CDPlus) |
| **WHU-CD (building)** | 1 giant ortho pair, tiled | 32,507x15,354 px, 0.2 m/px, 20.5 km2 | Aerial, Christchurch NZ 2012->2016 | not centrally published | **No official split** — literature uses 8:1:1, 7:1:2, or 1260/690 @256 | Not formally published; contact author | [gpcv.whu.edu.cn](https://gpcv.whu.edu.cn/data/building_dataset.html) |
| **PCD - TSUNAMI** | 100 | 224x1024 panoramic | Street-level, tsunami-damaged Japan | not published | **5-fold CV** (20-pair folds) | Research use, cite BMVC'15 | [sakuradaken.net/pcd_dataset.html](https://sakuradaken.net/pcd_dataset.html) |
| **PCD - GSV** | 100 | same | Google Street View | — | same 5-fold CV | Same, **but not hosted publicly** | same page (GSV link not live) |
| **ChangeSim** | 13,225 train / 8,212 test frames | 640x480 | Photorealistic indoor industrial sim | binary + multiclass masks | fixed train/test | **MIT** | [SAMMiCA/ChangeSim](https://github.com/SAMMiCA/ChangeSim) |
| **CLEVR-Change** | 79,606 | synthetic | CLEVR scenes, 6 change types incl. viewpoint Distractor | N/A — **bboxes only, no masks** | 67,660 / 3,976 / 7,970 | CLEVR base CC BY 4.0; repo has no LICENSE — verify | [Seth-Park/RobustChangeCaptioning](https://github.com/Seth-Park/RobustChangeCaptioning) |

---

## 2. SYSU-CD

- 20,000 pairs, 256x256, 0.5 m GSD aerial RGB, Hong Kong 2007-2014. Split 12,000/4,000/4,000.
- Change taxonomy: (a) new urban buildings, (b) suburban dilation, (c) pre-construction groundwork, (d) vegetation change, (e) road expansion, (f) sea/coastal construction.
- **Label-noise caveat:** categories (c) and (e) overlap conceptually with your nuisance classes (bare ground, vehicles, dust), so expect elevated inter-annotator disagreement there. **No published label-noise audit was found** — this is an inferred risk from the taxonomy, not a documented critique. Verify empirically on a sample before trusting the ground truth at face value.
- Changed-pixel ratio: 286,092,024 / 1,310,720,000 = **21.8%** — roughly 5x more change-dense than LEVIR-CD. Useful as a "less imbalanced" pretraining stage before fine-tuning on your sparser target domain.
- Download: BaiduYun (password `mlls`) + OneDrive, from the GitHub README.
- Baselines: SChanger F1 84.58%, MSGFNet F1 80.39%; typical modern CNN/Transformer methods land 75-85% F1.

## 3. LEVIR-CD and LEVIR-CD+

- 637 VHR (0.5 m/px) Google Earth pairs at 1024x1024, 5-14 year span, building growth/decline only, 31,333 change instances, double-checked by a second annotator.
- Official split 445/64/128 full pairs; standard protocol crops non-overlapping to **256x256** -> **7,120 / 1,024 / 2,048**. Essentially every CD paper reports against this — use it unmodified.
- Changed-pixel ratio: 31,066,643 / 667,942,912 = **4.65%** — severe imbalance, drives loss weighting (section 9).
- License: academic/non-commercial, plus Google Earth ToS on top.
- Baselines: FC-Siam-conc F1 87.67 / IoU 78.04; FC-Siam-diff F1 85.81 / IoU 75.14; ChangeFormer F1 91.11.
- **LEVIR-CD+**: 985 pairs (637 train + 348 test), same group. HF mirror lists license **unknown** — email the LEVIR group before any use outside a course context.

## 4. PCD / Panoramic Change Detection (Sakurada & Okatani, BMVC 2015)

- Two 100-pair subsets, TSUNAMI and GSV, each **224x1024** equirectangular panoramas with binary PNG change masks.
- **The original host (`vision.is.tohoku.ac.jp`) is confirmed dead** (404, per [sscdnet issue #1](https://github.com/kensakurada/sscdnet/issues/1)). Canonical page moved to `sakuradaken.net/pcd_dataset.html`.
- **Do not confuse with PSCD** at `sakuradaken.net/pscd/` — a newer, unrelated dataset (770 pairs, ICRA-2020 lineage, different annotation schema with semantic classes, instance labels, privacy masks).
- **Only TSUNAMI has a live Google Drive link; GSV is not hosted**, plausibly due to Street View redistribution restrictions. Plan to email the authors for GSV, or budget time to rebuild a GSV-analog via the Street View Static API.
- **No third-party mirror found** (no Kaggle/HF/Zenodo copy). This is a single-point-of-failure dataset. **Mitigation: archive it to your own storage the moment you get access; do not put GSV on the critical path.**
- Standard protocol: 5-fold CV per subset (train on 4 folds of 20 pairs, test on held-out), used by CSCDNet, DR-TANet, HPCFNet. Panoramas cropped into smaller training patches with flip augmentation.
- Baselines (F1, GSV / TSUNAMI): FC-Siam-diff 66.2/79.5, FC-Siam-conc 70.4/81.6, CSCDNet 72.8/87.8, DR-TANet 72.3/87.6 (DR-TANet whole-dataset avg F1 ~0.781). [arXiv:2103.00879](https://arxiv.org/pdf/2103.00879)

## 5. WHU-CD

- Christchurch NZ, pre- (2012) and post-reconstruction (2016) aerial ortho, 32,507x15,354 px @ 0.2 m/px over 20.5 km2; buildings 12,796 -> 16,077. CD subset ~5.43 GB.
- Use the `gpcv.whu.edu.cn` mirror — `study.rsgis.whu.edu.cn` has a cert/SAN mismatch.
- **No official train/test split exists.** Literature diverges: 8:1:1, 7:1:2, or a common 1260-train/690-test 256x256-patch convention. **Pick one, document it explicitly, and never compare directly against papers using a different split.** (See also Brief 1 section 7: the commonly circulated WHU-CD version has ~85% train/test leakage.)
- License not formally published; contact Shunping Ji (jishunping@whu.edu.cn), cite Ji et al., TGRS 2018.
- Baseline range: TinyCD F1 88.39/IoU 79.19; SNUNet F1 90.47/IoU 82.61; DSIFN F1 92.07/IoU 85.30; ChangerAD F1 92.77/IoU 86.52; 2DMCG F1 95.07/IoU 90.59.

## 6. ChangeSim and CLEVR-Change

**ChangeSim** (IROS 2021) — photorealistic indoor-industrial simulation, 640x480, multimodal (RGB, depth, semantic seg, **pixel-level change segmentation**, camera pose, 3D reconstruction, point clouds), 13,225 train / 8,212 test frames. **Environmental nuisance conditions are explicitly modelled** ("Dusty-air", "Low-illumination"). MIT license. ~156 GB total across 4 splits, SharePoint-hosted by KAIST. [sammica.github.io/ChangeSim](https://sammica.github.io/ChangeSim/)

Baselines: ChangeNet mIoU 45.4% binary / 23.0% multiclass; CSCDNet 55.1/26.8; C-3PO 59.6/27.8. Domain shift under dusty/low-light drops mIoU ~29.7% -> ~27-31%.

**This is your best off-the-shelf dataset for a controlled nuisance-robustness ablation** — it varies lighting/turbidity while holding ground-truth object change fixed, which is exactly your suppression target, even though the domain (industrial indoor robotics) is far from edited video.

**CLEVR-Change** (ICCV 2019, Park et al.) — synthetic CLEVR renders, 79,606 pairs / 493,735 captions, split 67,660/3,976/7,970. Six change types: Color, Texture, Movement, Add, Drop, **Distractor** (viewpoint-only change, zero semantic change — a literal nuisance-only pair). **Localisation is bounding boxes only, not pixel masks.** Google Drive script in the repo (`clevr_change.tar.gz`). Inherits CLEVR's CC BY 4.0 lineage but the repo has no explicit LICENSE — verify before redistribution.

**Usability verdict:** strong for a *pair-order-consistency unit test* (fully synthetic, you can render forward and reverse pairs with guaranteed-clean labels) and a nuisance sanity check via the Distractor category. Treat it as a controlled diagnostic, not a pretraining source, and **budget real engineering time to convert bboxes to masks** (threshold rendered per-object alpha/depth if you regenerate scenes via the Blender pipeline, or train a box-to-mask proxy) — this is not a free lunch.

## 7. Risk mitigation — substitutes if the authorised video set falls through

| Dataset | Content | Masks? | Fit |
|---|---|---|---|
| **VideoSham** (Adobe, arXiv:2207.13064, WACV'23 workshop) | 352 real + 352 professionally edited (After Effects) human-centric videos, 6 spatial+temporal attack types = 704 videos | Localisation GT per attack | **Best direct analog** — already frames the task as paired reference/edited video with professional realistic edits. [adobe-research/VideoSham-dataset](https://github.com/adobe-research/VideoSham-dataset) |
| **DAVIS-based video inpainting localisation**: DVI (DAVIS-2016, 30 train/20 test, methods VI/OP/CP, 50 videos each), YTVI (YouTube-VOS 2018, 3,471 videos / 5,945 instances), plus modern AI-inpainting variants via E2FGVI/FuseFormer | Objects removed per DAVIS GT masks | **Per-frame binary pixel masks**, F1/mIoU protocol | **Strong substitute for "object removed" edits** — pixel-accurate, large-scale, documented eval. See BMVC'21 arXiv:2101.11080, Mumpy arXiv:2404.11054 |
| **Video splicing localisation** (e.g. arXiv:2309.09482) | Composited/spliced regions | Pixel-level splicing masks | Datasets are largely self-constructed per paper — useful as method reference more than ready-made data |
| **Deepfake localisation**: LAV-DF (~136K), AV-Deepfake1M (~1.15M, ACM MM), Glitch in the Matrix (CVIU 2023) | Manipulated face/AV segments | Mostly **temporal** boundaries, some face-region masks | Weak fit — face-centric and temporal, not general scene-object masks. Lowest priority |
| **VCDB** (near-duplicate/partial copy detection) | 528 core videos / 9,236 annotated copied segments + 100K distractors, real transformations | Segment-level only | Useful only for the *video-alignment* sub-problem, not for change-mask supervision |

**Recommendation:** if the authorised set is delayed, use **VideoSham** as the closest structural substitute for final-domain testing, and **DAVIS/YTVI inpainting localisation** as a large-scale pixel-mask pretraining source for the object-removal case specifically.

## 8. Data engineering

**Scene/source-disjoint splitting.** RS-CD literature repeatedly documents that random patch-level splits leak information via spatial autocorrelation (Springer, *Machine Learning* 2021, "Spatial dependence between training and test sets: another pitfall of classification accuracy assessment in remote sensing"). For video-derived data this maps directly: **split by source video / scene ID, never by frame**, with a temporal buffer around cut points so adjacent or near-duplicate frames cannot straddle partitions. For WHU-CD and PCD (no official split), build the split at tile/panorama level *before* any cropping.

**Near-duplicate detection — two-tier pipeline.** Perceptual hash (`imagehash`, phash) as a cheap first pass to collapse near-pixel-identical frames, then CLIP embeddings (ViT-B/32, 512-d, L2-normalised) + cosine similarity (~0.95, tune per dataset) via a FAISS flat index to catch semantic near-duplicates the hash misses.

**Mining nuisance-only hard negatives.** Sample temporally-adjacent frame pairs from a single video (guaranteed zero semantic change) and apply synthetic nuisance perturbations — brightness/contrast jitter, Gaussian/motion blur, JPEG re-compression at varying quality, synthetic shadows. This manufactures labelled "changed-appearance, unchanged-content" pairs at near-zero annotation cost. **Validate your synthetic-nuisance choices against ChangeSim's native dust/illumination axes before trusting them on your own footage.**

**Cross-version frame correspondence when edits shift timing** — three techniques, in order of robustness-to-effort:
1. **Timecode/frame-index matching** — trivial, but only works if both versions share an unedited base timeline (rare once cuts are inserted/removed).
2. **Audio fingerprinting** (Chromaprint/AcoustID-style: 11,025 Hz resample, 4096-sample FFT window, 2/3 overlap) — robust to visual edits since it aligns on audio; achieves sub-second, often frame-exact alignment; the standard approach for subtitle/multi-cam resync. Fails if edits touch audio (music swaps, mutes) or footage is silent.
3. **Feature-based visual alignment** (ORB/SIFT + homography, or scene-embedding matching) as fallback — needed for camera-shift-only pairs where you want alignment despite viewpoint drift.

**Libraries:** `ffmpeg` (extraction/transcode), `PySceneDetect` (`detect-content` for cuts, `detect-hash` — internally imagehash phash — for near-duplicate-aware sampling), `decord` (fast random-access video reader; **note PyPI wheels are CPU-only, GPU decode requires building from source**), `imagehash`, `faiss`.

## 9. Annotation tooling and protocol

**Throughput.** No published number specific to *change-mask* annotation was found — treat the following as a baseline to validate on your own pilot, not a guarantee:
- Manual polygon segmentation: **20-50 images/hour**; instance segmentation (box+polygon): **10-20/hour**.
- SAM-per-frame assisted: **37.8 s/frame** (~95/hr); **SAM2 video-tracking assisted: 4.5 s/frame (~800/hr)**, an ~8.4x speedup, in Meta's SA-V pipeline (arXiv:2408.00714).
- SAM-assisted single-mask refinement, 14 domain experts: 9.7 s/mask -> **<2 s/mask** with AI pre-labelling.
- Roboflow publishes no images/hour for human annotators; vendor claims ~95% time reduction via Auto Label; per-annotation pricing $0.10/box, $0.20/polygon if outsourcing.

Change-mask annotation is *harder* than single-image segmentation — the annotator must compare two frames, discriminate real edits from nuisance, and often mark multiple regions per pair. A defensible working estimate: **8-20 pairs/hour manual**, **20-40 pairs/hour with a SAM2-assisted difference-mask workflow** once past the learning curve. Validate on your pilot before committing to a total-hours budget.

**Tooling recommendation: CVAT self-hosted** (Docker) — native SAM/SAM2 integration for click-to-mask, polygon/mask export, multi-annotator task assignment, review workflow ([CVAT SAM2 changelog](https://www.cvat.ai/resources/changelog/video-annotation-sam-2)). Label Studio is a reasonable alternative for mixed modality but its SAM integration is less mature. Roboflow is fastest to stand up but the free tier plus per-annotation pricing gets expensive at full-dataset scale and locks you into its export format.

**Suggested protocol:**
1. Write a guideline defining "meaningful change" vs each nuisance category, with 10-15 worked examples including edge cases (partial occlusion, shadow-only difference, re-graded colour).
2. **Pilot:** 2 annotators independently label the same 30-50 pairs blind.
3. Compute agreement: **mask IoU** (mean pairwise IoU) for the continuous overlap signal, **plus Cohen's kappa** at pixel level for a chance-corrected statistic. **Report both** — kappa alone misbehaves at the 5-20% changed-pixel prevalence these datasets exhibit.
4. Adjudicate disagreements as a group, revise the guideline, re-pilot on a fresh 20-30 pairs until agreement stabilises. Rough go/no-go: **kappa > 0.6-0.7** before scaling up.
5. Full annotation with periodic (every 10th pair) blind re-annotation for ongoing QA.

## 10. Changed-pixel ratio and loss weighting

| Dataset | Changed-px | Implication |
|---|---|---|
| SYSU-CD | 21.8% | Mildest imbalance — good "easy" pretraining stage |
| LEVIR-CD | 4.65% | Severe — weighted BCE/Focal (alpha ~ inverse frequency) or Dice/Tversky required; plain BCE collapses to all-background |
| WHU-CD | Not centrally published; reported as *more* imbalanced than LEVIR-CD | Same mitigation; verify empirically on your chosen split |
| Your target video-edit domain | Expect **lower** than LEVIR-CD (most of an edited frame is unedited background) | Budget aggressive class weighting / hard-negative mining from the start |

**Recommendation:** combined weighted-BCE + Dice (or Tversky with beta > 0.5 to penalise false negatives harder), with **weights recomputed per training stage** rather than one fixed weight — the ratio swings by >4x across your pretraining sources alone. (Cross-reference Brief 4 on how Dice affects calibration.)

---

## Data-pipeline recommendation

1. **Pretrain/benchmark:** SYSU-CD (less imbalanced, varied aerial nuisance) -> LEVIR-CD (standard, literature-comparable) -> WHU-CD (larger-scale sanity check, *fix and document your split first*).
2. **Nuisance-robustness ablation:** ChangeSim (real pixel masks + controlled lighting/turbidity) + CLEVR-Change Distractor subset (synthetic, clean pair-order test).
3. **Domain-transfer / fallback:** VideoSham (closest structural analog) and DAVIS/YTVI inpainting localisation (large-scale masks for object removal).
4. **Target-domain pipeline:** ffmpeg + PySceneDetect extraction -> Chromaprint audio alignment (fallback ORB/homography) between reference and edited video -> imagehash + CLIP/FAISS dedup -> scene-disjoint split -> CVAT+SAM2 annotation with the IoU/kappa-piloted protocol -> synthetic nuisance-only hard negatives validated against ChangeSim.

## Person-hour estimates (team-level, semester scope)

| Task | Estimate | Basis |
|---|---|---|
| Acquire + verify licenses + set up 5-6 public datasets | 8-12 h | ~1.5-2 h/dataset incl. download, checksum, license read |
| Fix WHU-CD / PCD split + document protocol | 4-6 h | No official split; needs scripting + validation |
| Near-duplicate dedup (imagehash + CLIP + FAISS) | 8-16 h | Build, threshold-tune, spot-validate |
| Frame extraction + scene detection + cross-version alignment | 20-40 h | **Hardest engineering task**; timing drift from edits is nontrivial |
| Nuisance-only hard-negative mining + synthetic aug | 6-10 h | Script + manual spot-check |
| Annotation tool setup (CVAT + SAM2, Dockerised) | 4-8 h | Infra + ontology config |
| Guideline + IoU/kappa pilot (30-50 pairs, 2 annotators, 1-2 rounds) | 10-15 h | Writing + pilot labelling + agreement analysis |
| Full annotation of target eval set (300-500 pairs @ ~15-25 pairs/hr) | 15-33 h | Scale with N/throughput; +20% for QA re-pass |
| Documentation of splits & protocol | 6-10 h | Needed given how many datasets lack official splits |
| **Total** | **~80-150 h** | Team of 3-5; annotation-set size is the main lever |

---

## Two risks to flag to the team

1. **PCD is a single point of failure.** Only TSUNAMI is confirmed downloadable, from one researcher's personal site; GSV is not hosted. Archive immediately on access and do not build a critical-path deliverable around GSV.
2. **WHU-CD has no official split**, so any number you report is comparable only to papers using your exact split. State your split explicitly rather than citing literature baselines as directly comparable.

## Verified sources

[liumency/SYSU-CD](https://github.com/liumency/SYSU-CD) · [justchenhao.github.io/LEVIR](https://justchenhao.github.io/LEVIR/) · [HF LEVIR_CDPlus](https://huggingface.co/datasets/blanchon/LEVIR_CDPlus) · [gpcv.whu.edu.cn](https://gpcv.whu.edu.cn/data/building_dataset.html) · [sakuradaken.net/pcd_dataset.html](https://sakuradaken.net/pcd_dataset.html) · [sscdnet issue #1](https://github.com/kensakurada/sscdnet/issues/1) · [DR-TANet arXiv:2103.00879](https://arxiv.org/pdf/2103.00879) · [ChangeSim](https://sammica.github.io/ChangeSim/) · [SAMMiCA/ChangeSim](https://github.com/SAMMiCA/ChangeSim) · [RobustChangeCaptioning](https://github.com/Seth-Park/RobustChangeCaptioning) · [VideoSham (arXiv:2207.13064)](https://github.com/adobe-research/VideoSham-dataset) · [Deep Video Inpainting Detection (arXiv:2101.11080)](https://arxiv.org/pdf/2101.11080) · [SAM 2 (arXiv:2408.00714)](https://arxiv.org/pdf/2408.00714) · [CVAT SAM2 changelog](https://www.cvat.ai/resources/changelog/video-annotation-sam-2) · [VCDB](https://link.springer.com/chapter/10.1007/978-3-319-10593-2_24) · [Spatial leakage in RS ML](https://link.springer.com/article/10.1007/s10994-021-05972-1)
