# Research Brief 1: Baselines, Architectures & Reference Implementations

*Agent-researched with live web verification. All URLs fetched and confirmed.*

## 1. FC-Siam-Diff / FC-Siam-Conc / FC-EF (Daudt et al., ICIP 2018)

**Paper:** "Fully Convolutional Siamese Networks for Change Detection," arXiv:1810.08462. Trained/evaluated on **OSCD** (Onera Satellite Change Detection, 24 Sentinel-2 pairs, 14 train / 10 test).

### Exact architecture (from official code, `rcdaudt/fully_convolutional_change_detection`)

All three are 4-stage U-Nets (not 5, unlike standard U-Net) with 3x3 convs, BatchNorm2d, Dropout2d(p=0.2), ReLU. Downsampling = MaxPool2d(2); upsampling = ConvTranspose2d(k=3, s=2, output_padding=1).

**Encoder (shared, Siamese for the two Siam variants):**

| Stage | Convs | Channels |
|---|---|---|
| 1 | 2x conv | in -> 16 -> 16 |
| 2 | 2x conv | 16 -> 32 -> 32 |
| 3 | 3x conv | 32 -> 64 -> 64 -> 64 |
| 4 | 3x conv | 64 -> 128 -> 128 -> 128 |

- **FC-EF**: no Siamese split — the two images are concatenated on the channel axis (`torch.cat((x1,x2),1)`) before a single encoder (early fusion). Same 4-stage channel plan with `2*in_nbr` input channels.
- **FC-Siam-Diff**: two encoder passes with shared weights; skip connections fused as `torch.abs(x_stage_1 - x_stage_2)`, concatenated with the upsampled decoder feature (decoder stage 4: 128 upsampled + 128 |diff| = 256 in).
- **FC-Siam-Conc**: skips concatenate **both** branch features plus the upsampled decoder feature (decoder stage 4: 128+128+128 = 384 in) — roughly 1.5x the decoder width/params of Diff.
- Final layer: `LogSoftmax`, trained with NLL/weighted cross-entropy, class weights inversely proportional to class frequency.

**Training recipe actually documented in the paper:** GTX 1070, PyTorch, flips + 90-degree rotation augmentation, dropout. Optimizer, LR, batch size, epoch count, and patch size are **not stated** — a genuine reproducibility gap you will have to fill by tuning.

**OSCD test-set numbers (paper):**

| Model | Precision | Recall | F1 | Global Acc. |
|---|---|---|---|---|
| FC-EF (3-ch RGB) | 44.72 | 53.92 | 48.89 | 94.23 |
| FC-Siam-conc (3-ch) | 42.89 | 47.77 | 45.20 | 94.07 |
| FC-Siam-diff (3-ch) | 49.81 | 47.94 | 48.86 | 94.86 |
| FC-EF (13-ch MS) | 64.42 | 50.97 | 56.91 | 96.05 |
| FC-Siam-conc (13-ch) | 42.39 | 65.15 | 51.36 | 93.68 |
| FC-Siam-diff (13-ch) | 57.84 | 57.99 | **57.92** | 95.68 |

The RGB-only rows (F1 ~45-49) are the relevant reference point for this project — paired RGB frames, not 13-band Sentinel-2.

**Param counts** (not in the paper; from later re-implementations on LEVIR-CD): FC-EF ~1.35M, FC-Siam-Diff ~1.35-4.4M, FC-Siam-Conc ~1.5-5.0M depending on repo. That variance is itself a reproducibility gotcha (see section 7).

### Public implementations

| Repo | License | Status | Encoder swap? |
|---|---|---|---|
| [rcdaudt/fully_convolutional_change_detection](https://github.com/rcdaudt/fully_convolutional_change_detection) | unspecified | Inactive (234 stars, 4 commits, no training code) | No — hardcoded channels |
| [Bobholamovic/FCN-CD-PyTorch](https://github.com/Bobholamovic/FCN-CD-PyTorch) | BSD-2-Clause | **Deprecated**, author redirects to CDLab | No |
| **[TorchGeo `torchgeo.models.fcsiam`](https://torchgeo.readthedocs.io/en/stable/api/models.html)** | MIT | Actively maintained (Microsoft) | **Yes — fully swappable** |
| [likyoo/change_detection.pytorch](https://github.com/likyoo/change_detection.pytorch) | MIT | Semi-active | Yes — SMP-style |
| **[likyoo/open-cd](https://github.com/likyoo/open-cd)** | Apache-2.0 | **Actively maintained** (ACM MM 2025 report) | Yes — MMSeg-based, 20+ methods |

**TorchGeo detail (the key finding):** `FCSiamDiff`/`FCSiamConc` are thin wrappers around `segmentation_models_pytorch`'s `Unet`/`UnetDecoder`, exposing `encoder_name` (any SMP/timm backbone: `resnet18`, `timm-efficientnet-b0`), `encoder_depth`, `encoder_weights="imagenet"`, `decoder_channels`, `decoder_attention_type`. **This is the single most useful implementation for this project** — you get FC-Siam-Diff's diff-fusion pattern with a modern ImageNet-pretrained swappable encoder, and it generalises directly to your proposed ResNet-18/EfficientNet encoder with almost no code change.

Caveat: it is *not* architecturally identical to the original 4-stage 16/32/64/128 net — it is a re-implementation on an SMP U-Net decoder. Do not cite its numbers as "the" FC-Siam-Diff without noting the substitution.

---

## 2. Siamese ResNet-18 CD encoder

`torchvision.models.resnet18` stage outputs:

| Stage | Name | Stride | Channels |
|---|---|---|---|
| stem (conv1+bn+relu+maxpool) | — | 4 | 64 |
| layer1 | C2 | 4 | 64 |
| layer2 | C3 | 8 | 128 |
| layer3 | C4 | 16 | 256 |
| layer4 | C5 | 32 | 512 |

Tap 4 skip levels at strides {4, 8, 16, 32}, channels [64, 128, 256, 512]. Backbone: **11.7M params**, ~1.8 GFLOPs @ 224x224.

**Documented pitfalls:**

- **BatchNorm sharing across two Siamese passes.** BN stats are per-batch; pushing I1 and I2 through as two separate forwards means BN sees half the effective batch each time and mixes running statistics from both time-points asymmetrically. Fix (also discussed in "On the Importance of Asymmetry for Siamese Representation Learning," arXiv:2204.00613): concatenate `[I1;I2]` along the **batch** dimension into one forward pass so BN sees a consistent 2N batch, then split the output. Cheaper and statistically more consistent than two sequential forwards.
- **Stride-32 is too coarse for thin change boundaries.** Common finding in CD literature (Changer, SARAS-Net arXiv:2212.01287): either drop layer4 and decode from C2-C4 only, or replace the stride-2 convs in the last stage(s) with stride-1 (dilated/atrous ResNet) to preserve resolution. Costs FLOPs but recovers thin/linear structures — directly relevant to detecting small inserted/removed objects.
- **ImageNet normalisation** (`mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]`) is directly appropriate here because you use natural video RGB, unlike most CD literature working with satellite radiometry where ImageNet norm is a known transfer problem. This is a genuine advantage of your domain — say so in the report.

---

## 3. EfficientNet-B0/B2 as CD encoder (timm)

`timm.create_model('efficientnet_b0', features_only=True)` returns 5 feature maps (`out_indices=(0,1,2,3,4)`); query at runtime with `model.feature_info.channels()` / `.reduction()`:

| out_index | stride | B0 ch | B2 ch (approx) |
|---|---|---|---|
| 0 | 2 | 16 | 16 |
| 1 | 4 | 24 | 24 |
| 2 | 8 | 40 | ~48 |
| 3 | 16 | 112 | ~120 |
| 4 | 32 | 320 | ~352 |

Confirm B2 numbers at runtime — timm's channel rounding uses a divisor-8 rule off base widths `[16,24,40,80,112,192,320]` and published figures vary slightly.

**Params/FLOPs (ImageNet, 224x224 unless noted):**
- ResNet-18: 11.7M params, ~1.8 GFLOPs
- EfficientNet-B0: **5.3M params**, ~0.39 GFLOPs (386M MACs)
- EfficientNet-B2 (native 260x260): **9.2M params**, ~1.0 GFLOPs

**Critical caveat for your compute budget.** EfficientNet's low FLOP count does **not** translate proportionally to wall-clock speed on GPU. Depthwise separable convs have low arithmetic intensity and fragmented memory access that cuDNN does not optimise as well as dense 3x3 convs (see Graphcore, "How we made EfficientNet more efficient"; Nature Sci Reports 2025 on depthwise conv memory cost). Combined with SiLU/Swish (extra sigmoid+multiply, not always kernel-fused in vanilla PyTorch), **EfficientNet-B0/B2 can run slower in wall-clock than ResNet-18 despite fewer FLOPs**, especially on P100. Benchmark actual step time early; do not assume the FLOP advantage is a speed advantage.

---

## 4. BIT and ChangeFormer

### BIT — Chen et al., TGRS 2021
Repo: [justchenhao/BIT_CD](https://github.com/justchenhao/BIT_CD) — official PyTorch, research-only license.

- ResNet-18 backbone (reduced stride), compresses spatial features into a small token set, transformer encoder over tokens, transformer decoder projects back to pixel space before differencing.
- **Params 2.99-3.55M** (2.99M per open-cd, 3.55M per original paper Table II), **FLOPs 4.35-8.75 G** @ 256x256.
- **LEVIR-CD test (paper):** P 89.24, R 89.37, **F1 89.31, IoU 80.68**, OA 98.92.
- **Training:** single V100, SGD momentum 0.99, wd 5e-4, LR 0.01 linearly decayed to 0 over **200 epochs**, batch 8, 256x256 crops, flip/rescale/crop/blur aug.

### ChangeFormer — Bandara & Patel, IGARSS 2022
Repo: [wgcban/ChangeFormer](https://github.com/wgcban/ChangeFormer) — official PyTorch, research-only license.

- Siamese hierarchical MixTransformer (MiT/SegFormer) encoder + lightweight MLP decoder fusing multi-scale differenced features.
- **Params ~41.03M**, **FLOPs ~202.87 G** @ 256x256 — **~40-45x BIT's FLOPs for ~2 F1 points**. A `ChangeFormer-b0` variant exists at 3.85M / 2.46 GFLOPs, F1 90.57 (open-cd) — nearly matching the full model at a fraction of cost.
- **LEVIR-CD test (paper):** P 92.05, R 88.80, **F1 90.40, IoU 82.48**, OA 99.04.
- **Training:** Quadro RTX 8000 (48GB), AdamW (wd 0.01), LR 1e-4 linear decay, batch 16, 200 epochs, 256x256, initialised from ADE20K-pretrained SegFormer — the paper notes this pretraining was needed to converge well.

### Free-Colab/Kaggle feasibility verdict

**BIT: yes, comfortably.** 3-3.5M params, ~4-9 GFLOPs/pair @ 256^2 — *lighter* than FC-Siam-Conc + ResNet decoder in FLOP terms. LEVIR-CD-256 has 7,120 train patches; batch 8 = ~890 iters/epoch x 200 epochs ~ 178K iterations. On a free T4 this is a few hours wall-clock — inside a single 12h Colab session, or 2-3 Kaggle sessions within the 30 GPU-hr/week quota. **BIT is the stretch model your team can actually finish training from scratch.**

**ChangeFormer: marginal.** ~14x BIT's params, ~40x its FLOPs. Original recipe used batch 16 on a 48GB card; on a 16GB T4/P100 expect batch 4-8 with gradient accumulation and AMP to fit at all. Extrapolating from the FLOP ratio, 200 epochs could plausibly run 15-25+ GPU-hours — **this is a reasoned estimate, not a documented figure** (neither repo states GPU-hours). Practical advice: use **ChangeFormer-b0** (3.85M, 2.46 GFLOPs) if you want the transformer story without burning the compute budget on one baseline.

---

## 5. Fusion strategies (relevant to the signed-fusion claim)

- **Absolute difference** `|f1 - f2|` (FC-Siam-Diff): order-invariant, half the decoder width of concat, **discards direction** — appeared and disappeared collapse to the same signal.
- **Concatenation** `[f1; f2]` (FC-Siam-Conc): preserves everything, order encoded by channel position, but 2-3x decoder input width.
- **Signed difference** `f2 - f1`: the literature you are extending explicitly notes that preserving sign encodes appeared-vs-disappeared direction. Citable precedents: **FCCDN** (arXiv:2105.10860) with its dual sum/difference branches; multiple 2024-2025 papers on dual sum-branch + subtraction-branch designs (HFNet; Nature Sci Reports 2025, "Siamese change detection based on information interaction and fusion network"). Frame your contribution as: abs-diff throws away directionality, concat keeps it at 2x cost, **signed difference + product is a middle ground — cheaper than concat, strictly more informative than abs-diff.**
- **Difference + concat + product**: the product term captures multiplicative/co-occurrence interactions that concat and diff alone miss, at the cost of one elementwise op.
- **Six-channel early fusion** (what FC-EF does): simplest, but cannot warm-start cleanly from 3-channel ImageNet weights without stem surgery (duplicate or average the pretrained stem conv into 6 channels — a well-known trick).
- **Detect-and-match**: alternative paradigm — "Spot the Difference by Object Detection" (arXiv:1801.01051), "Detecting Object-Level Scene Changes Using Graph Matching" (MDPI RS 2022, 10.3390/rs14174225). Detect objects per frame independently, match across frames, flag unmatched. Inherently more robust to misalignment than pixel-aligned diff. **Worth one related-work paragraph as the design point you deliberately did not take** — and worth stealing the idea of an explicit alignment/matching step ahead of dense fusion, which maps directly onto your bounded alignment module.

---

## 6. Leaderboards

### LEVIR-CD test — single consistent source: open-cd benchmark Table 3 (arXiv:2407.15317)

Use this table rather than mixing per-paper self-reported numbers.

| Method | Backbone | Params (M) | GFLOPs | F1 | IoU |
|---|---|---|---|---|---|
| FC-EF | — | 1.353 | 3.24 | 80.85 | 67.86 |
| FC-Siam-Diff | — | 4.385 | 1.35 | 84.68 | 73.44 |
| FC-Siam-Conc | — | 4.989 | 1.55 | 85.55 | 74.75 |
| STANet-PAM | ResNet-18 | 13.356 | 48.08 | 87.13 | 77.20 |
| SNUNet-c16 | — | 3.012 | 11.73 | 91.36 | 84.09 |
| BIT | ResNet-18 | 2.990 | 8.75 | 90.71 | 83.00 |
| ChangeStar | ResNet-18 | 16.965 | 19.21 | 91.26 | 83.92 |
| ChangeFormer-b0 | MiT-b0 | 3.847 | 2.46 | 90.57 | 82.76 |
| ChangeFormer-b1 | MiT-b1 | 13.941 | 5.83 | 91.14 | 83.71 |
| TinyCD | — | **0.285** | 1.45 | 90.87 | 83.26 |
| Changer | ResNet-18 | 11.391 | 5.96 | 91.81 | 84.86 |
| CGNet | VGG-16 | 38.989 | 87.55 | 92.10 | 85.36 |
| TTP (2024) | ViT-L | 6.21 | 929.8 | 92.10 | 85.60 |

Cross-check from original papers (different protocols — illustrates section 7): FC-Siam-Diff F1 86.31 (BIT paper's re-run), BIT F1 89.31, ChangeFormer F1 90.40, SNUNet F1 88.16. TinyCD's paper claims **"13 to 140x smaller"** than then-SOTA (~0.285M vs 4-40M) while matching or beating by >=1 F1 — a useful reference point for how small a good CD model can be.

### SYSU-CD

Far less standardised than LEVIR-CD — treat any single-paper number with suspicion.
- FC-Siam-Diff: F1 reported **75.15 to 84.68**, IoU **63.87 to 73.44** across re-implementations.
- BIT: F1 ~73.32-81.5, IoU ~57.9-68.8.
- ChangeFormer (MiT-b0): mIoU 71.19, mF1 78.41 (JL1-CD benchmark, arXiv:2502.13407 Table VI — **mean over both classes**, not change-class IoU; do not put this side by side with the LEVIR table without converting).
- SNUNet: mIoU 70.13, mF1 77.80 (same table).
- Dataset: **20,000 pairs, 256x256, 0.5m aerial**, Hong Kong 2007-2014, official split **12,000/4,000/4,000**. [liumency/SYSU-CD](https://github.com/liumency/SYSU-CD), Shi & Liu et al., TGRS 2021.

**Report advice:** cite open-cd's LEVIR table as your single-protocol reference; for SYSU-CD report a range and explicitly flag non-comparability. That flag is itself a finding worth stating.

---

## 7. Reproducibility gotchas

1. **LEVIR-CD crop convention.** Original: **637 pairs at 1024x1024**, 0.5m/px, author split **445/64/128 (70/10/20%)**. Near-universal convention crops non-overlapping into **256x256**, giving **7,120 / 1,024 / 2,048** patches. Different crop size or overlap policy makes your numbers incomparable to every row above. ChangeFormer's cropping script is public: [`levir_cd_256.m`](https://github.com/wgcban/ChangeFormer/blob/main/data_preparation/levir_cd_256.m).

2. **SYSU-CD split.** Use the official 12,000/4,000/4,000 from liumency/SYSU-CD. Ad hoc re-splitting is likely a real contributor to the wide F1 spread above.

3. **Aggregate vs per-image F1 — material.** **Aggregate/corpus F1** sums TP/FP/FN across the whole test set then computes one F1; this is what nearly all CD papers report. **Per-image F1** averages per-image F1s, giving equal weight to a nearly-empty tile and a heavily-changed one. Under heavy imbalance these diverge substantially. **"A Change Detection Reality Check"** (Corley et al., arXiv:2402.06994) flags this as a source of irreconcilable cross-paper comparisons. **Decide, state explicitly, and prefer aggregate F1** to match leaderboard convention.

4. **The Reality Check's other finding — read this one carefully.** On LEVIR-CD, a **plain U-Net with ResNet-50 encoder (no CD-specific tricks) scores F1 90.38**, and **U-Net-SiamDiff scores 90.46** — both matching or beating ChangeFormer (91.11) and essentially tying TinyCD (91.05), using no architectural novelty, purely from a modern well-tuned recipe. On WHU-CD with corrected splits (the commonly used version has **~85% train/test leakage** — a second major gotcha), simple U-Net baselines (F1 81.85-82.02) **outperform** BIT (72.67) and ChangeFormer (75.65). The paper recommends standardising on TorchGeo/open-cd/GEO-Bench rather than trusting cross-paper numbers.

   **Two actionable consequences for your project:** (a) this is a strong citable justification for giving your Siamese ResNet-18 baseline a properly tuned recipe rather than a strawman; (b) a naively-trained BIT/ChangeFormer may *underperform* a well-tuned FC-Siam-Diff in your own experiments for recipe reasons, not architectural ones — anticipate this in your write-up so it does not read as a bug.

5. **Non-DNN SSIM/RGB baseline.** SSIM decomposes into luminance/contrast/structure (range -1 to 1); threshold `SSIM(I1,I2) < T` or `|I1-I2| > T`. Expect it to work reasonably on clean aligned pairs and fail badly on exactly your nuisance list — camera displacement lights up every edge, lighting/grading shifts the whole map globally, blur/codec artifacts add high-frequency noise (SSIM is moderately robust to uniform blur since it is a perceptual-quality metric, but not immune). Cheap to implement in an afternoon; excellent for the "why we need learning" narrative. Budget near-zero engineering time.

---

## What this team should actually do

1. **Non-DNN baseline (RGB/SSIM):** implement first, **4-6 person-hours**. Zero training cost, establishes the floor, doubles as a data sanity check.
2. **FC-Siam-Diff (mandatory):** use **TorchGeo `FCSiamDiff`** rather than hand-rolling the original. **15-20 person-hours** (~4h data pipeline, ~4h training loop + logging, ~4h hyperparameter pass since the paper gives no recipe, ~4h eval/writeup). Training itself: a few T4 hours.
3. **Siamese ResNet-18 (stronger baseline):** same wrapper with `encoder_name="resnet18", encoder_weights="imagenet"`; tap C2-C5, watch the BN-sharing and stride-32 pitfalls. **20-25 person-hours.** Per the Reality Check, invest real tuning time here — a well-tuned version may already rival BIT, and that is a legitimate result.
4. **BIT (stretch):** realistic on free compute. **25-35 person-hours**, mostly environment setup and adapting the dataloader; training is only a few T4 hours.
5. **ChangeFormer (only if time remains):** **35-45 person-hours** with multi-session checkpoint/resume. Consider ChangeFormer-b0 instead.
6. **Proposed method:** build fusion as **signed difference + product (+ optional concat)**, not `|diff|` — that is the citable novelty angle. **Prefer ResNet-18 over EfficientNet-B0 as the default encoder** unless you benchmark step-time on your actual GPU first (see section 3's depthwise caveat); ResNet-18's BN behaviour is also simpler to reason about given the Siamese BN issue. **40-60 person-hours** for architecture + alignment + confidence head + ablations. This is the bulk of the work and should get the largest allocation.
7. **Evaluation discipline:** commit to **aggregate F1** up front, replicate the 256x256 non-overlapping crop convention, and cite open-cd Table 3 as your comparison table.

**Total: ~150-200 person-hours** across a 4-5 person team. Front-load the non-DNN and FC-Siam-Diff baselines to de-risk the data pipeline early; treat ChangeFormer as genuinely optional; protect the majority of hours for the proposed architecture.

---

## Verified sources

[Daudt et al. ICIP 2018 (arXiv:1810.08462)](https://arxiv.org/abs/1810.08462) · [rcdaudt/fully_convolutional_change_detection](https://github.com/rcdaudt/fully_convolutional_change_detection) · [Bobholamovic/FCN-CD-PyTorch](https://github.com/Bobholamovic/FCN-CD-PyTorch) · [TorchGeo fcsiam source](https://raw.githubusercontent.com/microsoft/torchgeo/main/torchgeo/models/fcsiam.py) · [likyoo/change_detection.pytorch](https://github.com/likyoo/change_detection.pytorch) · [likyoo/open-cd](https://github.com/likyoo/open-cd) · [Open-CD paper (arXiv:2407.15317)](https://arxiv.org/abs/2407.15317) · [BIT_CD](https://github.com/justchenhao/BIT_CD) · [BIT paper (arXiv:2103.00208)](https://arxiv.org/abs/2103.00208) · [ChangeFormer](https://github.com/wgcban/ChangeFormer) · [ChangeFormer paper (arXiv:2201.01293)](https://arxiv.org/abs/2201.01293) · [TinyCD (arXiv:2207.13159)](https://arxiv.org/abs/2207.13159) · [SNUNet, IEEE GRSL 2021](https://ieeexplore.ieee.org/document/9355573) · [ChangeStar](https://github.com/Z-Zheng/ChangeStar) · [SYSU-CD](https://github.com/liumency/SYSU-CD) · [JL1-CD (arXiv:2502.13407)](https://arxiv.org/abs/2502.13407) · [A Change Detection Reality Check (arXiv:2402.06994)](https://arxiv.org/abs/2402.06994) · [Spot the Difference (arXiv:1801.01051)](https://arxiv.org/abs/1801.01051) · [Graph matching object-level CD, MDPI RS 2022](https://www.mdpi.com/2072-4292/14/17/4225) · [timm EfficientNet docs](https://huggingface.co/docs/timm/models/efficientnet) · [SMP timm-efficientnet encoders](https://smp.readthedocs.io/en/latest/encoders_timm.html) · [Graphcore: How we made EfficientNet more efficient](https://www.graphcore.ai/posts/how-we-made-efficientnet-more-efficient)
