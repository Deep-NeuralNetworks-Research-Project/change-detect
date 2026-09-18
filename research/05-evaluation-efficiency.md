# Research Brief: Evaluation, Robustness, and Reproducibility Protocol for Pair-Order Consistent, Uncertainty-Aware Change Detection

**Prepared for:** University of Moratuwa CSE semester project — Siamese CNN change detection from paired RGB frames
**Scope:** Boundary metrics, CD-specific metric pitfalls, counting-based robustness metrics, nuisance corruption suite, efficiency measurement, few-seed statistics, experiment tracking on free compute.

---

## 1. Boundary Quality Metrics

Region-overlap metrics (IoU, F1, Dice) are dominated by interior pixels and are almost insensitive to boundary errors on large objects — a 2px boundary shift on a 500px-diameter blob barely moves IoU, while the same shift on a 20px blob devastates it. This is exactly backwards for a change-detection task with a mix of large façade changes and thin/small nuisance-adjacent changes (a removed pole, a new sign). Use boundary-specific metrics as secondary/diagnostic metrics alongside region IoU/F1.

### 1.1 Boundary IoU (Cheng, Girshick, Dollár, Berg, Kirillov — CVPR 2021)

**Definition.** For ground-truth mask `S` and prediction mask `S_d`... formally: extract a boundary band of pixel-width `d` from each mask (`G_d`, `P_d`), then compute IoU restricted to those bands:

```
Boundary_IoU(G, P) = |(G_d ∩ G) ∩ (P_d ∩ P)| / |(G_d ∩ G) ∪ (P_d ∩ P)|
```

`d` is set relative to image size (not a fixed pixel count) so the metric behaves consistently across resolutions: `d = dilation_ratio × image_diagonal`, default `dilation_ratio = 0.02` (2%).

**Why it suits this project:** Unlike Trimap-IoU/BF-score (below), Boundary IoU is symmetric in prediction/GT and does not over-penalize small objects while still being far more boundary-sensitive than Mask IoU for large ones — the paper shows it is the only measure whose sensitivity to boundary noise is roughly scale-invariant. This is the right property when the same model must score both a large façade repaint and a small removed object at comparable fidelity.

**Reference implementation** (from `bowenc0221/boundary-iou-api`, works directly on your binary change masks, no COCO/instance wrapper needed):

```python
import cv2
import numpy as np

def mask_to_boundary(mask, dilation_ratio=0.02):
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = max(1, int(round(dilation_ratio * img_diag)))
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1:h+1, 1:w+1]
    return mask - mask_erode          # boundary band

def boundary_iou(gt, dt, dilation_ratio=0.02):
    gt_b = mask_to_boundary(gt.astype(np.uint8), dilation_ratio)
    dt_b = mask_to_boundary(dt.astype(np.uint8), dilation_ratio)
    intersection = ((gt_b * dt_b) > 0).sum()
    union = ((gt_b + dt_b) > 0).sum()
    return intersection / union if union else 1.0
```
`pip install boundary-iou-api` if you want the packaged instance/panoptic wrappers too, but for binary change masks the above ~15 lines is the whole dependency.

### 1.2 Boundary F1 / BF-score (Csurka et al., BMVC 2013; MATLAB `bfscore`)

Match GT boundary points to predicted boundary points within a distance tolerance `θ` (pixels, e.g. 2–5px):
```
P = TP_b / (TP_b + FP_b)     # fraction of predicted boundary pixels within θ of a GT boundary pixel
R = TP_b / (TP_b + FN_b)     # fraction of GT boundary pixels within θ of a predicted boundary pixel
BF = 2·P·R / (P + R)
```
This is the same construction as the DAVIS video-segmentation `F-measure` (`fperazzi/davis`, `f_boundary.py`) and MATLAB's `bfscore`. **Caveat vs Boundary IoU:** BF is threshold-`θ`-sensitive and, per Csurka's own analysis and the Boundary IoU paper's comparison, biases toward large/simple contours — treat it as a secondary sanity metric, report Boundary IoU as primary.

### 1.3 Trimap accuracy (Kohli et al. 2009 style; "Trimap-IoU")

Compute plain pixel accuracy or IoU **only inside a band of width `w`** around the GT boundary (dilate/erode GT boundary by `w`), for several `w ∈ {3, 5, 10, 20}` px, and plot accuracy vs. `w`. This is the same construction the Cityscapes benchmark historically used. It is intuitive but width-dependent and, per the Boundary IoU paper, systematically favors large objects (a fixed-pixel band is a much smaller fraction of a big object's contour) — use it as a supplementary curve, not the headline number.

### 1.4 Hausdorff Distance / HD95 / Average Surface Distance

```
H(A,B) = max( max_{a∈∂A} min_{b∈∂B} d(a,b),  max_{b∈∂B} min_{a∈∂A} d(b,a) )   # directed-max, symmetrized by outer max
HD95   = 95th percentile of the pooled directed distances (both directions), instead of the max — robust to single-pixel outlier/annotation noise
ASD    = ( Σ_{a∈∂A} d(a,∂B) + Σ_{b∈∂B} d(b,∂A) ) / (|∂A| + |∂B|)              # average symmetric surface distance
```
These require extracting boundary contours (not filled masks) and computing a distance transform — degenerate/undefined when one mask is empty (common for `k` small or `no-change`-only image tiles), so guard with `if pred.sum()==0 and gt.sum()==0: skip` and `if exactly one is empty: return max_distance / image_diagonal` (do **not** silently drop these tiles — MONAI issue #2179 documents `inf` results from exactly this edge case).

**Implementations:**
- **MONAI** (`pip install monai`) — richest, medical-imaging-grade, batched, GPU-friendly:
```python
from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric
hd95 = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
asd  = SurfaceDistanceMetric(include_background=False, symmetric=True, reduction="mean")
hd95(y_pred=pred_onehot, y=gt_onehot)   # [B, C, H, W] one-hot binarized
asd(y_pred=pred_onehot, y=gt_onehot)
hd95.aggregate(); asd.aggregate()
```
- **seg-metrics** (`pip install seg-metrics`, Jia et al. 2024) — simplest API, CSV export, but built for volumetric NIfTI files (`.nii`/`.mha`), so for 2D PNGs wrap each mask as a `(1,H,W)` array; supports `dice`, `hd`, `hd95`, `msd` (mean surface distance) directly by name in `write_metrics(labels=[1], gdth_path=..., pred_path=..., metrics=['dice','hd95','msd'])`.
- **torchmetrics**: as of the current stable release, `torchmetrics.segmentation` ships `MeanIoU` and `GeneralizedDiceScore` but **no native Boundary IoU / HD95** — do not rely on it for boundary metrics; use MONAI or the boundary-iou-api snippet above instead.

**Recommendation for this project:** primary boundary metric = **Boundary IoU** (scale-balanced, cheap, no distance transform); secondary = **HD95 via MONAI** (interpretable in pixels, standard in the segmentation literature, good for the report's "boundary quality" bullet); skip full max-Hausdorff (too outlier-sensitive) and treat Trimap/BF as optional appendix curves.

---

## 2. Metric Definition Pitfalls in Change Detection

### 2.1 Global/pooled confusion matrix vs. per-image averaged F1 — this is the single biggest silent number-mover

Two fundamentally different protocols exist in the CD literature and they are **not interchangeable**:

**(A) Global/pooled ("micro") — the de-facto standard for LEVIR-CD/SYSU-CD/WHU-CD leaderboards.** Accumulate a single running `TP, FP, FN, TN` across *every pixel of every test image* (this is what tools like `BIT_CD`'s `ConfuseMatrixMeter` and the `open-cd` toolbox do), then compute P/R/F1/IoU **once** at the end:
```
P = ΣTP / (ΣTP + ΣFP)     R = ΣTP / (ΣTP + ΣFN)     F1 = 2PR/(P+R)     IoU = ΣTP / (ΣTP + ΣFP + ΣFN)
```
This is what "STANet: 83.81 P / 91.00 R / 87.26 F1 on LEVIR-CD" style leaderboard numbers mean. It implicitly weights each test image by how many changed pixels it contains — an image with a huge building complex dominates the sum, a nearly-empty-change tile contributes almost nothing.

**(B) Per-image macro (mF1).** Compute P/R/F1 independently *per image*, then average:
```
mF1 = (1/N) Σ_{n=1}^{N} 2·P_n·R_n / (P_n + R_n)
```
Every test pair counts equally regardless of its changed-pixel area. This is closer to what some competition leaderboards enforce, and is the protocol MMSegmentation-style tooling falls back to when asked to align with contest rules. **Undefined-denominator handling matters here**: for a genuine no-change pair with a perfect no-FP prediction, `TP=FP=FN=0` → `P,R,F1` undefined (0/0); the convention must be stated (commonly: define `F1=1` for a correct all-negative image, exclude it, or fold it into the false-alert-rate metric in §3 instead of the F1 average — do **not** silently `nan`-drop it, which inflates the mean).

Corley et al., *"A Change Detection Reality Check"* (2024, arXiv:2402.06994) is directly on point here: it shows reported SOTA gains in the CD literature shrink or vanish once evaluation protocol (among other things, exactly this per-image-vs-global choice) is held fixed across methods, and argues that a plain U-Net baseline is competitive with much of the "SOTA" once compared like-for-like. Cite this paper explicitly in your methodology section as justification for stating your protocol.

**Recommended explicit protocol for this project (state this verbatim in the report):**
1. Report the **global/pooled** changed-class P/R/F1/IoU as the primary number (comparable to LEVIR-CD/SYSU-CD leaderboards).
2. Report **per-image mF1** as a secondary number, with the zero-denominator convention stated (recommend: exclude true-negative-only images from the mF1 average and report them instead via the image-level false-alert rate in §3.3 — this avoids conflating "no false alarm" with "F1 of 1").
3. State the exact **changed-pixel ratio** `π = (Σ changed pixels)/(Σ total pixels)` of the test set, and separately for each nuisance stratum (§4) and each change-size bin, since π directly determines how meaningless Overall Accuracy is (below) and lets a reader sanity-check your F1 against class balance.
4. Never report a single "IoU" number computed with a different pooling convention than the "F1" number next to it — they are related by a closed form only when computed identically (see 2.2), so a mismatch is a protocol bug, not a finding.

### 2.2 IoU–F1 relationship

For the same TP/FP/FN counts (same pooling method), IoU and F1 for the changed class are a strict monotone bijection, not independent evidence:
```
F1 = 2·IoU / (1 + IoU)          IoU = F1 / (2 − F1)
```
Reporting both is customary (comparability with different papers that report one or the other) but do not present them as two independent pieces of evidence — if you want a second axis of information, use boundary IoU (§1) or object-level F1 (§3.4) instead, which are *not* algebraically determined by pixel F1.

### 2.3 Why Overall Accuracy is meaningless at 2–5% changed pixels

```
OA = (TP + TN) / (TP + TN + FP + FN)
```
At a changed-pixel ratio `π`, the trivial always-predict-no-change classifier scores `OA ≈ 1 − π`, i.e. **95–98% "accuracy" for doing nothing**. Never lead with OA; if reported at all, report it only alongside `π` and only as a redundancy check, never as an ablation-table headline metric. Cohen's Kappa (`κ = (OA − p_e)/(1 − p_e)`, `p_e` = chance agreement under the observed marginals) at least corrects for this and is used in some CD papers as a secondary metric (and Separated Kappa/SeK for multi-class semantic CD) — worth one column in an appendix table but changed-class F1/IoU remain primary.

---

## 3. Counting-Based Robustness Metrics

The promised deliverables (false-positive *pixels and connected regions* per no-change pair, image-level false-alert rate) require moving from per-pixel confusion counts to **connected-component** analysis.

### 3.1 Connected-component extraction

```python
import cv2, numpy as np
from scipy import ndimage as ndi

# --- Option A: scipy (4- or 8-connectivity via structure) ---
structure = np.ones((3, 3), dtype=int)          # 8-connectivity; np.array([[0,1,0],[1,1,1],[0,1,0]]) for 4-conn
labeled, n_components = ndi.label(pred_mask, structure=structure)
sizes = ndi.sum(pred_mask, labeled, range(1, n_components + 1))   # area per component

# --- Option B: OpenCV (gives centroids/bboxes for free, useful for qualitative figures) ---
n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    pred_mask.astype(np.uint8), connectivity=8)
areas = stats[1:, cv2.CC_STAT_AREA]              # skip background label 0
```
Prefer OpenCV when you also need bounding boxes/centroids for figures; prefer scipy when you need custom connectivity or are already in a `numpy`/scientific-stack pipeline. Both agree on component counts for the same connectivity choice — **state which connectivity (4 vs 8) you used**, since 8-connectivity merges diagonally-touching noise specks into fewer, larger blobs and materially changes both the FP-region count and minimum-size filtering behavior.

### 3.2 Minimum-region-size filtering

Small isolated FP specks from dithering/JPEG blockiness are a nuisance artifact of the *evaluation*, not necessarily the model — but the *choice of `k`* is itself a robustness statement, so report results **at multiple `k`** rather than picking one silently:
```python
min_size = k                      # e.g. k ∈ {1, 5, 20, 50} pixels — report as a small table/curve, not one number
filtered = np.isin(labeled, np.where(sizes >= min_size)[0] + 1)
```
Report raw (k=1, no filtering) and filtered (your chosen operating k) side by side — the *drop* between them is itself informative about whether your model's errors are "salt-and-pepper" (drops sharply) or "true structured false alarms" (barely changes).

### 3.3 Image-level false-alert rate

For the no-change-pair subset `N_neg`:
```
FalseAlertRate = |{ pairs in N_neg : max_region_area(pred) ≥ k }| / |N_neg|
```
i.e. a no-change pair counts as "alerted" iff *any* connected FP region exceeds the size threshold `k` — this is the natural complement to per-pixel FPR and is what an end user (someone reviewing alerts) actually experiences: one false alarm per pair, not "3.2% of pixels were wrong." Report this **stratified by nuisance condition** (§4) and as a function of `k` (a "false-alert rate vs. minimum-region-size" curve is a strong, cheap robustness figure).

### 3.4 Object/instance-level (matched-component) evaluation

Standard COCO-style greedy matching applied to connected components instead of pixels:
1. Extract GT components `{g_1..g_M}` and predicted components `{p_1..p_N}` (§3.1).
2. Compute pairwise IoU between every `(g_i, p_j)`.
3. Greedily match pairs with `IoU ≥ τ` (typical `τ = 0.5`), highest-IoU first, one-to-one.
4. `TP` = matched pairs, `FP` = unmatched predicted components, `FN` = unmatched GT components.
```
F1_obj = 2·TP / (2·TP + FP + FN)     at fixed IoU threshold τ (report τ=0.5 primary, 0.25/0.75 as sensitivity check)
```
**Citations for this exact style in change detection:** LoDA (arXiv:2608.05356, 2026) formalizes object-level CD evaluation for multi-class semantic change (per-class `IoU_y = TP_y/(TP_y+FP_y+FN_y)`, `F1_y` analogously, macro-averaged across change types, `ACC = ΣTP_y / N`); it notes most prior 3D/2D CD papers (Urb3DCD, DC3DCD) stick to **point/pixel-wise** protocols and that instance-level CD evaluation is comparatively rare — de Gélis et al. (2023) is cited as one of the few doing object-proxy-based evaluation. **Practical takeaway:** object-level F1 is a legitimate, citable, but non-standard addition — frame it in your report as "we additionally report object-level F1 at IoU≥0.5 following the object-centric evaluation philosophy of Cheng et al. 2021 and LoDA 2026, since pixel metrics can hide whether a model detects *distinct* change instances or bleeds one blob."

---

## 4. Nuisance-Robustness Benchmarking

### 4.1 Core design principle (must be explicit in the methodology)

**Apply every corruption to exactly one frame of the pair (never both).** This is what makes the test diagnostic of nuisance-robustness rather than generic image-quality robustness: if both frames are corrupted identically, a well-designed Siamese/differencing model can partially cancel the corruption (it appears in both branches and difference/attention mechanisms can suppress common-mode noise), so the test would under-report the failure mode you actually care about — a change detector that fires because *one* camera pass had different auto-exposure, motion blur, or compression than the other. Concretely:

```python
def apply_nuisance(img1, img2, corruption_fn, which="t2", **kwargs):
    if which == "t2":
        return img1, corruption_fn(img2, **kwargs)     # T1 clean, T2 corrupted
    elif which == "t1":
        return corruption_fn(img1, **kwargs), img2      # T1 corrupted, T2 clean
    # never corrupt both — that tests general robustness, not nuisance suppression
```
Run both directions (`t1`, `t2`) if the model is not perfectly symmetric (relevant to your pair-order-consistency objective — asymmetric corruption sensitivity between "corrupt earlier frame" vs. "corrupt later frame" is itself a robustness finding worth a table row) but keep the *pairing* asymmetric per corrupted sample.

Apply the suite to **both** no-change pairs (measures §3 false-alert metrics under nuisance) **and** change pairs (measures whether genuine-change F1/boundary quality degrades under nuisance) — these are two different rows in your results, both promised in the brief.

### 4.2 Corruption suite specification table

| Nuisance | Corruption(s) | Library call | Severity 1 → 5 |
|---|---|---|---|
| Brightness / exposure | `brightness` (ImageNet-C) | `imagecorruptions.corrupt(img, corruption_name='brightness', severity=s)` | ImageNet-C default schedule (additive HSV-V shift ≈ 0.1→0.5); *for domain realism*, prefer explicit gamma control below |
| Brightness / exposure (parametrized) | Gamma correction | `out = 255*(img/255.0)**(1/gamma)`; or `cv2.LUT(img, table)` with `table[i]=((i/255)**(1/gamma))*255`; Albumentations: `A.RandomGamma(gamma_limit=(g_lo,g_hi))` | γ ∈ {0.85–1.18, 0.7–1.4, 0.55–1.8, 0.4–2.2, 0.25–3.0} (symmetric under/over-expose ranges, widening with severity) |
| Colour grading / white-balance drift | ASC-CDL (Slope/Offset/Power + Saturation) | `out = clip((slope*in + offset), 0, 1) ** power`, then blend toward luma for saturation: `out = luma + sat*(out-luma)`; apply per-channel (R,G,B) | s∈{±3%,±6%,±10%,±15%,±25%} slope/offset jitter, power∈{0.9–1.1 → 0.6–1.6} |
| Contrast | `contrast` (ImageNet-C) or `A.RandomBrightnessContrast(contrast_limit=...)` | `imagecorruptions.corrupt(img,'contrast',s)` | package default 5-level schedule |
| Gaussian blur (defocus proxy) | `gaussian_blur` / `defocus_blur` | `imagecorruptions.corrupt(img,'gaussian_blur',s)` or `cv2.GaussianBlur(img,(0,0),sigmaX=σ)` | σ ∈ {0.5, 1, 2, 3, 4} px |
| Motion blur | `motion_blur` (ImageNet-C) or custom kernel | `imagecorruptions.corrupt(img,'motion_blur',s)`; custom: build a linear kernel of length `L` at angle `θ`, `cv2.filter2D(img, -1, kernel)` | `L` ∈ {3, 7, 15, 25, 35} px, random `θ`∈[0°,180°) per sample |
| JPEG compression | `jpeg_compression` (ImageNet-C) or direct | `imagecorruptions.corrupt(img,'jpeg_compression',s)`; direct: `cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])` → decode back | q ∈ {90, 70, 50, 30, 10} |
| H.264/HEVC video compression (real codec artifacts on a still) | ffmpeg round-trip (see §4.3) | shell out via `subprocess` | CRF ∈ {18, 23, 28, 35, 45} (H.264/libx264) and ∈ {20, 25, 30, 37, 47} (HEVC/libx265, roughly perceptually matched CRF offsets) |
| Synthetic shadows | `A.RandomShadow(shadow_roi=..., num_shadows_limit=(1,3))` or manual polygon-darkening (`cv2.fillPoly` mask + multiplicative darken) | `A.RandomShadow(num_shadows_limit=(1,2), shadow_dimension=5, shadow_intensity_range=(0.3,0.7))(image=img2)['image']` | shadow-intensity fraction ∈ {0.1, 0.2, 0.35, 0.5, 0.7}, area fraction growing with severity |
| Viewpoint jitter (camera displacement) | Small random homography / affine | `A.Affine(translate_percent=..., rotate=..., scale=..., shear=...)`; or manual: `cv2.getPerspectiveTransform` on 4 jittered corner points → `cv2.warpPerspective` | translate ∈ {1%,2%,4%,8%,16%} of W/H; rotate ∈ {0.5°,1°,2°,4°,8°}; corner jitter (perspective) ∈ {0.5%,1%,2%,4%,8%} of W/H |
| Occlusion | Cutout / paste patch | `A.CoarseDropout(num_holes_range=(1,3), hole_height_range=..., hole_width_range=..., fill=0/"random")`; or paste an unrelated crop for "paste" variant | occluded area fraction ∈ {2%, 5%, 10%, 20%, 35%} of frame |
| (Optional) Sensor/compression noise | `gaussian_noise`, `shot_noise`, `speckle_noise` | `imagecorruptions.corrupt(img, name, s)` | package default 5-level schedule |

**Albumentations vs `imagecorruptions`:** use `imagecorruptions` (`pip install imagecorruptions`, exposes `corrupt(image, corruption_name, severity)` and `get_corruption_names(subset='common'|'validation'|'all'|'noise'|'blur'|'weather'|'digital')`) when you want the literature-standard ImageNet-C severity schedule for direct comparability with robustness papers; use Albumentations (`A.RandomShadow`, `A.CoarseDropout`, `A.Affine/RandomPerspective`, `A.RandomGamma`) for the corruptions ImageNet-C doesn't cover (shadow, occlusion, viewpoint) and for GPU-friendly pipeline integration. Keep the two libraries' severity scales **independently calibrated** to your own 1–5 table above rather than assuming they line up with each other.

### 4.3 Real codec artifacts via ffmpeg round-trip (still image → compressed video → frame)

A single still run through a lossy image codec (JPEG) does not reproduce H.264/HEVC's block-motion-compensation and deblocking-filter artifacts. To get *real* video-codec artifacts on a still frame:
```bash
# Encode the still as a short static clip so the encoder produces representative I/P-frame structure,
# then extract the middle frame back out.
ffmpeg -loop 1 -i frame.png -t 1 -r 10 -c:v libx264 -crf 28 -pix_fmt yuv420p -y clip.mp4
ffmpeg -i clip.mp4 -vf "select=eq(n\,5)" -vframes 1 -y frame_compressed.png

# HEVC variant:
ffmpeg -loop 1 -i frame.png -t 1 -r 10 -c:v libx265 -crf 30 -pix_fmt yuv420p -y clip.mp4
```
Looping a single frame for ~10 encoded frames (rather than a 1-frame video) matters because a lone I-frame under-represents typical inter-frame codec behaviour; sampling a mid-sequence frame captures more realistic quantization/deblocking. Wrap this in a Python `subprocess.run([...], check=True)` call and cache outputs (ffmpeg round-trips are slow — precompute the corrupted image set once, don't do it inside the training/eval loop).

### 4.4 Severity-level design guidance

- 5 severities per corruption, monotonically increasing degradation, calibrated so severity 1 is "barely perceptible" and severity 5 is "clearly degraded but not pathological" (a human should still be able to tell the frames are of the same scene) — mirror the ImageNet-C philosophy but re-tune numeric parameters for your domain (RGB scene pairs, not ImageNet photos); do **not** blindly reuse ImageNet-C's pixel-noise-heavy schedule as-is, since your nuisance list (camera displacement, colour grading, compression, shadows) is dominated by *geometric and photometric* corruptions ImageNet-C barely covers.
- Keep corruption parameter ranges independent across corruption types (don't compound two corruptions in the main sweep — reserve compounding for a small "combined worst case" ablation) so each row of your results table isolates one nuisance axis, matching the brief's "stratified by nuisance condition" requirement.

### 4.5 Reporting relative robustness drop

Two options, pick one as primary and state it:

**(a) F1-retention ratio (simplest, no external baseline needed, recommended primary):**
```
Retention(c, s) = F1(model, corruption=c, severity=s) / F1(model, clean)
```
Average over severities per corruption, then over corruptions, for a single "mean retention" scalar per model; also keep the full per-corruption-per-severity table/heatmap — this is the actionable artifact for your robustness section.

**(b) mCE-style normalization (comparable across models the way ImageNet-C intends, needs a reference model):**
```
CE(c) = Σ_s (1 − F1(model, c, s)) / Σ_s (1 − F1(reference, c, s))
mCE   = (1/|C|) Σ_c CE(c)
RelCE(c) = Σ_s [(1−F1(model,c,s)) − (1−F1(model,clean))] / Σ_s [(1−F1(ref,c,s)) − (1−F1(ref,clean))]
```
Use your **RGB/SSIM baseline** (the weakest promised baseline) as the "AlexNet-equivalent" reference — this directly answers "does the proposed EfficientNet+fusion model degrade *less* than the naive baseline under nuisance," which is a stronger and more interesting claim than raw retention numbers alone, and is the paper-quality way to present §4's results. Report both if time allows: (a) for intuitive per-model reading, (b) for the headline robustness comparison across your baselines/proposed model.

---

## 5. Efficiency Measurement

### 5.1 Parameter counting

```python
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```
Report both, in millions, **per branch structure clearly stated** (e.g. "5.3M total, of which the two Siamese EfficientNet-B0 encoders share weights — report the *shared* encoder param count once, not doubled, plus the fusion/head params separately" — a common reporting error in Siamese-network papers is double-counting shared-weight encoder parameters).

### 5.2 FLOPs/MACs for a two-input (paired) model — and why the tools disagree

**The core confusion:** a fused multiply-accumulate (`a += b*c`) is *one hardware operation* but is counted as either **1** ("1 MAC") or **2** ("1 mul + 1 add = 2 FLOPs") depending on convention. `fvcore` counts fused multiply-adds as 1 (i.e. its "flops" number is numerically a **MACs** count); `ptflops` (current versions) explicitly reports **MACs** and labels them as such; `thop` historically reports MACs too but many downstream users misreport its output as "FLOPs" without the ×2; PyTorch's native `torch.utils.flop_counter.FlopCounterMode` counts mul and add separately, i.e. **true FLOPs = 2× MACs**. **Always state which convention you used and report both MACs and the derived FLOPs (=2×MACs) explicitly in a footnote** — this single sentence prevents your efficiency table from being silently wrong by 2× relative to a paper you're comparing against.

Two-input model, three tool options:

```python
# --- fvcore (reports MAC-convention "flops") ---
from fvcore.nn import FlopCountAnalysis
flops = FlopCountAnalysis(model, (t1_tensor, t2_tensor))
macs = flops.total()                      # this is MACs despite the API name
true_flops = 2 * macs

# --- ptflops (two inputs via input_constructor) ---
from ptflops import get_model_complexity_info
def input_constructor(res):
    return {"t1": torch.randn(1, 3, *res).to(device),
            "t2": torch.randn(1, 3, *res).to(device)}
macs, params = get_model_complexity_info(
    model, (256, 256), input_constructor=input_constructor,
    as_strings=False, print_per_layer_stat=False)

# --- torch.utils.flop_counter (native PyTorch 2.x, true FLOPs incl. backward if needed) ---
from torch.utils.flop_counter import FlopCounterMode
fcm = FlopCounterMode(display=False)
with fcm:
    model(t1_tensor, t2_tensor)
true_flops = fcm.get_total_flops()
```
`thop` is simplest for a quick sanity number (`from thop import profile; macs, params = profile(model, inputs=(t1, t2))`) but is the least actively maintained of the four — use it only as a cross-check, not primary.

**Recipe:** run at least two of the four tools on the *same* input resolution and batch size, confirm they agree to within the expected 2× (or ~5% at equal convention, per known small fvcore/native discrepancies on some op types), and report the MACs number with an explicit "(≈ X GFLOPs at 2×MACs convention)" annotation. Report FLOPs **per pair** (both frames through both encoder passes, plus fusion/head), not per single image — this is the actual promised "pair-input FLOPs" deliverable and is frequently mis-reported as half its true value by evaluating only one branch.

### 5.3 Peak GPU memory

```python
torch.cuda.reset_peak_memory_stats(device)
# ... run one representative forward(+backward) pass at your chosen batch size ...
allocated = torch.cuda.max_memory_allocated(device) / 1024**2   # MB — tensors PyTorch actually placed
reserved  = torch.cuda.max_memory_reserved(device) / 1024**2    # MB — allocator's cached pool, closer to what nvidia-smi shows
```
`max_memory_allocated` undercounts real GPU usage (fragmentation, caching allocator overhead); `max_memory_reserved` is closer to `nvidia-smi` but still excludes the CUDA context itself (typically 300–800MB fixed overhead per process, driver/CUDA-version dependent). For the number that matches what a Colab/Kaggle user would actually see as "will this fit," cross-check with `nvidia-smi --query-gpu=memory.used --format=csv` (via `subprocess`) or `pynvml.nvmlDeviceGetMemoryInfo` sampled during the run — **report reserved (or nvidia-smi) as the headline "peak GPU memory," with allocated as a secondary tensor-only figure**, and state the fixed batch size and input resolution used, since memory scales roughly linearly with batch size and quadratically-ish with resolution for conv nets.

### 5.4 Latency benchmarking recipe (reusable)

```python
import torch, time, numpy as np

@torch.no_grad()
def benchmark_latency(model, t1, t2, warmup=30, iters=200, use_amp=False):
    model.eval()
    if use_amp:
        ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        from contextlib import nullcontext; ctx = nullcontext()

    with ctx:
        for _ in range(warmup):              # NOT timed — lets cuDNN autotune, clocks ramp up
            model(t1, t2)
        torch.cuda.synchronize()

        times_ms = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            model(t1, t2)
            end.record()
            torch.cuda.synchronize()          # must sync before reading event time
            times_ms.append(start.elapsed_time(end))

    times_ms = np.array(times_ms)
    median = np.median(times_ms)
    q25, q75 = np.percentile(times_ms, [25, 75])
    return {"median_ms": median, "iqr_ms": (q25, q75), "raw": times_ms}
```
Non-negotiable disclosure list for every latency number reported (Colab/Kaggle GPU assignment is random per session — this is exactly the "hardware must be logged" requirement in the brief):
- `torch.cuda.get_device_name(0)` + driver/CUDA version (`nvidia-smi` header) — **log this every run**, not just once, since Colab/Kaggle silently hand out T4/P100/V100/A100/L4 depending on availability and this alone can shift latency 2–5×.
- Fixed batch size (report at pair-batch=1 for "single-pair latency" and separately at a realistic batch e.g. 8/16 for "throughput").
- AMP on/off (`torch.autocast`) and TF32 on/off (`torch.backends.cuda.matmul.allow_tf32`, `torch.backends.cudnn.allow_tf32`) — both silently change numeric precision and speed by large margins; state both explicitly next to every latency table.
- Report **median + IQR** (25th/75th percentile), not mean±std — GPU latency distributions are right-skewed (occasional thermal/scheduler stalls), so mean±std overstates spread and median is the standard robust choice recommended in benchmarking literature (also report p95 if you want a tail-latency figure).
- ≥30 warmup iterations excluded from timing, ≥100–200 measured iterations retained.

---

## 6. Statistical Reporting with 3+ Seeds

- **What to report:** mean ± std *and* median + [min, max] range side by side. With only n=3, the sample standard deviation has just 2 degrees of freedom and is itself extremely noisy (its own relative standard error is roughly `1/√(2(n−1))` ≈ 50% at n=3) — std is still worth reporting as the community-standard summary, but median+range communicates the actual spread without implying a stable Gaussian estimate, and the raw 3 numbers should simply be listed in an appendix table (at n=3, "show the data" beats "summarize the data").
- **Why significance tests are inappropriate at n=3:** a two-sample t-test with `df=2` (per group) has enormous critical values and essentially no power to detect anything but a huge effect; the deep-RL literature's "How Many Random Seeds?" analysis (Colas et al., arXiv:1806.08295) and related "Deep RL that Matters"-style work make the general point that properly powered seed-based significance testing typically needs on the order of 10–20+ seeds for realistic effect sizes — n=3 is far below the threshold where a t-test's assumptions (or even a non-parametric alternative like Mann-Whitney, which needs comparable sample sizes to have any power) are meaningful. **Do not report p-values from 3-seed comparisons; do not claim "statistically significant improvement" from n=3.** State plainly in the report: "with 3 seeds per configuration, formal significance testing is underpowered; we instead report [bootstrap CIs / effect-size-relative-to-spread] as the more honest alternative."
- **Bootstrap confidence intervals over the *test set*, not over seeds — the better alternative:** for a single trained model (one seed), resample the test set (images/pairs) **with replacement**, `B = 1000–10000` times, recompute the metric on each resample, take the 2.5th/97.5th percentiles as a 95% CI:
```python
import numpy as np
def bootstrap_ci(per_image_scores, B=5000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = len(per_image_scores)
    boot_means = np.array([
        rng.choice(per_image_scores, size=n, replace=True).mean()
        for _ in range(B)
    ])
    lo, hi = np.percentile(boot_means, [100*alpha/2, 100*(1-alpha/2)])
    return per_image_scores.mean(), (lo, hi)
```
This captures **test-set sampling uncertainty** — a genuinely different, complementary source of variance to **seed-to-seed (training stochasticity) variance**. Present both, not conflated: (1) a bootstrap CI per seed (or pooled across seeds' per-image scores) answering "how precisely do we know this model's true performance given a finite test set," and (2) the 3-seed spread answering "how much does training randomness alone move the number." A model whose seed-spread is *smaller* than its own bootstrap CI width is one where seed variance is not the limiting uncertainty — worth stating explicitly, it's a stronger and more defensible claim than an unsupported significance test.
- **Ablation tables, honestly:** report mean ± std (3+ seeds) for **every row**, including every baseline, using the exact same seed set `{0,1,2,...}` across all rows for comparability; bold/flag a row's improvement only when the mean±std bands are clearly separated (or, better, when a paired bootstrap on the *difference* between two models' per-image scores excludes 0) — never bold a sub-noise delta (e.g. +0.3 F1 points when std is ±1.5) as if it were a finding. State the exact seed values used, not just "n=3," for reproducibility.

---

## 7. Experiment Tracking and Reproducibility on Colab/Kaggle

### 7.1 Tool choice

| Tool | Free-tier limits (2026) | Fit for this project |
|---|---|---|
| **Weights & Biases (Personal/free)** | Unlimited experiments/tracked hours, **100GB** storage | **Recommended primary.** Zero-setup logging (`wandb.init`), automatic system metrics (GPU util/mem), built-in run-comparison dashboards, artifact versioning for checkpoints. Manage the 100GB cap by logging only `best.pt` per run (not every epoch) and pushing large raw datasets/full checkpoint sets to Kaggle Datasets or Drive instead. |
| **MLflow** | Free/open-source, self-hosted, no usage caps | Full data ownership, no vendor dependency — but on ephemeral Colab/Kaggle VMs the tracking server/UI has no persistent state across sessions unless you point its file/SQLite backend at a Drive-mounted directory, and sharing the UI outside the notebook needs a tunnel (ngrok, ssh) — more setup overhead for a 4-person team than it's worth for a semester project; use as a fallback if W&B account/storage becomes a blocker. |
| **TensorBoard + CSV** | Free, local | Lowest friction (`torch.utils.tensorboard.SummaryWriter`), but **TensorBoard.dev (the hosted sharing service) was shut down January 1, 2024** and no longer accepts uploads — do not plan around it. Viable pattern: write event logs to a Drive-mounted folder so they survive VM resets, view via `%load_ext tensorboard; %tensorboard --logdir <drive_path>` inside the notebook; also dump a parallel flat CSV of per-epoch metrics (trivial `csv.writer` append) as the durable, tool-independent source of truth for your report's tables/plots. |

**Recommendation:** W&B free tier as primary (best team-visibility-to-setup-cost ratio, generous limits for a single-semester project of this size), with a parallel plain-CSV log (belt-and-suspenders — CSVs never get deprecated, rate-limited, or require re-authentication) for every run's final metrics.

### 7.2 Checkpoint/artefact storage given ephemeral disk and Drive quota

- Colab/Kaggle local disk is wiped on every session reset — never treat it as durable storage for anything you can't afford to regenerate.
- Mount Drive (Colab) or use Kaggle's persistent `/kaggle/working` output (persists per-notebook-version, not per-session) as the checkpoint target; save a checkpoint every N epochs *and* immediately before any known session boundary.
- Avoid many small read/write ops directly against the Drive mount (Google enforces per-file operation-count/bandwidth quotas that trigger I/O errors under heavy small-file traffic) — copy datasets to local VM disk as a single `.zip`/`.tar.gz`, unarchive locally, and only write back *aggregated* checkpoint files (not per-batch logs) to Drive.
- Google Drive's trash retains deleted large checkpoint files for 30 days by default and still counts against quota — empty trash explicitly (`Drive UI → Trash → Empty`) as part of routine cleanup, or the team will hit the quota wall mid-semester despite "deleting" old checkpoints.
- Checkpoint-resume pattern (both platforms): save `{epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict, rng_states, best_metric}` every N epochs; on relaunch, check for an existing checkpoint file first and resume from it automatically rather than restarting — this should be a ~15-line utility written once and reused by every team member's training script.

### 7.3 Seeding and determinism — the speed cost

```python
def set_seed(seed, deterministic=False):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False       # forces deterministic algo selection — slower
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True          # lets cuDNN autotune fastest conv algo for fixed input size — faster but run-to-run algo choice can vary slightly
```
`cudnn.deterministic=True` forces bitwise-reproducible convolution algorithms, which are frequently **not** the fastest ones cuDNN would otherwise autotune — expect a real (sometimes 10–30%) throughput hit. Given the free-GPU-hours constraint, the pragmatic policy for this project is: **seed everything for initialization/data-order reproducibility (`set_seed(seed, deterministic=False)`), but do not force full cuDNN determinism** — run-to-run bitwise identity is not the actual requirement (statistical stability across your ≥3 seeds is), and the speed cost directly eats into your scarce Colab/Kaggle quota. State this choice explicitly in the reproducibility section of the report so a reader knows numbers may vary by a fraction of a point even at fixed seed.

### 7.4 Environment pinning

`pip freeze > requirements.txt` (or better, a minimal hand-curated `requirements.txt` with pinned major-version numbers for `torch`, `torchvision`, `timm`/EfficientNet source, `albumentations`, `opencv-python`, `fvcore`/`ptflops`, `monai`) committed alongside the code; log the exact Colab/Kaggle base-image CUDA/driver/Python version at the top of every notebook run (`!python --version`, `!nvidia-smi`, `torch.__version__`, `torch.version.cuda`) into the same experiment-tracking run, not just into scrollback that gets lost.

### 7.5 Session limits as of 2025–2026 (log these, they will vary run to run)

- **Colab free tier:** dynamic, unpublished weekly GPU allowance, informally reported around 15–30 GPU-hours/week; single-session cap ~12 hours max runtime, disconnects after ~90 min idle; GPU type (T4 typically, occasionally better/worse) assigned dynamically and can drop to CPU-only under load.
- **Kaggle free tier:** ~30 GPU-hours/week (≈20 TPU-hours/week), session cap ~9–12 hours for GPU/TPU sessions; GPU choice among P100 (1×16GB) or T4×2 (2×16GB) selectable per session.
- Given both are dynamic/unpublished and can change, **do not hardcode a quota number into the report as fact** — instead state the policy you followed ("we logged actual GPU-hours consumed per run via W&B system metrics and stayed within observed weekly caps of ~X hours") and let the checkpoint-resume pattern (7.2) absorb any mid-run disconnects.

---

## Reusable Efficiency Benchmarking Recipe (consolidated)

```python
def benchmark_model(model, t1, t2, device="cuda", batch_size=1, use_amp=False):
    model = model.to(device).eval()
    t1, t2 = t1.to(device), t2.to(device)

    # 1. Params
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 2. FLOPs/MACs (fvcore convention: numeric value = MACs)
    from fvcore.nn import FlopCountAnalysis
    macs = FlopCountAnalysis(model, (t1, t2)).total()

    # 3. Peak memory
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        model(t1, t2)
    peak_alloc_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    peak_reserv_mb = torch.cuda.max_memory_reserved(device) / 1024**2

    # 4. Latency (median + IQR, warm-up excluded, CUDA events)
    lat = benchmark_latency(model, t1, t2, use_amp=use_amp)   # see §5.4

    return {
        "params_total_M": total_p / 1e6, "params_trainable_M": train_p / 1e6,
        "MACs_G": macs / 1e9, "FLOPs_G_approx": 2 * macs / 1e9,
        "peak_mem_allocated_MB": peak_alloc_mb, "peak_mem_reserved_MB": peak_reserv_mb,
        "latency_median_ms": lat["median_ms"], "latency_iqr_ms": lat["iqr_ms"],
        "gpu": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda, "amp": use_amp,
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "batch_size": batch_size,
    }
```
Run this once per model (baselines + proposed, all seeds share the same architecture so one benchmark run per architecture suffices — efficiency numbers don't need ×3 seeds, only accuracy/robustness numbers do), and dump the returned dict straight into your experiment tracker as run metadata.

---

## Person-Hour Effort Estimates

Assumes a ~4-person team, semester-scope (~10–12 working weeks for the evaluation infrastructure specifically, run in parallel with model development). Hours are **total team person-hours** per task; suggest splitting metric/robustness-library tasks to 1–2 owners each so work parallelizes rather than serializes.

| # | Task | Deliverable | Person-hours | Notes |
|---|---|---|---|---|
| 1 | Core metric library (P/R/F1/IoU, both pooling protocols, π reporting) | `metrics.py` + unit tests on toy masks | 8–12 | Get the global-vs-per-image distinction (§2.1) right once, reuse everywhere |
| 2 | Boundary metrics (Boundary IoU, HD95/ASD via MONAI) | `boundary_metrics.py` | 6–10 | Boundary IoU snippet is small; MONAI wiring + empty-mask edge cases take the rest |
| 3 | Connected-component / object-level metrics (§3) | `region_metrics.py`, false-alert-rate + object-F1 | 8–12 | Include a min-size-vs-metric sweep utility, not just one k |
| 4 | Corruption suite implementation (§4 table) | `corruptions.py` covering all 10 nuisance types + ffmpeg wrapper | 16–24 | ffmpeg round-trip + homography jitter are the fiddliest; budget extra time for calibrating severity levels visually |
| 5 | Nuisance-stratified evaluation harness (apply suite to one frame, run inference, aggregate by condition/severity/change-size bin) | end-to-end robustness eval script + results tables/heatmaps | 12–18 | Depends on 1–4 being done first |
| 6 | Efficiency benchmarking harness (§5) | `benchmark.py` reusable across all models | 6–10 | Mostly integration/plumbing once tools are chosen |
| 7 | Calibration/reliability metrics (ECE, Brier, risk-coverage/AURC) | `calibration.py` | 6–10 | Standard formulas, low risk, but needs a well-defined per-pixel or per-region confidence score from the model first |
| 8 | Statistical reporting pipeline (bootstrap CI, ablation table formatting) | `stats.py` + ablation table generator | 6–8 | Bootstrap function is short; the discipline is in consistently applying it everywhere |
| 9 | Experiment tracking setup (W&B project, checkpoint-resume utility, seeding utility, requirements pinning) | working W&B project + `utils/reproducibility.py` | 6–10 | One-time setup, do this in week 1–2 not late |
| 10 | Running full evaluation across all baselines × ≥3 seeds × full nuisance suite × efficiency benchmark | populated results CSVs/W&B tables | 20–30 (mostly GPU-bound wall-clock, not active work) | Budget Colab/Kaggle quota (§7.5) carefully around this — largest wall-clock risk item in the whole plan |
| 11 | Report writing: turning tables into the promised figures (P/R/F1/IoU, boundary quality, robustness curves, reliability diagrams/risk-coverage curves, efficiency table) | final report sections + plots | 16–24 | Don't underbudget plotting/figure polish — 5 metric categories × multiple models × multiple conditions is a lot of figures |
| **Total** | | | **~110–160 person-hours** | Across 4 people over ~10–12 weeks ≈ 3–4 hrs/person/week on infrastructure alone, on top of model development time — front-load tasks 1, 4, 9 in weeks 1–3 since everything else depends on them |

---

### Sources

- [Boundary IoU: Improving Object-Centric Image Segmentation Evaluation (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.pdf) / [arXiv](https://arxiv.org/pdf/2103.16562) / [boundary-iou-api](https://github.com/bowenc0221/boundary-iou-api/blob/master/README.md) / [reference gist](https://gist.github.com/bowenc0221/71f7a02afee92646ca05efeeb14d687d)
- [What is a good evaluation measure for semantic segmentation? (Csurka et al., BMVC 2013)](https://www.bmva-archive.org.uk/bmvc/2013/Papers/paper0032/abstract0032.pdf)
- [bfscore — MATLAB](https://www.mathworks.com/help/images/ref/bfscore.html); [DAVIS f_boundary.py](https://github.com/fperazzi/davis/blob/main/python/lib/davis/measures/f_boundary.py)
- [MONAI metrics documentation](https://docs.monai.io/en/stable/metrics.html); [Hausdorff 'inf' discussion #2179](https://github.com/Project-MONAI/MONAI/discussions/2179)
- [seg-metrics: a Python package to compute segmentation metrics (2024)](https://www.medrxiv.org/content/10.1101/2024.02.22.24303215v1)
- [A Change Detection Reality Check (Corley et al., 2024)](https://arxiv.org/pdf/2402.06994)
- [LEVIR-CD dataset](https://justchenhao.github.io/LEVIR/)
- [Fully Convolutional Siamese Networks for Change Detection (Daudt et al., ICIP 2018)](https://rcdaudt.github.io/files/2018icip-fully-convolutional.pdf) / [code](https://github.com/rcdaudt/fully_convolutional_change_detection)
- [LoDA: Level of Detection Aware Method for Object Level Change Detection (2026)](https://arxiv.org/html/2608.05356)
- [OpenCV connectedComponentsWithStats guide](https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/)
- [imagecorruptions package](https://github.com/bethgelab/imagecorruptions)
- [Benchmarking Neural Network Robustness to Common Corruptions (Hendrycks & Dietterich, ImageNet-C)](https://arxiv.org/pdf/1903.12261)
- [ASC CDL — Wikipedia](https://en.wikipedia.org/wiki/ASC_CDL); [FFmpeg CRF explained](https://shotstack.io/learn/ffmpegcrf/)
- [Albumentations RandomShadow](https://explore.albumentations.ai/transform/RandomShadow); [CoarseDropout](https://albumentations.ai/docs/api-reference/albumentations/augmentations/dropout/coarse_dropout/)
- [fvcore flop_count docs](https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md); [ptflops](https://github.com/LukasHedegaard/ptflops); [torch.utils.flop_counter source](https://github.com/pytorch/pytorch/blob/main/torch/utils/flop_counter.py); [PyTorch forum: MACs vs FLOPs](https://discuss.pytorch.org/t/clarification-of-flops-macs-in-model-descriptions/198948)
- [PyTorch forum: max_memory_allocated vs nvidia-smi](https://discuss.pytorch.org/t/pytorchs-torch-cuda-max-memory-allocated-showing-different-results-from-nvidia-smi/165706); [reserved vs allocated issue #40989](https://github.com/pytorch/pytorch/issues/40989)
- [PyTorch Benchmark best practices — Lei Mao](https://leimao.github.io/blog/PyTorch-Benchmark/)
- [How Many Random Seeds? Statistical Power Analysis in Deep RL (Colas et al.)](https://arxiv.org/pdf/1806.08295)
- [Confidence intervals for performance estimates via bootstrapping in 3D medical image segmentation](https://www.researchgate.net/publication/372488161_Confidence_intervals_for_performance_estimates_in_3D_medical_image_segmentation)
- [Reliability diagrams / ECE — Towards Data Science](https://towardsdatascience.com/expected-calibration-error-ece-a-step-by-step-visual-explanation-with-python-code-c3e9aa12937d/); [AURC — TorchUncertainty docs](https://torch-uncertainty.github.io/generated/torch_uncertainty.metrics.classification.AURC.html)
- [MLflow vs W&B vs TensorBoard (2026)](https://mlopslab.org/mlflow-vs-tensorboard-which-experiment-tracker-should-you-use-in-2026/); [W&B pricing 2026](https://www.g2.com/products/weights-biases/pricing)
- [Google Colab free tier limits 2026](https://joshthompson.co.uk/ai/google-colab-2026-guide-free-compute-automations-pro-tips/); [Kaggle weekly GPU quota](https://www.kaggle.com/general/108481)
- [PyTorch Reproducibility notes](https://docs.pytorch.org/docs/main/notes/randomness.html)
