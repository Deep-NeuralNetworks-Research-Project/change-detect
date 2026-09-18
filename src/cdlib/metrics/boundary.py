"""Boundary IoU (Cheng et al., CVPR 2021) plus HD95 / ASD.

Primary boundary metric = Boundary IoU with dilation_ratio=0.02 (scale-
balanced, cheap). Secondary = HD95 and average symmetric surface
distance via a distance-transform implementation (MONAI-equivalent
formulae; we do not require MONAI at import time).

Empty-mask policy (MONAI issue #2179 — do not return inf, do not
silently drop):
- both empty → skip in the running mean (count as 1.0 Biou / 0 HD)
- exactly one empty → HD95 = image diagonal, ASD = image diagonal,
  Boundary IoU = 0
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cdlib.metrics._tensor import as_numpy, valid_mask
from cdlib.metrics.base import Metric
from cdlib.metrics.segmentation import _squeeze_mask


def mask_to_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    """Boundary band of width d = dilation_ratio × image diagonal."""
    mask = (mask.astype(np.uint8) > 0).astype(np.uint8)
    h, w = mask.shape
    img_diag = float(np.sqrt(h**2 + w**2))
    dilation = max(1, int(round(dilation_ratio * img_diag)))
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    return mask - mask_erode


def boundary_iou(gt: np.ndarray, dt: np.ndarray, dilation_ratio: float = 0.02) -> float:
    gt_b = mask_to_boundary(gt, dilation_ratio)
    dt_b = mask_to_boundary(dt, dilation_ratio)
    intersection = int(((gt_b * dt_b) > 0).sum())
    union = int(((gt_b + dt_b) > 0).sum())
    if union == 0:
        return 1.0
    return intersection / union


def _surface_distances(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Directed boundary-to-boundary distances both ways (pixels)."""
    pred_u = (pred.astype(np.uint8) > 0).astype(np.uint8)
    gt_u = (gt.astype(np.uint8) > 0).astype(np.uint8)
    pred_b = mask_to_boundary(pred_u, dilation_ratio=0)  # 1-px contour via erode-1
    # dilation_ratio=0 still yields max(1, 0)=1 iteration — good, 1-px band.
    gt_b = mask_to_boundary(gt_u, dilation_ratio=0)
    # Distance to GT boundary: invert GT boundary (0 on boundary).
    inv_gt = np.where(gt_b > 0, 0, 1).astype(np.uint8)
    inv_pr = np.where(pred_b > 0, 0, 1).astype(np.uint8)
    dist_gt = cv2.distanceTransform(inv_gt, cv2.DIST_L2, 5)
    dist_pr = cv2.distanceTransform(inv_pr, cv2.DIST_L2, 5)
    pred_to_gt = dist_gt[pred_b > 0]
    gt_to_pred = dist_pr[gt_b > 0]
    return pred_to_gt, gt_to_pred


def hd95_asd(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    h, w = pred.shape
    diag = float(np.sqrt(h**2 + w**2))
    pred_any = bool(pred.astype(bool).any())
    gt_any = bool(gt.astype(bool).any())
    if not pred_any and not gt_any:
        return 0.0, 0.0
    if pred_any != gt_any:
        return diag, diag
    a, b = _surface_distances(pred, gt)
    pooled = np.concatenate([a, b]) if (a.size + b.size) else np.array([0.0])
    hd95 = float(np.percentile(pooled, 95)) if pooled.size else 0.0
    asd = float(pooled.mean()) if pooled.size else 0.0
    return hd95, asd


class BoundaryMetric(Metric):
    def __init__(self, threshold: float = 0.5, dilation_ratio: float = 0.02):
        self.threshold = float(threshold)
        self.dilation_ratio = float(dilation_ratio)
        self.reset()

    def reset(self) -> None:
        self._biou: list[float] = []
        self._hd95: list[float] = []
        self._asd: list[float] = []

    def update(self, outputs: dict[str, Any], batch: dict[str, Any]) -> None:
        logits = _squeeze_mask(as_numpy(outputs["logits"]))
        gt_raw = _squeeze_mask(as_numpy(batch["mask"]))
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
        pred = probs >= self.threshold
        gt_pos = gt_raw > 0.5
        valid = valid_mask(gt_raw)
        for i in range(logits.shape[0]):
            # Ignore pixels: treat as background so they don't grow a fake boundary.
            p = np.where(valid[i], pred[i], False)
            g = np.where(valid[i], gt_pos[i], False)
            self._biou.append(boundary_iou(g, p, self.dilation_ratio))
            hd, asd = hd95_asd(p, g)
            self._hd95.append(hd)
            self._asd.append(asd)

    def compute(self) -> dict[str, float]:
        if not self._biou:
            return {
                "boundary_iou": 0.0,
                "hd95": 0.0,
                "asd": 0.0,
                "dilation_ratio": self.dilation_ratio,
            }
        return {
            "boundary_iou": float(np.mean(self._biou)),
            "hd95": float(np.mean(self._hd95)),
            "asd": float(np.mean(self._asd)),
            "dilation_ratio": self.dilation_ratio,
        }
