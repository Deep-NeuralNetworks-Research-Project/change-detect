"""Connected-component / object-level metrics.

8-connectivity (stated, because 4 vs 8 changes FP-region counts).
A pair is "alerted" if any predicted region exceeds size k.
Object-level F1: greedy IoU matching at τ (primary 0.5).
Min-size sweep k ∈ {1, 5, 20, 50} — not a single silent k.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cdlib.metrics._tensor import as_numpy, valid_mask
from cdlib.metrics.base import Metric
from cdlib.metrics.segmentation import _squeeze_mask

DEFAULT_K = (1, 5, 20, 50)
CONNECTIVITY = 8


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (label_map, areas_per_label starting at label 1)."""
    m = (mask.astype(np.uint8) > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=CONNECTIVITY)
    areas = stats[1:, cv2.CC_STAT_AREA] if n_labels > 1 else np.zeros((0,), dtype=np.int32)
    return labels, areas


def filter_by_min_size(labels: np.ndarray, areas: np.ndarray, k: int) -> np.ndarray:
    keep = np.where(areas >= k)[0] + 1
    return np.isin(labels, keep)


def _component_masks(labels: np.ndarray, areas: np.ndarray, k: int) -> list[np.ndarray]:
    keep = np.where(areas >= k)[0] + 1
    return [(labels == lab) for lab in keep]


def _pairwise_iou(a: list[np.ndarray], b: list[np.ndarray]) -> np.ndarray:
    if not a or not b:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    iou = np.zeros((len(a), len(b)), dtype=np.float64)
    for i, am in enumerate(a):
        for j, bm in enumerate(b):
            inter = np.logical_and(am, bm).sum()
            union = np.logical_or(am, bm).sum()
            iou[i, j] = inter / union if union else 0.0
    return iou


def greedy_match_f1(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray], tau: float) -> float:
    iou = _pairwise_iou(pred_masks, gt_masks)
    n_p, n_g = iou.shape
    if n_p == 0 and n_g == 0:
        return 1.0
    matched_p = set()
    matched_g = set()
    pairs = [(iou[i, j], i, j) for i in range(n_p) for j in range(n_g)]
    pairs.sort(reverse=True)
    tp = 0
    for val, i, j in pairs:
        if val < tau:
            break
        if i in matched_p or j in matched_g:
            continue
        matched_p.add(i)
        matched_g.add(j)
        tp += 1
    fp = n_p - tp
    fn = n_g - tp
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 1.0


class RegionMetric(Metric):
    def __init__(
        self,
        threshold: float = 0.5,
        min_sizes: tuple[int, ...] = DEFAULT_K,
        iou_tau: float = 0.5,
    ):
        self.threshold = float(threshold)
        self.min_sizes = tuple(int(k) for k in min_sizes)
        self.iou_tau = float(iou_tau)
        self.reset()

    def reset(self) -> None:
        self._alerted = {k: 0 for k in self.min_sizes}
        self._n_neg = 0  # no-change pairs (no GT component at k=1)
        self._n_images = 0
        self._obj_f1 = {k: [] for k in self.min_sizes}
        self._n_fp_regions = {k: [] for k in self.min_sizes}

    def update(self, outputs: dict[str, Any], batch: dict[str, Any]) -> None:
        logits = _squeeze_mask(as_numpy(outputs["logits"]))
        gt_raw = _squeeze_mask(as_numpy(batch["mask"]))
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
        pred = probs >= self.threshold
        gt_pos = gt_raw > 0.5
        valid = valid_mask(gt_raw)
        for i in range(logits.shape[0]):
            p = np.where(valid[i], pred[i], False)
            g = np.where(valid[i], gt_pos[i], False)
            p_lab, p_areas = connected_components(p)
            g_lab, g_areas = connected_components(g)
            is_neg = g_areas.size == 0 or (g_areas < 1).all()
            if is_neg:
                self._n_neg += 1
            self._n_images += 1
            for k in self.min_sizes:
                p_kept = _component_masks(p_lab, p_areas, k)
                g_kept = _component_masks(g_lab, g_areas, k)
                self._n_fp_regions[k].append(float(len(p_kept)))
                self._obj_f1[k].append(greedy_match_f1(p_kept, g_kept, self.iou_tau))
                if is_neg:
                    max_area = float(p_areas.max()) if p_areas.size else 0.0
                    if max_area >= k:
                        self._alerted[k] += 1

    def compute(self) -> dict[str, float]:
        out: dict[str, float] = {
            "connectivity": float(CONNECTIVITY),
            "iou_tau": self.iou_tau,
            "n_images": float(self._n_images),
            "n_neg_pairs": float(self._n_neg),
        }
        for k in self.min_sizes:
            far = (self._alerted[k] / self._n_neg) if self._n_neg else 0.0
            out[f"false_alert_rate_k{k}"] = far
            vals = self._obj_f1[k]
            out[f"object_f1_k{k}"] = float(np.mean(vals)) if vals else 0.0
            nfp = self._n_fp_regions[k]
            out[f"mean_pred_regions_k{k}"] = float(np.mean(nfp)) if nfp else 0.0
        return out
