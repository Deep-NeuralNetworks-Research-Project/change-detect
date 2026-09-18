"""Changed-class P/R/F1/IoU with both pooling protocols.

Primary number is **aggregate (corpus / micro) F1**: one TP/FP/FN sum
across every valid pixel of every test pair, then P/R/F1/IoU once.
Per-image (macro) mF1 is secondary. True-negative-only images
(TP=FP=FN=0) are excluded from mF1 and counted via the region metric's
false-alert rate — they are never silently nan-dropped.

Changed-pixel ratio π is returned beside every result. Overall accuracy
is computed but is not a headline key (at π=2–5% the all-negative
classifier scores 95–98%).

Threshold is recorded; it must be fixed on validation scenes only.
``threshold_protocol`` is encoded as 0.0 = F1-optimal, 1.0 = precision-constrained.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np

from cdlib.metrics._tensor import as_numpy, valid_mask
from cdlib.metrics.base import Metric


def _squeeze_mask(x: np.ndarray) -> np.ndarray:
    """[B,1,H,W] or [B,H,W] or [H,W] -> [B,H,W]."""
    x = np.asarray(x)
    if x.ndim == 2:
        x = x[None, ...]
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 3:
        raise ValueError(f"expected mask-like array with 2–4 dims, got {x.shape}")
    return x


def prf1_from_counts(tp: float, fp: float, fn: float) -> tuple[float, float, float, float]:
    """Return precision, recall, F1, IoU from confusion counts.

    Undefined-denominator convention (used only when the pair is *not*
    true-negative-only): a zero denominator yields 0, except the
    TP=FP=FN=0 case which is 1/1/1/1 and should be excluded from mF1
    by the caller.
    """
    tp, fp, fn = float(tp), float(fp), float(fn)
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0, 1.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    denom = tp + fp + fn
    iou = tp / denom if denom > 0 else 1.0
    return prec, rec, f1, iou


def confusion_counts(
    pred: np.ndarray, gt: np.ndarray, valid: np.ndarray
) -> tuple[int, int, int, int]:
    pred_b = pred.astype(bool) & valid.astype(bool)
    gt_b = gt.astype(bool) & valid.astype(bool)
    val = valid.astype(bool)
    tp = int((pred_b & gt_b).sum())
    fp = int((pred_b & ~gt_b).sum())
    fn = int((~pred_b & gt_b).sum())
    tn = int((~pred_b & ~gt_b & val).sum())
    return tp, fp, fn, tn


class SegmentationMetric(Metric):
    def __init__(
        self,
        threshold: float = 0.5,
        threshold_protocol: Literal["f1_optimal", "precision_constrained"] = "f1_optimal",
    ):
        self.threshold = float(threshold)
        if threshold_protocol not in ("f1_optimal", "precision_constrained"):
            raise ValueError(threshold_protocol)
        self.threshold_protocol = threshold_protocol
        self.reset()

    def reset(self) -> None:
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._tn = 0
        self._changed = 0
        self._valid = 0
        self._per_image_f1: list[float] = []
        self._per_image_p: list[float] = []
        self._per_image_r: list[float] = []
        self._per_image_iou: list[float] = []
        self._n_images = 0
        self._n_tn_only = 0

    def update(self, outputs: dict[str, Any], batch: dict[str, Any]) -> None:
        logits = _squeeze_mask(as_numpy(outputs["logits"]))
        gt_raw = _squeeze_mask(as_numpy(batch["mask"]))
        if logits.shape != gt_raw.shape:
            raise ValueError(f"logits {logits.shape} vs mask {gt_raw.shape}")
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
        pred = probs >= self.threshold
        gt_pos = gt_raw > 0.5
        valid = valid_mask(gt_raw)
        b = logits.shape[0]
        for i in range(b):
            tp, fp, fn, tn = confusion_counts(pred[i], gt_pos[i], valid[i])
            self._tp += tp
            self._fp += fp
            self._fn += fn
            self._tn += tn
            n_valid = int(valid[i].sum())
            n_changed = int((gt_pos[i] & valid[i]).sum())
            self._valid += n_valid
            self._changed += n_changed
            self._n_images += 1
            if tp == 0 and fp == 0 and fn == 0:
                self._n_tn_only += 1
                continue
            p, r, f1, iou = prf1_from_counts(tp, fp, fn)
            self._per_image_p.append(p)
            self._per_image_r.append(r)
            self._per_image_f1.append(f1)
            self._per_image_iou.append(iou)

    def compute(self) -> dict[str, float]:
        p, r, f1, iou = prf1_from_counts(self._tp, self._fp, self._fn)
        total = self._tp + self._fp + self._fn + self._tn
        oa = (self._tp + self._tn) / total if total else 1.0
        pi = self._changed / self._valid if self._valid else 0.0
        n_macro = len(self._per_image_f1)
        return {
            "precision": p,
            "recall": r,
            "f1": f1,
            "iou": iou,
            "precision_macro": float(np.mean(self._per_image_p)) if n_macro else 0.0,
            "recall_macro": float(np.mean(self._per_image_r)) if n_macro else 0.0,
            "f1_macro": float(np.mean(self._per_image_f1)) if n_macro else 0.0,
            "iou_macro": float(np.mean(self._per_image_iou)) if n_macro else 0.0,
            "changed_pixel_ratio": pi,
            "overall_accuracy": oa,
            "threshold": self.threshold,
            "threshold_protocol": 0.0 if self.threshold_protocol == "f1_optimal" else 1.0,
            "n_images": float(self._n_images),
            "n_tn_only_excluded": float(self._n_tn_only),
            "tp": float(self._tp),
            "fp": float(self._fp),
            "fn": float(self._fn),
            "tn": float(self._tn),
        }
