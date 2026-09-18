"""Colour-blind-safe qualitative overlays. No red-green.

Okabe-Ito categorical: TP blue, FP vermillion/orange, FN magenta/purple.
Confidence maps: diverging blue↔orange, not red↔green.
"""
from __future__ import annotations

import numpy as np

# Okabe-Ito (sRGB 0–1)
TP_BLUE = np.array([0.0, 114, 178], dtype=np.float64) / 255.0
FP_ORANGE = np.array([213, 94, 0], dtype=np.float64) / 255.0
FN_PURPLE = np.array([204, 121, 167], dtype=np.float64) / 255.0
TN_GRAY = np.array([0.85, 0.85, 0.85])


def overlay_tp_fp_fn(
    image: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    """Blend a CHW or HWC [0,1] RGB frame with TP/FP/FN colours.

    Returns HWC float32 in [0,1].
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    pred_b = np.asarray(pred).astype(bool)
    gt_b = np.asarray(gt).astype(bool)
    if pred_b.ndim == 3:
        pred_b = pred_b.squeeze()
    if gt_b.ndim == 3:
        gt_b = gt_b.squeeze()
    tp = pred_b & gt_b
    fp = pred_b & ~gt_b
    fn = ~pred_b & gt_b
    colour = np.zeros_like(img)
    colour[tp] = TP_BLUE
    colour[fp] = FP_ORANGE
    colour[fn] = FN_PURPLE
    painted = tp | fp | fn
    out = img.copy()
    out[painted] = (1.0 - alpha) * img[painted] + alpha * colour[painted]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def confidence_colormap(conf: np.ndarray) -> np.ndarray:
    """Diverging blue↔orange on [0,1] confidence. Returns HWC float32."""
    x = np.clip(np.asarray(conf, dtype=np.float64), 0.0, 1.0)
    if x.ndim == 3:
        x = x.squeeze()
    # 0 → blue, 0.5 → light gray, 1 → orange
    blue = np.array([0.0, 114, 178]) / 255.0
    orange = np.array([213, 94, 0]) / 255.0
    mid = np.array([0.92, 0.92, 0.92])
    t = x[..., None]
    lo = (1 - 2 * t) * blue + (2 * t) * mid
    hi = (1 - 2 * (t - 0.5)) * mid + (2 * (t - 0.5)) * orange
    rgb = np.where(t <= 0.5, lo, hi)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def pick_best_median_worst(
    per_image_scores: np.ndarray,
) -> dict[str, int]:
    """Indices for qualitative export. Ties → first occurrence."""
    x = np.asarray(per_image_scores, dtype=np.float64)
    if x.size == 0:
        return {"best": 0, "median": 0, "worst": 0}
    order = np.argsort(x)
    return {
        "worst": int(order[0]),
        "median": int(order[len(order) // 2]),
        "best": int(order[-1]),
    }
