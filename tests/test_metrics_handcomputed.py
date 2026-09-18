"""Hand-built 4×4 and 5×5 mask pairs with calculator-verified P/R/F1/IoU.

Written *before* trusting any number. Update these fixtures if you change
the pooling protocol — do not "fix" a failing test by loosening atol
without re-doing the arithmetic.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cdlib.metrics.segmentation import SegmentationMetric, prf1_from_counts


def _logits_from_hard(pred: np.ndarray) -> torch.Tensor:
    """Hard 0/1 mask → extreme logits so threshold 0.5 is a no-op."""
    x = np.where(pred > 0.5, 20.0, -20.0).astype(np.float32)
    return torch.from_numpy(x)[None, None]  # [1,1,H,W]


def _run(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    m = SegmentationMetric(threshold=0.5)
    m.update({"logits": _logits_from_hard(pred)}, {"mask": torch.from_numpy(gt.astype(np.float32))[None, None]})
    return m.compute()


# ---------------------------------------------------------------------------
# Pair A — 4×4  (calculator)
#   gt                pred
#   1 1 0 0           1 1 0 0
#   1 1 0 0           1 0 0 0
#   0 0 0 0           0 0 1 0
#   0 0 0 1           0 0 0 1
# TP={(0,0),(0,1),(1,0),(3,3)} = 4
# FP={(2,2)} = 1
# FN={(1,1)} = 1
# TN = 10
# P = 4/5 = 0.8
# R = 4/5 = 0.8
# F1 = 0.8
# IoU = 4/6 = 2/3
# π = 5/16 = 0.3125
# ---------------------------------------------------------------------------
GT_A = np.array(
    [
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)
PRED_A = np.array(
    [
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)

# Pair B — 4×4 true-negative-only (perfect). Excluded from mF1.
GT_B = np.zeros((4, 4), dtype=np.float32)
PRED_B = np.zeros((4, 4), dtype=np.float32)

# Pair C — 4×4
# gt: single changed pixel at (0,0); pred: that pixel + one FP at (0,1)
# TP=1 FP=1 FN=0 TN=14
# P=1/2  R=1  F1=2/3  IoU=1/2  π=1/16
GT_C = np.zeros((4, 4), dtype=np.float32)
GT_C[0, 0] = 1
PRED_C = np.zeros((4, 4), dtype=np.float32)
PRED_C[0, 0] = 1
PRED_C[0, 1] = 1

# ---------------------------------------------------------------------------
# 5×5 pair D
# gt positives (9): (0,0)(0,1)(0,2)(1,0)(1,1)(2,0)(3,4)(4,3)(4,4)
# pred positives (8): (0,0)(0,1)(1,0)(1,1)(1,2)(3,3)(3,4)(4,4)
# TP=6  FP=2  FN=3  TN=14
# P=6/8=0.75  R=6/9=2/3  F1=12/17  IoU=6/11  π=9/25
# ---------------------------------------------------------------------------
GT_D = np.array(
    [
        [1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 1],
    ],
    dtype=np.float32,
)
PRED_D = np.array(
    [
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1],
    ],
    dtype=np.float32,
)


def test_prf1_from_counts_closed_form():
    p, r, f1, iou = prf1_from_counts(4, 1, 1)
    assert p == pytest.approx(0.8)
    assert r == pytest.approx(0.8)
    assert f1 == pytest.approx(0.8)
    assert iou == pytest.approx(4 / 6)
    # IoU–F1 bijection at identical pooling: F1 = 2 IoU / (1+IoU)
    assert f1 == pytest.approx(2 * iou / (1 + iou))


def test_pair_a_4x4():
    s = _run(PRED_A, GT_A)
    assert s["tp"] == 4
    assert s["fp"] == 1
    assert s["fn"] == 1
    assert s["tn"] == 10
    assert s["precision"] == pytest.approx(0.8)
    assert s["recall"] == pytest.approx(0.8)
    assert s["f1"] == pytest.approx(0.8)
    assert s["iou"] == pytest.approx(2 / 3)
    assert s["changed_pixel_ratio"] == pytest.approx(5 / 16)
    assert s["n_tn_only_excluded"] == 0


def test_pair_b_true_negative_excluded_from_macro():
    s = _run(PRED_B, GT_B)
    assert s["tp"] == 0 and s["fp"] == 0 and s["fn"] == 0
    assert s["tn"] == 16
    assert s["n_tn_only_excluded"] == 1
    assert s["f1_macro"] == 0.0  # excluded, not nan-dropped into a fake 1.0 mean
    assert s["changed_pixel_ratio"] == pytest.approx(0.0)
    # aggregate of a perfect TN image: P/R/F1 conventionally 1
    assert s["f1"] == pytest.approx(1.0)


def test_pair_c_4x4():
    s = _run(PRED_C, GT_C)
    assert s["tp"] == 1 and s["fp"] == 1 and s["fn"] == 0
    assert s["precision"] == pytest.approx(0.5)
    assert s["recall"] == pytest.approx(1.0)
    assert s["f1"] == pytest.approx(2 / 3)
    assert s["iou"] == pytest.approx(0.5)
    assert s["changed_pixel_ratio"] == pytest.approx(1 / 16)


def test_pair_d_5x5():
    s = _run(PRED_D, GT_D)
    assert s["tp"] == 6 and s["fp"] == 2 and s["fn"] == 3
    assert s["precision"] == pytest.approx(0.75)
    assert s["recall"] == pytest.approx(2 / 3)
    assert s["f1"] == pytest.approx(12 / 17)
    assert s["iou"] == pytest.approx(6 / 11)
    assert s["changed_pixel_ratio"] == pytest.approx(9 / 25)


def test_aggregate_vs_macro_on_a_plus_c():
    """A and C together: aggregate F1 ≠ mean of per-image F1.

    Aggregate: TP=5 FP=2 FN=1 → P=5/7 R=5/6 F1=10/13
    mF1: (0.8 + 2/3) / 2 = 11/15
    """
    m = SegmentationMetric()
    for pred, gt in ((PRED_A, GT_A), (PRED_C, GT_C)):
        m.update(
            {"logits": _logits_from_hard(pred)},
            {"mask": torch.from_numpy(gt.astype(np.float32))[None, None]},
        )
    s = m.compute()
    assert s["tp"] == 5 and s["fp"] == 2 and s["fn"] == 1
    assert s["precision"] == pytest.approx(5 / 7)
    assert s["recall"] == pytest.approx(5 / 6)
    assert s["f1"] == pytest.approx(10 / 13)
    assert s["iou"] == pytest.approx(5 / 8)
    assert s["f1_macro"] == pytest.approx(11 / 15)
    assert s["f1"] != pytest.approx(s["f1_macro"])
    assert s["changed_pixel_ratio"] == pytest.approx(6 / 32)


def test_tn_pair_does_not_inflate_macro():
    m = SegmentationMetric()
    for pred, gt in ((PRED_A, GT_A), (PRED_B, GT_B)):
        m.update(
            {"logits": _logits_from_hard(pred)},
            {"mask": torch.from_numpy(gt.astype(np.float32))[None, None]},
        )
    s = m.compute()
    assert s["n_tn_only_excluded"] == 1
    assert s["f1_macro"] == pytest.approx(0.8)  # only pair A
    # aggregate includes B's zeros, so same TP/FP/FN as A
    assert s["f1"] == pytest.approx(0.8)


def test_ignore_label_minus_one():
    """Last row of pair A marked ignore (−1). Valid 3×4 = 12 pixels.

    Remaining: TP=3 FP=1 FN=1  (the (3,3) TP is ignored)
    P=R=F1=0.75  IoU=3/5  π=4/12
    """
    gt = GT_A.copy()
    gt[3, :] = -1
    s = _run(PRED_A, gt)
    assert s["tp"] == 3 and s["fp"] == 1 and s["fn"] == 1
    assert s["precision"] == pytest.approx(0.75)
    assert s["recall"] == pytest.approx(0.75)
    assert s["f1"] == pytest.approx(0.75)
    assert s["iou"] == pytest.approx(0.6)
    assert s["changed_pixel_ratio"] == pytest.approx(4 / 12)
    assert s["tn"] + s["tp"] + s["fp"] + s["fn"] == 12
