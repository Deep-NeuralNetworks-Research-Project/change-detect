from __future__ import annotations

import numpy as np
import pytest
import torch

from cdlib.metrics.boundary import BoundaryMetric, boundary_iou, hd95_asd


def _hard(pred, gt):
    logits = torch.from_numpy(np.where(pred > 0.5, 20.0, -20.0).astype(np.float32))[None, None]
    mask = torch.from_numpy(gt.astype(np.float32))[None, None]
    m = BoundaryMetric()
    m.update({"logits": logits}, {"mask": mask})
    return m.compute()


def test_identical_masks_boundary_iou_one():
    m = np.zeros((8, 8), dtype=np.uint8)
    m[2:6, 2:6] = 1
    assert boundary_iou(m, m) == pytest.approx(1.0)


def test_empty_empty_boundary_iou_one():
    z = np.zeros((8, 8), dtype=np.uint8)
    assert boundary_iou(z, z) == pytest.approx(1.0)
    hd, asd = hd95_asd(z, z)
    assert hd == 0.0 and asd == 0.0


def test_one_empty_does_not_return_inf():
    g = np.zeros((8, 8), dtype=np.uint8)
    g[2:6, 2:6] = 1
    p = np.zeros((8, 8), dtype=np.uint8)
    hd, asd = hd95_asd(p, g)
    assert np.isfinite(hd) and np.isfinite(asd)
    assert hd == pytest.approx(np.sqrt(8**2 + 8**2))


def test_metric_on_shifted_square():
    gt = np.zeros((16, 16), dtype=np.float32)
    pred = np.zeros((16, 16), dtype=np.float32)
    gt[4:12, 4:12] = 1
    pred[5:13, 4:12] = 1  # 1px vertical shift
    s = _hard(pred, gt)
    assert 0.0 < s["boundary_iou"] < 1.0
    assert s["hd95"] >= 1.0
    assert np.isfinite(s["asd"])
