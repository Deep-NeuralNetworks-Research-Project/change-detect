from __future__ import annotations

import numpy as np
import pytest
import torch

from cdlib.metrics.region import RegionMetric, connected_components, greedy_match_f1


def _hard(pred, gt):
    logits = torch.from_numpy(np.where(pred > 0.5, 20.0, -20.0).astype(np.float32))[None, None]
    mask = torch.from_numpy(gt.astype(np.float32))[None, None]
    m = RegionMetric()
    m.update({"logits": logits}, {"mask": mask})
    return m.compute()


def test_two_blobs_object_f1():
    gt = np.zeros((16, 16), dtype=np.float32)
    pred = np.zeros((16, 16), dtype=np.float32)
    gt[1:4, 1:4] = 1
    gt[10:14, 10:14] = 1
    pred[1:4, 1:4] = 1  # match one
    pred[10:14, 8:12] = 1  # shifted, may or may not match at 0.5
    s = _hard(pred, gt)
    assert s["connectivity"] == 8
    assert 0.0 <= s["object_f1_k1"] <= 1.0


def test_false_alert_on_no_change():
    gt = np.zeros((16, 16), dtype=np.float32)
    pred = np.zeros((16, 16), dtype=np.float32)
    pred[4:10, 4:10] = 1  # 36-pixel blob
    s = _hard(pred, gt)
    assert s["n_neg_pairs"] == 1
    assert s["false_alert_rate_k1"] == pytest.approx(1.0)
    assert s["false_alert_rate_k20"] == pytest.approx(1.0)
    assert s["false_alert_rate_k50"] == pytest.approx(0.0)  # 36 < 50


def test_min_size_filters_speck():
    gt = np.zeros((16, 16), dtype=np.float32)
    pred = np.zeros((16, 16), dtype=np.float32)
    pred[0, 0] = 1
    s = _hard(pred, gt)
    assert s["false_alert_rate_k1"] == pytest.approx(1.0)
    assert s["false_alert_rate_k5"] == pytest.approx(0.0)


def test_greedy_match_identical():
    blob = np.zeros((8, 8), dtype=bool)
    blob[2:5, 2:5] = True
    assert greedy_match_f1([blob], [blob], 0.5) == pytest.approx(1.0)


def test_connected_components_count():
    m = np.zeros((8, 8), dtype=np.uint8)
    m[0:2, 0:2] = 1
    m[6:8, 6:8] = 1
    _, areas = connected_components(m)
    assert len(areas) == 2
