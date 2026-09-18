"""Registry integrity + frozen metric contract on dummy tensors."""
from __future__ import annotations

import torch

from cdlib.metrics.base import Metric
from cdlib.metrics.registry import METRIC_REGISTRY, build_metric


def _dummy(b=2, h=32, w=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "outputs": {
            "logits": torch.randn(b, 1, h, w, generator=g),
            "confidence": None,
            "aux": {},
        },
        "batch": {
            "img1": torch.rand(b, 3, h, w, generator=g),
            "img2": torch.rand(b, 3, h, w, generator=g),
            "mask": (torch.rand(b, 1, h, w, generator=g) > 0.9).float(),
            "nuisance_label": torch.zeros(b, dtype=torch.int64),
        },
    }


def test_expected_keys_present():
    for key in ("segmentation", "boundary", "region", "calibration", "robustness"):
        assert key in METRIC_REGISTRY


def test_unknown_key_is_helpful():
    try:
        METRIC_REGISTRY.get("not_a_metric")
        raise AssertionError("should have raised")
    except KeyError as e:
        assert "segmentation" in str(e)


def test_every_metric_update_compute_reset():
    dummy = _dummy()
    for key in METRIC_REGISTRY.keys():
        if key == "swap_consistency":
            continue  # needs logits_swapped
        m = build_metric(key)
        assert isinstance(m, Metric)
        m.update(dummy["outputs"], dummy["batch"])
        out = m.compute()
        assert isinstance(out, dict)
        assert out, f"{key} returned empty dict"
        for k, v in out.items():
            assert isinstance(k, str)
            assert isinstance(v, float), f"{key}.{k} is {type(v)}"
        m.reset()
        z = m.compute()
        assert isinstance(z, dict)


def test_swap_consistency_contract():
    dummy = _dummy()
    dummy["outputs"]["logits_swapped"] = dummy["outputs"]["logits"].flip(0)
    m = build_metric("swap_consistency")
    m.update(dummy["outputs"], dummy["batch"])
    out = m.compute()
    assert 0.0 <= out["swap_consistency"] <= 1.0
