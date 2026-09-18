from __future__ import annotations

import numpy as np

from cdlib.metrics.overlays import (
    FN_PURPLE,
    FP_ORANGE,
    TP_BLUE,
    overlay_tp_fp_fn,
    pick_best_median_worst,
)
from cdlib.metrics.stats import ablation_table, bootstrap_ci, summarize_seeds


def test_bootstrap_ci_covers_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.05, size=200)
    mean, (lo, hi) = bootstrap_ci(x, B=1000, seed=0)
    assert lo < mean < hi
    assert abs(mean - x.mean()) < 1e-9


def test_summarize_seeds_n3_no_pvalue():
    s = summarize_seeds([0.70, 0.72, 0.71])
    assert s["n"] == 3
    assert "p" not in s
    assert s["min"] <= s["median"] <= s["max"]


def test_ablation_table_no_significance():
    md = ablation_table(
        {
            "rgb_ssim": [0.40, 0.41, 0.39],
            "proposed": [0.80, 0.81, 0.79],
        }
    )
    assert "proposed" in md
    assert "p-value" not in md.lower() or "not reported" in md
    assert "underpowered" in md


def test_overlay_colours_are_okabe_ito_not_red_green():
    img = np.zeros((8, 8, 3), dtype=np.float32)
    pred = np.zeros((8, 8), dtype=bool)
    gt = np.zeros((8, 8), dtype=bool)
    pred[0, 0] = True
    gt[0, 0] = True  # TP
    pred[0, 1] = True  # FP
    gt[1, 0] = True  # FN
    out = overlay_tp_fp_fn(img, pred, gt, alpha=1.0)
    np.testing.assert_allclose(out[0, 0], TP_BLUE, atol=1e-5)
    np.testing.assert_allclose(out[0, 1], FP_ORANGE, atol=1e-5)
    np.testing.assert_allclose(out[1, 0], FN_PURPLE, atol=1e-5)
    # Not a red-green scheme
    assert not (FP_ORANGE[0] > 0.8 and FP_ORANGE[1] < 0.2 and FP_ORANGE[2] < 0.2)
    assert not (FN_PURPLE[1] > 0.8 and FN_PURPLE[0] < 0.2)


def test_best_median_worst():
    idx = pick_best_median_worst(np.array([0.2, 0.9, 0.5, 0.1, 0.7]))
    assert idx["best"] == 1
    assert idx["worst"] == 3
    assert idx["median"] in {0, 2, 4}
