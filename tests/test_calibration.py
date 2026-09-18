from __future__ import annotations

import numpy as np
import pytest
import torch

from cdlib.metrics.calibration import (
    CalibrationMetric,
    equal_mass_ece,
    msr_pair_score,
    nll_binary,
    risk_coverage_from_scores,
    soft_dice_confidence,
)
from cdlib.metrics.temperature import TemperatureScaler


def test_perfect_calibration_ece_zero():
    p = np.array([0.1, 0.1, 0.9, 0.9])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    conf = np.where(p >= 0.5, p, 1 - p)
    correct = ((p >= 0.5) == (y >= 0.5)).astype(float)
    ece, *_ = equal_mass_ece(conf, correct, n_bins=2)
    # conf is 0.9 everywhere, accuracy 1.0 → ECE = 0.1
    assert ece == pytest.approx(0.1)


def test_all_ones_brier_nll_zero():
    p = np.ones(10)
    y = np.ones(10)
    assert nll_binary(p, y) == pytest.approx(0.0, abs=1e-5)


def test_foreground_ece_not_washed_by_background():
    """Easy TN majority would hide foreground miscalibration in pooled ECE."""
    h = w = 8
    logits = torch.full((1, 1, h, w), -4.0)  # p≈0.018, confident background
    mask = torch.zeros(1, 1, h, w)
    mask[0, 0, 0, 0] = 1
    logits[0, 0, 0, 0] = 4.0  # confident and correct on the one changed pixel
    m = CalibrationMetric(n_bins=4)
    m.update({"logits": logits}, {"mask": mask})
    s = m.compute()
    assert s["ece_pooled"] < 0.05
    assert "ece_foreground" in s
    assert "ece_changed" in s
    assert "brier_reliability" in s
    assert "aurc" in s and "e_aurc" in s


def test_sdc_and_msr_range():
    p = np.linspace(0, 1, 16).reshape(4, 4)
    sdc = soft_dice_confidence(p, 0.5)
    msr = msr_pair_score(p, 0.5)
    assert 0.0 <= sdc <= 1.0
    assert 0.5 <= msr <= 1.0


def test_aurc_perfect_ranking():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    errors = np.array([0.0, 0.0, 1.0, 1.0])
    aurc, e_aurc, _, _ = risk_coverage_from_scores(scores, errors)
    scores_bad = np.array([0.1, 0.2, 0.8, 0.9])
    aurc_bad, _, _, _ = risk_coverage_from_scores(scores_bad, errors)
    assert aurc < aurc_bad
    assert e_aurc == pytest.approx(0.0, abs=1e-9)


def test_temperature_rejects_clean_val():
    with pytest.raises(ValueError, match="shift-representative"):
        TemperatureScaler(fitted_on="clean_val")


def test_temperature_fit_on_overconfident_logits():
    # Overconfident: large |logits| but labels mixed → T should increase.
    rng = np.random.default_rng(0)
    y = (rng.random((1, 1, 16, 16)) > 0.5).astype(np.float32)
    logits = np.where(y > 0.5, 8.0, -8.0).astype(np.float32)
    # Flip 30% of labels so the model is overconfidently wrong.
    flip = rng.random(y.shape) < 0.3
    y_noisy = np.where(flip, 1.0 - y, y)
    scaler = TemperatureScaler(fitted_on="shift_val")
    t = scaler.fit(logits, y_noisy, max_iter=50)
    assert t > 1.0
    assert scaler.nll_after <= scaler.nll_before + 1e-5
    assert scaler.summary()["fitted_on_shift"] == 1.0
