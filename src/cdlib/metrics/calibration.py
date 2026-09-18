"""Calibration metrics under extreme class imbalance.

Naive all-pixel ECE is dominated by easy background at 2–5% change.
We report:
- class-wise ECE (changed vs unchanged), equal-mass (adaptive) bins
- foreground-restricted ECE (GT-changed pixels only)
- pooled ECE (for the trap, not the headline)
- Brier (total + reliability term of Murphy's decomposition)
- NLL
- pair-level risk–coverage: AURC / E-AURC with MSR-aggregate and
  GT-free Soft-Dice-Confidence ranking
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np

from cdlib.metrics._tensor import as_numpy, valid_mask
from cdlib.metrics.base import Metric
from cdlib.metrics.segmentation import _squeeze_mask


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))


def equal_mass_ece(
    conf: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Adaptive/quantile-binned ECE. Returns (ece, bin_conf, bin_acc, bin_count)."""
    conf = np.asarray(conf, dtype=np.float64).ravel()
    correct = np.asarray(correct, dtype=np.float64).ravel()
    n = conf.size
    if n == 0:
        empty = np.zeros(0)
        return 0.0, empty, empty, empty
    n_bins = max(1, min(n_bins, n))
    order = np.argsort(conf)
    conf_s = conf[order]
    corr_s = correct[order]
    splits = np.array_split(np.arange(n), n_bins)
    ece = 0.0
    bin_conf, bin_acc, bin_count = [], [], []
    for sl in splits:
        if sl.size == 0:
            continue
        c = conf_s[sl]
        a = corr_s[sl]
        bc = float(c.mean())
        ba = float(a.mean())
        ece += (sl.size / n) * abs(ba - bc)
        bin_conf.append(bc)
        bin_acc.append(ba)
        bin_count.append(float(sl.size))
    return (
        float(ece),
        np.asarray(bin_conf),
        np.asarray(bin_acc),
        np.asarray(bin_count),
    )


def brier_decomposition(prob: np.ndarray, y: np.ndarray, n_bins: int = 15) -> dict[str, float]:
    """Murphy (1973): BS = reliability − resolution + uncertainty."""
    p = np.asarray(prob, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = p.size
    if n == 0:
        return {"brier": 0.0, "brier_reliability": 0.0, "brier_resolution": 0.0, "brier_uncertainty": 0.0}
    bs = float(np.mean((p - y) ** 2))
    ybar = float(y.mean())
    uncertainty = ybar * (1.0 - ybar)
    n_bins = max(1, min(n_bins, n))
    order = np.argsort(p)
    p_s, y_s = p[order], y[order]
    reliability = 0.0
    resolution = 0.0
    for sl in np.array_split(np.arange(n), n_bins):
        if sl.size == 0:
            continue
        pk = float(p_s[sl].mean())
        yk = float(y_s[sl].mean())
        wk = sl.size / n
        reliability += wk * (pk - yk) ** 2
        resolution += wk * (yk - ybar) ** 2
    return {
        "brier": bs,
        "brier_reliability": float(reliability),
        "brier_resolution": float(resolution),
        "brier_uncertainty": float(uncertainty),
    }


def nll_binary(prob: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(np.asarray(prob, dtype=np.float64).ravel(), eps, 1 - eps)
    y = np.asarray(y, dtype=np.float64).ravel()
    if y.size == 0:
        return 0.0
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def msr_pair_score(prob: np.ndarray, tau: float = 0.5) -> float:
    """Max-softmax-response aggregate over predicted-positive pixels."""
    p = np.asarray(prob, dtype=np.float64)
    conf = np.maximum(p, 1.0 - p)
    changed = p >= tau
    if changed.any():
        return float(conf[changed].mean())
    return float(conf.mean())


def soft_dice_confidence(prob: np.ndarray, tau: float = 0.5, eps: float = 1e-7) -> float:
    """GT-free Soft-Dice-Confidence: 2 Σ p·ŷ / (Σ p + Σ ŷ).

    Linear-time ranking signal for pair-level deferral (arXiv:2402.10665).
    Does **not** look at the ground-truth mask.
    """
    p = np.asarray(prob, dtype=np.float64)
    yhat = (p >= tau).astype(np.float64)
    num = 2.0 * float((p * yhat).sum())
    den = float(p.sum()) + float(yhat.sum()) + eps
    return num / den


def risk_coverage_from_scores(
    scores: np.ndarray, errors: np.ndarray
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """AURC and E-AURC. ``errors`` is 0/1 per pair (1 = incorrect / high risk).

    Coverage runs from 1/N to 1 (most-confident first). Oracle AURC* puts
    all correct pairs first.
    """
    scores = np.asarray(scores, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    n = scores.size
    if n == 0:
        return 0.0, 0.0, np.zeros(0), np.zeros(0)
    order = np.argsort(-scores)
    err_sorted = errors[order]
    coverage = np.arange(1, n + 1, dtype=np.float64) / n
    risk = np.cumsum(err_sorted) / np.arange(1, n + 1)
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    aurc = float(trapz(risk, coverage))
    oracle = np.sort(errors)  # 0s first
    oracle_risk = np.cumsum(oracle) / np.arange(1, n + 1)
    aurc_star = float(trapz(oracle_risk, coverage))
    return aurc, aurc - aurc_star, coverage, risk


class CalibrationMetric(Metric):
    def __init__(
        self,
        threshold: float = 0.5,
        n_bins: int = 15,
        pair_score: Literal["msr", "sdc"] = "sdc",
        pair_error: Literal["pair_f1", "any_fn"] = "pair_f1",
    ):
        self.threshold = float(threshold)
        self.n_bins = int(n_bins)
        self.pair_score = pair_score
        self.pair_error = pair_error
        self.reset()

    def reset(self) -> None:
        self._prob: list[np.ndarray] = []
        self._y: list[np.ndarray] = []
        self._pair_scores: list[float] = []
        self._pair_errors: list[float] = []

    def update(self, outputs: dict[str, Any], batch: dict[str, Any]) -> None:
        logits = _squeeze_mask(as_numpy(outputs["logits"]))
        gt_raw = _squeeze_mask(as_numpy(batch["mask"]))
        probs = _sigmoid(logits)
        valid = valid_mask(gt_raw)
        y = (gt_raw > 0.5) & valid
        for i in range(logits.shape[0]):
            v = valid[i]
            p = probs[i][v]
            yi = y[i][v].astype(np.float64)
            self._prob.append(p.astype(np.float64))
            self._y.append(yi)
            full_p = probs[i]
            if self.pair_score == "sdc":
                score = soft_dice_confidence(full_p, self.threshold)
            else:
                score = msr_pair_score(full_p, self.threshold)
            self._pair_scores.append(score)
            pred = full_p >= self.threshold
            tp = float((pred & y[i] & v).sum())
            fp = float((pred & ~y[i] & v).sum())
            fn = float((~pred & y[i] & v).sum())
            if self.pair_error == "any_fn":
                err = 1.0 if fn > 0 else 0.0
            else:
                f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 1.0
                err = 1.0 - f1
            self._pair_errors.append(err)

    def compute(self) -> dict[str, float]:
        if not self._prob:
            return {
                "ece_pooled": 0.0,
                "ece_changed": 0.0,
                "ece_unchanged": 0.0,
                "ece_foreground": 0.0,
                "brier": 0.0,
                "brier_reliability": 0.0,
                "nll": 0.0,
                "aurc": 0.0,
                "e_aurc": 0.0,
                "n_bins": float(self.n_bins),
            }
        p = np.concatenate(self._prob)
        y = np.concatenate(self._y)
        conf_pos = np.where(p >= 0.5, p, 1.0 - p)
        pred_pos = p >= 0.5
        correct = (pred_pos == (y >= 0.5)).astype(np.float64)
        ece_pooled, _, _, _ = equal_mass_ece(conf_pos, correct, self.n_bins)
        ch = y >= 0.5
        unch = ~ch
        ece_ch, _, _, _ = equal_mass_ece(p[ch], y[ch], self.n_bins) if ch.any() else (0.0, None, None, None)
        # For unchanged class, confidence that pixel is unchanged = 1-p
        ece_un, _, _, _ = (
            equal_mass_ece(1.0 - p[unch], 1.0 - y[unch], self.n_bins) if unch.any() else (0.0, None, None, None)
        )
        ece_fg, _, _, _ = equal_mass_ece(p[ch], y[ch], self.n_bins) if ch.any() else (0.0, None, None, None)
        brier = brier_decomposition(p, y, self.n_bins)
        nll = nll_binary(p, y)
        aurc, e_aurc, _, _ = risk_coverage_from_scores(
            np.asarray(self._pair_scores), np.asarray(self._pair_errors)
        )
        return {
            "ece_pooled": ece_pooled,
            "ece_changed": float(ece_ch),
            "ece_unchanged": float(ece_un),
            "ece_foreground": float(ece_fg),
            "ece_classwise_mean": 0.5 * (float(ece_ch) + float(ece_un)),
            "brier": brier["brier"],
            "brier_reliability": brier["brier_reliability"],
            "brier_resolution": brier["brier_resolution"],
            "brier_uncertainty": brier["brier_uncertainty"],
            "nll": nll,
            "aurc": aurc,
            "e_aurc": e_aurc,
            "n_bins": float(self.n_bins),
        }

    def reliability_diagram_data(self, which: str = "foreground") -> dict[str, np.ndarray]:
        """Bin centres for publication reliability diagrams (one per class)."""
        if not self._prob:
            return {"conf": np.zeros(0), "acc": np.zeros(0), "count": np.zeros(0)}
        p = np.concatenate(self._prob)
        y = np.concatenate(self._y)
        if which == "foreground":
            m = y >= 0.5
            ece, bc, ba, cnt = equal_mass_ece(p[m], y[m], self.n_bins) if m.any() else (0.0, np.zeros(0), np.zeros(0), np.zeros(0))
        elif which == "changed":
            m = y >= 0.5
            ece, bc, ba, cnt = equal_mass_ece(p[m], y[m], self.n_bins) if m.any() else (0.0, np.zeros(0), np.zeros(0), np.zeros(0))
        else:
            conf = np.where(p >= 0.5, p, 1.0 - p)
            pred = p >= 0.5
            correct = (pred == (y >= 0.5)).astype(np.float64)
            ece, bc, ba, cnt = equal_mass_ece(conf, correct, self.n_bins)
        return {"conf": bc, "acc": ba, "count": cnt, "ece": np.array([ece])}
