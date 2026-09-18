"""Few-seed statistics. Do **not** run significance tests at n=3.

Report mean ± std *and* median + [min, max], plus bootstrap CIs over the
test set (resample pairs, not seeds). Ablation tables bold a row only
when mean±std bands are clearly separated — never a sub-noise delta.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np


def bootstrap_ci(
    per_image_scores: np.ndarray,
    B: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
    statistic: str = "mean",
) -> tuple[float, tuple[float, float]]:
    """95% CI of the mean (or median) by resampling the test set with replacement."""
    x = np.asarray(per_image_scores, dtype=np.float64)
    n = x.size
    if n == 0:
        return 0.0, (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    samples = x[idx]
    if statistic == "median":
        boot = np.median(samples, axis=1)
        point = float(np.median(x))
    else:
        boot = samples.mean(axis=1)
        point = float(x.mean())
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, (float(lo), float(hi))


def summarize_seeds(values: Sequence[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "n": 0.0,
        }
    return {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "median": float(np.median(x)),
        "min": float(x.min()),
        "max": float(x.max()),
        "n": float(x.size),
    }


def ablation_table(
    rows: Mapping[str, Sequence[float]],
    metric_name: str = "f1",
    float_fmt: str = "{:.4f}",
) -> str:
    """Markdown table: model | mean±std | median [min, max]. No p-values."""
    lines = [
        f"| model | {metric_name} mean ± std | {metric_name} median [min, max] | n |",
        "|---|---|---|---|",
    ]
    summaries = {k: summarize_seeds(v) for k, v in rows.items()}
    # Flag a row if its mean-std sits entirely above every other mean+std.
    means_plus = {k: s["mean"] + s["std"] for k, s in summaries.items()}
    for name, s in summaries.items():
        band_lo = s["mean"] - s["std"]
        separated = all(band_lo > means_plus[o] for o in summaries if o != name) and len(summaries) > 1
        label = f"**{name}**" if separated else name
        mean_std = f"{float_fmt.format(s['mean'])} ± {float_fmt.format(s['std'])}"
        med = (
            f"{float_fmt.format(s['median'])} "
            f"[{float_fmt.format(s['min'])}, {float_fmt.format(s['max'])}]"
        )
        lines.append(f"| {label} | {mean_std} | {med} | {int(s['n'])} |")
    lines.append("")
    lines.append(
        "_n=3 seeds: significance tests are underpowered and are not reported. "
        "Bootstrap CIs over the test set are the uncertainty estimate of a single seed._"
    )
    return "\n".join(lines)


def stratified_heatmap_values(
    cells: Mapping[tuple[str, int], float],
    conditions: Iterable[str],
    severities: Iterable[int] = (1, 2, 3, 4, 5),
) -> np.ndarray:
    """Return a (n_conditions × n_severities) array for a retention heatmap."""
    conds = list(conditions)
    sevs = list(severities)
    arr = np.full((len(conds), len(sevs)), np.nan)
    for i, c in enumerate(conds):
        for j, s in enumerate(sevs):
            if (c, s) in cells:
                arr[i, j] = cells[(c, s)]
    return arr
