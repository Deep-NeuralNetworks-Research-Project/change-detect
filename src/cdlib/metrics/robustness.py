"""Nuisance-stratified evaluation and F1-retention.

Primary robustness scalar: Retention(c,s) = F1(corrupted) / F1(clean).
Also accumulates per (condition × severity × direction) F1 via the
shared SegmentationMetric so pooling protocol stays consistent.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from cdlib.metrics._tensor import as_numpy
from cdlib.metrics.base import Metric
from cdlib.metrics.corruptions import (
    CODEC_CORRUPTIONS,
    EVAL_CORRUPTIONS,
    apply_to_pair,
)
from cdlib.metrics.segmentation import SegmentationMetric, _squeeze_mask


def retention(f1_corrupted: float, f1_clean: float, eps: float = 1e-8) -> float:
    if f1_clean <= eps:
        return 0.0
    return float(f1_corrupted / f1_clean)


class RobustnessMetric(Metric):
    """Wraps SegmentationMetric; records clean F1 plus optional strata keys.

    Stratification is driven by ``batch['nuisance_label']`` and optional
    ``batch['meta']['corruption']`` / ``severity`` / ``which`` fields.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)
        self.reset()

    def reset(self) -> None:
        self._overall = SegmentationMetric(threshold=self.threshold)
        self._by_label: dict[int, SegmentationMetric] = {}
        self._clean = SegmentationMetric(threshold=self.threshold)

    def update(self, outputs: dict[str, Any], batch: dict[str, Any]) -> None:
        self._overall.update(outputs, batch)
        labels = as_numpy(batch["nuisance_label"]).reshape(-1).astype(int)
        # Split the batch per nuisance id so strata stay honest.
        logits = _squeeze_mask(as_numpy(outputs["logits"]))
        mask = _squeeze_mask(as_numpy(batch["mask"]))
        for lab in np.unique(labels):
            idx = np.where(labels == lab)[0]
            if lab not in self._by_label:
                self._by_label[int(lab)] = SegmentationMetric(threshold=self.threshold)
            sub_out = {"logits": logits[idx]}
            sub_batch = {"mask": mask[idx]}
            self._by_label[int(lab)].update(sub_out, sub_batch)
            if int(lab) == 0:
                self._clean.update(sub_out, sub_batch)

    def compute(self) -> dict[str, float]:
        overall = self._overall.compute()
        out = {f"overall/{k}": v for k, v in overall.items()}
        clean_f1 = self._clean.compute()["f1"] if 0 in self._by_label else overall["f1"]
        out["clean_f1"] = clean_f1
        for lab, m in self._by_label.items():
            stats = m.compute()
            out[f"label{lab}/f1"] = stats["f1"]
            out[f"label{lab}/changed_pixel_ratio"] = stats["changed_pixel_ratio"]
            out[f"label{lab}/retention"] = retention(stats["f1"], clean_f1)
        return out


def run_corruption_eval(
    predict_fn,
    img1: np.ndarray,
    img2: np.ndarray,
    mask: np.ndarray,
    names: Iterable[str] | None = None,
    severities: Iterable[int] = (1, 2, 3, 4, 5),
    directions: Iterable[str] = ("t1", "t2"),
    include_codecs: bool = False,
    cache_dir=None,
    seed: int = 0,
) -> dict[str, float]:
    """Apply the suite to one pair and score F1 per (name, severity, direction).

    ``predict_fn(img1, img2) -> logits [1,H,W] or [H,W]``.
    """
    names = tuple(names) if names is not None else EVAL_CORRUPTIONS
    if include_codecs:
        names = names + CODEC_CORRUPTIONS
    rng = np.random.default_rng(seed)
    logits_clean = np.asarray(predict_fn(img1, img2))
    clean_m = SegmentationMetric()
    clean_m.update({"logits": logits_clean}, {"mask": mask})
    clean_f1 = clean_m.compute()["f1"]
    rows: dict[str, float] = {"clean_f1": clean_f1}
    for name in names:
        for s in severities:
            for which in directions:
                c1, c2 = apply_to_pair(img1, img2, name, int(s), which=which, rng=rng, cache_dir=cache_dir)
                logits = np.asarray(predict_fn(c1, c2))
                m = SegmentationMetric()
                m.update({"logits": logits}, {"mask": mask})
                f1 = m.compute()["f1"]
                key = f"{name}/s{s}/{which}"
                rows[f"{key}/f1"] = f1
                rows[f"{key}/retention"] = retention(f1, clean_f1)
    rets = [v for k, v in rows.items() if k.endswith("/retention")]
    rows["mean_retention"] = float(np.mean(rets)) if rets else 0.0
    return rows
