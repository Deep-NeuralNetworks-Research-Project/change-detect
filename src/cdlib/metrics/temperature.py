"""Post-hoc temperature scaling.

Fit a single scalar T by minimising NLL on a held-out set. Ovadia et al.
(NeurIPS 2019): a T fitted on *clean* validation does not transfer under
shift. Callers must pass shift-representative logits (video-like /
nuisance-corrupted val), not the clean split.

Does not change argmax / ranking at a fixed threshold of 0.5 on
temperature-scaled probabilities only if the threshold is 0.5; the
decision threshold itself is still chosen on validation (research/04 §7).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from cdlib.metrics._tensor import as_numpy, valid_mask
from cdlib.metrics.segmentation import _squeeze_mask


class TemperatureScaler:
    def __init__(self, fitted_on: str = "shift_val"):
        if fitted_on == "clean_val":
            raise ValueError(
                "Temperature scaling must be fitted on shift-representative data "
                "(Ovadia et al., NeurIPS 2019). Use fitted_on='shift_val' (or "
                "'nuisance_val' / 'video_val'), not clean_val."
            )
        self.fitted_on = fitted_on
        self.temperature = 1.0
        self.nll_before = float("nan")
        self.nll_after = float("nan")

    def fit(
        self,
        logits: Any,
        mask: Any,
        max_iter: int = 200,
        lr: float = 0.05,
    ) -> float:
        z = torch.as_tensor(_squeeze_mask(as_numpy(logits)), dtype=torch.float32).reshape(-1)
        gt = torch.as_tensor(_squeeze_mask(as_numpy(mask)), dtype=torch.float32).reshape(-1)
        valid = torch.as_tensor(valid_mask(_squeeze_mask(as_numpy(mask))).reshape(-1))
        z = z[valid]
        y = (gt[valid] > 0.5).float()
        if z.numel() == 0:
            self.temperature = 1.0
            return 1.0
        with torch.no_grad():
            self.nll_before = float(F.binary_cross_entropy_with_logits(z, y))
        log_t = torch.nn.Parameter(torch.zeros(1))  # T = exp(log_t) > 0
        opt = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            t = torch.exp(log_t).clamp(min=1e-4)
            loss = F.binary_cross_entropy_with_logits(z / t, y)
            loss.backward()
            return loss

        opt.step(closure)
        t = float(torch.exp(log_t).detach().clamp(min=1e-4))
        self.temperature = t
        with torch.no_grad():
            self.nll_after = float(F.binary_cross_entropy_with_logits(z / t, y))
        return t

    def scale(self, logits: Any) -> np.ndarray:
        z = _squeeze_mask(as_numpy(logits))
        return z / self.temperature

    def summary(self) -> dict[str, float]:
        return {
            "temperature": float(self.temperature),
            "nll_before": float(self.nll_before),
            "nll_after": float(self.nll_after),
            "fitted_on_shift": 1.0,
        }
