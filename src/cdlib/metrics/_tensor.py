"""Tiny conversion helpers so metrics accept torch tensors *or* numpy arrays."""
from __future__ import annotations

from typing import Any

import numpy as np


def as_numpy(x: Any) -> np.ndarray:
    if x is None:
        raise TypeError("expected array/tensor, got None")
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def binarize(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return mask >= threshold


def valid_mask(gt: np.ndarray) -> np.ndarray:
    """Pixels with label >= 0 are evaluated; -1 is ignore (frozen dataset contract)."""
    return gt >= 0
