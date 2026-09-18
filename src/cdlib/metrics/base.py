"""torchmetrics-style metric contract.

We implement the contract ourselves rather than subclassing
``torchmetrics.Metric`` so CI stays CPU-light and ``compute()`` can return
``dict[str, float]`` (torchmetrics wants a tensor). The state/reset/update
shape is the same, so swapping to a torchmetrics base later is mechanical.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Metric(ABC):
    """Frozen metric interface (CLAUDE.md)."""

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def update(self, outputs: dict[str, Any], batch: dict[str, Any]) -> None: ...

    @abstractmethod
    def compute(self) -> dict[str, float]: ...
