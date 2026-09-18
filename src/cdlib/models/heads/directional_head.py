"""Appeared / disappeared head. Emitted as ``aux['directional_logits']`` [B,2,H,W]."""

from __future__ import annotations

import torch
import torch.nn as nn


class DirectionalHead(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
