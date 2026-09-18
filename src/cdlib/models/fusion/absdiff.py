"""Absolute-difference fusion — the FC-Siam-Diff baseline behaviour."""
from __future__ import annotations

import torch
import torch.nn as nn

from .registry import FUSION_REGISTRY


@FUSION_REGISTRY.register("absdiff")
class AbsDiffFusion(nn.Module):
    """`|a - b|` per scale.

    Order-invariant for free: `|a-b| == |b-a|` identically, for any
    downstream head. Cheap (output width == input width per scale) but
    throws away direction — appeared and disappeared collapse to the
    same signal. This is what `signed_fusion.py` is written to improve
    on; see that module's docstring for the argument in full.
    """

    def __init__(self, in_channels: list[int]):
        super().__init__()
        self.in_channels = list(in_channels)
        self.out_channels = list(in_channels)

    def forward(
        self, feats1: list[torch.Tensor], feats2: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        return [torch.abs(a - b) for a, b in zip(feats1, feats2)]
