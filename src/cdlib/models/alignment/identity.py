"""No-op alignment — ablation A1 is a one-line config override to this key."""

from __future__ import annotations

import torch
import torch.nn as nn

from cdlib.models.alignment.registry import ALIGNMENT_REGISTRY


@ALIGNMENT_REGISTRY.register("identity")
class IdentityAlignment(nn.Module):
    """Pass feature lists through unchanged. Offset map is zeros at the finest scale."""

    def __init__(self, in_channels: list[int] | None = None, **_ignored: object) -> None:
        super().__init__()
        self.in_channels = list(in_channels or [])

    def forward(
        self, feats1: list[torch.Tensor], feats2: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        ref = feats1[-1]
        offset = ref.new_zeros(ref.shape[0], 2, ref.shape[2], ref.shape[3])
        return feats1, feats2, offset
