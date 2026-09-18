"""Siamese EfficientNet-B0/B2 encoder (timm, features_only).

Channel counts are read at runtime via `feature_info`, never copied from
a blog post or table: timm's divisor-8 channel-rounding rule makes
published numbers for B2 in particular unreliable (research/01 §3).

Reminder for whoever picks the default encoder (research/01 §3,
CLAUDE.md gotcha): EfficientNet's ~4x FLOP advantage over ResNet-18 does
not reliably translate into wall-clock speed on T4/P100 — depthwise
separable convs have low arithmetic intensity and SiLU often isn't
kernel-fused. Benchmark step time (`scripts/benchmark_encoders.py`)
before committing to this as the default; that measurement is a
reportable result, not just an implementation detail.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

from .registry import ENCODER_REGISTRY

# out_indices (1,2,3,4) skip timm's stride-2 stem tap (index 0) so the
# returned strides line up with ResNet's C2-C5 convention: {4, 8, 16, 32}.
_OUT_INDICES = (1, 2, 3, 4)


class EfficientNetEncoder(nn.Module):
    """Base class; `EfficientNetB0Encoder` / `EfficientNetB2Encoder` below
    are the registered, concretely-named variants (registry keys must be
    picklable/config-friendly strings, not free-form kwargs).
    """

    def __init__(self, variant: str, pretrained: bool = True, use_c5: bool = True):
        super().__init__()
        self.variant = variant
        self.use_c5 = use_c5
        self.backbone = timm.create_model(
            variant, pretrained=pretrained, features_only=True, out_indices=_OUT_INDICES
        )

        channels = list(self.backbone.feature_info.channels())
        strides = list(self.backbone.feature_info.reduction())
        if not use_c5:
            channels, strides = channels[:3], strides[:3]
        self.out_channels = channels
        self.out_strides = strides

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = list(self.backbone(x))
        if not self.use_c5:
            feats = feats[:3]
        return feats

    def forward_pair(
        self, img1: torch.Tensor, img2: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        b = img1.shape[0]
        x = torch.cat([img1, img2], dim=0)
        feats = self.forward(x)
        feats1 = [f[:b] for f in feats]
        feats2 = [f[b:] for f in feats]
        return feats1, feats2


@ENCODER_REGISTRY.register("efficientnet_b0")
class EfficientNetB0Encoder(EfficientNetEncoder):
    def __init__(self, pretrained: bool = True, use_c5: bool = True):
        super().__init__("efficientnet_b0", pretrained=pretrained, use_c5=use_c5)


@ENCODER_REGISTRY.register("efficientnet_b2")
class EfficientNetB2Encoder(EfficientNetEncoder):
    def __init__(self, pretrained: bool = True, use_c5: bool = True):
        super().__init__("efficientnet_b2", pretrained=pretrained, use_c5=use_c5)
