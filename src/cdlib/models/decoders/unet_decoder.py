"""U-Net/FPN-style decoder over multi-scale fused features."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import DECODER_REGISTRY


class _ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


@DECODER_REGISTRY.register("unet_decoder")
class UNetDecoder(nn.Module):
    """Top-down decoder over N skip levels, coarsest-to-finest, each step
    upsampling and concatenating with the next skip connection.

    Input: a list of fused feature maps ordered shallow -> deep (i.e. the
    same [C2, C3, C4, (C5)] order encoders/fusion produce). Output:
    `[B, 1, H, W]` pre-sigmoid logits at the *finest* input skip's
    resolution — callers (e.g. `siamese_resnet18.py`) upsample to full
    image size if that isn't already stride 1.
    """

    def __init__(self, in_channels: list[int], decoder_channels: list[int] | None = None):
        super().__init__()
        n = len(in_channels)
        if decoder_channels is None:
            decoder_channels = [max(32, c // 2) for c in reversed(in_channels)]
        if len(decoder_channels) != n:
            raise ValueError("decoder_channels must have exactly one entry per skip level")

        rev_in = list(reversed(in_channels))  # coarsest -> finest
        self.blocks = nn.ModuleList()
        prev_ch = 0
        for i, skip_ch in enumerate(rev_in):
            block_in = skip_ch if i == 0 else prev_ch + skip_ch
            out_ch = decoder_channels[i]
            self.blocks.append(_ConvBNReLU(block_in, out_ch))
            prev_ch = out_ch

        self.head = nn.Conv2d(prev_ch, 1, kernel_size=1)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        rev = list(reversed(feats))  # coarsest -> finest, matches self.blocks order
        x = self.blocks[0](rev[0])
        for i in range(1, len(rev)):
            x = F.interpolate(x, size=rev[i].shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, rev[i]], dim=1)
            x = self.blocks[i](x)
        return self.head(x)
