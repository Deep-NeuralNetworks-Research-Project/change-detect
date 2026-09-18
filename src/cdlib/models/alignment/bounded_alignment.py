"""Bounded change-aware alignment of Siamese feature maps.

Offset is ``tanh(·) * max_disp`` (hard bound). The gate is derived from a
*decoupled* signal (abs-diff magnitude), not from the change head, so we
do not train a circular collapse (research/03, P4 role A4 vs A5).

``grid_sample`` uses ``align_corners=True`` and ``padding_mode='zeros'``.
The last offset conv is zero-init so the module starts at identity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cdlib.models.alignment.registry import ALIGNMENT_REGISTRY


def _warp(feat: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Warp ``feat`` by a pixel-space offset map ``[B, 2, H, W]`` (dx, dy)."""
    b, _, h, w = feat.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=feat.device, dtype=feat.dtype),
        torch.linspace(-1.0, 1.0, w, device=feat.device, dtype=feat.dtype),
        indexing="ij",
    )
    base = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(b, -1, -1, -1)
    # offset is in pixels; convert to normalised grid units
    scale = torch.tensor([2.0 / max(w - 1, 1), 2.0 / max(h - 1, 1)], device=feat.device, dtype=feat.dtype)
    scale = scale.view(1, 2, 1, 1)
    grid = (base + offset * scale).permute(0, 2, 3, 1)
    return F.grid_sample(
        feat, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )


@ALIGNMENT_REGISTRY.register("bounded")
class BoundedAlignment(nn.Module):
    """Per-scale offset head + optional abs-diff gate, applied to ``feats2``."""

    def __init__(
        self,
        in_channels: list[int] | None = None,
        max_disp: float = 4.0,
        max_offset: float | None = None,
        bounded: bool = True,
        gated: bool = True,
        gate_source: str = "decoupled",
    ) -> None:
        super().__init__()
        if in_channels is None:
            in_channels = [64, 128, 256, 512]
        self.in_channels = list(in_channels)
        self.max_disp = float(max_disp if max_offset is None else max_offset)
        self.bounded = bool(bounded)
        self.gated = bool(gated)
        if gate_source not in ("decoupled", "own_logits"):
            raise ValueError(f"gate_source must be decoupled|own_logits, got {gate_source!r}")
        self.gate_source = gate_source

        self.offset_heads = nn.ModuleList()
        self.gate_heads = nn.ModuleList()
        for c in self.in_channels:
            head = nn.Sequential(
                nn.Conv2d(2 * c, c, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, 2, kernel_size=3, padding=1, bias=True),
            )
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            self.offset_heads.append(head)
            ghead = nn.Conv2d(c, 1, kernel_size=1)
            nn.init.zeros_(ghead.weight)
            nn.init.constant_(ghead.bias, 2.0)  # sigmoid(2) ≈ 0.88, near-pass-through
            self.gate_heads.append(ghead)

    def forward(
        self, feats1: list[torch.Tensor], feats2: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        aligned2: list[torch.Tensor] = []
        last_offset: torch.Tensor | None = None
        for f1, f2, ohead, ghead in zip(
            feats1, feats2, self.offset_heads, self.gate_heads, strict=True
        ):
            raw = ohead(torch.cat([f1, f2], dim=1))
            if self.bounded:
                offset = torch.tanh(raw) * self.max_disp
            else:
                offset = raw
            warped = _warp(f2, offset)
            if self.gated:
                if self.gate_source == "decoupled":
                    gate = torch.sigmoid(ghead(torch.abs(f1 - f2)))
                else:
                    gate = torch.sigmoid(ghead(f1))
                warped = gate * warped + (1.0 - gate) * f2
            aligned2.append(warped)
            last_offset = offset
        assert last_offset is not None
        return feats1, aligned2, last_offset
