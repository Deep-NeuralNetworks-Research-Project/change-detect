"""Tiny two-input CNN so benchmark/evaluate run without P2/P3 models.

Not a baseline. Replace with ``build_model(cfg)`` when the team monorepo lands.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DummyPairCNN(nn.Module):
    def __init__(self, in_ch: int = 3, hidden: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(hidden, 1, 1)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> dict:
        # Siamese-BN pattern: concat on batch, one forward, split.
        b = img1.shape[0]
        x = torch.cat([img1, img2], dim=0)
        feat = self.encoder(x)
        f1, f2 = feat[:b], feat[b:]
        logits = self.head(torch.abs(f1 - f2))
        return {
            "logits": logits,
            "confidence": None,
            "aux": {"directional_logits": None, "alignment_offset": None},
        }
