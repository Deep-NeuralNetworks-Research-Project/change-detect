"""Signed-difference fusion family — the project's cheapest novelty claim.

Treat this file as a contribution, not plumbing (per CLAUDE.local.md).

**Why signed, not `|a-b|`:**

`|a-b|` is swap-symmetric for *any* downstream head `h`, because
`|a-b| == |b-a|` identically — you get order-invariance for free, but
appeared-vs-disappeared direction is destroyed in the process.

Plain signed difference `a-b` is *not* swap-symmetric in general.
Swapping input order gives:

    h(b - a) = h(-(a - b))

which equals `h(a - b)` only when `h` is an *even* function. Nothing in
standard supervised training enforces that — weight sharing between the
two Siamese towers buys symmetry only for symmetric fusion ops
(`|a-b|`, `a+b`, `max`), not for signed difference. This is the crux
fact reviewers will look for: it's exactly why the pair-order
consistency loss (P4's contribution) is doing real work here rather than
decorating a property the architecture already had.

**Why add the product term:** `a*b` is a multiplicative/co-occurrence
term neither plain difference nor concatenation captures on its own.
Citable precedent: FCCDN (arXiv:2105.10860), dual sum/difference
branches.

**Framing:** `|a-b|` is order-invariant but destroys direction. `concat`
keeps direction at 2-3x decoder width. Signed `a-b` (+ optionally
product) is the middle ground — cheaper than concat, strictly more
informative than abs-diff.

All four modes must stay selectable by config: P4's ablation B6 sweeps
them.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .registry import FUSION_REGISTRY

_VALID_MODES = ("signed", "signed_product", "concat", "signed_concat")
_OUT_MULTIPLIER = {"signed": 1, "signed_product": 2, "concat": 2, "signed_concat": 3}


@FUSION_REGISTRY.register("signed_fusion")
class SignedFusion(nn.Module):
    """Configurable signed-difference fusion.

    Modes (selected per scale, applied identically at every scale):
      - "signed":         a - b                    (out width = C)
      - "signed_product": concat(a - b, a * b)      (out width = 2C)
      - "concat":          concat(a, b)              (out width = 2C)
      - "signed_concat":   concat(a - b, a, b)        (out width = 3C)

    `in_channels` is the per-scale channel list coming out of fusion's
    two input feature lists (i.e. the encoder's `out_channels`); the
    resulting `out_channels` is what the decoder must be built against.
    """

    def __init__(self, in_channels: list[int], mode: str = "signed_product"):
        super().__init__()
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
        self.mode = mode
        self.in_channels = list(in_channels)
        self.out_channels = [c * _OUT_MULTIPLIER[mode] for c in in_channels]

    def _fuse_one(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.mode == "signed":
            return a - b
        if self.mode == "signed_product":
            return torch.cat([a - b, a * b], dim=1)
        if self.mode == "concat":
            return torch.cat([a, b], dim=1)
        # "signed_concat"
        return torch.cat([a - b, a, b], dim=1)

    def forward(
        self, feats1: list[torch.Tensor], feats2: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        return [self._fuse_one(a, b) for a, b in zip(feats1, feats2)]
