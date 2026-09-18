"""Pair-order consistency loss (P4).

Binary term: hard-change mask of (I1,I2) should match (I2,I1).
Directional term: appeared/disappeared channels permute under swap.

Requires ``outputs['logits_swapped']`` from a second forward (ProposedModel
emits this in train mode). If it is missing the pair-order term is zero so
baselines can share this loss config.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cdlib.losses.bce_dice import BCEDiceLoss
from cdlib.losses.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("pair_order")
class PairOrderConsistencyLoss(nn.Module):
    def __init__(self, lambda_swap: float = 1.0, directional_weight: float = 1.0) -> None:
        super().__init__()
        self.lambda_swap = float(lambda_swap)
        self.directional_weight = float(directional_weight)

    def compute(self, outputs: dict, batch: dict) -> dict:
        logits = outputs["logits"]
        zero = logits.sum() * 0.0
        swapped = outputs.get("logits_swapped")
        if swapped is None:
            return {"loss": zero, "loss/pairorder": zero, "loss/pairorder_dir": zero}

        valid = batch["mask"] != -1
        p_fwd = torch.sigmoid(logits)
        p_rev = torch.sigmoid(swapped)
        if valid.any():
            binary = (p_fwd - p_rev).abs()[valid].mean()
        else:
            binary = zero

        directional = zero
        aux = outputs.get("aux") or {}
        d_fwd = aux.get("directional_logits")
        d_rev = aux.get("directional_logits_swapped")
        if d_fwd is not None and d_rev is not None:
            # Swap should permute appeared ↔ disappeared (channel 0 ↔ 1).
            permuted = torch.stack([d_rev[:, 1], d_rev[:, 0]], dim=1)
            directional = F.mse_loss(d_fwd, permuted)

        total = self.lambda_swap * (binary + self.directional_weight * directional)
        return {
            "loss": total,
            "loss/pairorder": binary,
            "loss/pairorder_dir": directional,
        }


@LOSS_REGISTRY.register("bce_dice_pairorder")
class BCEDicePairOrderLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pairorder_weight: float = 0.1,
        lambda_swap: float | None = None,
        pos_weight: float | None = None,
        directional_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.seg = BCEDiceLoss(
            bce_weight=bce_weight, dice_weight=dice_weight, pos_weight=pos_weight
        )
        weight = pairorder_weight if lambda_swap is None else lambda_swap
        self.pairorder_weight = float(weight)
        self.pair = PairOrderConsistencyLoss(
            lambda_swap=1.0, directional_weight=directional_weight
        )

    def compute(self, outputs: dict, batch: dict) -> dict:
        seg = self.seg.compute(outputs, batch)
        po = self.pair.compute(outputs, batch)
        loss = seg["loss"] + self.pairorder_weight * po["loss"]
        out = {**seg, **po, "loss": loss}
        return out
