"""BCE + Dice loss for semantic change detection.

WARNING: Dice hurts calibration (Mehrtash et al., IEEE TMI 2020).
P5 will decide on temperature scaling or a calibration-aware auxiliary term
before finalisation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cdlib.losses.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("bce_dice")
class BCEDiceLoss(nn.Module):
    """Combination of Binary Cross-Entropy and Soft Dice Loss.

    Args:
        bce_weight: Weight for the BCE term.
        dice_weight: Weight for the Dice term.
        pos_weight: Weight for the positive class in BCE loss.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        pos_weight_tensor = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight_tensor)

    def compute(self, outputs: dict, batch: dict) -> dict:
        """Compute the loss according to the frozen contract.

        Ignores pixels where mask == -1.
        """
        logits = outputs["logits"]  # [B, 1, H, W]
        mask = batch["mask"]  # [B, 1, H, W], {0, 1, -1}

        # Ignore regions (-1)
        valid_mask = mask != -1

        if not valid_mask.any():
            zero = logits.sum() * 0.0
            return {"loss": zero, "loss/bce": zero, "loss/dice": zero}

        # Flatten valid regions
        valid_logits = logits[valid_mask]
        valid_targets = mask[valid_mask].float()

        # BCE Loss
        # nn.BCEWithLogitsLoss with reduction='none' allows us to only average over valid
        bce_loss_all = self.bce(valid_logits, valid_targets)
        loss_bce = bce_loss_all.mean()

        # Soft Dice Loss
        probs = torch.sigmoid(logits)
        valid_probs = probs[valid_mask]

        intersection = (valid_probs * valid_targets).sum()
        union = valid_probs.sum() + valid_targets.sum()
        loss_dice = 1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)

        loss = self.bce_weight * loss_bce + self.dice_weight * loss_dice

        return {
            "loss": loss,
            "loss/bce": loss_bce,
            "loss/dice": loss_dice,
        }
