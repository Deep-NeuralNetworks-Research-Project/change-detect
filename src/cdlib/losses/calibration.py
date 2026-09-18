"""Calibration auxiliary term (P4).

Dice hurts calibration (Mehrtash TMI 2020). The default fix is P5's
post-hoc temperature scaling on shift-representative data. This module
is a placeholder regulariser so the loss registry and CODEOWNERS path
exist; it currently returns zero and must not be treated as a result.
"""

from __future__ import annotations

import torch.nn as nn

from cdlib.losses.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("calibration")
class CalibrationAuxLoss(nn.Module):
    def compute(self, outputs: dict, batch: dict) -> dict:
        zero = outputs["logits"].sum() * 0.0
        return {"loss": zero, "loss/calibration": zero}
