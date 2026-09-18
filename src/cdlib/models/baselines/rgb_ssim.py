"""RGB/SSIM non-DNN baseline — the floor for 'why we need learning'.

Threshold SSIM(I1,I2) < T and |I1-I2| > T. Expect it to fail on exactly
the nuisance list (camera displacement, lighting, grading, blur, codec).
That failure IS the narrative.

~4h of work. Zero training cost.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cdlib.models._model_registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("rgb_ssim")
class RGBSSIMBaseline(nn.Module):
    """Non-DNN change detection baseline using pixel difference and SSIM.

    This is a non-trainable baseline. The forward pass computes change masks
    using simple thresholding on pixel differences and/or SSIM scores.
    """

    def __init__(self, diff_threshold: float = 0.3, ssim_threshold: float = 0.5) -> None:
        super().__init__()
        self.diff_threshold = diff_threshold
        self.ssim_threshold = ssim_threshold

    def forward(
        self, img1: torch.Tensor, img2: torch.Tensor
    ) -> dict[str, torch.Tensor | None | dict]:
        """Compute change mask via pixel difference thresholding.

        Matches the frozen model forward contract.
        """
        B, C, H, W = img1.shape

        # Simple absolute difference across channels, then mean
        diff = torch.abs(img1 - img2).mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # Logits: positive = change, negative = no change
        # Shift so threshold becomes the decision boundary at 0
        logits = diff - self.diff_threshold  # [B, 1, H, W]

        return {
            "logits": logits,
            "confidence": None,
            "aux": {
                "directional_logits": None,
                "alignment_offset": None,
            },
        }
