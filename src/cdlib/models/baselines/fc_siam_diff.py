"""FC-Siam-Diff baseline via TorchGeo.

Uses TorchGeo's FCSiamDiff which wraps segmentation_models_pytorch,
exposing encoder_name for backbone swapping. This means one code path
serves the mandatory baseline, P3's ResNet-18 baseline, and P4's
proposed model.

IMPORTANT CAVEATS:
- This is NOT architecturally identical to Daudt's original 4-stage
  16/32/64/128 net. Note the substitution in the paper.
- Siamese BN: concatenate [I1;I2] along batch dim into one forward pass.
- The original paper gives NO training recipe (no optimizer, LR, batch
  size, epochs). Budget real tuning time.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cdlib.models._model_registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("fc_siam_diff")
class FCSiamDiffWrapper(nn.Module):
    """Wrapper around TorchGeo's FCSiamDiff matching the frozen forward contract.

    Args:
        encoder_name: SMP/timm backbone name (e.g., "resnet18", "efficientnet-b0").
        encoder_weights: Pretrained weights (e.g., "imagenet" or None).
        in_channels: Number of input channels per image (default: 3 for RGB).
    """

    def __init__(
        self,
        encoder_name: str = "resnet18",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
    ) -> None:
        super().__init__()

        # Lazy import to avoid hard dependency at module load time
        try:
            from torchgeo.models import FCSiamDiff
        except ImportError:
            raise ImportError(
                "TorchGeo is required for FCSiamDiff. "
                "Install with: pip install torchgeo"
            )

        self.model = FCSiamDiff(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
        )

    def forward(
        self, img1: torch.Tensor, img2: torch.Tensor
    ) -> dict[str, torch.Tensor | None | dict]:
        """Forward pass matching the frozen model contract.

        TorchGeo v0.6+ expects a single stacked tensor of shape
        [B, T, C, H, W] where T=2 (two temporal images).
        """
        # Stack into [B, 2, C, H, W] for TorchGeo's API
        x = torch.stack([img1, img2], dim=1)  # [B, 2, C, H, W]
        logits = self.model(x)  # [B, 1, H, W]

        return {
            "logits": logits,
            "confidence": None,
            "aux": {
                "directional_logits": None,
                "alignment_offset": None,
            },
        }

