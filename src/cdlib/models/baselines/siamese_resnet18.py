"""The mandatory "stronger baseline": Siamese ResNet-18 encoder +
abs-diff fusion + U-Net decoder.

Per "A Change Detection Reality Check" (Corley et al., arXiv:2402.06994),
a *properly tuned* plain-recipe baseline like this one may already rival
BIT on LEVIR-CD — that would be a legitimate, publishable result, not a
failure. Give it a real LR sweep and >=3 seeds (see checklist), don't
treat it as a strawman.

Satisfies the frozen model `forward` contract from CLAUDE.md:
    {"logits": [B,1,H,W], "confidence": ... | None, "aux": {...}}
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cdlib.models._model_registry import MODEL_REGISTRY
from cdlib.models.decoders.unet_decoder import UNetDecoder
from cdlib.models.encoders.resnet import ResNet18Encoder
from cdlib.models.fusion.absdiff import AbsDiffFusion


@MODEL_REGISTRY.register("siamese_resnet18")
class SiameseResNet18(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        use_c5: bool = True,
        dilate_last: bool = False,
        decoder_channels: list[int] | None = None,
    ):
        super().__init__()
        self.encoder = ResNet18Encoder(pretrained=pretrained, use_c5=use_c5, dilate_last=dilate_last)
        self.fusion = AbsDiffFusion(in_channels=self.encoder.out_channels)
        self.decoder = UNetDecoder(in_channels=self.fusion.out_channels, decoder_channels=decoder_channels)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> dict:
        feats1, feats2 = self.encoder.forward_pair(img1, img2)
        fused = self.fusion(feats1, feats2)
        logits = self.decoder(fused)

        if logits.shape[-2:] != img1.shape[-2:]:
            logits = F.interpolate(logits, size=img1.shape[-2:], mode="bilinear", align_corners=False)

        return {
            "logits": logits,
            "confidence": None,
            "aux": {"directional_logits": None, "alignment_offset": None},
        }
