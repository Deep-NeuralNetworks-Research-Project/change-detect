"""Proposed model: encoder + fusion + alignment + decoder + optional heads.

Composes P3 components with P4 alignment/heads. Registered as both
``proposed`` and ``proposed_effnet`` (the Hydra config name).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from cdlib.models._model_registry import MODEL_REGISTRY
from cdlib.models.alignment.registry import ALIGNMENT_REGISTRY
from cdlib.models.decoders.registry import DECODER_REGISTRY
from cdlib.models.encoders.registry import ENCODER_REGISTRY
from cdlib.models.fusion.registry import FUSION_REGISTRY
from cdlib.models.heads.confidence_head import ConfidenceHead
from cdlib.models.heads.directional_head import DirectionalHead


def _plain(node: Any) -> dict[str, Any]:
    if node is None:
        return {}
    if isinstance(node, dict):
        return dict(node)
    if OmegaConf.is_config(node):
        resolved = OmegaConf.to_container(node, resolve=True)
        return dict(resolved) if isinstance(resolved, dict) else {}
    return dict(node)


def _build_encoder(spec: Any) -> nn.Module:
    cfg = _plain(spec) or {"name": "resnet18", "pretrained": False}
    name = str(cfg.pop("name", "resnet18"))
    if "weights" in cfg:
        weights = cfg.pop("weights")
        cfg["pretrained"] = weights not in (None, False, "none", "None")
    cfg.setdefault("pretrained", True)
    return ENCODER_REGISTRY.build(name, **cfg)


class ProposedModel(nn.Module):
    """Siamese CD model matching the frozen forward contract."""

    def __init__(
        self,
        encoder: Any | None = None,
        fusion: Any | None = None,
        alignment: Any | None = None,
        decoder: Any | None = None,
        heads: Any | None = None,
        compute_swap: bool = True,
        **_ignored: object,
    ) -> None:
        super().__init__()
        self.encoder = _build_encoder(encoder)

        fcfg = _plain(fusion) or {"name": "signed_fusion"}
        fname = str(fcfg.pop("name", "signed_fusion"))
        fcfg.pop("in_channels", None)
        self.fusion = FUSION_REGISTRY.build(
            fname, in_channels=self.encoder.out_channels, **fcfg
        )

        acfg = _plain(alignment) or {"name": "identity"}
        aname = str(acfg.pop("name", "identity"))
        acfg.pop("in_channels", None)
        self.alignment = ALIGNMENT_REGISTRY.build(
            aname, in_channels=self.encoder.out_channels, **acfg
        )

        dcfg = _plain(decoder) or {"name": "unet_decoder"}
        dname = str(dcfg.pop("name", "unet_decoder"))
        dcfg.pop("in_channels", None)
        self.decoder = DECODER_REGISTRY.build(
            dname, in_channels=self.fusion.out_channels, **dcfg
        )

        hcfg = _plain(heads)
        self.confidence_head: ConfidenceHead | None = None
        self.directional_head: DirectionalHead | None = None
        if hcfg.get("confidence"):
            self.confidence_head = ConfidenceHead(in_channels=1)
        if hcfg.get("directional"):
            self.directional_head = DirectionalHead(in_channels=self.fusion.out_channels[-1])
        self.compute_swap = bool(compute_swap)

    def _forward_once(
        self, img1: torch.Tensor, img2: torch.Tensor
    ) -> tuple[dict, torch.Tensor]:
        feats1, feats2 = self.encoder.forward_pair(img1, img2)
        feats1, feats2, offset = self.alignment(feats1, feats2)
        fused = self.fusion(feats1, feats2)
        logits = self.decoder(fused)
        if logits.shape[-2:] != img1.shape[-2:]:
            logits = F.interpolate(
                logits, size=img1.shape[-2:], mode="bilinear", align_corners=False
            )
        confidence = None
        if self.confidence_head is not None:
            confidence = self.confidence_head(logits)
            if confidence.shape[-2:] != img1.shape[-2:]:
                confidence = F.interpolate(
                    confidence, size=img1.shape[-2:], mode="bilinear", align_corners=False
                )
        directional = None
        if self.directional_head is not None:
            directional = self.directional_head(fused[-1])
            if directional.shape[-2:] != img1.shape[-2:]:
                directional = F.interpolate(
                    directional, size=img1.shape[-2:], mode="bilinear", align_corners=False
                )
        out = {
            "logits": logits,
            "confidence": confidence,
            "aux": {
                "directional_logits": directional,
                "alignment_offset": offset,
            },
        }
        return out, offset

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> dict:
        out, _ = self._forward_once(img1, img2)
        if self.training and self.compute_swap:
            rev, _ = self._forward_once(img2, img1)
            out["logits_swapped"] = rev["logits"]
            if rev["aux"]["directional_logits"] is not None:
                out["aux"]["directional_logits_swapped"] = rev["aux"]["directional_logits"]
        return out


MODEL_REGISTRY.register_explicit("proposed", ProposedModel)
MODEL_REGISTRY.register_explicit("proposed_effnet", ProposedModel)
