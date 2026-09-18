"""Siamese ResNet-18 encoder.

Taps C2-C5 (layer1-layer4). Backbone is 11.7M params, ~1.8 GFLOPs @
224x224 (research/01-baselines-architectures.md §2).

ImageNet normalisation (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
is genuinely correct for this project: we use natural video RGB, not
satellite radiometry, so unlike most change-detection literature we don't
need to caveat the pretrained-norm mismatch. P1's data pipeline should
use IMAGENET_MEAN / IMAGENET_STD below.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights

from .registry import ENCODER_REGISTRY

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _dilate_stage(layer: nn.Sequential, dilation: int = 2) -> None:
    """In-place: convert a ResNet stage from stride-2 to stride-1 +
    dilated ("atrous"), the classic DeepLab-style trick for recovering
    resolution in dense prediction.

    Not done via torchvision's `replace_stride_with_dilation`: that
    kwarg raises `NotImplementedError: Dilation > 1 not supported in
    BasicBlock` for ResNet-18/34 specifically (it only supports
    Bottleneck-based nets like ResNet-50+). The restriction is
    torchvision's API, not an actual architectural limit — a plain 3x3
    conv dilates fine regardless of block type — so we do the
    conversion by hand: drop every stride-2 conv in the stage to
    stride-1 (main path's first 3x3 conv, and the downsample's 1x1
    conv), then dilate every 3x3 conv in the stage so padding=dilation
    keeps spatial size unchanged (receptive field preserved, no
    resolution loss).
    """
    for block in layer:
        for module in block.modules():
            if not isinstance(module, nn.Conv2d):
                continue
            if module.stride == (2, 2):
                module.stride = (1, 1)
            if module.kernel_size == (3, 3):
                module.dilation = (dilation, dilation)
                module.padding = (dilation, dilation)


@ENCODER_REGISTRY.register("resnet18")
class ResNet18Encoder(nn.Module):
    """Siamese ResNet-18 encoder tapping C2 (stride 4, 64ch) through
    C5 (stride 32, 512ch).

    Stride-32 is too coarse for thin change boundaries (standard finding
    in the CD literature: Changer, SARAS-Net). Two ablatable escape
    hatches, both exposed as config flags per the P3 checklist:
      - `use_c5=False`  -> drop C5, decode from C2-C4 only.
      - `dilate_last=True` -> keep 4 stages but make layer4 stride-1/
        dilated (atrous), so C5 is emitted at stride 16 instead of 32.
    Both cost FLOPs and recover thin structures; ablate, don't just pick
    one (see the stride-32 ablation in the Weeks 5+ checklist).

    Use `forward_pair`, not two sequential `forward` calls, for the
    Siamese pass: BatchNorm sees a consistent 2N-sized batch that way,
    instead of two N-sized batches with asymmetrically mixed running
    stats.
    """

    def __init__(self, pretrained: bool = True, use_c5: bool = True, dilate_last: bool = False):
        super().__init__()
        self.use_c5 = use_c5
        self.dilate_last = dilate_last

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet18(weights=weights)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1  # C2, stride 4,  64ch
        self.layer2 = net.layer2  # C3, stride 8,  128ch
        self.layer3 = net.layer3  # C4, stride 16, 256ch
        self.layer4 = net.layer4  # C5, stride 32 (or 16 if dilate_last), 512ch

        if dilate_last:
            _dilate_stage(self.layer4, dilation=2)  # weight shapes unaffected, pretrained-safe

        strides = [4, 8, 16, 16 if dilate_last else 32]
        channels = [64, 128, 256, 512]
        if not use_c5:
            strides, channels = strides[:3], channels[:3]
        self.out_strides = strides
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        feats = [c2, c3, c4]
        if self.use_c5:
            feats.append(self.layer4(c4))
        return feats

    def forward_pair(
        self, img1: torch.Tensor, img2: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Siamese BN pattern: concat [I1;I2] along the batch dim, one
        forward pass, then split — never two sequential forward() calls.
        """
        b = img1.shape[0]
        x = torch.cat([img1, img2], dim=0)
        feats = self.forward(x)
        feats1 = [f[:b] for f in feats]
        feats2 = [f[b:] for f in feats]
        return feats1, feats2
