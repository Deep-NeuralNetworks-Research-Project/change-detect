"""Shape tests for the P3 component set: encoders, fusion, decoder,
and the Siamese ResNet-18 baseline.

Parametrized over every registry key, per research/06's test strategy.
Uses `pretrained=False` and small tensors so this runs fast on CPU/CI.
"""
from __future__ import annotations

import pytest
import torch

from cdlib.models.baselines.siamese_resnet18 import SiameseResNet18
from cdlib.models.decoders.unet_decoder import UNetDecoder
from cdlib.models.encoders import ENCODER_REGISTRY
from cdlib.models.fusion import FUSION_REGISTRY

B, H, W = 2, 64, 64


@pytest.mark.parametrize("key", ENCODER_REGISTRY.keys())
@pytest.mark.parametrize("use_c5", [True, False])
def test_encoder_shapes(key, use_c5):
    enc = ENCODER_REGISTRY.build(key, pretrained=False, use_c5=use_c5)
    img1 = torch.rand(B, 3, H, W)
    img2 = torch.rand(B, 3, H, W)

    feats1, feats2 = enc.forward_pair(img1, img2)

    assert len(feats1) == len(feats2) == len(enc.out_channels) == len(enc.out_strides)
    for f1, f2, c, s in zip(feats1, feats2, enc.out_channels, enc.out_strides):
        expected = (B, c, H // s, W // s)
        assert f1.shape == expected, f"{key}: {f1.shape} != {expected}"
        assert f2.shape == expected, f"{key}: {f2.shape} != {expected}"


def test_resnet18_dilate_last_keeps_c5_at_stride16():
    enc = ENCODER_REGISTRY.build("resnet18", pretrained=False, dilate_last=True)
    assert enc.out_strides == [4, 8, 16, 16]
    img1 = torch.rand(B, 3, H, W)
    img2 = torch.rand(B, 3, H, W)
    feats1, _ = enc.forward_pair(img1, img2)
    assert feats1[-1].shape[-2:] == (H // 16, W // 16)


@pytest.mark.parametrize("key", FUSION_REGISTRY.keys())
def test_fusion_shapes(key):
    in_channels = [64, 128, 256, 512]
    kwargs = {"mode": "signed_concat"} if key == "signed_fusion" else {}
    fusion = FUSION_REGISTRY.build(key, in_channels=in_channels, **kwargs)

    feats1 = [torch.rand(B, c, 8, 8) for c in in_channels]
    feats2 = [torch.rand(B, c, 8, 8) for c in in_channels]
    fused = fusion(feats1, feats2)

    assert len(fused) == len(in_channels) == len(fusion.out_channels)
    for f, c in zip(fused, fusion.out_channels):
        assert f.shape == (B, c, 8, 8)


@pytest.mark.parametrize("mode", ["signed", "signed_product", "concat", "signed_concat"])
def test_signed_fusion_all_modes_selectable(mode):
    fusion = FUSION_REGISTRY.build("signed_fusion", in_channels=[16], mode=mode)
    a = torch.rand(B, 16, 4, 4)
    b = torch.rand(B, 16, 4, 4)
    out = fusion([a], [b])[0]
    assert out.shape[1] == fusion.out_channels[0]


def test_absdiff_is_order_invariant():
    fusion = FUSION_REGISTRY.build("absdiff", in_channels=[16])
    a, b = torch.rand(B, 16, 4, 4), torch.rand(B, 16, 4, 4)
    torch.testing.assert_close(fusion([a], [b])[0], fusion([b], [a])[0])


def test_signed_fusion_breaks_swap_symmetry():
    """The crux fact from CLAUDE.md: h(b-a) == h(a-b) only if h is even.
    Plain signed diff is anti-symmetric under swap — verify that here so
    nobody "fixes" it back into abs-diff by accident.
    """
    fusion = FUSION_REGISTRY.build("signed_fusion", in_channels=[16], mode="signed")
    a, b = torch.rand(B, 16, 4, 4), torch.rand(B, 16, 4, 4)
    out_ab = fusion([a], [b])[0]
    out_ba = fusion([b], [a])[0]
    assert not torch.allclose(out_ab, out_ba)
    torch.testing.assert_close(out_ab, -out_ba)


def test_decoder_shape():
    in_channels = [64, 128, 256, 512]
    strides = [4, 8, 16, 32]
    decoder = UNetDecoder(in_channels=in_channels)
    feats = [torch.rand(B, c, H // s, W // s) for c, s in zip(in_channels, strides)]
    logits = decoder(feats)
    assert logits.shape == (B, 1, H // 4, W // 4)  # finest skip's resolution


def test_decoder_handles_three_levels_when_c5_dropped():
    in_channels = [64, 128, 256]
    strides = [4, 8, 16]
    decoder = UNetDecoder(in_channels=in_channels)
    feats = [torch.rand(B, c, H // s, W // s) for c, s in zip(in_channels, strides)]
    logits = decoder(feats)
    assert logits.shape == (B, 1, H // 4, W // 4)


def test_siamese_resnet18_forward_contract():
    model = SiameseResNet18(pretrained=False)
    img1 = torch.rand(B, 3, H, W)
    img2 = torch.rand(B, 3, H, W)

    out = model(img1, img2)

    assert set(out.keys()) == {"logits", "confidence", "aux"}
    assert out["logits"].shape == (B, 1, H, W)  # full input resolution
    assert out["confidence"] is None
    assert out["aux"] == {"directional_logits": None, "alignment_offset": None}


def test_siamese_resnet18_gradients_flow():
    model = SiameseResNet18(pretrained=False)
    img1 = torch.rand(B, 3, H, W, requires_grad=False)
    img2 = torch.rand(B, 3, H, W, requires_grad=False)
    out = model(img1, img2)
    loss = out["logits"].sum()
    loss.backward()
    grad_norms = [p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0 and all(g >= 0 for g in grad_norms)
