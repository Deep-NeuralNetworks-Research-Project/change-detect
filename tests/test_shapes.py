"""Test model output shapes — verify frozen forward contract.

Uses small synthetic tensors (batch=2, H=W=32) for fast CPU testing.
"""

from __future__ import annotations

import torch

# Contract: forward(img1, img2) -> {"logits": [B,1,H,W], "confidence": ..., "aux": {...}}
BATCH = 2
CHANNELS = 3
HEIGHT = 32
WIDTH = 32


def _make_dummy_inputs():
    """Create synthetic inputs matching the dataset contract."""
    return (
        torch.rand(BATCH, CHANNELS, HEIGHT, WIDTH),
        torch.rand(BATCH, CHANNELS, HEIGHT, WIDTH),
    )


class TestRGBSSIMShapes:
    """Verify RGB/SSIM baseline output shapes match the frozen contract."""

    def setup_method(self):
        from cdlib.models._model_registry import MODEL_REGISTRY
        from cdlib.models.baselines import _import_all_baselines

        _import_all_baselines()
        self.model = MODEL_REGISTRY.build("rgb_ssim")
        self.model.eval()

    def test_output_is_dict(self):
        img1, img2 = _make_dummy_inputs()
        out = self.model(img1, img2)
        assert isinstance(out, dict), f"Expected dict, got {type(out)}"

    def test_logits_shape(self):
        img1, img2 = _make_dummy_inputs()
        out = self.model(img1, img2)
        assert "logits" in out, "Missing 'logits' key"
        assert out["logits"].shape == (BATCH, 1, HEIGHT, WIDTH), (
            f"Expected logits shape {(BATCH, 1, HEIGHT, WIDTH)}, "
            f"got {out['logits'].shape}"
        )

    def test_confidence_key_exists(self):
        img1, img2 = _make_dummy_inputs()
        out = self.model(img1, img2)
        assert "confidence" in out, "Missing 'confidence' key"

    def test_aux_keys_exist(self):
        img1, img2 = _make_dummy_inputs()
        out = self.model(img1, img2)
        assert "aux" in out, "Missing 'aux' key"
        assert "directional_logits" in out["aux"], "Missing 'aux.directional_logits'"
        assert "alignment_offset" in out["aux"], "Missing 'aux.alignment_offset'"


class TestFCSiamDiffShapes:
    """Verify FC-Siam-Diff baseline output shapes match the frozen contract."""

    def setup_method(self):
        from cdlib.models._model_registry import MODEL_REGISTRY
        from cdlib.models.baselines import _import_all_baselines

        _import_all_baselines()
        self.model = MODEL_REGISTRY.build(
            "fc_siam_diff",
            encoder_name="resnet18",
            encoder_weights=None,  # Don't download weights in CI
            in_channels=CHANNELS,
        )
        self.model.eval()

    def test_output_is_dict(self):
        img1, img2 = _make_dummy_inputs()
        out = self.model(img1, img2)
        assert isinstance(out, dict), f"Expected dict, got {type(out)}"

    def test_logits_shape(self):
        img1, img2 = _make_dummy_inputs()
        out = self.model(img1, img2)
        assert "logits" in out, "Missing 'logits' key"
        # FCSiamDiff may output slightly different spatial dims due to encoder
        B, C_out, H_out, W_out = out["logits"].shape
        assert B == BATCH, f"Expected batch={BATCH}, got {B}"
        assert C_out == 1, f"Expected 1 output channel, got {C_out}"
        # H,W should match input (FCN is fully convolutional)
        assert H_out == HEIGHT, f"Expected H={HEIGHT}, got {H_out}"
        assert W_out == WIDTH, f"Expected W={WIDTH}, got {W_out}"

    def test_confidence_key_exists(self):
        img1, img2 = _make_dummy_inputs()
        out = self.model(img1, img2)
        assert "confidence" in out, "Missing 'confidence' key"
        # Baselines have confidence=None
        assert out["confidence"] is None, "Baseline should have confidence=None"

    def test_aux_keys_exist(self):
        img1, img2 = _make_dummy_inputs()
        out = self.model(img1, img2)
        assert "aux" in out, "Missing 'aux' key"
        assert isinstance(out["aux"], dict), "aux must be a dict"
        assert "directional_logits" in out["aux"]
        assert "alignment_offset" in out["aux"]
