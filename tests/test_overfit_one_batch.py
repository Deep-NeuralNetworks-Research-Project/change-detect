"""Test overfit on one batch — verify training actually reduces loss.

Uses FC-Siam-Diff with a tiny synthetic batch (2 pairs, 32x32).
Runs ~50 steps and asserts loss drops significantly.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


# Minimal BCEDice loss for testing (matches the loss contract)
class _TestBCEDiceLoss(nn.Module):
    """Minimal loss matching the frozen contract for testing."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def compute(self, outputs: dict, batch: dict) -> dict:
        logits = outputs["logits"]
        mask = batch["mask"]

        # Handle ignore regions (mask == -1)
        valid = (mask >= 0).float()
        logits_masked = logits * valid
        mask_masked = mask.clamp(min=0) * valid

        bce_loss = self.bce(logits_masked, mask_masked)
        return {
            "loss": bce_loss,
            "loss/bce": bce_loss,
        }


def _make_synthetic_batch(batch_size=2, img_size=32, in_channels=3):
    """Create a synthetic batch that has a learnable pattern."""
    # Create images where img2 has a bright square that img1 doesn't
    img1 = torch.rand(batch_size, in_channels, img_size, img_size) * 0.3
    img2 = img1.clone()
    mask = torch.zeros(batch_size, 1, img_size, img_size)

    # Add a bright square to img2 in a known region
    h_start, h_end = img_size // 4, 3 * img_size // 4
    w_start, w_end = img_size // 4, 3 * img_size // 4
    img2[:, :, h_start:h_end, w_start:w_end] += 0.7
    mask[:, :, h_start:h_end, w_start:w_end] = 1.0

    return {
        "img1": img1,
        "img2": img2,
        "mask": mask,
        "nuisance_label": torch.zeros(batch_size, dtype=torch.int64),
    }


@pytest.mark.slow
def test_overfit_one_batch_fcsiamdiff():
    """FC-Siam-Diff must overfit a single batch — loss must drop by >50%."""
    from cdlib.models._model_registry import MODEL_REGISTRY
    from cdlib.models.baselines import _import_all_baselines

    _import_all_baselines()

    model = MODEL_REGISTRY.build(
        "fc_siam_diff",
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
    )
    model.train()

    loss_fn = _TestBCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch = _make_synthetic_batch(batch_size=2, img_size=32)

    # Record initial loss
    with torch.no_grad():
        out = model(batch["img1"], batch["img2"])
        initial_loss = loss_fn.compute(out, batch)["loss"].item()

    # Train for 50 steps on the same batch
    for step in range(50):
        optimizer.zero_grad()
        out = model(batch["img1"], batch["img2"])
        loss_dict = loss_fn.compute(out, batch)
        loss_dict["loss"].backward()
        optimizer.step()

    # Final loss
    with torch.no_grad():
        out = model(batch["img1"], batch["img2"])
        final_loss = loss_fn.compute(out, batch)["loss"].item()

    # Loss must drop significantly
    assert final_loss < initial_loss * 0.5, (
        f"Loss did not drop enough. Initial: {initial_loss:.4f}, "
        f"Final: {final_loss:.4f}. Expected >50% reduction."
    )


def test_overfit_one_batch_rgb_ssim_no_crash():
    """RGB/SSIM is non-trainable — just verify forward + loss doesn't crash."""
    from cdlib.models._model_registry import MODEL_REGISTRY
    from cdlib.models.baselines import _import_all_baselines

    _import_all_baselines()

    model = MODEL_REGISTRY.build("rgb_ssim")
    loss_fn = _TestBCEDiceLoss()

    batch = _make_synthetic_batch(batch_size=2, img_size=32)
    out = model(batch["img1"], batch["img2"])
    loss_dict = loss_fn.compute(out, batch)

    assert "loss" in loss_dict
    assert loss_dict["loss"].ndim == 0, "Loss must be a scalar"
    assert not torch.isnan(loss_dict["loss"]), "Loss is NaN"
