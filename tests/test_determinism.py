"""Test determinism — same seed must produce same results.

Runs two forward passes with the same seed and asserts logits match
within floating-point tolerance.
"""

from __future__ import annotations

import torch

from cdlib.utils.seed import set_seed


def _run_forward_with_seed(seed: int):
    """Run a forward pass with a specific seed, return logits."""
    from cdlib.models._model_registry import MODEL_REGISTRY
    from cdlib.models.baselines import _import_all_baselines

    set_seed(seed, deterministic=True)
    _import_all_baselines()

    model = MODEL_REGISTRY.build(
        "fc_siam_diff",
        encoder_name="resnet18",
        encoder_weights=None,  # Don't download weights in tests
        in_channels=3,
    )
    model.eval()

    # Fixed random input (deterministic because we seeded)
    img1 = torch.rand(2, 3, 32, 32)
    img2 = torch.rand(2, 3, 32, 32)

    with torch.no_grad():
        out = model(img1, img2)

    return out["logits"]


def test_same_seed_same_output():
    """Two runs with the same seed must produce identical logits."""
    logits_a = _run_forward_with_seed(42)
    logits_b = _run_forward_with_seed(42)

    assert torch.allclose(logits_a, logits_b, atol=1e-6), (
        f"Logits differ with same seed. "
        f"Max diff: {(logits_a - logits_b).abs().max().item():.2e}"
    )


def test_different_seed_different_output():
    """Two runs with different seeds should (almost certainly) differ."""
    logits_a = _run_forward_with_seed(42)
    logits_b = _run_forward_with_seed(123)

    # These should differ — if they don't, seeding is broken
    assert not torch.allclose(logits_a, logits_b, atol=1e-4), (
        "Logits are identical with different seeds — seeding may be broken."
    )


def test_rng_state_save_restore():
    """Saving and restoring RNG state must produce identical random tensors."""
    from cdlib.utils.seed import get_rng_state, set_rng_state

    set_seed(42)
    state = get_rng_state()

    # Generate some random values
    t1 = torch.rand(10)

    # Restore state and regenerate
    set_rng_state(state)
    t2 = torch.rand(10)

    assert torch.equal(t1, t2), "RNG state restore did not reproduce same values"
