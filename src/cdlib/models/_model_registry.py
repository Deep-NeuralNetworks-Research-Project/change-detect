"""MODEL_REGISTRY — top-level model registry for complete CD models.

This is separate from ENCODER_REGISTRY (which holds backbone encoders only).
MODEL_REGISTRY holds complete models (baselines + proposed) that implement
the full forward(img1, img2) -> dict contract.
"""

from cdlib.utils.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
