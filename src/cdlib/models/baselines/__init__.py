"""Baseline model imports — triggers registration in MODEL_REGISTRY."""

_imported = False


def _import_all_baselines() -> None:
    """Import all baseline modules to trigger their @register decorators."""
    global _imported
    if _imported:
        return

    from cdlib.models.baselines import fc_siam_diff as _fc_siam  # noqa: F401
    from cdlib.models.baselines import rgb_ssim as _rgb_ssim  # noqa: F401
    from cdlib.models.baselines import siamese_resnet18 as _siamese  # noqa: F401

    _imported = True
