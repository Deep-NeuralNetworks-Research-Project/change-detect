"""Models package. Importing this does not auto-register components.

Call :func:`import_all_models` (or go through :func:`cdlib.models.build.build_model`)
so every ``@*_REGISTRY.register`` decorator has run.
"""

from cdlib.models._model_registry import MODEL_REGISTRY


def import_all_models() -> None:
    """Import every model-side module so registry decorators fire."""
    from cdlib.models.alignment import ALIGNMENT_REGISTRY  # noqa: F401
    from cdlib.models.baselines import _import_all_baselines
    from cdlib.models.decoders import DECODER_REGISTRY  # noqa: F401
    from cdlib.models.encoders import ENCODER_REGISTRY  # noqa: F401
    from cdlib.models.fusion import FUSION_REGISTRY  # noqa: F401

    _import_all_baselines()
    from cdlib.models import proposed as _proposed  # noqa: F401


__all__ = ["MODEL_REGISTRY", "import_all_models"]
