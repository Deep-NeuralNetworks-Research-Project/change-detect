# Import submodules for their registration side-effects: each module
# registers its encoder key(s) into ENCODER_REGISTRY at import time.
from . import (
    efficientnet,  # noqa: F401,E402
    resnet,  # noqa: F401,E402
)
from .registry import ENCODER_REGISTRY

__all__ = ["ENCODER_REGISTRY"]
