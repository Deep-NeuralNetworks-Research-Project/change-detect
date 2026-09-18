"""Deterministic seeding for reproducibility.

Seeds python, numpy, torch, and CUDA. Enables deterministic algorithms
where feasible. Document any non-deterministic ops in MODEL_CARD.md.

Usage:
    from cdlib.utils.seed import set_seed
    set_seed(42)
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: The seed value.
        deterministic: If True, enable torch.use_deterministic_algorithms.
            May cause errors with some ops (e.g., some scatter ops). Set False
            if you hit compatibility issues and document in MODEL_CARD.md.
    """
    # Python stdlib
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        # CUBLAS workspace config for deterministic CUDA ops
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Disable benchmark mode for deterministic convolutions
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_rng_state() -> dict:
    """Capture all RNG states for checkpoint/resume.

    Returns:
        Dict containing python, numpy, torch, and cuda RNG states.
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return state


def set_rng_state(state: dict) -> None:
    """Restore RNG states from checkpoint for deterministic resume.

    Args:
        state: Dict from get_rng_state().
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])

    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
