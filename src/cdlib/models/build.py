"""Central builder functions — the ONLY entrypoints the CLI calls.

Frozen contracts:
    build_model(cfg) -> nn.Module
    build_dataset(cfg, split) -> Dataset
    build_loss(cfg) -> Loss
    build_optimizer(cfg, params) -> Optimizer
"""

from __future__ import annotations

from typing import Any, Iterator

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset


def _plain(node: Any) -> dict[str, Any]:
    if node is None:
        return {}
    if isinstance(node, dict):
        return dict(node)
    if OmegaConf.is_config(node):
        resolved = OmegaConf.to_container(node, resolve=True)
        return dict(resolved) if isinstance(resolved, dict) else {}
    return {k: v for k, v in vars(node).items() if not str(k).startswith("_")}


def _kwargs(node: Any, *, drop: tuple[str, ...] = ("name",)) -> dict[str, Any]:
    raw = _plain(node)
    return {k: v for k, v in raw.items() if k not in drop and not str(k).startswith("_")}


def build_model(cfg: DictConfig) -> nn.Module:
    """Build a model from config. Requires ``model.name`` in a registry key."""
    from cdlib.models import import_all_models
    from cdlib.models._model_registry import MODEL_REGISTRY

    import_all_models()

    model_cfg = cfg.model
    model_name = model_cfg.name
    return MODEL_REGISTRY.build(model_name, **_kwargs(model_cfg))


def build_dataset(cfg: DictConfig, split: str) -> Dataset:
    """Delegate to P1's data-layer builder (root resolution + shared augs)."""
    from cdlib.data.registry import build_dataset as _build_dataset

    return _build_dataset(cfg, split)


def build_loss(cfg: DictConfig) -> Any:
    """Build a loss module. Injects ``pos_weight`` from dataset π when present."""
    # Side-effect imports so @register decorators run.
    import cdlib.losses.bce_dice  # noqa: F401
    import cdlib.losses.calibration  # noqa: F401
    import cdlib.losses.pair_order_consistency  # noqa: F401
    from cdlib.data.registry import DATASET_REGISTRY
    from cdlib.losses.registry import LOSS_REGISTRY

    loss_cfg = cfg.loss
    loss_name = loss_cfg.name
    loss_params = _kwargs(loss_cfg)

    if "data" in cfg and getattr(cfg.data, "name", None):
        dataset_cls = DATASET_REGISTRY.get(cfg.data.name)
        ratio = getattr(dataset_cls, "published_changed_pixel_ratio", None)
        if ratio is not None and 0 < ratio < 1:
            loss_params["pos_weight"] = (1.0 - ratio) / ratio

    return LOSS_REGISTRY.build(loss_name, **loss_params)


def build_optimizer(
    cfg: DictConfig, params: Iterator[nn.Parameter]
) -> torch.optim.Optimizer:
    """Build a standard PyTorch optimizer from ``train.optimizer``."""
    optim_cfg = cfg.train.optimizer
    optim_name = str(optim_cfg.name).lower()
    optim_params = _kwargs(optim_cfg)

    optimizers = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }
    if optim_name not in optimizers:
        raise ValueError(
            f"Unknown optimizer '{optim_name}'. Available: {list(optimizers.keys())}"
        )
    return optimizers[optim_name](params, **optim_params)
