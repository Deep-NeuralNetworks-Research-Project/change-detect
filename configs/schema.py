"""Light structured-config dataclasses for CI typo-catching only.

These are NOT used for runtime config resolution — Hydra YAML is the
source of truth. These dataclasses validate that YAML files contain
expected keys, catching typos before they hit training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DataConfig:
    name: str = "sysu_cd"
    root: str = "./data/sysu_cd"
    img_size: int = 256
    num_workers: int = 4
    crop_size: Optional[int] = None
    subset: Optional[str] = None


@dataclass
class ModelConfig:
    name: str = "fc_siam_diff"
    encoder_name: Optional[str] = None
    encoder_weights: Optional[str] = None
    in_channels: int = 3

    # Proposed model sub-configs (ignored for baselines)
    encoder: Optional[dict[str, Any]] = None
    fusion: Optional[dict[str, Any]] = None
    alignment: Optional[dict[str, Any]] = None
    decoder: Optional[dict[str, Any]] = None
    heads: Optional[dict[str, Any]] = None

    # Non-DNN baseline params
    diff_threshold: Optional[float] = None
    ssim_threshold: Optional[float] = None


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.01


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    T_max: int = 200
    eta_min: float = 1e-6


@dataclass
class CheckpointConfig:
    interval_minutes: int = 15
    keep_last: int = 2
    keep_best: int = 1
    resume_from: Optional[str] = None


@dataclass
class WandbConfig:
    project: str = "cd-project"
    entity: Optional[str] = None
    log_interval: int = 10


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 200
    batch_size: int = 8
    val_interval: int = 1
    amp: bool = True
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class LossConfig:
    name: str = "bce_dice"
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    pairorder_weight: float = 0.0


@dataclass
class CDLibConfig:
    """Top-level config schema — for CI validation only."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
