"""Wall-clock checkpoint system for preemption-safe training.

Saves model/optimizer/scheduler/RNG state every N minutes of wall-clock
time (not per-epoch). Designed for Colab/Kaggle where sessions can be
killed without warning.

Features:
    - Wall-clock cadence (default 15 min, configurable)
    - ``resume_from=auto``: picks newest checkpoint by mtime
    - Keeps last-K + best checkpoints, prunes the rest
    - Restores full RNG state for deterministic resume
    - SHA256 hash of checkpoint for reproducibility auditing
    - Config hash verification: warns on resume if config changed

Usage:
    saver = CheckpointSaver(cfg, run_dir="results/exp_001")
    saver.maybe_save(model, optimizer, scheduler, epoch, step, metrics)
    # ... on resume:
    state = CheckpointSaver.load_for_resume(cfg, run_dir="results/exp_001")
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from cdlib.utils.seed import get_rng_state, set_rng_state

logger = logging.getLogger(__name__)


def _cfg_hash(cfg: DictConfig) -> str:
    """Deterministic hash of the resolved config for change detection."""
    yaml_str = OmegaConf.to_yaml(cfg, resolve=True)
    return hashlib.sha256(yaml_str.encode("utf-8")).hexdigest()[:16]


def _sha256_file(path: Path) -> str:
    """Compute SHA256 of a file (for reproducibility auditing)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class CheckpointSaver:
    """Wall-clock checkpoint manager.

    Args:
        cfg: Full Hydra config (used for interval, keep counts, and hashing).
        run_dir: Directory to save checkpoints into (e.g. ``results/exp_001``).
    """

    def __init__(self, cfg: DictConfig, run_dir: str | Path) -> None:
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Config-driven settings with sensible defaults
        ckpt_cfg = cfg.train.get("checkpoint", {})
        self.interval_seconds = ckpt_cfg.get("interval_minutes", 15) * 60
        self.keep_last = ckpt_cfg.get("keep_last", 2)
        self.keep_best = ckpt_cfg.get("keep_best", 1)

        self._cfg_hash = _cfg_hash(cfg)
        self._last_save_time = time.monotonic()
        self._best_metric: float | None = None

    def should_save(self) -> bool:
        """Check if enough wall-clock time has elapsed since last save."""
        elapsed = time.monotonic() - self._last_save_time
        return elapsed >= self.interval_seconds

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        global_step: int,
        metrics: dict[str, float] | None = None,
        is_best: bool = False,
    ) -> Path:
        """Save a checkpoint and prune old ones.

        Args:
            model: The model to save.
            optimizer: The optimizer to save.
            scheduler: The LR scheduler to save (can be None).
            epoch: Current epoch number.
            global_step: Current global training step.
            metrics: Optional dict of metric values (for logging).
            is_best: If True, also save as ``best.pt``.

        Returns:
            Path to the saved checkpoint file.
        """
        state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "global_step": global_step,
            "rng_states": get_rng_state(),
            "best_metric": self._best_metric,
            "cfg_hash": self._cfg_hash,
            "metrics": metrics or {},
        }

        # Filename with step for easy sorting
        ckpt_path = self.ckpt_dir / f"ckpt_step{global_step:08d}.pt"
        torch.save(state, ckpt_path)

        # Write SHA256 for reproducibility auditing
        sha = _sha256_file(ckpt_path)
        sha_path = self.ckpt_dir / "checkpoint_sha256.txt"
        with open(sha_path, "a") as f:
            f.write(f"{sha}  {ckpt_path.name}\n")

        logger.info(
            f"Checkpoint saved: {ckpt_path.name} "
            f"(epoch={epoch}, step={global_step}, sha256={sha[:12]}...)"
        )

        # Save as best if applicable
        if is_best:
            best_path = self.ckpt_dir / "best.pt"
            torch.save(state, best_path)
            logger.info(f"Best checkpoint updated: {best_path.name}")

        # Prune old checkpoints (keep last-K + best)
        self._prune()

        self._last_save_time = time.monotonic()
        return ckpt_path

    def maybe_save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        global_step: int,
        metrics: dict[str, float] | None = None,
        is_best: bool = False,
    ) -> Path | None:
        """Save only if enough wall-clock time has elapsed."""
        if self.should_save():
            return self.save(
                model, optimizer, scheduler, epoch, global_step, metrics, is_best
            )
        return None

    def update_best(self, metric_value: float) -> bool:
        """Update best metric. Returns True if this is a new best.

        Uses higher-is-better convention (e.g., F1 score).
        """
        if self._best_metric is None or metric_value > self._best_metric:
            self._best_metric = metric_value
            return True
        return False

    def _prune(self) -> None:
        """Keep only the last-K checkpoints plus best.pt."""
        # List all ckpt_step*.pt files, sorted by step number (oldest first)
        ckpts = sorted(self.ckpt_dir.glob("ckpt_step*.pt"))
        best_path = self.ckpt_dir / "best.pt"

        # Don't count best.pt in the keep_last budget
        to_prune = ckpts[: -self.keep_last] if len(ckpts) > self.keep_last else []

        for p in to_prune:
            if p != best_path:
                p.unlink()
                logger.debug(f"Pruned old checkpoint: {p.name}")

    @staticmethod
    def find_latest(run_dir: str | Path) -> Path | None:
        """Find the most recent checkpoint in a run directory.

        Used by ``resume_from=auto`` to pick up where training left off.

        Returns:
            Path to the newest checkpoint, or None if no checkpoints found.
        """
        ckpt_dir = Path(run_dir) / "checkpoints"
        if not ckpt_dir.exists():
            return None

        ckpts = sorted(ckpt_dir.glob("ckpt_step*.pt"))
        return ckpts[-1] if ckpts else None

    @staticmethod
    def load(
        path: str | Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        cfg: DictConfig | None = None,
        restore_rng: bool = True,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Load a checkpoint and restore state.

        Args:
            path: Path to the checkpoint file.
            model: Model to load state into.
            optimizer: Optimizer to restore (optional, skip for eval-only).
            scheduler: Scheduler to restore (optional).
            cfg: Current config (used to verify cfg_hash hasn't changed).
            restore_rng: Whether to restore RNG states (for deterministic resume).
            device: Device to map tensors to.

        Returns:
            Dict with 'epoch', 'global_step', 'best_metric', 'metrics'.
        """
        path = Path(path)
        logger.info(f"Loading checkpoint: {path}")
        state = torch.load(path, map_location=device, weights_only=False)

        # Verify config hash if we have a current config
        if cfg is not None:
            current_hash = _cfg_hash(cfg)
            saved_hash = state.get("cfg_hash", "")
            if current_hash != saved_hash:
                logger.warning(
                    f"Config hash mismatch! "
                    f"Saved: {saved_hash}, Current: {current_hash}. "
                    f"Resuming with a different config may cause unexpected behavior."
                )

        # Restore model
        model.load_state_dict(state["model_state"])

        # Restore optimizer
        if optimizer is not None and "optimizer_state" in state:
            optimizer.load_state_dict(state["optimizer_state"])

        # Restore scheduler
        if scheduler is not None and state.get("scheduler_state") is not None:
            scheduler.load_state_dict(state["scheduler_state"])

        # Restore RNG states for deterministic resume
        if restore_rng and "rng_states" in state:
            set_rng_state(state["rng_states"])
            logger.info("RNG states restored for deterministic resume.")

        return {
            "epoch": state["epoch"],
            "global_step": state["global_step"],
            "best_metric": state.get("best_metric"),
            "metrics": state.get("metrics", {}),
        }
