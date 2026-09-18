"""Contract-blind training engine.

The trainer is *contract-blind*: it backprops ``out["loss"]`` and auto-logs
every ``loss/*`` key to W&B. It never inspects what's inside a loss dict.
If you find yourself special-casing a particular loss or model, the contract
is wrong — fix the contract, not the trainer.

Usage:
    trainer = Trainer(cfg, model, train_loader, val_loader, loss_fn,
                      optimizer, scheduler, metric_fns)
    trainer.train()
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from cdlib.utils.capture import capture_run_metadata
from cdlib.utils.checkpoint import CheckpointSaver
from cdlib.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _try_wandb_init(cfg: DictConfig, run_dir: Path) -> Any:
    """Initialize W&B if available and configured, otherwise return None."""
    try:
        import wandb

        wandb_cfg = cfg.train.get("wandb", {})
        project = wandb_cfg.get("project", "cd-project")
        entity = wandb_cfg.get("entity", None)

        run = wandb.init(
            project=project,
            entity=entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=str(run_dir),
            resume="allow",
        )
        logger.info(f"W&B initialized: {run.url}")
        return run
    except Exception as e:
        logger.warning(f"W&B init failed (training will continue without it): {e}")
        return None


def _build_scheduler(
    cfg: DictConfig, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler | None:
    """Build a learning rate scheduler from config."""
    sched_cfg = cfg.train.get("scheduler", None)
    if sched_cfg is None:
        return None

    name = sched_cfg.get("name", "cosine").lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.get("T_max", cfg.train.epochs),
            eta_min=sched_cfg.get("eta_min", 1e-6),
        )
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get("step_size", 50),
            gamma=sched_cfg.get("gamma", 0.1),
        )
    else:
        logger.warning(f"Unknown scheduler '{name}', skipping.")
        return None


class Trainer:
    """Contract-blind training engine.

    The trainer orchestrates the training loop without knowledge of
    what the model, loss, or metrics compute internally. It relies
    entirely on the frozen contracts defined in ``docs/architecture.md``.

    Args:
        cfg: Full Hydra config.
        model: Model implementing the forward contract.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader (can be None for train-only).
        loss_fn: Loss module implementing the loss contract.
        optimizer: Optimizer for model parameters.
        scheduler: LR scheduler (optional, built internally if None).
        metric_fns: List of metric objects implementing reset/update/compute.
        run_dir: Directory for results (checkpoints, logs, metadata).
        device: Device to train on.
    """

    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        loss_fn: Any,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None = None,
        metric_fns: list[Any] | None = None,
        run_dir: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metric_fns = metric_fns or []

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = self.model.to(self.device)

        # Scheduler — build from config if not provided
        if scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = _build_scheduler(cfg, optimizer)

        # Run directory
        if run_dir is None:
            exp_name = cfg.get("experiment_name", "default")
            ts = time.strftime("%Y%m%d_%H%M%S")
            self.run_dir = Path("results") / f"{exp_name}_{ts}"
        else:
            self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Training config with defaults
        train_cfg = cfg.train
        self.epochs = train_cfg.get("epochs", 200)
        self.val_interval = train_cfg.get("val_interval", 1)
        self.log_interval = train_cfg.get("wandb", {}).get("log_interval", 10)
        self.use_amp = train_cfg.get("amp", True)

        # AMP scaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp and self.device.type == "cuda")

        # Checkpoint saver
        self.ckpt_saver = CheckpointSaver(cfg, self.run_dir)

        # State
        self.start_epoch = 0
        self.global_step = 0
        self.wandb_run = None

    def train(self) -> dict[str, Any]:
        """Main training loop.

        Returns:
            Dict with final metrics and best metric value.
        """
        # Seed everything
        seed = self.cfg.train.get("seed", 42)
        set_seed(seed)

        # Auto-capture run metadata
        capture_run_metadata(self.cfg, self.run_dir)

        # Handle resume
        self._maybe_resume()

        # Initialize W&B
        self.wandb_run = _try_wandb_init(self.cfg, self.run_dir)

        logger.info(
            f"Training started: epochs={self.epochs}, device={self.device}, "
            f"amp={self.use_amp}, run_dir={self.run_dir}"
        )

        best_metrics = {}

        try:
            for epoch in range(self.start_epoch, self.epochs):
                # Train one epoch
                train_loss = self._train_one_epoch(epoch)

                # Step scheduler
                if self.scheduler is not None:
                    self.scheduler.step()

                # Log epoch-level info
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch}/{self.epochs - 1} — "
                    f"train_loss={train_loss:.4f}, lr={lr:.2e}"
                )
                if self.wandb_run:
                    try:
                        import wandb

                        wandb.log(
                            {"epoch": epoch, "train/loss_epoch": train_loss, "lr": lr},
                            step=self.global_step,
                        )
                    except Exception:
                        pass

                # Validate
                val_metrics = {}
                if (
                    self.val_loader is not None
                    and (epoch + 1) % self.val_interval == 0
                ):
                    val_metrics = self._validate(epoch)

                    # Check for best model (use F1 if available, else -val_loss)
                    primary = val_metrics.get("F1", -val_metrics.get("val_loss", 0))
                    is_best = self.ckpt_saver.update_best(primary)
                    if is_best:
                        best_metrics = val_metrics.copy()
                        self.ckpt_saver.save(
                            self.model,
                            self.optimizer,
                            self.scheduler,
                            epoch,
                            self.global_step,
                            metrics=val_metrics,
                            is_best=True,
                        )

                # Wall-clock checkpoint (separate from best — fires on time)
                self.ckpt_saver.maybe_save(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    self.global_step,
                    metrics=val_metrics,
                )

        except KeyboardInterrupt:
            logger.info("Training interrupted by user. Saving emergency checkpoint...")
            self.ckpt_saver.save(
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                self.global_step,
                metrics={},
            )

        finally:
            # Final checkpoint
            self.ckpt_saver.save(
                self.model,
                self.optimizer,
                self.scheduler,
                self.epochs - 1,
                self.global_step,
                metrics=best_metrics,
            )

            if self.wandb_run:
                try:
                    import wandb

                    wandb.finish()
                except Exception:
                    pass

        logger.info(f"Training complete. Best metrics: {best_metrics}")
        return best_metrics

    def _train_one_epoch(self, epoch: int) -> float:
        """One training epoch: forward → loss → backward → step → log.

        CONTRACT-BLIND: backprops only ``out["loss"]`` and auto-logs
        every ``loss/*`` key. Never inspects loss internals.

        Returns:
            Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            leave=False,
            disable=not logger.isEnabledFor(logging.INFO),
        )

        for batch in pbar:
            # Move batch to device
            img1 = batch["img1"].to(self.device, non_blocking=True)
            img2 = batch["img2"].to(self.device, non_blocking=True)
            batch_device = {
                k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            self.optimizer.zero_grad(set_to_none=True)

            # Mixed precision forward + loss
            with torch.amp.autocast(
                self.device.type, enabled=self.use_amp and self.device.type == "cuda"
            ):
                outputs = self.model(img1, img2)
                loss_dict = self.loss_fn.compute(outputs, batch_device)

            # Contract-blind: backprop only out["loss"]
            loss = loss_dict["loss"]
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Progress bar
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            # Auto-log every loss/* key to W&B — never inspect internals
            if self.global_step % self.log_interval == 0 and self.wandb_run:
                try:
                    import wandb

                    log_dict = {
                        k: v.item() if isinstance(v, torch.Tensor) else v
                        for k, v in loss_dict.items()
                    }
                    log_dict["step"] = self.global_step
                    wandb.log(log_dict, step=self.global_step)
                except Exception:
                    pass

            # Wall-clock checkpoint check (within epoch)
            self.ckpt_saver.maybe_save(
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                self.global_step,
            )

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict[str, float]:
        """Run validation and compute metrics.

        Returns:
            Dict of metric name → value.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # Reset metrics
        for m in self.metric_fns:
            m.reset()

        for batch in tqdm(
            self.val_loader,
            desc=f"Val {epoch}",
            leave=False,
            disable=not logger.isEnabledFor(logging.INFO),
        ):
            img1 = batch["img1"].to(self.device, non_blocking=True)
            img2 = batch["img2"].to(self.device, non_blocking=True)
            batch_device = {
                k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with torch.amp.autocast(
                self.device.type, enabled=self.use_amp and self.device.type == "cuda"
            ):
                outputs = self.model(img1, img2)
                loss_dict = self.loss_fn.compute(outputs, batch_device)

            total_loss += loss_dict["loss"].item()
            num_batches += 1

            # Update metrics
            for m in self.metric_fns:
                m.update(outputs, batch_device)

        # Compute metrics
        metrics = {"val_loss": total_loss / max(num_batches, 1)}
        for m in self.metric_fns:
            computed = m.compute()
            metrics.update(computed)

        # Log to W&B
        if self.wandb_run:
            try:
                import wandb

                val_log = {f"val/{k}": v for k, v in metrics.items()}
                val_log["epoch"] = epoch
                wandb.log(val_log, step=self.global_step)
            except Exception:
                pass

        logger.info(f"Validation — {metrics}")
        return metrics

    def _maybe_resume(self) -> None:
        """Handle resume_from config: 'auto', explicit path, or None."""
        resume_from = self.cfg.train.get("checkpoint", {}).get("resume_from", None)

        if resume_from is None:
            return

        if resume_from == "auto":
            ckpt_path = CheckpointSaver.find_latest(self.run_dir)
            if ckpt_path is None:
                logger.info("resume_from=auto but no checkpoint found. Starting fresh.")
                return
        else:
            ckpt_path = Path(resume_from)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        logger.info(f"Resuming from: {ckpt_path}")
        state = CheckpointSaver.load(
            ckpt_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            cfg=self.cfg,
            restore_rng=True,
            device=self.device,
        )

        self.start_epoch = state["epoch"] + 1
        self.global_step = state["global_step"]
        if state.get("best_metric") is not None:
            self.ckpt_saver._best_metric = state["best_metric"]

        logger.info(
            f"Resumed: epoch={self.start_epoch}, step={self.global_step}, "
            f"best_metric={state.get('best_metric')}"
        )
