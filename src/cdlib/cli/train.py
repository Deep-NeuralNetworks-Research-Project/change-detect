"""CLI entrypoint for training.

Usage:
    python -m cdlib.cli.train +experiment=baseline_fcsiamdiff_sysu
    python -m cdlib.cli.train data=levir_cd model=siamese_resnet18
    python -m cdlib.cli.train -m train.lr=1e-3,3e-4,1e-4 model=fc_siam_diff
    python -m cdlib.cli.train resume_from=auto
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra-powered training entrypoint with multirun support.

    Hydra's ``-m`` flag enables sweeps automatically:
        python -m cdlib.cli.train -m train.lr=1e-3,3e-4,1e-4
    """

    from cdlib.engine.trainer import Trainer
    from cdlib.models.build import build_loss, build_model, build_optimizer

    logger.info(f"Starting training: model={cfg.model.name}, data={cfg.data.name}")

    # Build components through the frozen builder contracts
    model = build_model(cfg)
    loss_fn = build_loss(cfg)
    optimizer = build_optimizer(cfg, model.parameters())

    # Build dataset + dataloader
    # NOTE: dataset builders depend on P1's data loaders (not yet available).
    # For now, if DATASET_REGISTRY has the requested dataset, use it.
    # Otherwise, fall back to a synthetic dataset for testing.
    train_loader, val_loader = _build_dataloaders(cfg)

    # Run directory from Hydra's output dir
    run_dir = hydra.utils.get_original_cwd() + "/results/" + cfg.get(
        "experiment_name", f"{cfg.model.name}_{cfg.data.name}"
    )

    # Create and run trainer
    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        run_dir=run_dir,
    )

    metrics = trainer.train()
    logger.info(f"Training complete. Final metrics: {metrics}")
    return metrics


def _build_dataloaders(cfg: DictConfig) -> tuple:
    """Build train and val DataLoaders.

    Falls back to synthetic data if the requested dataset isn't registered yet
    (P1's data loaders are expected by end of week 2).
    """
    from torch.utils.data import DataLoader


    batch_size = cfg.train.get("batch_size", 8)
    num_workers = cfg.data.get("num_workers", 4)

    from cdlib.models.build import build_dataset

    train_ds = build_dataset(cfg, split="train")
    val_ds = build_dataset(cfg, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader




if __name__ == "__main__":
    main()
