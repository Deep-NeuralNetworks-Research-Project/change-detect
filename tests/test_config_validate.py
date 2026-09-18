"""Test config validation — all YAML configs must load via Hydra compose."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_base_config_composes():
    """The root config.yaml must compose without errors."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    configs_dir = str(Path(__file__).resolve().parent.parent / "configs")
    GlobalHydra.instance().clear()

    try:
        with initialize_config_dir(config_dir=configs_dir, version_base=None):
            cfg = compose(config_name="config")

        assert hasattr(cfg, "model"), "Missing 'model' key"
        assert hasattr(cfg, "data"), "Missing 'data' key"
        assert hasattr(cfg, "loss"), "Missing 'loss' key"
        assert hasattr(cfg, "train"), "Missing 'train' key"
    finally:
        GlobalHydra.instance().clear()


def test_all_experiment_configs_compose():
    """Every experiment config must compose without errors."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    configs_dir = str(Path(__file__).resolve().parent.parent / "configs")
    exp_dir = Path(configs_dir) / "experiment"

    if not exp_dir.exists():
        pytest.skip("No experiment configs found")

    for exp_file in sorted(exp_dir.glob("*.yaml")):
        name = exp_file.stem
        GlobalHydra.instance().clear()

        try:
            with initialize_config_dir(config_dir=configs_dir, version_base=None):
                cfg = compose(
                    config_name="config", overrides=[f"+experiment={name}"]
                )

            assert hasattr(cfg, "model"), f"experiment/{name}: Missing 'model'"
            assert hasattr(cfg, "data"), f"experiment/{name}: Missing 'data'"
        finally:
            GlobalHydra.instance().clear()


def test_all_model_configs_compose():
    """Every model config must compose without errors."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    configs_dir = str(Path(__file__).resolve().parent.parent / "configs")
    model_dir = Path(configs_dir) / "model"

    for model_file in sorted(model_dir.glob("*.yaml")):
        name = model_file.stem
        GlobalHydra.instance().clear()

        try:
            with initialize_config_dir(config_dir=configs_dir, version_base=None):
                cfg = compose(
                    config_name="config", overrides=[f"model={name}"]
                )

            assert hasattr(cfg.model, "name"), (
                f"model={name}: Missing 'model.name'"
            )
        finally:
            GlobalHydra.instance().clear()


def test_all_data_configs_compose():
    """Every data config must compose without errors."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    configs_dir = str(Path(__file__).resolve().parent.parent / "configs")
    data_dir = Path(configs_dir) / "data"

    for data_file in sorted(data_dir.glob("*.yaml")):
        name = data_file.stem
        GlobalHydra.instance().clear()

        try:
            with initialize_config_dir(config_dir=configs_dir, version_base=None):
                cfg = compose(
                    config_name="config", overrides=[f"data={name}"]
                )

            assert hasattr(cfg.data, "name"), f"data={name}: Missing 'data.name'"
        finally:
            GlobalHydra.instance().clear()


def test_all_loss_configs_compose():
    """Every loss config must compose without errors."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    configs_dir = str(Path(__file__).resolve().parent.parent / "configs")
    loss_dir = Path(configs_dir) / "loss"

    for loss_file in sorted(loss_dir.glob("*.yaml")):
        name = loss_file.stem
        GlobalHydra.instance().clear()

        try:
            with initialize_config_dir(config_dir=configs_dir, version_base=None):
                cfg = compose(
                    config_name="config", overrides=[f"loss={name}"]
                )

            assert hasattr(cfg.loss, "name"), f"loss={name}: Missing 'loss.name'"
        finally:
            GlobalHydra.instance().clear()
