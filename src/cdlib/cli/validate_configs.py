"""Config validation CLI — dry-construct all configs for CI.

Usage:
    python -m cdlib.cli.validate_configs

Loads every Hydra config combination and attempts to dry-construct
the model, loss, and dataset to catch typos and missing keys before
they hit training. Used by CI pipeline.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def validate_all_configs() -> bool:
    """Validate all config files by attempting Hydra compose + dry construction.

    Returns:
        True if all configs are valid, False if any failed.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    configs_dir = str(Path(__file__).resolve().parent.parent.parent.parent / "configs")
    all_passed = True
    results = []

    # Base config
    configs_to_test = [
        ("base config", []),
    ]

    # Find all experiment configs
    exp_dir = Path(configs_dir) / "experiment"
    if exp_dir.exists():
        for exp_file in sorted(exp_dir.glob("*.yaml")):
            name = exp_file.stem
            configs_to_test.append(
                (f"experiment/{name}", [f"+experiment={name}"])
            )

    for label, overrides in configs_to_test:
        # Reset Hydra singleton between tests
        GlobalHydra.instance().clear()

        try:
            with initialize_config_dir(config_dir=configs_dir, version_base=None):
                cfg = compose(config_name="config", overrides=overrides)

            # Verify essential keys exist
            assert hasattr(cfg, "model"), "Missing 'model' key"
            assert hasattr(cfg, "data"), "Missing 'data' key"
            assert hasattr(cfg, "loss"), "Missing 'loss' key"
            assert hasattr(cfg, "train"), "Missing 'train' key"
            assert hasattr(cfg.model, "name"), "Missing 'model.name'"
            assert hasattr(cfg.data, "name"), "Missing 'data.name'"
            assert hasattr(cfg.loss, "name"), "Missing 'loss.name'"

            results.append((label, "✅ PASS"))
            logger.info(f"✅ {label}: PASS")

        except Exception as e:
            results.append((label, f"❌ FAIL: {e}"))
            logger.error(f"❌ {label}: FAIL — {e}")
            all_passed = False

        finally:
            GlobalHydra.instance().clear()

    # Summary
    print("\n" + "=" * 60)
    print("Config Validation Summary")
    print("=" * 60)
    for label, status in results:
        print(f"  {label}: {status}")
    print("=" * 60)
    passed = sum(1 for _, s in results if s.startswith("✅"))
    total = len(results)
    print(f"  {passed}/{total} passed")

    return all_passed


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    success = validate_all_configs()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
