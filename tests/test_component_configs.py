"""Config-shape validation for configs/model/*.yaml — no Hydra needed.

These are draft configs (see configs/model/README.md): P2's Hydra root
doesn't exist yet, so nothing composes/validates them automatically.
This test is the guard against them silently drifting out of sync with
the actual constructors as the P3 modules change — each config's
`_target_` must resolve to a real, importable class, and its remaining
keys must be valid kwargs for that class's `__init__` (checked by
actually building it, with `pretrained` forced False to avoid
downloading weights in CI).
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "model"
# Hydra group YAMLs at this directory's top level use `name:`, not `_target_`.
# Only the P3 component drafts (encoder/, fusion/) are constructor-shaped.
CONFIG_FILES = sorted(
    p
    for p in CONFIG_ROOT.rglob("*.yaml")
    if p.parent.name in {"encoder", "fusion"}
)


def _build_from_config(path: Path):
    cfg = dict(yaml.safe_load(path.read_text()))
    target = cfg.pop("_target_")
    module_name, class_name = target.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)

    if "pretrained" in cfg:
        cfg["pretrained"] = False  # never download weights in a test
    if path.parent.name == "fusion" and "in_channels" not in cfg:
        # in_channels comes from the encoder's out_channels at real
        # model-build time, not stored statically in a fusion config —
        # substitute a dummy value just to validate the rest of the shape.
        cfg["in_channels"] = [8]

    return cls(**cfg)


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: str(p.relative_to(CONFIG_ROOT)))
def test_config_builds(path: Path):
    assert _build_from_config(path) is not None


def test_at_least_the_expected_configs_exist():
    names = {str(p.relative_to(CONFIG_ROOT)).replace("\\", "/") for p in CONFIG_FILES}
    expected = {
        "encoder/resnet18.yaml",
        "encoder/efficientnet_b0.yaml",
        "encoder/efficientnet_b2.yaml",
        "fusion/absdiff.yaml",
        "fusion/signed.yaml",
        "fusion/signed_product.yaml",
        "fusion/concat.yaml",
        "fusion/signed_concat.yaml",
    }
    assert expected <= names
