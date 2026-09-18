"""``DATASET_REGISTRY`` and ``build_dataset`` — the data layer's single entrypoint.

``research/06`` names four registries (``ENCODER_``, ``FUSION_``, ``ALIGNMENT_``,
``LOSS_``) and refers to a fifth without ever naming it. P1 owns this file, so P1
claims the name: :data:`DATASET_REGISTRY`. ``build_dataset(cfg, split)`` lives here
too — it is one of the only four entrypoints the CLI ever calls, and the data half
of that contract belongs with the data.

Adding a dataset is one new file in ``datasets/`` plus one line here. Nothing else
in the repo changes (root ``CLAUDE.md``, "The registry pattern").

Two things this module must get right, both driven by constraints from outside:

* **Construction must not need the data.** CI dry-constructs every
  ``configs/data/*.yaml`` through this function on a runner with no datasets
  attached (``research/06`` §4). Nothing here touches the filesystem.
* **``split`` is an argument, not a config key.** The frozen builder signature is
  ``build_dataset(cfg, split)``, so one config serves all five splits.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from cdlib.data.base import PairedChangeDataset
from cdlib.data.contract import SPLITS
from cdlib.data.datasets.dvi import DVIDataset
from cdlib.data.datasets.levir_cd import LevirCDDataset
from cdlib.data.datasets.pcd import PCDDataset
from cdlib.data.datasets.sysu_cd import SysuCDDataset
from cdlib.data.datasets.video_pairs import VideoPairsDataset
from cdlib.data.datasets.videosham import VideoShamDataset
from cdlib.utils.registry import Registry

log = logging.getLogger(__name__)

#: The fifth registry. Keys are what a user types as ``data=<key>`` on the CLI and
#: what ``configs/data/<key>.yaml`` is named after.
DATASET_REGISTRY: Registry = Registry("DATASET_REGISTRY")

DATASET_REGISTRY.register_explicit("sysu_cd", SysuCDDataset)
DATASET_REGISTRY.register_explicit("levir_cd", LevirCDDataset)
DATASET_REGISTRY.register_explicit("pcd", PCDDataset)
# NOTE: `research/06` §1 lists the config file as `video_set.yaml` while naming the
# module `video_pairs.py` and the output directory `data/video_pairs/`. Two of the
# three say video_pairs, so that is the key. Documented in docs/HANDOFF.md §7.
DATASET_REGISTRY.register_explicit("video_pairs", VideoPairsDataset)

# The "if the video set falls through" escape hatch (P1 role file). Registered now,
# with neither dataset downloaded, so switching is a config change rather than a week
# of work at the moment the fallback is actually needed.
DATASET_REGISTRY.register_explicit("videosham", VideoShamDataset)
DATASET_REGISTRY.register_explicit("dvi", DVIDataset)

#: Constructor arguments that are ours to interpret, not the dataset's.
_RESERVED_KEYS = frozenset({"name", "root", "transform", "transforms", "_target_"})


def _resolve_root(cfg: Any, data_cfg: Any) -> Path:
    """Resolve the dataset root, preferring an explicit ``data.root``.

    Falls back to ``<cfg.data_root>/<name>`` so a machine with datasets in an unusual
    place overrides one value rather than four. Never returns a hardcoded runtime
    path — a pre-commit hook rejects ``/content/`` and ``/kaggle/`` outside
    ``notebooks/``, and those belong in a CLI override.
    """
    root = _get(data_cfg, "root", None)
    if root:
        return Path(str(root)).expanduser()
    base = _get(cfg, "data_root", None)
    name = _get(data_cfg, "name", "unnamed")
    if base:
        return Path(str(base)).expanduser() / str(name)
    return Path("data") / str(name)


def _get(node: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` off a dict or an OmegaConf node without importing hydra."""
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _to_plain(node: Any) -> dict[str, Any]:
    """Best-effort conversion of a config node to a plain dict."""
    if node is None:
        return {}
    if isinstance(node, dict):
        return dict(node)
    try:  # OmegaConf without importing it at module scope
        from omegaconf import OmegaConf

        if OmegaConf.is_config(node):
            resolved = OmegaConf.to_container(node, resolve=True)
            return dict(resolved) if isinstance(resolved, dict) else {}
    except ImportError:  # pragma: no cover - omegaconf is a hard dependency
        pass
    return {k: v for k, v in vars(node).items() if not k.startswith("_")}


def build_dataset(cfg: Any, split: str) -> PairedChangeDataset:
    """Construct the dataset named by ``cfg.data.name`` for ``split``.

    Args:
        cfg: The composed config. Either the root node (with a ``data`` group) or the
            data node itself — both are accepted, because CI's dry-construction job
            and a training run reach this function from different depths.
        split: One of :data:`cdlib.data.contract.SPLITS`.

    Returns:
        A constructed dataset. It may be empty: a missing root is a warning, not an
        error, so that CI can validate every config with no data present.

    Raises:
        ValueError: if ``split`` is not a known split name, or ``cfg`` names no
            dataset.
        KeyError: if the named dataset is not in :data:`DATASET_REGISTRY`.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {list(SPLITS)}")

    # Accept either the root config or the data node, so callers need not care.
    data_cfg = _get(cfg, "data", None)
    if data_cfg is None or _get(data_cfg, "name", None) is None:
        if _get(cfg, "name", None) is not None:
            data_cfg = cfg
        else:
            raise ValueError(
                "config names no dataset: expected cfg.data.name or cfg.name to be a "
                f"key in DATASET_REGISTRY {DATASET_REGISTRY.keys()}"
            )

    name = str(_get(data_cfg, "name"))
    cls = DATASET_REGISTRY.get(name)
    root = _resolve_root(cfg, data_cfg)

    kwargs = {k: v for k, v in _to_plain(data_cfg).items() if k not in _RESERVED_KEYS}

    # The transform is built here rather than in each loader so every dataset gets the
    # shared-geometric / independent-photometric split (root CLAUDE.md rule 4) from
    # one code path instead of four chances to get it backwards. Which paths are
    # active for a given split is `transforms.SPLIT_AUGMENTATION_POLICY`, so eval
    # splits are already augmentation-free without a branch here.
    transform = _get(data_cfg, "transform", None)
    if not callable(transform):
        if transform is False:
            transform = None
        else:
            from cdlib.data.transforms import build_transforms

            transform = build_transforms(data_cfg, split)

    log.debug("build_dataset(%s, %s) -> %s(root=%s)", name, split, cls.__name__, root)
    return cls(root=root, split=split, transform=transform, **kwargs)


def available_datasets() -> list[str]:
    """Registered dataset keys, for error messages and ``tests/test_registries.py``."""
    return DATASET_REGISTRY.keys()


__all__ = ["DATASET_REGISTRY", "available_datasets", "build_dataset"]
