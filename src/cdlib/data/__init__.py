"""Data layer (P1).

Start with ``docs/HANDOFF.md`` — it records the eight interface decisions the rest of
the team depends on. The three that bite hardest if ignored:

* ``mask`` may contain ``-1``. Gate every metric on :func:`valid_pixel_mask` before
  the confusion matrix, or ``-1`` becomes ``255`` and the numbers are wrong.
* ``nuisance_label`` is ``-1`` (unknown) on the public datasets, never ``0`` (clean).
* Geometric augmentation is shared across the pair; photometric is independent.
"""

from cdlib.data.base import PairedChangeDataset, PairRecord
from cdlib.data.contract import (
    IGNORE_INDEX,
    NUISANCE_NAMES,
    SPLITS,
    NuisanceLabel,
    changed_pixel_counts,
    is_negative_pair,
    nuisance_name,
    valid_pixel_mask,
    validate_sample,
)
from cdlib.data.registry import DATASET_REGISTRY, available_datasets, build_dataset
from cdlib.data.splits import (
    DROPPED_KEY,
    assert_no_group_leakage,
    scene_disjoint_split,
    split_summary,
)
from cdlib.data.transforms import (
    IndependentPhotometric,
    PairedGeometric,
    PairedTransform,
    build_transforms,
)

__all__ = [
    "DATASET_REGISTRY",
    "DROPPED_KEY",
    "IGNORE_INDEX",
    "NUISANCE_NAMES",
    "SPLITS",
    "IndependentPhotometric",
    "NuisanceLabel",
    "PairRecord",
    "PairedChangeDataset",
    "PairedGeometric",
    "PairedTransform",
    "assert_no_group_leakage",
    "available_datasets",
    "build_dataset",
    "build_transforms",
    "changed_pixel_counts",
    "is_negative_pair",
    "nuisance_name",
    "scene_disjoint_split",
    "split_summary",
    "valid_pixel_mask",
    "validate_sample",
]
