"""SYSU-CD — 20,000 aerial pairs, Hong Kong 2007-2014 (``research/02`` §2).

On-disk layout::

    <root>/{train,val,test}/{time1,time2,label}/NNNNN.png

The split *is* the directory structure. ``research/02`` §2 gives the published
partition as 12,000 / 4,000 / 4,000 and every published SYSU-CD result is reported
against it, so this loader reads whichever split directory it was asked for and
never re-partitions. Masks are 8-bit ``0``/``255``; SYSU-CD publishes no ignore
convention, so no pixel here is ever ``IGNORE_INDEX``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cdlib.data.base import PairedChangeDataset, PairRecord, binarize_mask
from cdlib.data.contract import NuisanceLabel

log = logging.getLogger(__name__)

#: Extensions accepted for images and masks, matched case-insensitively. The
#: upstream archive is PNG throughout, but mirrors re-encode, and the archive
#: cannot be verified until someone downloads it (BaiduYun, password ``mlls``) —
#: so this is a constructor parameter, not a literal buried in the walk.
DEFAULT_IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif")

#: Published partition sizes (``research/02`` §2). Recorded for the DATA_CARD and
#: for anyone tempted to re-split: this partition is given, not sampled.
OFFICIAL_SPLIT_SIZES: dict[str, int] = {"train": 12000, "val": 4000, "test": 4000}


def _normalise_extensions(extensions: Sequence[str]) -> tuple[str, ...]:
    """Lower-case, dot-prefix and de-duplicate an extension list."""
    out: list[str] = []
    for ext in extensions:
        e = ext.lower()
        if not e.startswith("."):
            e = f".{e}"
        if e not in out:
            out.append(e)
    return tuple(out)


def _index_by_stem(directory: Path, extensions: tuple[str, ...]) -> dict[str, Path]:
    """Map ``stem -> path`` for every accepted image file directly in *directory*.

    Iteration is over a sorted listing so that a stem carried by two encodings
    (``0001.png`` and ``0001.jpg``, as some mirrors ship) resolves to the same file
    on every run — the index order feeds the stats-cache fingerprint.
    """
    found: dict[str, Path] = {}
    if not directory.is_dir():
        return found
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in extensions:
            found.setdefault(path.stem, path)
    return found


def _read_rgb(path: Path) -> np.ndarray:
    """Decode one image as HWC uint8 RGB.

    PIL rather than ``cv2.imread``, which returns BGR: a silent channel swap in one
    loader would reach the model as a permanent colour-grading shift on that
    dataset, which is precisely the nuisance the model is supposed to be invariant
    to — it would be learned as signal instead.
    """
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _read_binary_mask(path: Path, threshold: int | None = None) -> np.ndarray:
    """Decode one label as HW int16 in ``{0, 1}``.

    Thresholding is delegated to :func:`cdlib.data.base.binarize_mask` so that every
    dataset applies the same rule to a soft mask edge. See that function for why
    ``arr > 0`` is not it.
    """
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"))
    return binarize_mask(arr, threshold=threshold)


class SysuCDDataset(PairedChangeDataset):
    """SYSU-CD, read split-by-split from the published directory partition.

    **Nuisance label.** Every record is :attr:`~cdlib.data.contract.NuisanceLabel.UNKNOWN`
    (``-1``). SYSU-CD's nuisance content is undocumented; emitting ``CLEAN`` would
    silently corrupt P5's nuisance-stratified evaluation by filling the clean
    stratum with pairs nobody has checked.

    **Grouping keys.** SYSU-CD is aerial tiles with no scene metadata, so
    ``scene_id`` and ``source_video`` are both the image stem, qualified by the
    split directory it came from (file numbering restarts at ``00000`` in each
    split, and an unqualified stem would make ``tests/test_splits.py`` fire on
    duplicate *file names* rather than on shared imagery). This makes the official
    split trivially group-disjoint only because we honour the published partition;
    it is **not** evidence that the underlying imagery is scene-disjoint — tiles
    from one Hong Kong district may well straddle the split. State that limitation
    in DATA_CARD rather than claiming scene-disjointness we cannot demonstrate.

    **Label-noise caveat** (``research/02`` §2, **inferred**, not documented).
    SYSU-CD's change taxonomy includes (c) pre-construction groundwork and
    (e) road expansion, which overlap conceptually with our nuisance classes (bare
    ground, vehicles, dust), so elevated inter-annotator disagreement is expected
    there. No published label-noise audit was found: this is a risk read off the
    taxonomy, not a critique anyone has run. Spot-check a sample before trusting
    the ground truth at face value, and keep the *inferred* marker until someone
    does.

    Args:
        root: Dataset root. Read from config; never hardcoded (``base`` module docs).
        split: One of ``train`` / ``val`` / ``test``. ``val_shift`` and ``cal`` are
            accepted and yield an empty dataset — SYSU-CD publishes no such
            partition, and inventing one would not be the shift-heavy slice
            ``research/04`` §4 asks for.
        dir_time1: Reference-frame subdirectory. Default ``"time1"``.
        dir_time2: Edited-frame subdirectory. Default ``"time2"``.
        dir_label: Change-mask subdirectory. Default ``"label"``.
        extensions: Accepted file extensions, matched case-insensitively.
        **kwargs: Forwarded to :class:`~cdlib.data.base.PairedChangeDataset`
            (``transform``, ``validate``, ``stats_cache``).
    """

    name = "sysu_cd"

    #: 286,092,024 / 1,310,720,000 changed pixels (``research/02`` §2).
    published_changed_pixel_ratio = 0.218

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "train",
        *,
        dir_time1: str = "time1",
        dir_time2: str = "time2",
        dir_label: str = "label",
        extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
        mask_threshold: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(root, split, **kwargs)
        self.dir_time1 = dir_time1
        self.dir_time2 = dir_time2
        self.dir_label = dir_label
        self.extensions = _normalise_extensions(extensions)
        self.mask_threshold = mask_threshold

    # -- index --------------------------------------------------------------------

    def _build_index(self) -> list[PairRecord]:
        if self.split not in OFFICIAL_SPLIT_SIZES:
            log.warning(
                "sysu_cd[%s]: SYSU-CD publishes only %s; returning an empty index "
                "rather than inventing a partition.",
                self.split,
                sorted(OFFICIAL_SPLIT_SIZES),
            )
            return []

        split_dir = self.root / self.split
        if not split_dir.is_dir():
            log.warning(
                "sysu_cd[%s]: %s is missing — returning an empty index.", self.split, split_dir
            )
            return []

        time1 = _index_by_stem(split_dir / self.dir_time1, self.extensions)
        time2 = _index_by_stem(split_dir / self.dir_time2, self.extensions)
        label = _index_by_stem(split_dir / self.dir_label, self.extensions)

        stems = sorted(set(time1) & set(time2) & set(label))
        orphans = len(set(time1) | set(time2) | set(label)) - len(stems)
        if orphans:
            log.warning(
                "sysu_cd[%s]: %d stem(s) lack a time1/time2/label counterpart and were "
                "skipped. Check dir_time1/dir_time2/dir_label against the archive.",
                self.split,
                orphans,
            )
        if not stems:
            log.warning("sysu_cd[%s]: no complete triples found under %s", self.split, split_dir)

        expected = OFFICIAL_SPLIT_SIZES[self.split]
        if stems and len(stems) != expected:
            log.warning(
                "sysu_cd[%s]: found %d pairs, published partition is %d (research/02 §2). "
                "Expected on a fixture or a partial download; suspicious on the real archive.",
                self.split,
                len(stems),
                expected,
            )

        return [
            PairRecord(
                pair_id=f"{self.split}/{stem}",
                scene_id=f"{self.split}/{stem}",
                source_video=f"{self.split}/{stem}",
                nuisance_label=int(NuisanceLabel.UNKNOWN),
                payload={"time1": time1[stem], "time2": time2[stem], "label": label[stem]},
            )
            for stem in stems
        ]

    # -- pixels -------------------------------------------------------------------

    def _read_images(self, rec: PairRecord) -> tuple[np.ndarray, np.ndarray]:
        return _read_rgb(rec.payload["time1"]), _read_rgb(rec.payload["time2"])

    def _read_mask(self, rec: PairRecord) -> np.ndarray:
        return _read_binary_mask(rec.payload["label"], self.mask_threshold)


__all__ = [
    "DEFAULT_IMAGE_EXTENSIONS",
    "OFFICIAL_SPLIT_SIZES",
    "SysuCDDataset",
]
