"""LEVIR-CD — 637 VHR building-change pairs, cropped to the standard protocol.

On-disk layout::

    <root>/{train,val,test}/{A,B,label}/<name>.png     # full pairs, 1024x1024

**The crop policy is not negotiable** (root ``CLAUDE.md`` rule 3, ``research/02``
§3): 256x256 **non-overlapping** crops of the official 445 / 64 / 128 full pairs,
giving **7,120 / 1,024 / 2,048** crops. Essentially every change-detection paper
reports against exactly this, and any other crop policy — overlap, a different
size, resizing instead of cropping — makes our numbers comparable to nothing.
There is deliberately no stride or overlap parameter.

Masks are 8-bit ``0``/``255``; LEVIR-CD publishes no ignore convention, so no pixel
here is ever ``IGNORE_INDEX``.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cdlib.data.base import PairedChangeDataset, PairRecord, binarize_mask
from cdlib.data.contract import NuisanceLabel

log = logging.getLogger(__name__)

#: Extensions accepted for images and masks, matched case-insensitively. The
#: upstream release is PNG; the HuggingFace and Kaggle mirrors re-encode, and none
#: of it is verifiable until someone downloads the archive — so this is a
#: constructor parameter, not a literal buried in the walk.
DEFAULT_IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif")

#: Side of the published full tiles, in pixels (``research/02`` §3).
FULL_TILE_SIZE: int = 1024

#: Side of the standard crop, in pixels. 1024 / 256 = 4, so 4x4 = 16 crops per tile.
DEFAULT_CROP_SIZE: int = 256

#: Official partition, in *full* pairs (``research/02`` §3).
OFFICIAL_FULL_PAIRS: dict[str, int] = {"train": 445, "val": 64, "test": 128}

#: The protocol crop counts, derived rather than typed so the arithmetic behind
#: 7,120 / 1,024 / 2,048 is visible and testable instead of asserted.
OFFICIAL_CROP_COUNTS: dict[str, int] = {
    split: n * (FULL_TILE_SIZE // DEFAULT_CROP_SIZE) ** 2
    for split, n in OFFICIAL_FULL_PAIRS.items()
}


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
    resolves to the same file on every run — the index order feeds the stats-cache
    fingerprint.
    """
    found: dict[str, Path] = {}
    if not directory.is_dir():
        return found
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in extensions:
            found.setdefault(path.stem, path)
    return found


def _image_size(path: Path) -> tuple[int, int]:
    """``(height, width)`` read from the file header, without decoding pixels."""
    with Image.open(path) as im:
        width, height = im.size
    return height, width


def _read_rgb(path: Path) -> np.ndarray:
    """Decode one full tile as HWC uint8 RGB.

    PIL rather than ``cv2.imread``, which returns BGR: a silent channel swap in one
    loader reaches the model as a permanent colour shift on that dataset — the very
    nuisance the model is meant to be invariant to, learned as signal instead.
    """
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _read_label(path: Path) -> np.ndarray:
    """Decode one full label tile as HW int16 in ``{0, 1}``.

    LEVIR-CD's real masks are exactly ``{0, 255}``, so the adaptive threshold in
    :func:`cdlib.data.base.binarize_mask` is a no-op here — verified against the real
    archive, where the measured ratio is unchanged to four decimal places. It is used
    anyway so that one rule governs every dataset.
    """
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"))
    return binarize_mask(arr)


class _BoundedImageCache:
    """LRU of decoded full tiles, keyed by path.

    One crop is one record, so a sequential pass over a tile asks for the same
    three files 16 times; without a cache that is 16 PNG decodes of a 1024x1024
    image to serve 16 crops. The bound is what keeps it from becoming an unbounded
    leak — at ~3 MB per RGB tile, the default holds a couple of tiles' worth of
    A/B/label and nothing more. It is per-instance, so ``DataLoader`` workers
    (processes) each keep their own and nothing is shared across them.
    """

    def __init__(self, maxsize: int) -> None:
        self.maxsize = max(1, int(maxsize))
        self._entries: OrderedDict[Path, np.ndarray] = OrderedDict()

    def get(self, path: Path, decode: Callable[[Path], np.ndarray]) -> np.ndarray:
        cached = self._entries.get(path)
        if cached is not None:
            self._entries.move_to_end(path)
            return cached
        arr = decode(path)
        self._entries[path] = arr
        while len(self._entries) > self.maxsize:
            self._entries.popitem(last=False)
        return arr


class LevirCDDataset(PairedChangeDataset):
    """LEVIR-CD as non-overlapping crops of the official full pairs.

    **Nuisance label.** Every record is :attr:`~cdlib.data.contract.NuisanceLabel.UNKNOWN`
    (``-1``). LEVIR-CD's nuisance content is undocumented; emitting ``CLEAN`` would
    silently corrupt P5's nuisance-stratified evaluation by filling the clean
    stratum with pairs nobody has checked.

    **Grouping keys are the parent tile, never the crop.** The 16 crops of one
    1024x1024 tile are spatially autocorrelated near-duplicates of each other:
    same sensor pass, same date pair, same district, often the same building
    complex spanning a crop boundary. Letting them cross a split is exactly the
    leakage failure mode root ``CLAUDE.md`` rule 2 forbids — a model that memorised
    crop ``r0c1`` scores on ``r0c2`` for free. So ``scene_id`` and ``source_video``
    are the split-qualified stem of the **parent full pair**, shared by all 16
    crops, and ``cdlib.data.splits`` partitions on that; only ``pair_id`` identifies
    the individual crop (``<split>/<stem>_r<row>c<col>``).

    Args:
        root: Dataset root. Read from config; never hardcoded.
        split: One of ``train`` / ``val`` / ``test``. ``val_shift`` and ``cal`` are
            accepted and yield an empty dataset — LEVIR-CD publishes no such
            partition.
        crop_size: Side of the non-overlapping crop, default 256. Parameterised
            only so the test fixture can reproduce the same NxN grid on small
            stand-in tiles; the grid is derived from each image's real size, never
            hardcoded to 4x4. **Do not change it for a reported run** (rule 3).
        dir_a: Reference-frame subdirectory. Default ``"A"``.
        dir_b: Edited-frame subdirectory. Default ``"B"``.
        dir_label: Change-mask subdirectory. Default ``"label"``.
        extensions: Accepted file extensions, matched case-insensitively.
        cache_size: Decoded full tiles held by the LRU. Default 8 — enough for the
            A/B/label of the current tile plus the next one.
        **kwargs: Forwarded to :class:`~cdlib.data.base.PairedChangeDataset`
            (``transform``, ``validate``, ``stats_cache``).
    """

    name = "levir_cd"

    #: 31,066,643 / 667,942,912 changed pixels (``research/02`` §3) — ~5x sparser
    #: than SYSU-CD, which is what drives the loss weighting in ``research/02`` §10.
    published_changed_pixel_ratio = 0.0465

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "train",
        *,
        crop_size: int = DEFAULT_CROP_SIZE,
        dir_a: str = "A",
        dir_b: str = "B",
        dir_label: str = "label",
        extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
        cache_size: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(root, split, **kwargs)
        if int(crop_size) <= 0:
            raise ValueError(f"crop_size must be positive, got {crop_size!r}")
        self.crop_size = int(crop_size)
        self.dir_a = dir_a
        self.dir_b = dir_b
        self.dir_label = dir_label
        self.extensions = _normalise_extensions(extensions)
        self._cache = _BoundedImageCache(cache_size)

    # -- index --------------------------------------------------------------------

    def _build_index(self) -> list[PairRecord]:
        if self.split not in OFFICIAL_FULL_PAIRS:
            log.warning(
                "levir_cd[%s]: LEVIR-CD publishes only %s; returning an empty index "
                "rather than inventing a partition.",
                self.split,
                sorted(OFFICIAL_FULL_PAIRS),
            )
            return []

        split_dir = self.root / self.split
        if not split_dir.is_dir():
            log.warning(
                "levir_cd[%s]: %s is missing — returning an empty index.", self.split, split_dir
            )
            return []

        images_a = _index_by_stem(split_dir / self.dir_a, self.extensions)
        images_b = _index_by_stem(split_dir / self.dir_b, self.extensions)
        labels = _index_by_stem(split_dir / self.dir_label, self.extensions)

        stems = sorted(set(images_a) & set(images_b) & set(labels))
        orphans = len(set(images_a) | set(images_b) | set(labels)) - len(stems)
        if orphans:
            log.warning(
                "levir_cd[%s]: %d stem(s) lack an A/B/label counterpart and were skipped. "
                "Check dir_a/dir_b/dir_label against the archive.",
                self.split,
                orphans,
            )

        records: list[PairRecord] = []
        discarded_px = 0
        n_ragged = 0
        n_too_small = 0
        n_mismatched = 0

        for stem in stems:
            size = _image_size(images_a[stem])
            if _image_size(images_b[stem]) != size or _image_size(labels[stem]) != size:
                n_mismatched += 1
                continue
            height, width = size
            rows, cols = height // self.crop_size, width // self.crop_size
            if rows == 0 or cols == 0:
                n_too_small += 1
                continue
            if rows * self.crop_size != height or cols * self.crop_size != width:
                # The real 1024/256 divides exactly; this only guards a surprise in
                # the archive. Dropping the remainder strip beats padding it, which
                # would feed the model invented pixels, or resizing, which breaks the
                # 0.5 m/px GSD every reported LEVIR-CD number assumes.
                n_ragged += 1
                discarded_px += height * width - rows * cols * self.crop_size**2
            group_id = f"{self.split}/{stem}"
            for row in range(rows):
                for col in range(cols):
                    records.append(
                        PairRecord(
                            pair_id=f"{group_id}_r{row}c{col}",
                            scene_id=group_id,
                            source_video=group_id,
                            nuisance_label=int(NuisanceLabel.UNKNOWN),
                            payload={
                                "a": images_a[stem],
                                "b": images_b[stem],
                                "label": labels[stem],
                                "crop": (
                                    row * self.crop_size,
                                    col * self.crop_size,
                                    self.crop_size,
                                    self.crop_size,
                                ),
                            },
                        )
                    )

        if n_mismatched:
            log.warning(
                "levir_cd[%s]: %d pair(s) whose A/B/label sizes disagree were skipped.",
                self.split,
                n_mismatched,
            )
        if n_too_small:
            log.warning(
                "levir_cd[%s]: %d image(s) smaller than crop_size=%d were skipped.",
                self.split,
                n_too_small,
                self.crop_size,
            )
        if n_ragged:
            log.warning(
                "levir_cd[%s]: %d image(s) are not a whole multiple of crop_size=%d; "
                "dropped %d remainder pixel(s).",
                self.split,
                n_ragged,
                self.crop_size,
                discarded_px,
            )

        expected = OFFICIAL_CROP_COUNTS[self.split]
        if records and self.crop_size == DEFAULT_CROP_SIZE and len(records) != expected:
            log.warning(
                "levir_cd[%s]: built %d crops, the standard protocol is %d "
                "(%d full pairs x %d crops, research/02 §3). Every published LEVIR-CD "
                "number assumes the latter.",
                self.split,
                len(records),
                expected,
                OFFICIAL_FULL_PAIRS[self.split],
                (FULL_TILE_SIZE // DEFAULT_CROP_SIZE) ** 2,
            )

        return records

    # -- pixels -------------------------------------------------------------------

    def _crop(self, full: np.ndarray, rec: PairRecord) -> np.ndarray:
        """Cut this record's crop out of a cached full tile.

        Always a copy: the cache hands the same array to all 16 crops of a tile, and
        the augmentation pipeline is free to write in place.
        """
        y, x, height, width = rec.payload["crop"]
        return full[y : y + height, x : x + width].copy()

    def _read_images(self, rec: PairRecord) -> tuple[np.ndarray, np.ndarray]:
        full_a = self._cache.get(rec.payload["a"], _read_rgb)
        full_b = self._cache.get(rec.payload["b"], _read_rgb)
        return self._crop(full_a, rec), self._crop(full_b, rec)

    def _read_mask(self, rec: PairRecord) -> np.ndarray:
        return self._crop(self._cache.get(rec.payload["label"], _read_label), rec)


__all__ = [
    "DEFAULT_CROP_SIZE",
    "DEFAULT_IMAGE_EXTENSIONS",
    "FULL_TILE_SIZE",
    "OFFICIAL_CROP_COUNTS",
    "OFFICIAL_FULL_PAIRS",
    "LevirCDDataset",
]
