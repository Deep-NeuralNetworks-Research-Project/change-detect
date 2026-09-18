"""PCD / Panoramic Change Detection (Sakurada & Okatani, BMVC 2015).

Two 100-pair subsets of 224x1024 equirectangular street-level panoramas, TSUNAMI
and GSV, with binary change masks (``research/02`` §4).

Three things about this dataset drive the whole module:

* **There are no train/val/test directories.** The published protocol is 5-fold
  cross-validation with 20-pair folds (``research/02`` §4, the protocol CSCDNet,
  DR-TANet and HPCFNet all report against). The split is therefore *computed*:
  ``fold`` selects which 20 panoramas are held out and ``split`` selects which side
  of that partition you get.
* **Folds are assigned per panorama, before any cropping** (``research/02`` §8:
  build the split at panorama level *before* cropping). Every patch of a panorama
  inherits its panorama's fold and carries the panorama as its ``scene_id``. A
  panorama whose patches straddle a fold boundary puts near-duplicate pixels on both
  sides of the evaluation — the exact leakage failure mode of root ``CLAUDE.md``
  rule 2.
* **GSV is not hosted anywhere** (``research/02`` §4 and the P1 role file: "do not
  let GSV gate any deliverable"). Asking for it when it is absent logs a warning
  naming that fact and yields an empty dataset. It never raises, so nothing
  downstream can be blocked by a dataset that cannot be downloaded.

``nuisance_label`` is :attr:`~cdlib.data.contract.NuisanceLabel.UNKNOWN` for every
pair: PCD's nuisance content is undocumented, and per the frozen contract "if I
don't know, it's -1, not 0". No changed-pixel ratio has been published for PCD
(``research/02`` §1 lists it as "not published"), so
:attr:`published_changed_pixel_ratio` stays ``None`` and the base class reports
``source: "unavailable"`` rather than a made-up number when the data is absent.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cdlib.data.base import PairedChangeDataset, PairRecord, PairTransform, binarize_mask
from cdlib.data.contract import SPLITS, NuisanceLabel

log = logging.getLogger(__name__)

#: The published protocol's fold count (``research/02`` §4).
N_FOLDS: int = 5

#: Fold-assignment schemes. ``contiguous`` is the default because consecutive PCD
#: panorama ids are consecutive street-level captures: keeping neighbours in the
#: same fold is what stops a near-duplicate viewpoint appearing in train and test of
#: the same fold. ``interleaved`` (``id index % n_folds``) is offered only to
#: reproduce a paper that used it, and is documented as leakage-prone.
FOLD_SCHEMES: tuple[str, ...] = ("contiguous", "interleaved")


def pcd_fold_assignment(
    panorama_ids: Sequence[str],
    *,
    n_folds: int = N_FOLDS,
    scheme: str = "contiguous",
) -> dict[str, int]:
    """Map each panorama id to its fold index.

    Args:
        panorama_ids: Panorama ids; sorted internally so the result never depends on
            filesystem enumeration order.
        n_folds: Number of cross-validation folds. ``5`` is the published protocol.
        scheme: One of :data:`FOLD_SCHEMES`.

    Returns:
        ``{panorama_id: fold_index}`` covering every input id exactly once.

    Raises:
        ValueError: If ``n_folds`` < 1 or ``scheme`` is not in :data:`FOLD_SCHEMES`.
    """
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    if scheme not in FOLD_SCHEMES:
        raise ValueError(f"unknown fold scheme {scheme!r}; expected one of {list(FOLD_SCHEMES)}")

    ids = sorted(panorama_ids)
    assignment: dict[str, int] = {}
    if scheme == "interleaved":
        for i, pid in enumerate(ids):
            assignment[pid] = i % n_folds
        return assignment

    # Contiguous blocks. 100 panoramas / 5 folds = the published 20-pair folds; a
    # non-divisible count (a partial download, or the test fixtures) puts the
    # remainder in the earliest folds so the sizes stay within one of each other.
    base, extra = divmod(len(ids), n_folds)
    start = 0
    for fold in range(n_folds):
        size = base + (1 if fold < extra else 0)
        for pid in ids[start : start + size]:
            assignment[pid] = fold
        start += size
    return assignment


def _crop_positions(total: int, size: int, stride: int) -> list[int]:
    """Window origins along one axis, with the final window clamped to the border.

    Clamping rather than padding keeps every patch a real image region: PCD
    panoramas are 1024 wide, which is not a multiple of the usual 224 crop, and
    padding would feed the network a synthetic border that exists in no real pair.
    """
    if size >= total:
        return [0]
    positions = list(range(0, total - size + 1, max(1, stride)))
    if positions[-1] != total - size:
        positions.append(total - size)
    return positions


class PCDDataset(PairedChangeDataset):
    """PCD panoramas under the published 5-fold cross-validation protocol.

    On-disk layout (``research/02`` §4)::

        <root>/<subset>/t0/NNNNN.jpg
        <root>/<subset>/t1/NNNNN.jpg
        <root>/<subset>/mask/NNNNN.bmp

    Every directory name and file extension is a constructor argument, because the
    only live copy of this dataset is one researcher's Drive link and the archive's
    internal naming cannot be re-verified until it is downloaded — a surprise there
    must be a config change, not a code change.

    **Split semantics.** ``fold`` picks the held-out 20-pair fold; ``split`` picks
    which side of it you get:

    * ``test`` — the held-out fold. This is the published protocol.
    * ``train`` — the other four folds, minus whatever ``val_fraction`` and
      ``cal_fraction`` carve off.
    * ``val`` — **our decision, not part of the published protocol.** The published
      protocol trains on all four remaining folds and reports on the held-out one;
      it names no validation set, but we need one for early stopping and threshold
      selection, and reusing ``test`` for that would leak. So ``val`` is a
      deterministic tail slice of the training folds, taken at *panorama* level:
      with the default ``val_fraction=0.125`` that is the last 10 of the 80 training
      panoramas, which under the default contiguous fold scheme is a spatially
      contiguous block rather than a scatter of near-duplicates. Set
      ``val_fraction=0.0`` to reproduce the published protocol exactly (``train``
      becomes the full four folds and ``val`` is empty).
    * ``cal`` — same mechanism, disjoint from ``val``, off by default
      (``cal_fraction=0.0``). ``research/04`` §7 wants a third split for conformal
      prediction; enabling it costs training panoramas, so it is opt-in.
    * ``val_shift`` — always empty. ``research/04`` §4 defines it as a
      *nuisance-shift-heavy* slice, and PCD's nuisance content is undocumented
      (every pair is ``UNKNOWN``), so there is no honest way to build one here.

    Args:
        root: Dataset root containing the ``<subset>`` directory.
        split: One of :data:`~cdlib.data.contract.SPLITS`.
        fold: Held-out fold index, ``0..n_folds-1``.
        n_folds: Fold count; ``5`` is the published protocol.
        subset: ``"TSUNAMI"`` or ``"GSV"``. GSV is not hosted (``research/02`` §4);
            an absent subset directory yields an empty dataset with a warning.
        fold_scheme: See :data:`FOLD_SCHEMES`.
        val_fraction: Fraction of the training panoramas carved off as ``val``.
        cal_fraction: Fraction of the training panoramas carved off as ``cal``.
        dir_t0, dir_t1, dir_mask: Subdirectory names.
        image_ext, mask_ext: File extensions for frames and masks.
        crop_size: ``(height, width)`` patch size, or ``None`` for whole panoramas.
            Patches inherit their panorama's fold and ``scene_id``.
        crop_stride: ``(dy, dx)``; defaults to ``crop_size`` (non-overlapping).
        transform: Paired augmentation pipeline.
        validate: Validate every assembled sample against the frozen contract.
        stats_cache: Cache the changed-pixel-ratio scan under ``<root>/.cdlib_stats``.

    Raises:
        ValueError: For an unknown ``split``, an out-of-range ``fold``, or holdout
            fractions that leave no training panoramas.
    """

    name = "pcd"

    #: ``research/02`` §1 lists PCD's changed-pixel ratio as "not published". Leaving
    #: this ``None`` makes ``changed_pixel_stats()["source"]`` report ``"unavailable"``
    #: when the data is absent, instead of laundering a guess into P2's loss weights.
    published_changed_pixel_ratio: float | None = None

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        fold: int = 0,
        n_folds: int = N_FOLDS,
        subset: str = "TSUNAMI",
        fold_scheme: str = "contiguous",
        val_fraction: float = 0.125,
        cal_fraction: float = 0.0,
        dir_t0: str = "t0",
        dir_t1: str = "t1",
        dir_mask: str = "mask",
        image_ext: str = ".jpg",
        mask_ext: str = ".png",
        mask_threshold: int | None = None,
        crop_size: tuple[int, int] | None = None,
        crop_stride: tuple[int, int] | None = None,
        transform: PairTransform | None = None,
        validate: bool = False,
        stats_cache: bool = True,
        **kwargs: Any,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {list(SPLITS)}")
        if n_folds < 1:
            raise ValueError(f"n_folds must be >= 1, got {n_folds}")
        if not 0 <= fold < n_folds:
            raise ValueError(f"fold must be in 0..{n_folds - 1}, got {fold}")
        if fold_scheme not in FOLD_SCHEMES:
            raise ValueError(
                f"unknown fold scheme {fold_scheme!r}; expected one of {list(FOLD_SCHEMES)}"
            )
        if val_fraction < 0 or cal_fraction < 0 or val_fraction + cal_fraction >= 1:
            raise ValueError(
                f"val_fraction + cal_fraction must be in [0, 1), got "
                f"{val_fraction} + {cal_fraction}"
            )

        super().__init__(
            root,
            split,
            transform=transform,
            validate=validate,
            stats_cache=stats_cache,
            **kwargs,
        )
        self.fold = int(fold)
        self.n_folds = int(n_folds)
        self.subset = subset
        self.fold_scheme = fold_scheme
        self.val_fraction = float(val_fraction)
        self.cal_fraction = float(cal_fraction)
        self.dir_t0 = dir_t0
        self.dir_t1 = dir_t1
        self.dir_mask = dir_mask
        self.image_ext = image_ext
        self.mask_ext = mask_ext
        self.mask_threshold = mask_threshold
        self.crop_size = tuple(crop_size) if crop_size is not None else None  # type: ignore[assignment]
        self.crop_stride = tuple(crop_stride) if crop_stride is not None else None  # type: ignore[assignment]
        # One-panorama decode cache. With cropping on, the index is patch-major
        # within a panorama, so caching the most recent panorama turns N decodes per
        # panorama back into one. Off when reading whole panoramas, where it would
        # only pin memory.
        self._cache_pano: str | None = None
        self._cache: dict[str, np.ndarray] = {}

    # -- fold plumbing ----------------------------------------------------------------

    @classmethod
    def cv_folds(
        cls,
        root: str | Path,
        *,
        splits: Sequence[str] = ("train", "val", "test"),
        n_folds: int = N_FOLDS,
        **kwargs: Any,
    ) -> list[dict[str, PCDDataset]]:
        """Instantiate the whole cross-validation protocol.

        P5 runs PCD as 5-fold CV and averages; this exists so that loop is
        ``for fold in PCDDataset.cv_folds(root): ...`` rather than a reimplementation
        of the fold arithmetic at the call site.

        Args:
            root: Dataset root.
            splits: Splits to build per fold.
            n_folds: Fold count.
            **kwargs: Forwarded to every constructed dataset.

        Returns:
            One ``{split_name: dataset}`` dict per fold, in fold order.
        """
        return [
            {s: cls(root, s, fold=f, n_folds=n_folds, **kwargs) for s in splits}
            for f in range(n_folds)
        ]

    @property
    def panorama_ids(self) -> list[str]:
        """Sorted panorama ids present in this split (not patch ids)."""
        return sorted({rec.scene_id for rec in self.records})

    def fold_assignment(self) -> dict[str, int]:
        """``{panorama_id: fold}`` over every panorama on disk, split-independent."""
        return pcd_fold_assignment(
            self._available_panoramas(), n_folds=self.n_folds, scheme=self.fold_scheme
        )

    # -- index ------------------------------------------------------------------------

    def _subset_dir(self) -> Path:
        return self.root / self.subset

    def _available_panoramas(self) -> list[str]:
        """Panorama ids with all three files present, sorted."""
        subset_dir = self._subset_dir()
        if not subset_dir.is_dir():
            return []
        t0_dir, t1_dir, mask_dir = (
            subset_dir / self.dir_t0,
            subset_dir / self.dir_t1,
            subset_dir / self.dir_mask,
        )
        stems = sorted(p.stem for p in t0_dir.glob(f"*{self.image_ext}"))
        complete, incomplete = [], []
        for stem in stems:
            if (t1_dir / f"{stem}{self.image_ext}").is_file() and (
                mask_dir / f"{stem}{self.mask_ext}"
            ).is_file():
                complete.append(stem)
            else:
                incomplete.append(stem)
        if incomplete:
            log.warning(
                "%s[%s]: skipping %d panorama(s) missing a t1 frame or a mask (e.g. %s)",
                self.name,
                self.split,
                len(incomplete),
                incomplete[:3],
            )
        return complete

    def _split_panoramas(self, panorama_ids: Sequence[str]) -> list[str]:
        """Select the panoramas belonging to ``self.split`` for ``self.fold``."""
        if self.split == "val_shift":
            # No honest nuisance-shift slice exists for a dataset whose nuisance
            # content is undocumented; see the class docstring.
            log.info(
                "%s[val_shift]: PCD carries no nuisance annotation, so there is no "
                "shift-heavy slice to hold out — returning an empty dataset "
                "(research/04 §4)",
                self.name,
            )
            return []

        assignment = pcd_fold_assignment(
            panorama_ids, n_folds=self.n_folds, scheme=self.fold_scheme
        )
        if self.split == "test":
            return [pid for pid in sorted(panorama_ids) if assignment[pid] == self.fold]

        pool = [pid for pid in sorted(panorama_ids) if assignment[pid] != self.fold]
        n_val = int(round(self.val_fraction * len(pool)))
        n_cal = int(round(self.cal_fraction * len(pool)))
        n_train = len(pool) - n_val - n_cal
        if n_train < 0:  # pragma: no cover - guarded by the constructor
            raise ValueError(
                f"val_fraction + cal_fraction carve {n_val + n_cal} of {len(pool)} "
                "training panoramas, leaving none to train on"
            )
        if self.split == "train":
            return pool[:n_train]
        if self.split == "val":
            return pool[n_train : n_train + n_val]
        return pool[n_train + n_val :]  # cal

    def _build_index(self) -> list[PairRecord]:
        subset_dir = self._subset_dir()
        if not subset_dir.is_dir():
            if self.subset.upper() == "GSV":
                # research/02 §4: only TSUNAMI has a live link; the GSV half of PCD is
                # not hosted anywhere and no third-party mirror exists. This is a
                # warning and an empty dataset by design — GSV must never gate a
                # deliverable (P1 role file, risk 1).
                log.warning(
                    "%s[%s]: PCD-GSV is not hosted anywhere (research/02 §4: the GSV "
                    "link on sakuradaken.net is not live and no mirror exists), and %s "
                    "is absent. Returning an empty dataset — run TSUNAMI instead, or "
                    "email the authors for GSV.",
                    self.name,
                    self.split,
                    subset_dir,
                )
            else:
                log.warning(
                    "%s[%s]: subset directory %s does not exist — returning an empty "
                    "dataset. Expected layout <root>/%s/{%s,%s,%s}/.",
                    self.name,
                    self.split,
                    subset_dir,
                    self.subset,
                    self.dir_t0,
                    self.dir_t1,
                    self.dir_mask,
                )
            return []

        panoramas = self._available_panoramas()
        if not panoramas:
            log.warning(
                "%s[%s]: no complete panorama triples under %s", self.name, self.split, subset_dir
            )
            return []
        selected = self._split_panoramas(panoramas)

        records: list[PairRecord] = []
        for pid in selected:
            scene = f"{self.subset}_{pid}"  # subset-qualified: TSUNAMI and GSV reuse ids
            payload_base = {
                "t0": str(self._subset_dir() / self.dir_t0 / f"{pid}{self.image_ext}"),
                "t1": str(self._subset_dir() / self.dir_t1 / f"{pid}{self.image_ext}"),
                "mask": str(self._subset_dir() / self.dir_mask / f"{pid}{self.mask_ext}"),
                "panorama": pid,
                "fold": self.fold,
            }
            if self.crop_size is None:
                records.append(
                    PairRecord(
                        pair_id=scene,
                        scene_id=scene,
                        source_video=scene,
                        # PCD is a t0/t1 pair, not a video: (0, 1) reads as
                        # "first capture, second capture".
                        frame_idx=(0, 1),
                        nuisance_label=int(NuisanceLabel.UNKNOWN),
                        payload=dict(payload_base),
                    )
                )
                continue
            for y, x, ch, cw in self._boxes_for(payload_base["mask"]):
                records.append(
                    PairRecord(
                        pair_id=f"{scene}_y{y:04d}x{x:04d}",
                        # scene_id stays the *panorama*: every patch of a panorama
                        # must land on the same side of every split (research/02 §8).
                        scene_id=scene,
                        source_video=scene,
                        frame_idx=(0, 1),
                        nuisance_label=int(NuisanceLabel.UNKNOWN),
                        payload={**payload_base, "box": (y, x, ch, cw)},
                    )
                )
        return records

    def _boxes_for(self, mask_path: str) -> list[tuple[int, int, int, int]]:
        """Crop boxes ``(y, x, h, w)`` for one panorama, read from the mask's size."""
        assert self.crop_size is not None
        with Image.open(mask_path) as im:
            width, height = im.size
        crop_h = min(int(self.crop_size[0]), height)
        crop_w = min(int(self.crop_size[1]), width)
        stride_h, stride_w = self.crop_stride or (crop_h, crop_w)
        return [
            (y, x, crop_h, crop_w)
            for y in _crop_positions(height, crop_h, int(stride_h))
            for x in _crop_positions(width, crop_w, int(stride_w))
        ]

    # -- pixels -----------------------------------------------------------------------

    def _load(self, rec: PairRecord, key: str, mode: str) -> np.ndarray:
        """Decode one file for ``rec``, via the one-panorama cache when cropping."""
        if self.crop_size is None:
            with Image.open(rec.payload[key]) as im:
                return np.array(im.convert(mode))

        pano = str(rec.payload["panorama"])
        if self._cache_pano != pano:
            self._cache.clear()
            self._cache_pano = pano
        cached = self._cache.get(key)
        if cached is None:
            with Image.open(rec.payload[key]) as im:
                cached = np.array(im.convert(mode))
            self._cache[key] = cached
        y, x, h, w = rec.payload["box"]
        # Copy: the cached panorama is shared across patches and a transform may
        # write into what we return.
        return np.ascontiguousarray(cached[y : y + h, x : x + w])

    def _read_images(self, rec: PairRecord) -> tuple[np.ndarray, np.ndarray]:
        # PIL, not cv2.imread: cv2 hands back BGR and a silently channel-swapped
        # pair is invisible in every shape test.
        return self._load(rec, "t0", "RGB"), self._load(rec, "t1", "RGB")

    def _read_mask(self, rec: PairRecord) -> np.ndarray:
        raw = self._load(rec, "mask", "L")
        # PCD masks are ANTI-ALIASED, not binary — verified against the real archive:
        # 98.99% of pixels are 0 or 255, and 1.01% form a soft ring around every
        # changed region. `raw > 0` would promote that whole ring to "changed", which
        # biases exactly the boundary P5's boundary-IoU measures. Hence the shared
        # threshold. PCD has no ignore convention, so nothing here yields IGNORE_INDEX.
        return binarize_mask(raw, threshold=self.mask_threshold)

    # -- stats ------------------------------------------------------------------------

    def _stats_cache_path(self) -> Path | None:
        # The base path keys on (name, split), but here five folds and two subsets
        # share those two strings and would all overwrite one file. The fingerprint
        # check would catch the staleness, but only by rescanning every time.
        path = super()._stats_cache_path()
        if path is None:
            return None
        crop = "full" if self.crop_size is None else f"{self.crop_size[0]}x{self.crop_size[1]}"
        return path.with_name(f"{self.name}_{self.subset}_fold{self.fold}_{self.split}_{crop}.json")


__all__ = ["FOLD_SCHEMES", "N_FOLDS", "PCDDataset", "pcd_fold_assignment"]
