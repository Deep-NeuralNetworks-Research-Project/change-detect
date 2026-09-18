"""``PairedChangeDataset`` — the base every dataset loader subclasses.

It owns the two things that must behave identically across all four datasets:

1. **Assembling the frozen ``__getitem__`` dict** (see :mod:`cdlib.data.contract`), so a
   loader author only writes "how do I read this dataset's files off disk", never
   "what shape does the trainer want".
2. **The changed-pixel ratio** that P2 wires into loss weighting and P5 reports
   alongside every result.

Subclasses implement three methods: :meth:`_build_index`, :meth:`_read_images` and
:meth:`_read_mask`.

Two constraints come from outside and are load-bearing:

* **Construction must succeed with no data on disk.** CI dry-constructs every
  ``configs/data/*.yaml`` through ``build_dataset`` on a runner with no datasets
  attached (``research/06`` §4). The index is therefore built lazily and a missing
  root yields an empty dataset, not an exception.
* **Roots come from config, never from a hardcoded runtime path.** A pre-commit hook
  rejects ``/content/`` and ``/kaggle/`` outside ``notebooks/``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from torch.utils.data import Dataset

from cdlib.data.contract import (
    IGNORE_INDEX,
    NuisanceLabel,
    changed_pixel_counts,
    nuisance_name,
    validate_sample,
)

log = logging.getLogger(__name__)

#: Bumped whenever mask decoding changes, so cached stats computed under the old rule
#: are recognised as stale. The index fingerprint alone cannot see this: fixing the
#: anti-aliasing bug changed every PCD ratio while leaving pair ids and counts
#: identical, which would otherwise have served the pre-fix number forever.
STATS_SCHEMA_VERSION: int = 2

#: Change-size bins, as fractions of *valid* pixels changed in a pair. P5 reports
#: results by condition x severity x change-size bin (``research/05`` §2.1), and the
#: bin edges have to be agreed somewhere — here.
CHANGE_SIZE_BINS: tuple[tuple[str, float, float], ...] = (
    ("none", 0.0, 0.0),
    ("tiny", 0.0, 0.001),
    ("small", 0.001, 0.01),
    ("medium", 0.01, 0.05),
    ("large", 0.05, 0.20),
    ("huge", 0.20, 1.01),
)


def change_size_bin(fraction: float) -> str:
    """Name the change-size bin a per-pair changed fraction falls into."""
    if fraction <= 0.0:
        return "none"
    for name, lo, hi in CHANGE_SIZE_BINS[1:]:
        if lo < fraction <= hi:
            return name
    return CHANGE_SIZE_BINS[-1][0]


@dataclass(slots=True)
class PairRecord:
    """One (reference, edited) pair, before any pixels are read.

    ``scene_id`` and ``source_video`` are what :mod:`cdlib.data.splits` partitions on —
    never the pair index, because adjacent frames are near-duplicates and a
    frame-level split silently invalidates the benchmark (root ``CLAUDE.md`` rule 2).

    ``payload`` carries whatever the specific loader needs to read the pair (file
    paths, a crop box, a fold number). Nothing outside the owning loader reads it.
    """

    pair_id: str
    scene_id: str
    source_video: str
    frame_idx: tuple[int, int] = (0, 0)
    nuisance_label: int = int(NuisanceLabel.UNKNOWN)
    payload: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class PairTransform(Protocol):
    """Calling convention for the augmentation pipeline.

    Implemented by :class:`cdlib.data.transforms.PairedTransform`. Takes and returns
    ``(img1 HWC uint8, img2 HWC uint8, mask HW int16 in {0,1,-1})``. The transform is
    responsible for keeping geometric ops shared across the pair and photometric ops
    independent per frame (root ``CLAUDE.md`` rule 4).
    """

    def __call__(
        self, img1: np.ndarray, img2: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    """HWC uint8 RGB -> CHW float32 in ``[0,1]``."""
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
    return t.to(torch.float32).div_(255.0)


def binarize_mask(arr: np.ndarray, *, threshold: int | None = None) -> np.ndarray:
    """Decode a raw label image to ``HW int16`` in ``{0, 1}``.

    Change masks do not ship in one encoding. Across the datasets we actually use:
    LEVIR-CD is ``{0, 255}`` bar 709 stray pixels (see below); SYSU-CD is ``{0, 1}``;
    and **PCD/TSUNAMI masks are anti-aliased** — 98.99% of pixels are 0 or 255, but
    1.01% form a soft ring of intermediate values around every changed region.

    The obvious rule, ``arr > 0``, silently promotes that entire ring to "changed".
    It inflates the changed-pixel ratio, and because the error lies exactly on object
    boundaries it lands on precisely what P5's boundary-IoU measures
    (``research/05`` §1). Over PCD's 100 masks it moves pi from 28.3170% to 28.4452%.

    So the default threshold adapts to the observed encoding: ``1`` when the peak
    value is ``<= 1`` (a ``{0,1}`` mask, where ``>= 1`` is the only sensible rule),
    otherwise the midpoint ``(peak + 1) // 2``. On a near-binary mask this is a
    no-op — LEVIR's measured ratio is unchanged to the digit — and on an
    anti-aliased one it thresholds instead of promoting.

    On LEVIR the no-op is a measured result, not an assumption. 32 of its 637 label
    images do carry stray values (709 pixels total at 156 and 254, 0.0001% of the
    corpus, concentrated in two annotation batches). Both values sit above the
    midpoint 128, so this rule and ``arr > 0`` agree on every one of them and pi is
    identical either way. A stricter rule such as ``arr == 255`` would not agree —
    which is the reason to route every mask through this helper rather than
    open-coding a comparison.

    Pass ``threshold`` explicitly to pin the rule when a dataset's encoding is known
    and you would rather not depend on what happens to be in one image.

    Args:
        arr: Raw label image, any integer dtype.
        threshold: Values ``>= threshold`` become 1. ``None`` selects the adaptive
            rule above.

    Returns:
        ``HW int16`` in ``{0, 1}``.
    """
    a = np.asarray(arr)
    if a.ndim == 3:
        a = a[..., 0]
    if threshold is None:
        peak = int(a.max()) if a.size else 0
        threshold = 1 if peak <= 1 else (peak + 1) // 2
    return (a >= threshold).astype(np.int16)


def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    """HW ``{0,1,-1}`` -> ``[1,H,W]`` float32.

    Expects a mask a loader has **already** binarised with :func:`binarize_mask`.
    The ``> 0`` rule here is correct for that alphabet and only for it — raw label
    images must never reach this function, or an anti-aliased boundary becomes
    changed ground truth.

    ``IGNORE_INDEX`` is preserved exactly.
    """
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    out = np.zeros(m.shape, dtype=np.float32)
    out[m > 0] = 1.0
    out[m == IGNORE_INDEX] = float(IGNORE_INDEX)
    return torch.from_numpy(out).unsqueeze(0)


class PairedChangeDataset(Dataset, ABC):
    """Base class for every dataset in ``cdlib.data.datasets``."""

    #: Registry key / value of ``meta["dataset"]``. Subclasses must set it.
    name: str = "unnamed"

    #: Published changed-pixel ratio, used as a clearly-labelled fallback when the
    #: data is not on disk. ``None`` when no ratio has been published.
    published_changed_pixel_ratio: float | None = None

    def __init__(
        self,
        root: str | Path | None,
        split: str,
        *,
        transform: PairTransform | None = None,
        validate: bool = False,
        stats_cache: bool = True,
        **kwargs: Any,
    ) -> None:
        if root is None:
            import os
            base_dir = os.environ.get("CDLIB_DATA_ROOT", "data")
            self.root = Path(base_dir).expanduser() / self.name
        else:
            self.root = Path(root).expanduser()

        self.split = split
        self.transform = transform
        self.validate = validate
        self.stats_cache = stats_cache
        self.options: dict[str, Any] = dict(kwargs)
        self._records: list[PairRecord] | None = None
        self._stats: dict[str, Any] | None = None

    # -- subclass hooks -----------------------------------------------------------

    @abstractmethod
    def _build_index(self) -> list[PairRecord]:
        """Enumerate the pairs in ``self.split``. Called once, lazily.

        Must return ``[]`` rather than raise when ``self.root`` does not exist — CI
        constructs every dataset with no data present.
        """

    @abstractmethod
    def _read_images(self, rec: PairRecord) -> tuple[np.ndarray, np.ndarray]:
        """Read one pair as ``(HWC uint8 RGB, HWC uint8 RGB)``."""

    @abstractmethod
    def _read_mask(self, rec: PairRecord) -> np.ndarray:
        """Read one change mask as ``HW`` with values in ``{0, 1, IGNORE_INDEX}``.

        Kept separate from :meth:`_read_images` so that the changed-pixel-ratio scan
        never decodes the RGB frames — that is the difference between seconds and
        minutes on SYSU-CD's 20,000 pairs.
        """

    # -- index --------------------------------------------------------------------

    @property
    def records(self) -> list[PairRecord]:
        """The pair index, built on first access."""
        if self._records is None:
            if not self.root.exists():
                log.warning(
                    "%s[%s]: root %s does not exist — constructing an empty dataset. "
                    "This is expected in CI; run scripts/download_%s.sh to populate it.",
                    self.name,
                    self.split,
                    self.root,
                    self.name,
                )
                self._records = []
            else:
                self._records = self._build_index()
        return self._records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        img1, img2 = self._read_images(rec)
        mask = self._read_mask(rec)

        if self.transform is not None:
            img1, img2, mask = self.transform(img1, img2, mask)

        sample = {
            "img1": image_to_tensor(img1),
            "img2": image_to_tensor(img2),
            "mask": mask_to_tensor(mask),
            "nuisance_label": torch.tensor(int(rec.nuisance_label), dtype=torch.int64),
            "meta": {
                "source_video": rec.source_video,
                "scene_id": rec.scene_id,
                "frame_idx": (int(rec.frame_idx[0]), int(rec.frame_idx[1])),
                "pair_id": rec.pair_id,
                "dataset": self.name,
            },
        }
        if self.validate:
            validate_sample(sample, name=f"{self.name}[{self.split}][{idx}]")
        return sample

    # -- changed-pixel ratio ------------------------------------------------------

    @property
    def changed_pixel_ratio(self) -> float:
        """``pi`` for this split: changed valid pixels / total valid pixels.

        This is the number P2 recomputes loss weights from at each training stage
        (``research/02`` §10) and P5 reports next to every result. Ignore pixels are
        excluded from *both* numerator and denominator.

        Falls back to :attr:`published_changed_pixel_ratio` when the data is absent;
        :meth:`changed_pixel_stats` says which of the two you got via its ``source``
        field, so a published constant can never be mistaken for a measurement.
        """
        return float(self.changed_pixel_stats()["overall"])

    def changed_pixel_stats(self, *, force: bool = False) -> dict[str, Any]:
        """Full ratio breakdown, cached to disk.

        Returns ``overall``, ``by_nuisance``, ``by_change_size_bin``, ``n_pairs``,
        ``n_negative_pairs`` and ``source`` (``"computed"`` or ``"published"``).
        The per-stratum breakdowns are what ``research/05`` §2.1 asks for: ``pi``
        stated per nuisance stratum and per change-size bin, not one scalar.
        """
        if self._stats is not None and not force:
            return self._stats

        cache_path = self._stats_cache_path()
        if cache_path is not None and cache_path.exists() and not force:
            try:
                cached = json.loads(cache_path.read_text())
                if cached.get("index_fingerprint") == self._index_fingerprint():
                    self._stats = cached
                    return cached
                log.info("%s[%s]: stats cache is stale, recomputing", self.name, self.split)
            except (OSError, json.JSONDecodeError) as exc:
                log.warning(
                    "%s[%s]: unreadable stats cache (%s), recomputing", self.name, self.split, exc
                )

        self._stats = self._compute_stats()
        self._write_stats_cache(cache_path, self._stats)
        return self._stats

    def _compute_stats(self) -> dict[str, Any]:
        records = self.records
        if not records:
            return {
                "dataset": self.name,
                "split": self.split,
                "overall": float(self.published_changed_pixel_ratio or 0.0),
                "by_nuisance": {},
                "by_change_size_bin": {},
                "n_pairs": 0,
                "n_negative_pairs": 0,
                "source": "published" if self.published_changed_pixel_ratio else "unavailable",
                "index_fingerprint": self._index_fingerprint(),
            }

        total_changed = 0
        total_valid = 0
        n_negative = 0
        per_nuisance: dict[str, list[int]] = {}
        per_bin: dict[str, int] = {}

        for rec in records:
            mask = mask_to_tensor(self._read_mask(rec))
            changed, valid = changed_pixel_counts(mask)
            total_changed += changed
            total_valid += valid
            if valid > 0 and changed == 0:
                n_negative += 1

            key = nuisance_name(rec.nuisance_label)
            acc = per_nuisance.setdefault(key, [0, 0])
            acc[0] += changed
            acc[1] += valid

            frac = changed / valid if valid else 0.0
            bin_name = change_size_bin(frac)
            per_bin[bin_name] = per_bin.get(bin_name, 0) + 1

        return {
            "dataset": self.name,
            "split": self.split,
            "overall": (total_changed / total_valid) if total_valid else 0.0,
            "by_nuisance": {
                k: {"ratio": (c / v) if v else 0.0, "changed_px": c, "valid_px": v}
                for k, (c, v) in sorted(per_nuisance.items())
            },
            "by_change_size_bin": {
                name: per_bin.get(name, 0) for name, _, _ in CHANGE_SIZE_BINS if name in per_bin
            },
            "n_pairs": len(records),
            "n_negative_pairs": n_negative,
            "changed_px": total_changed,
            "valid_px": total_valid,
            "source": "computed",
            "index_fingerprint": self._index_fingerprint(),
        }

    # -- stats cache plumbing -----------------------------------------------------

    def _index_fingerprint(self) -> str:
        """Cheap signature of the current index, so a stale cache is detected."""
        records = self._records if self._records is not None else []
        h = hashlib.sha256()
        h.update(f"{self.name}|{self.split}|{len(records)}".encode())
        # The decoding rule is part of the identity of a cached statistic, not just the
        # index it was computed over.
        h.update(f"|v{STATS_SCHEMA_VERSION}|thr={getattr(self, 'mask_threshold', None)}".encode())
        for rec in records[:64]:
            h.update(f"|{rec.pair_id}|{rec.nuisance_label}".encode())
        if records:
            h.update(f"|{records[-1].pair_id}".encode())
        return h.hexdigest()[:16]

    def _stats_cache_path(self) -> Path | None:
        if not self.stats_cache:
            return None
        return self.root / ".cdlib_stats" / f"{self.name}_{self.split}.json"

    def _write_stats_cache(self, path: Path | None, stats: dict[str, Any]) -> None:
        if path is None or stats.get("source") != "computed":
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(stats, indent=2, sort_keys=True))
        except OSError as exc:
            # Read-only dataset roots are normal on Kaggle (/kaggle/input is mounted
            # read-only). Losing the cache costs time, never correctness.
            log.info("%s[%s]: could not write stats cache (%s)", self.name, self.split, exc)

    def __repr__(self) -> str:
        n = len(self._records) if self._records is not None else "?"
        return f"{type(self).__name__}(name={self.name!r}, split={self.split!r}, n={n}, root={str(self.root)!r})"


__all__ = [
    "CHANGE_SIZE_BINS",
    "STATS_SCHEMA_VERSION",
    "PairRecord",
    "PairTransform",
    "PairedChangeDataset",
    "binarize_mask",
    "change_size_bin",
    "image_to_tensor",
    "mask_to_tensor",
]
