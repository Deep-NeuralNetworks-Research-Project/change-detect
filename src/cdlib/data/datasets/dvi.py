"""DAVIS / YouTube-VOS video-inpainting localisation — the scale fallback.

`research/02` §7 pairs this with VideoSham as the second escape hatch: where VideoSham
is the closest *structural* analog (few pairs, professionally edited), DVI/YTVI is the
**scale** option — pixel-accurate per-frame masks for the object-removal case
specifically, with a documented F1/mIoU protocol.

Composition, per `research/02` §7:

* **DVI** — DAVIS-2016 based, 30 train / 20 test sequences, three inpainting methods
  (VI, OP, CP), 50 videos each (BMVC'21, arXiv:2101.11080).
* **YTVI** — YouTube-VOS 2018 based, 3,471 videos / 5,945 instances.

The pair is *(inpainted frame, original frame)* and the mask is the removed object's
DAVIS ground-truth segmentation, so unlike our other sources the label comes for free
and is exact. That makes it a pretraining source for object removal, **not** a
substitute for the target domain: it contains one edit type, applied synthetically,
with none of the grading or codec nuisance real post-production carries.

**Not downloaded.** **Layout is provisional** — the sources ship as separate video and
mask archives per inpainting method, so a build step is required; every directory name
below is a constructor argument.

**Nuisance label.** ``UNKNOWN`` (-1). The inpainting method is a generation artefact,
not one of our nuisance classes, and mapping it in would corrupt P5's strata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cdlib.data.base import PairedChangeDataset, PairRecord, binarize_mask
from cdlib.data.contract import NuisanceLabel

log = logging.getLogger(__name__)

#: Inpainting methods DVI is built from (`research/02` §7). Each is a separate
#: rendering of the same DAVIS sequences, so they must never be split apart —
#: three renderings of one sequence are near-duplicates by construction.
INPAINTING_METHODS: tuple[str, ...] = ("VI", "OP", "CP")

#: DAVIS-2016 sequence counts (`research/02` §7).
DVI_SEQUENCES: dict[str, int] = {"train": 30, "test": 20}

DEFAULT_IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg")


class DVIDataset(PairedChangeDataset):
    """Inpainted/original frame pairs with the removed object's mask as ground truth.

    Expected layout, one directory per sequence::

        <root>/<method>/<sequence>/inpainted/00000.png
        <root>/<method>/<sequence>/original/00000.png
        <root>/<method>/<sequence>/mask/00000.png

    ``scene_id`` is the **sequence**, not the method — all three inpainting renderings
    of one DAVIS sequence share it, so the splitter cannot separate them. Letting VI
    and OP renderings of the same clip land on opposite sides of a split would be the
    near-duplicate leak of root ``CLAUDE.md`` rule 2 in its purest form: identical
    footage, identical mask, different filter.
    """

    name = "dvi"
    published_changed_pixel_ratio = None

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        methods: tuple[str, ...] = INPAINTING_METHODS,
        dir_inpainted: str = "inpainted",
        dir_original: str = "original",
        dir_mask: str = "mask",
        extensions: tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS,
        split_ratios: dict[str, float] | None = None,
        split_seed: int = 0,
        mask_threshold: int | None = None,
        frame_stride: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(root, split, **kwargs)
        self.methods = tuple(methods)
        self.dir_inpainted = dir_inpainted
        self.dir_original = dir_original
        self.dir_mask = dir_mask
        self.extensions = tuple(e.lower() for e in extensions)
        self.split_ratios = dict(split_ratios or {"train": 0.7, "val": 0.15, "test": 0.15})
        self.split_seed = int(split_seed)
        self.mask_threshold = mask_threshold
        self.frame_stride = max(1, int(frame_stride))

    # -- index ------------------------------------------------------------------------

    def _build_index(self) -> list[PairRecord]:
        sequences = self._discover_sequences()
        if not sequences:
            log.warning(
                "dvi[%s]: no <method>/<sequence>/%s directories under %s. DVI ships as "
                "separate video and mask archives per inpainting method and needs a "
                "build step first; see docs/DATA_CARD.md.",
                self.split,
                self.dir_mask,
                self.root,
            )
            return []

        assignment = self._assign_sequences(sorted(sequences))
        records: list[PairRecord] = []

        for (method, sequence), frames in sorted(sequences.items()):
            if assignment.get(sequence) != self.split:
                continue
            for i, stem in enumerate(frames):
                if i % self.frame_stride:
                    continue
                base = self.root / method / sequence
                records.append(
                    PairRecord(
                        # The pair is (original, inpainted): img1 is the reference
                        # version, img2 the edited one, matching the task statement.
                        pair_id=f"{method}/{sequence}/{stem}",
                        # Sequence only — every method renders the same footage.
                        scene_id=sequence,
                        source_video=sequence,
                        frame_idx=(i, i),
                        nuisance_label=int(NuisanceLabel.UNKNOWN),
                        payload={
                            "original": str(base / self.dir_original / f"{stem}.png"),
                            "inpainted": str(base / self.dir_inpainted / f"{stem}.png"),
                            "mask": str(base / self.dir_mask / f"{stem}.png"),
                        },
                    )
                )
        records.sort(key=lambda r: r.pair_id)
        return records

    def _discover_sequences(self) -> dict[tuple[str, str], list[str]]:
        """``(method, sequence) -> [frame stems]``, from whatever is on disk."""
        found: dict[tuple[str, str], list[str]] = {}
        for method in self.methods:
            method_dir = self.root / method
            if not method_dir.is_dir():
                continue
            for seq_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
                mask_dir = seq_dir / self.dir_mask
                if not mask_dir.is_dir():
                    continue
                stems = sorted(
                    p.stem
                    for p in mask_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in self.extensions
                )
                # Only frames with all three files present; a partial extraction
                # should shrink the index, never produce a missing-file crash at
                # __getitem__ time deep into a training run.
                stems = [
                    s
                    for s in stems
                    if (seq_dir / self.dir_original / f"{s}.png").is_file()
                    and (seq_dir / self.dir_inpainted / f"{s}.png").is_file()
                ]
                if stems:
                    found[(method, seq_dir.name)] = stems
        return found

    def _assign_sequences(self, keys: list[tuple[str, str]]) -> dict[str, str]:
        """Partition at sequence level, pooling every method's rendering together."""
        from cdlib.data.splits import DROPPED_KEY, scene_disjoint_split

        sequences = sorted({seq for _, seq in keys})
        splits = scene_disjoint_split(
            [PairRecord(pair_id=s, scene_id=s, source_video=s) for s in sequences],
            self.split_ratios,
            seed=self.split_seed,
        )
        return {
            r.scene_id: name for name, recs in splits.items() if name != DROPPED_KEY for r in recs
        }

    # -- io ---------------------------------------------------------------------------

    def _read_images(self, rec: PairRecord) -> tuple[np.ndarray, np.ndarray]:
        with Image.open(rec.payload["original"]) as im:
            img1 = np.asarray(im.convert("RGB"))
        with Image.open(rec.payload["inpainted"]) as im:
            img2 = np.asarray(im.convert("RGB"))
        return img1, img2

    def _read_mask(self, rec: PairRecord) -> np.ndarray:
        with Image.open(rec.payload["mask"]) as im:
            raw = np.asarray(im.convert("L"))
        # DAVIS masks are exact object segmentations with no ignore convention.
        return binarize_mask(raw, threshold=self.mask_threshold)


__all__ = ["DVI_SEQUENCES", "INPAINTING_METHODS", "DVIDataset"]
