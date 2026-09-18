"""VideoSham — the escape hatch if the authorised video set falls through.

`research/02` §7 rates this the **closest structural analog** to our target domain:
352 real and 352 professionally edited (After Effects) human-centric videos, 704 in
total, across six spatial and temporal attack types, with per-attack localisation
ground truth. It already frames the task the way we do — a reference version and an
edited version of the same footage — which no aerial or street dataset does.

The P1 role file's "If the video set falls through" section names it first, and the
proposal's own escape hatch is to report as paired-image change detection with no
post-production claim. This loader exists so that switching is a config change rather
than a week of work at the worst possible moment.

**Not downloaded.** Obtain from `adobe-research/VideoSham-dataset` (arXiv:2207.13064,
WACV'23 workshop). Until then this constructs empty, like every other loader.

**Layout is provisional.** The repo distributes videos plus per-attack annotation, not
a pre-extracted mask directory, so an extraction step is required first — reuse
`scripts/prepare_video_set.py`, whose output this loader reads. The directory names
below are therefore *our* convention for that extracted form, not VideoSham's own,
and every one is a constructor argument.

**Nuisance label.** ``UNKNOWN`` (-1) by default. VideoSham documents an *attack type*
per pair, which is not the same axis as our nuisance taxonomy — an attack is the
change we want to find, not the nuisance we want to ignore. If a future mapping is
agreed, pass ``attack_nuisance_map``; do not invent one silently.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cdlib.data.base import PairedChangeDataset, PairRecord, binarize_mask
from cdlib.data.contract import IGNORE_INDEX, NuisanceLabel

log = logging.getLogger(__name__)

#: `research/02` §7. 352 real + 352 edited = 704 videos, six attack types.
PUBLISHED_PAIR_COUNT: int = 352

#: Attack types named in `research/02` §7 / arXiv:2207.13064. Recorded for the data
#: card and for stratified reporting; deliberately NOT mapped onto the nuisance
#: taxonomy, because an attack is the signal, not the nuisance.
ATTACK_TYPES: tuple[str, ...] = (
    "add_object",
    "remove_object",
    "swap_object",
    "change_background",
    "audio_replace",
    "frame_reorder",
)

DEFAULT_IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg")


class VideoShamDataset(PairedChangeDataset):
    """VideoSham reference/edited frame pairs, read from an extracted manifest.

    Reads the same ``manifest.jsonl`` schema ``prepare_video_set.py`` emits, so the
    extraction pipeline we already have feeds it directly.
    """

    name = "videosham"
    #: Not published. `changed_pixel_stats()` reports "unavailable" rather than guessing.
    published_changed_pixel_ratio = None

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        manifest_name: str = "manifest.jsonl",
        splits_file: str = "splits.json",
        pairs_dir: str = "pairs",
        ref_suffix: str = "_ref.png",
        edit_suffix: str = "_edit.png",
        mask_suffix: str = "_mask.png",
        ignore_grey_level: int = 128,
        mask_threshold: int | None = None,
        attack_nuisance_map: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(root, split, **kwargs)
        self.manifest_name = manifest_name
        self.splits_file = splits_file
        self.pairs_dir = pairs_dir
        self.ref_suffix = ref_suffix
        self.edit_suffix = edit_suffix
        self.mask_suffix = mask_suffix
        self.ignore_grey_level = int(ignore_grey_level)
        self.mask_threshold = mask_threshold
        self.attack_nuisance_map = dict(attack_nuisance_map or {})

    # -- index ------------------------------------------------------------------------

    def _build_index(self) -> list[PairRecord]:
        manifest = self.root / self.manifest_name
        if not manifest.is_file():
            log.warning(
                "videosham[%s]: no %s under %s. VideoSham ships videos plus per-attack "
                "annotation, not extracted frames — run scripts/prepare_video_set.py "
                "over the downloaded videos first.",
                self.split,
                self.manifest_name,
                self.root,
            )
            return []

        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        assignment = self._split_assignment(rows)

        records: list[PairRecord] = []
        for row in rows:
            scene = str(row.get("scene_id") or row["pair_id"])
            if assignment.get(scene) != self.split:
                continue
            records.append(
                PairRecord(
                    pair_id=str(row["pair_id"]),
                    scene_id=scene,
                    source_video=str(row.get("source_video") or scene),
                    frame_idx=tuple(int(v) for v in row.get("frame_idx", (0, 0))),
                    nuisance_label=self._nuisance_of(row),
                    payload={
                        k: str(row.get(k) or f"{self.pairs_dir}/{scene}/{row['pair_id']}{sfx}")
                        for k, sfx in (
                            ("ref", self.ref_suffix),
                            ("edit", self.edit_suffix),
                            ("mask", self.mask_suffix),
                        )
                    },
                )
            )
        records.sort(key=lambda r: r.pair_id)
        return records

    def _split_assignment(self, rows: list[dict[str, Any]]) -> dict[str, str]:
        """Scene -> split. Never a pair-level partition (root ``CLAUDE.md`` rule 2)."""
        path = self.root / self.splits_file
        if path.is_file():
            return {str(k): str(v) for k, v in json.loads(path.read_text()).items()}

        scenes = sorted({str(r.get("scene_id") or r["pair_id"]) for r in rows})
        log.warning(
            "videosham[%s]: no %s — falling back to a deterministic scene-level "
            "partition over %d scene(s). Write one with "
            "cdlib.data.splits.scene_disjoint_split to pin it.",
            self.split,
            self.splits_file,
            len(scenes),
        )
        from cdlib.data.base import PairRecord as _R
        from cdlib.data.splits import DROPPED_KEY, scene_disjoint_split

        splits = scene_disjoint_split(
            [_R(pair_id=s, scene_id=s, source_video=s) for s in scenes],
            {"train": 0.7, "val": 0.15, "test": 0.15},
            seed=0,
        )
        return {
            r.scene_id: name for name, recs in splits.items() if name != DROPPED_KEY for r in recs
        }

    def _nuisance_of(self, row: dict[str, Any]) -> int:
        """UNKNOWN unless an explicit attack->nuisance mapping was supplied.

        VideoSham's attack type describes the *edit*, which is our positive class.
        Reading it as a nuisance label would put real changes into P5's nuisance
        strata and quietly corrupt the stratified evaluation.
        """
        if "nuisance_label" in row and row["nuisance_label"] is not None:
            return int(row["nuisance_label"])
        attack = str(row.get("attack_type") or "")
        if attack and attack in self.attack_nuisance_map:
            return int(self.attack_nuisance_map[attack])
        return int(NuisanceLabel.UNKNOWN)

    # -- io ---------------------------------------------------------------------------

    def _load(self, rel: str, mode: str) -> np.ndarray:
        with Image.open(self.root / rel) as im:
            return np.asarray(im.convert(mode))

    def _read_images(self, rec: PairRecord) -> tuple[np.ndarray, np.ndarray]:
        return self._load(rec.payload["ref"], "RGB"), self._load(rec.payload["edit"], "RGB")

    def _read_mask(self, rec: PairRecord) -> np.ndarray:
        raw = self._load(rec.payload["mask"], "L")
        ignore = raw == self.ignore_grey_level
        out = binarize_mask(np.where(ignore, 0, raw), threshold=self.mask_threshold)
        out[ignore] = IGNORE_INDEX
        return out


__all__ = ["ATTACK_TYPES", "PUBLISHED_PAIR_COUNT", "VideoShamDataset"]
