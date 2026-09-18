"""``video_pairs`` — our own annotated reference/edited video pairs.

This is the only dataset in the project whose ``nuisance_label`` is *real*. SYSU-CD,
LEVIR-CD and PCD all emit ``UNKNOWN``; here the label comes from the annotation
manifest, and P5's entire nuisance-stratified evaluation and P2's hard-negative
training pool key off it. Two consequences run through this module:

* **A label outside the taxonomy is an error, not a warning.** An unrecognised
  integer silently reassigned to ``UNKNOWN`` would quietly empty a stratum in P5's
  results table, so :class:`VideoPairsDataset` raises and names the offending pair.
* **The mask encoding is ours to define and ours to get right.** ``0`` unchanged,
  ``255`` changed, ``128`` ignore -> :data:`~cdlib.data.contract.IGNORE_INDEX`. The
  public change-detection sets have no ignore convention, so nothing external will
  catch a mistake here: reading ``128`` as "changed" turns *uncertain annotation*
  into *fabricated ground truth*. The grey levels are constructor parameters and, by
  default, any fourth grey level is rejected rather than guessed at (that is what a
  mask resampled with interpolation looks like).

On-disk layout::

    <root>/manifest.jsonl
    <root>/pairs/<scene_id>/<pair_id>_{ref,edit,mask}.png

One JSON object per manifest line::

    {"pair_id", "scene_id", "source_video", "frame_idx": [i, j], "nuisance_label",
     "ref", "edit", "mask"}          # ref/edit/mask are paths relative to <root>

**Splitting is scene-disjoint, always.** Adjacent frames of one scene are
near-duplicates, so a pair-level split puts the test set in the training set and
every number we report becomes fiction (root ``CLAUDE.md`` rule 2, ``research/02``
§8, risk-register item 4). The split assignment is resolved per scene, in this
order: a ``splits.json`` at the root mapping ``scene_id -> split``; failing that a
``split`` field on the manifest rows; failing that a deterministic scene-level hash
partition implemented below.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cdlib.data.base import PairedChangeDataset, PairRecord, PairTransform
from cdlib.data.contract import IGNORE_INDEX, NUISANCE_NAMES, SPLITS, NuisanceLabel

log = logging.getLogger(__name__)

#: Grey levels in our masks. Ours to define — the public sets have no ignore level.
MASK_UNCHANGED: int = 0
MASK_CHANGED: int = 255
MASK_IGNORE: int = 128

#: Fallback split ratios, used only when neither ``splits.json`` nor a manifest
#: ``split`` field says where a scene belongs.
DEFAULT_SPLIT_RATIOS: dict[str, float] = {"train": 0.7, "val": 0.15, "test": 0.15}

#: Salt for the fallback hash partition. Changing it repartitions every scene, so it
#: is versioned rather than anonymous.
DEFAULT_SPLIT_SEED: str = "cdlib.video_pairs.v1"

_PATH_FIELDS: tuple[str, ...] = ("ref", "edit", "mask")


def scene_hash_split(
    scene_ids: list[str],
    ratios: dict[str, float],
    *,
    seed: str = DEFAULT_SPLIT_SEED,
) -> dict[str, str]:
    """Deterministically assign whole scenes to splits by hashing the scene id.

    TEMPORARY. This duplicates, in miniature, what
    ``cdlib.data.splits.scene_disjoint_split`` will do properly (with the temporal
    buffer around cut points that ``research/02`` §8 asks for). Replace the call site
    with that function once it lands; this exists so ``video_pairs`` is not blocked
    on it, and it is deliberately scene-level so the fallback cannot leak either.

    The assignment depends only on ``(seed, scene_id)`` — not on the set of scenes
    present — so adding scenes later never moves an existing scene across splits.

    Args:
        scene_ids: Scene ids to assign.
        ratios: ``{split_name: weight}``; normalised internally.
        seed: Salt, so a repartition is possible without renaming scenes.

    Returns:
        ``{scene_id: split_name}``.

    Raises:
        ValueError: If ``ratios`` is empty, names an unknown split, or has a
            negative or all-zero weight.
    """
    if not ratios:
        raise ValueError("split_ratios must name at least one split")
    unknown = sorted(set(ratios) - set(SPLITS))
    if unknown:
        raise ValueError(f"split_ratios names unknown split(s) {unknown}; valid: {list(SPLITS)}")
    if any(v < 0 for v in ratios.values()):
        raise ValueError(f"split_ratios must be non-negative, got {ratios}")
    total = float(sum(ratios.values()))
    if total <= 0:
        raise ValueError(f"split_ratios must sum to > 0, got {ratios}")

    # Iterate in SPLITS order, not dict order, so the bucket boundaries do not depend
    # on how the config happened to spell the mapping.
    ordered = [(name, ratios[name] / total) for name in SPLITS if name in ratios]
    assignment: dict[str, str] = {}
    for scene in scene_ids:
        digest = hashlib.sha256(f"{seed}|{scene}".encode()).digest()
        u = int.from_bytes(digest[:8], "big") / float(1 << 64)
        acc = 0.0
        assignment[scene] = ordered[-1][0]
        for name, weight in ordered:
            acc += weight
            if u < acc:
                assignment[scene] = name
                break
    return assignment


class VideoPairsDataset(PairedChangeDataset):
    """Reference/edited video pairs with real nuisance labels and ignore regions.

    Args:
        root: Dataset root holding ``manifest.jsonl`` and ``pairs/``.
        split: One of :data:`~cdlib.data.contract.SPLITS`. ``val_shift`` is the
            nuisance-shift-heavy split ``research/04`` §4 requires for temperature
            scaling — a temperature fitted on a clean ``val`` under-corrects under
            shift — and ``cal`` is the conformal-prediction split of §7. Both are
            just more split names in the mapping; both may legitimately be empty.
        manifest_name: Manifest filename.
        splits_file: Filename of the optional ``scene_id -> split`` mapping. Also
            accepts the inverse shape, ``split -> [scene_id, ...]``.
        split_ratios: Ratios for the fallback hash partition. Defaults to
            :data:`DEFAULT_SPLIT_RATIOS`.
        split_seed: Salt for the fallback hash partition.
        pairs_dir, ref_suffix, edit_suffix, mask_suffix: Used only to reconstruct a
            path when a manifest row omits ``ref``/``edit``/``mask``. Rows that carry
            explicit relative paths always win.
        ignore_grey_level: Grey level meaning "ignore" -> ``IGNORE_INDEX``.
        changed_grey_level: Grey level meaning "changed" -> ``1``.
        strict_mask_levels: Reject a mask carrying any other non-zero grey level.
            Keep this on: our masks are authored with exactly three levels, so a
            fourth one means the mask was resampled with interpolation, and every
            interpolated pixel would otherwise be silently promoted to "changed".
        transform: Paired augmentation pipeline.
        validate: Validate every assembled sample against the frozen contract.
        stats_cache: Cache the changed-pixel-ratio scan under ``<root>/.cdlib_stats``.

    Raises:
        ValueError: For an unknown ``split``, a malformed manifest, a
            ``nuisance_label`` outside the taxonomy, or a scene whose rows disagree
            about which split it belongs to.
    """

    name = "video_pairs"

    #: Nothing is published about our own set, and ``research/02`` §10 only says to
    #: expect a ratio *below* LEVIR-CD's 4.65%. A guess here would end up in P2's
    #: loss weights, so the base class reports "unavailable" instead.
    published_changed_pixel_ratio: float | None = None

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        manifest_name: str = "manifest.jsonl",
        splits_file: str = "splits.json",
        split_ratios: dict[str, float] | None = None,
        split_seed: str = DEFAULT_SPLIT_SEED,
        pairs_dir: str = "pairs",
        ref_suffix: str = "_ref.png",
        edit_suffix: str = "_edit.png",
        mask_suffix: str = "_mask.png",
        ignore_grey_level: int = MASK_IGNORE,
        changed_grey_level: int = MASK_CHANGED,
        strict_mask_levels: bool = True,
        transform: PairTransform | None = None,
        validate: bool = False,
        stats_cache: bool = True,
        **kwargs: Any,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {list(SPLITS)}")

        super().__init__(
            root,
            split,
            transform=transform,
            validate=validate,
            stats_cache=stats_cache,
            **kwargs,
        )
        self.manifest_name = manifest_name
        self.splits_file = splits_file
        self.split_ratios = dict(split_ratios) if split_ratios else dict(DEFAULT_SPLIT_RATIOS)
        self.split_seed = split_seed
        self.pairs_dir = pairs_dir
        self.ref_suffix = ref_suffix
        self.edit_suffix = edit_suffix
        self.mask_suffix = mask_suffix
        self.ignore_grey_level = int(ignore_grey_level)
        self.changed_grey_level = int(changed_grey_level)
        self.strict_mask_levels = bool(strict_mask_levels)
        self._hard_negatives: list[int] | None = None

    # -- manifest ---------------------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.root / self.manifest_name

    def _read_manifest(self) -> list[dict[str, Any]]:
        """Parse every manifest line. Returns ``[]`` (with a warning) if it is absent."""
        path = self.manifest_path
        if not path.is_file():
            log.warning(
                "%s[%s]: no manifest at %s — returning an empty dataset. Run "
                "scripts/prepare_video_set.py to build one.",
                self.name,
                self.split,
                path,
            )
            return []

        rows: list[dict[str, Any]] = []
        with path.open() as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno}: malformed JSON ({exc})") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{lineno}: expected a JSON object, got {type(row)}")
                rows.append(self._normalise_row(row, path, lineno))
        return rows

    def _normalise_row(self, row: dict[str, Any], path: Path, lineno: int) -> dict[str, Any]:
        """Validate one manifest row and fill in its defaults."""
        where = f"{path}:{lineno}"
        pair_id = str(row.get("pair_id") or "").strip()
        scene_id = str(row.get("scene_id") or "").strip()
        if not pair_id or not scene_id:
            raise ValueError(f"{where}: every row needs a non-empty 'pair_id' and 'scene_id'")

        out: dict[str, Any] = {
            "pair_id": pair_id,
            "scene_id": scene_id,
            # The contract needs a non-empty source_video; a set exported per scene
            # legitimately has no separate video id.
            "source_video": str(row.get("source_video") or scene_id).strip() or scene_id,
            "nuisance_label": self._parse_nuisance(row, pair_id, where),
            "frame_idx": self._parse_frame_idx(row, where),
        }

        default_paths = {
            "ref": f"{self.pairs_dir}/{scene_id}/{pair_id}{self.ref_suffix}",
            "edit": f"{self.pairs_dir}/{scene_id}/{pair_id}{self.edit_suffix}",
            "mask": f"{self.pairs_dir}/{scene_id}/{pair_id}{self.mask_suffix}",
        }
        for field in _PATH_FIELDS:
            rel = str(row.get(field) or default_paths[field])
            out[field] = str(self.root / rel)

        if "split" in row and row["split"] is not None:
            value = str(row["split"])
            if value not in SPLITS:
                raise ValueError(
                    f"{where}: pair {pair_id!r} has split {value!r}, which is not one "
                    f"of {list(SPLITS)}"
                )
            out["split"] = value
        return out

    def _parse_nuisance(self, row: dict[str, Any], pair_id: str, where: str) -> int:
        """Validate ``nuisance_label`` against the taxonomy.

        This is the one dataset where the label is real, so an out-of-taxonomy value
        is a hard error: coercing it to ``UNKNOWN`` would silently drop the pair out
        of P5's stratified tables instead of failing loudly at load time.
        """
        if "nuisance_label" not in row or row["nuisance_label"] is None:
            log.warning(
                "%s: pair %r carries no 'nuisance_label' — recording UNKNOWN (-1). "
                "This dataset is supposed to be labelled; check the annotation export.",
                where,
                pair_id,
            )
            return int(NuisanceLabel.UNKNOWN)

        raw = row["nuisance_label"]
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"{where}: pair {pair_id!r} has non-integer nuisance_label {raw!r}. "
                f"Valid values are {sorted(NUISANCE_NAMES)} "
                "(cdlib.data.contract.NuisanceLabel)."
            ) from None
        if value not in NUISANCE_NAMES:
            raise ValueError(
                f"{where}: pair {pair_id!r} has nuisance_label {value}, which is not in "
                f"the taxonomy {sorted(NUISANCE_NAMES)} "
                "(cdlib.data.contract.NuisanceLabel). P5's stratified evaluation bins "
                "map 1:1 onto that enum, so an unknown label cannot be accepted."
            )
        return value

    @staticmethod
    def _parse_frame_idx(row: dict[str, Any], where: str) -> tuple[int, int]:
        raw = row.get("frame_idx", (0, 0))
        try:
            i, j = raw  # type: ignore[misc]
            return int(i), int(j)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}: frame_idx must be [int, int], got {raw!r}") from exc

    # -- split resolution -------------------------------------------------------------

    def _load_splits_file(self) -> dict[str, str]:
        """Read ``splits.json`` if present. Accepts both mapping directions."""
        path = self.root / self.splits_file
        if not path.is_file():
            return {}
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}: unreadable split map ({exc})") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a JSON object, got {type(raw).__name__}")

        mapping: dict[str, str] = {}
        for key, value in raw.items():
            if isinstance(value, (list, tuple)):  # split -> [scene_id, ...]
                for scene in value:
                    mapping[str(scene)] = str(key)
            else:  # scene_id -> split
                mapping[str(key)] = str(value)

        bad = sorted({v for v in mapping.values() if v not in SPLITS})
        if bad:
            raise ValueError(f"{path}: unknown split name(s) {bad}; valid: {list(SPLITS)}")
        return mapping

    def _scene_splits(self, rows: list[dict[str, Any]]) -> dict[str, str]:
        """Resolve ``{scene_id: split}`` for every scene in the manifest.

        Precedence per scene: ``splits.json`` > a manifest ``split`` field > the
        deterministic hash fallback. Every path here assigns a *whole scene*.
        """
        scenes = sorted({row["scene_id"] for row in rows})
        assignment = {s: v for s, v in self._load_splits_file().items() if s in scenes}

        # A manifest that splits one scene two ways is the leakage bug this module
        # exists to prevent, so it fails loudly rather than picking a winner.
        from_rows: dict[str, str] = {}
        for row in rows:
            if "split" not in row:
                continue
            scene, value = row["scene_id"], row["split"]
            previous = from_rows.setdefault(scene, value)
            if previous != value:
                raise ValueError(
                    f"{self.manifest_path}: scene {scene!r} is assigned to both "
                    f"{previous!r} and {value!r}. Splits are per scene, never per pair "
                    "— adjacent frames of one scene are near-duplicates."
                )
        for scene, value in from_rows.items():
            assignment.setdefault(scene, value)

        missing = [s for s in scenes if s not in assignment]
        if missing:
            log.warning(
                "%s[%s]: %d of %d scene(s) have no split assignment in %s or the "
                "manifest (e.g. %s) — falling back to a deterministic scene-level "
                "hash partition with ratios %s. Write a %s to pin the split.",
                self.name,
                self.split,
                len(missing),
                len(scenes),
                self.splits_file,
                missing[:3],
                self.split_ratios,
                self.splits_file,
            )
            # TODO(P1): replace with cdlib.data.splits.scene_disjoint_split once A1's
            # module lands — it adds the temporal buffer around cut points that a pure
            # hash partition cannot express (research/02 §8).
            assignment.update(scene_hash_split(missing, self.split_ratios, seed=self.split_seed))
        return assignment

    # -- index ------------------------------------------------------------------------

    def _build_index(self) -> list[PairRecord]:
        rows = self._read_manifest()
        if not rows:
            return []
        assignment = self._scene_splits(rows)

        records = [
            PairRecord(
                pair_id=row["pair_id"],
                scene_id=row["scene_id"],
                source_video=row["source_video"],
                frame_idx=row["frame_idx"],
                nuisance_label=row["nuisance_label"],
                payload={field: row[field] for field in _PATH_FIELDS},
            )
            for row in rows
            if assignment.get(row["scene_id"]) == self.split
        ]
        # Manifest order is already deterministic, but sorting makes the index
        # independent of how the manifest was appended to.
        records.sort(key=lambda rec: (rec.scene_id, rec.pair_id))
        return records

    @property
    def scenes(self) -> list[str]:
        """Sorted scene ids in this split. The unit of the split, and of leakage."""
        return sorted({rec.scene_id for rec in self.records})

    # -- pixels -----------------------------------------------------------------------

    @staticmethod
    def _load(path: str, mode: str) -> np.ndarray:
        # PIL, not cv2.imread: cv2 returns BGR and a channel-swapped pair passes
        # every shape test in the suite.
        with Image.open(path) as im:
            return np.array(im.convert(mode))

    def _read_images(self, rec: PairRecord) -> tuple[np.ndarray, np.ndarray]:
        return self._load(rec.payload["ref"], "RGB"), self._load(rec.payload["edit"], "RGB")

    def _read_mask(self, rec: PairRecord) -> np.ndarray:
        """Decode our three-level mask: ``0`` -> 0, ``255`` -> 1, ``128`` -> ``-1``."""
        raw = self._load(rec.payload["mask"], "L")
        if self.strict_mask_levels:
            allowed = {MASK_UNCHANGED, self.ignore_grey_level, self.changed_grey_level}
            extra = sorted(int(v) for v in np.unique(raw) if int(v) not in allowed)
            if extra:
                raise ValueError(
                    f"{self.name}: mask for pair {rec.pair_id!r} carries grey level(s) "
                    f"{extra[:8]}, outside the documented encoding "
                    f"{{{MASK_UNCHANGED}: unchanged, {self.changed_grey_level}: changed, "
                    f"{self.ignore_grey_level}: ignore}}. That normally means the mask "
                    "was resized with interpolation, which would silently promote "
                    "interpolated pixels to 'changed'. Re-export with nearest-neighbour "
                    "resampling, or pass strict_mask_levels=False if you accept the "
                    "thresholding."
                )
        out = np.zeros(raw.shape, dtype=np.int16)
        # Ignore first: it is a positive grey level too, so order matters.
        ignore = raw == self.ignore_grey_level
        out[(raw > 0) & ~ignore] = 1
        out[ignore] = IGNORE_INDEX
        return out

    # -- nuisance-only hard negatives -------------------------------------------------

    def _is_negative_record(self, rec: PairRecord) -> bool:
        """Record-level :func:`~cdlib.data.contract.is_negative_pair`.

        Same definition — no *valid* pixel is marked changed, and an all-ignore mask
        is undefined rather than negative — but it reads only the mask, skipping the
        RGB decode and tensor assembly that ``__getitem__`` would do.
        """
        mask = self._read_mask(rec)
        valid = mask != IGNORE_INDEX
        return bool(valid.any()) and not bool((mask[valid] > 0).any())

    def hard_negative_indices(self) -> list[int]:
        """Indices of nuisance-only hard negatives, cached after the first scan.

        A hard negative is a pair with a real nuisance (``nuisance_label != CLEAN``)
        and no semantic change at all. These are P5's ``N_neg`` — the denominator of
        the image-level false-alert rate (``research/05`` §3.3) — and P2's
        hard-negative training pool, so ``Subset(ds, ds.hard_negative_indices())``
        is the whole handoff.
        """
        if self._hard_negatives is None:
            self._hard_negatives = [
                i
                for i, rec in enumerate(self.records)
                if rec.nuisance_label != int(NuisanceLabel.CLEAN) and self._is_negative_record(rec)
            ]
        return list(self._hard_negatives)

    def hard_negative_records(self) -> list[PairRecord]:
        """The :class:`~cdlib.data.base.PairRecord`s behind :meth:`hard_negative_indices`."""
        return [self.records[i] for i in self.hard_negative_indices()]


__all__ = [
    "DEFAULT_SPLIT_RATIOS",
    "DEFAULT_SPLIT_SEED",
    "MASK_CHANGED",
    "MASK_IGNORE",
    "MASK_UNCHANGED",
    "VideoPairsDataset",
    "scene_hash_split",
]
