"""The frozen data contract — single source of truth for the data layer.

Everything in ``cdlib.data`` imports from here, and so should P2's trainer, P4's
losses and P5's metrics when they need to reason about ignore regions or nuisance
strata. Changing anything in this module is a contract change and requires a
team-wide PR review (root ``CLAUDE.md``, "Frozen interfaces").

The contract itself (root ``CLAUDE.md``)::

    {
      "img1": Tensor[C,H,W] float32 [0,1],
      "img2": Tensor[C,H,W] float32 [0,1],
      "mask": Tensor[1,H,W] float32 {0,1,-1},    # -1 = ignore
      "nuisance_label": Tensor[] int64,           # 0=clean, 1..k=nuisance type, -1=unknown
      "meta": {"source_video": str, "scene_id": str, "frame_idx": tuple[int,int],
               "pair_id": str, "dataset": str},
    }

P1 decisions recorded here, and announced in ``docs/HANDOFF.md``:

* **The nuisance taxonomy** (:class:`NuisanceLabel`). ``1..7`` are exactly the
  nuisances named in the task statement; ``8`` covers P5's optional sensor-noise
  corruption row; ``9`` covers real video pairs that carry several at once.
* **Unknown means -1, not 0.** SYSU-CD, LEVIR-CD and PCD do not document their
  nuisance content, so their loaders emit ``UNKNOWN``. Emitting ``CLEAN`` there
  would silently corrupt P5's nuisance-stratified evaluation.
* **Ignore pixels are excluded from both numerator and denominator** of every
  count (:func:`valid_pixel_mask`, :func:`changed_pixel_counts`).
* **Split names** (:data:`SPLITS`) include ``val_shift`` and ``cal``, which
  ``research/04`` requires but nobody had named.
* **No new ``meta`` key** for P5's no-change-pair denominator — the derived
  :func:`is_negative_pair` answers it without touching the frozen contract.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import torch

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

#: Value marking a pixel that must be excluded from every loss and every metric.
IGNORE_INDEX: int = -1

#: The exact keys a ``__getitem__`` must return. Order is irrelevant; membership is not.
SAMPLE_KEYS: tuple[str, ...] = ("img1", "img2", "mask", "nuisance_label", "meta")

#: The exact keys ``sample["meta"]`` must carry.
META_KEYS: tuple[str, ...] = ("source_video", "scene_id", "frame_idx", "pair_id", "dataset")

#: Split names accepted by ``build_dataset(cfg, split)``.
#:
#: ``train`` / ``val`` / ``test`` are the usual three. The other two exist because
#: ``research/04`` needs them and no document had named them:
#:
#: * ``val_shift`` — a held-out slice that is *more nuisance-shift-heavy than the
#:   training distribution*. Temperature scaling fitted on clean ``val``
#:   under-corrects or even miscorrects under shift (``research/04`` §4).
#: * ``cal`` — a third disjoint split for conformal prediction, kept out of
#:   threshold tuning (``research/04`` §7).
SPLITS: tuple[str, ...] = ("train", "val", "test", "val_shift", "cal")

#: Splits that must exist for every dataset. ``val_shift`` and ``cal`` are optional
#: and a loader may legitimately return an empty dataset for them.
REQUIRED_SPLITS: tuple[str, ...] = ("train", "val", "test")


class NuisanceLabel(IntEnum):
    """``nuisance_label`` taxonomy. P5's stratification bins must match this 1:1.

    Values ``1..8`` line up with the corruption rows in ``research/05`` §4.2, so a
    stratum in P5's retention table maps onto exactly one integer here:

    ===========================  ==========================================================
    This enum                    ``research/05`` §4.2 corruption row(s)
    ===========================  ==========================================================
    ``CAMERA_DISPLACEMENT``      viewpoint jitter
    ``LIGHTING``                 brightness/exposure, brightness (gamma), contrast
    ``COLOUR_GRADING``           colour grading / white-balance drift
    ``BLUR``                     Gaussian (defocus) blur, motion blur
    ``SHADOW``                   synthetic shadows
    ``OCCLUSION``                occlusion
    ``CODEC_ARTIFACT``           JPEG, H.264/HEVC round-trip
    ``SENSOR_NOISE``             (optional) gaussian / shot / speckle noise
    ===========================  ==========================================================

    ``MIXED`` has no counterpart in P5's suite by design: ``research/05`` says not to
    compound corruptions in the main sweep. It exists for *real* annotated video
    pairs, which routinely carry a re-grade and a camera move at once, and it keeps
    those honest rather than forcing them into a single wrong bin.
    """

    UNKNOWN = -1
    CLEAN = 0
    CAMERA_DISPLACEMENT = 1
    LIGHTING = 2
    COLOUR_GRADING = 3
    BLUR = 4
    SHADOW = 5
    OCCLUSION = 6
    CODEC_ARTIFACT = 7
    SENSOR_NOISE = 8
    MIXED = 9


#: Human-readable names, for DATA_CARD tables and P5's stratified result rows.
NUISANCE_NAMES: dict[int, str] = {int(m): m.name.lower() for m in NuisanceLabel}


def nuisance_name(value: int | torch.Tensor) -> str:
    """Map a ``nuisance_label`` value to its taxonomy name."""
    if isinstance(value, torch.Tensor):
        value = int(value.item())
    try:
        return NUISANCE_NAMES[int(value)]
    except KeyError:
        raise ValueError(
            f"{value!r} is not in the nuisance taxonomy. Valid: {sorted(NUISANCE_NAMES)}"
        ) from None


# --------------------------------------------------------------------------------------
# Ignore-region handling
# --------------------------------------------------------------------------------------


def valid_pixel_mask(mask: torch.Tensor) -> torch.Tensor:
    """Boolean mask of pixels that participate in losses and metrics.

    ``research/05`` defines TP/FP/FN with no ignore term and its snippets call
    ``gt.astype(np.uint8)``, which turns ``-1`` into ``255`` and silently corrupts
    the confusion matrix. Any consumer of ``mask`` must gate on this first — before
    the confusion matrix, before ``mask_to_boundary``, and before one-hot encoding.
    """
    return mask != IGNORE_INDEX


def changed_pixel_counts(mask: torch.Tensor) -> tuple[int, int]:
    """Return ``(changed, valid)`` pixel counts, with ignore pixels excluded from both.

    ``changed / valid`` is the per-sample contribution to the changed-pixel ratio
    ``pi``. Excluding ignore pixels from the *denominator* as well as the numerator
    is what keeps ``pi`` interpretable when a dataset has large ignore borders
    (PCD panorama seams, uncertain video annotation).
    """
    valid = valid_pixel_mask(mask)
    changed = (mask > 0) & valid
    return int(changed.sum().item()), int(valid.sum().item())


def is_negative_pair(sample: dict[str, Any]) -> bool:
    """True when the pair carries no semantic change at all.

    P5 needs ``N_neg`` — the no-change-pair denominator of the image-level
    false-alert rate (``research/05`` §3.3) — and the ``meta`` dict is frozen, so
    there is no key to add. This derives it instead: a pair is negative when no
    *valid* pixel is marked changed. A pair whose mask is entirely ignore is not
    negative; it is undefined, and returns False so it cannot silently pad
    ``N_neg``.

    A mined nuisance-only hard negative is the case ``is_negative_pair(s) and
    s["nuisance_label"] != NuisanceLabel.CLEAN``.
    """
    mask = sample["mask"]
    changed, valid = changed_pixel_counts(mask)
    return valid > 0 and changed == 0


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------


def validate_sample(sample: Any, *, name: str = "sample") -> None:
    """Assert that ``sample`` satisfies the frozen ``__getitem__`` contract.

    Raises ``AssertionError`` with a message naming the offending field. Called by
    ``tests/test_shapes.py`` for every registry key, and optionally per-item by
    :class:`cdlib.data.base.PairedChangeDataset` when ``validate=True``.
    """
    assert isinstance(sample, dict), f"{name}: expected dict, got {type(sample).__name__}"

    missing = set(SAMPLE_KEYS) - set(sample)
    extra = set(sample) - set(SAMPLE_KEYS)
    assert not missing, f"{name}: missing contract keys {sorted(missing)}"
    assert not extra, (
        f"{name}: unexpected keys {sorted(extra)}. The dataset contract is frozen — "
        "adding a key requires a team-wide PR review."
    )

    img1, img2, mask = sample["img1"], sample["img2"], sample["mask"]

    for key, img in (("img1", img1), ("img2", img2)):
        assert isinstance(img, torch.Tensor), f"{name}[{key}]: expected Tensor"
        assert img.ndim == 3, f"{name}[{key}]: expected [C,H,W], got shape {tuple(img.shape)}"
        assert img.dtype == torch.float32, f"{name}[{key}]: expected float32, got {img.dtype}"
        if img.numel():
            lo, hi = float(img.min()), float(img.max())
            assert (
                -1e-6 <= lo and hi <= 1.0 + 1e-6
            ), f"{name}[{key}]: expected range [0,1], got [{lo:.4f}, {hi:.4f}]"

    assert (
        img1.shape == img2.shape
    ), f"{name}: img1 {tuple(img1.shape)} and img2 {tuple(img2.shape)} must match"

    assert isinstance(mask, torch.Tensor), f"{name}[mask]: expected Tensor"
    assert (
        mask.ndim == 3 and mask.shape[0] == 1
    ), f"{name}[mask]: expected [1,H,W], got shape {tuple(mask.shape)}"
    assert mask.dtype == torch.float32, f"{name}[mask]: expected float32, got {mask.dtype}"
    assert mask.shape[1:] == img1.shape[1:], (
        f"{name}[mask]: spatial shape {tuple(mask.shape[1:])} "
        f"!= image spatial shape {tuple(img1.shape[1:])}"
    )
    if mask.numel():
        allowed = torch.tensor([float(IGNORE_INDEX), 0.0, 1.0], dtype=mask.dtype)
        bad = ~torch.isin(mask, allowed)
        assert not bool(bad.any()), (
            f"{name}[mask]: values must be in {{0, 1, {IGNORE_INDEX}}}, "
            f"found {sorted({float(v) for v in mask[bad].unique()[:5]})}"
        )

    nl = sample["nuisance_label"]
    assert isinstance(nl, torch.Tensor), f"{name}[nuisance_label]: expected Tensor"
    assert nl.ndim == 0, f"{name}[nuisance_label]: expected scalar Tensor[], got {tuple(nl.shape)}"
    assert nl.dtype == torch.int64, f"{name}[nuisance_label]: expected int64, got {nl.dtype}"
    assert int(nl.item()) in NUISANCE_NAMES, (
        f"{name}[nuisance_label]: {int(nl.item())} is not in the taxonomy "
        f"{sorted(NUISANCE_NAMES)}"
    )

    meta = sample["meta"]
    assert isinstance(meta, dict), f"{name}[meta]: expected dict"
    meta_missing = set(META_KEYS) - set(meta)
    meta_extra = set(meta) - set(META_KEYS)
    assert not meta_missing, f"{name}[meta]: missing keys {sorted(meta_missing)}"
    assert (
        not meta_extra
    ), f"{name}[meta]: unexpected keys {sorted(meta_extra)}. The meta dict is frozen."
    for key in ("source_video", "scene_id", "pair_id", "dataset"):
        assert isinstance(
            meta[key], str
        ), f"{name}[meta][{key}]: expected str, got {type(meta[key]).__name__}"
        assert meta[key], f"{name}[meta][{key}]: must be non-empty"
    fi = meta["frame_idx"]
    assert (
        isinstance(fi, tuple) and len(fi) == 2
    ), f"{name}[meta][frame_idx]: expected tuple[int,int], got {fi!r}"
    assert all(
        isinstance(v, int) for v in fi
    ), f"{name}[meta][frame_idx]: both entries must be int, got {fi!r}"


__all__ = [
    "IGNORE_INDEX",
    "META_KEYS",
    "NUISANCE_NAMES",
    "REQUIRED_SPLITS",
    "SAMPLE_KEYS",
    "SPLITS",
    "NuisanceLabel",
    "changed_pixel_counts",
    "is_negative_pair",
    "nuisance_name",
    "valid_pixel_mask",
    "validate_sample",
]
