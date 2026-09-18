"""Scene/source-disjoint splitting — the one thing that keeps the benchmark honest.

Root ``CLAUDE.md`` rule 2 and risk-register item 4: **never split at frame level**.
Adjacent video frames are near-duplicates, so a random frame-level partition puts
paraphrases of the test set into the training set and every number the project
reports becomes fiction. ``research/02`` §8 records the same finding from the
remote-sensing literature (spatial autocorrelation across randomly split patches)
and prescribes the fix used here: partition by source video / scene, with a
temporal buffer around cut points.

Three things follow from that, and all three are load-bearing:

* **Groups, not records, are the unit of allocation.** Ratios are therefore over
  groups. Because groups differ in size, the realised *record* proportions will not
  match the requested ratios exactly. That is correct and expected; the alternative
  — topping a split up to its record quota — reintroduces the leak.
* **Records near a cut point are dropped, never reassigned** (see
  :func:`scene_disjoint_split`, ``temporal_buffer``). Reassigning them would move a
  near-duplicate across the partition, which is the leak wearing a hat.
* **Dropped records stay visible.** They are returned under :data:`DROPPED_KEY`, the
  key is always present, and the splitter refuses to return unless every input
  record is accounted for exactly once.

Typical use::

    splits = scene_disjoint_split(records, {"train": 0.7, "val": 0.1, "test": 0.2},
                                  group_key="scene_id", temporal_buffer=8, seed=0)
    assert_no_group_leakage(splits, "scene_id")
    print(split_summary(splits))
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import fields

from cdlib.data.base import PairRecord
from cdlib.data.contract import SPLITS

log = logging.getLogger(__name__)

#: Key under which :func:`scene_disjoint_split` returns the records the temporal
#: buffer removed. Deliberately *not* a member of :data:`cdlib.data.contract.SPLITS`,
#: so it can never collide with a requested split name, and always present in the
#: result even when empty — a drop that is not in the returned dict is a drop nobody
#: notices.
DROPPED_KEY: str = "dropped"

#: The group keys that actually defeat near-duplicate leakage. Anything else is
#: allowed but warned about.
GROUP_KEYS: tuple[str, ...] = ("scene_id", "source_video")

#: Grouping on either of these *is* the frame-level split rule 2 forbids, so they are
#: refused outright rather than warned about.
_FRAME_LEVEL_KEYS: frozenset[str] = frozenset({"pair_id", "frame_idx"})

_RATIO_SUM_TOL: float = 1e-6

StratifyBy = str | Callable[[PairRecord], Hashable]


# --------------------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------------------


def _validate_group_key(group_key: str) -> None:
    """Reject group keys that would silently produce a leaking split."""
    if group_key in _FRAME_LEVEL_KEYS:
        raise ValueError(
            f"group_key={group_key!r} would partition at frame level, which root "
            "CLAUDE.md rule 2 forbids: adjacent frames are near-duplicates and a "
            f"frame-level split invalidates the benchmark. Use one of {list(GROUP_KEYS)}."
        )
    valid = {f.name for f in fields(PairRecord)}
    if group_key not in valid:
        raise ValueError(
            f"group_key={group_key!r} is not a PairRecord field. Valid fields: "
            f"{sorted(valid)}; the sensible choices are {list(GROUP_KEYS)}."
        )
    if group_key not in GROUP_KEYS:
        log.warning(
            "group_key=%r is not one of %s — check it really is coarse enough to "
            "separate near-duplicate frames (root CLAUDE.md rule 2).",
            group_key,
            list(GROUP_KEYS),
        )


def _normalised_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    """Validate split names against the contract and normalise weights to sum to 1.

    Ratios are *normalised*, not required to sum to 1, so ``{"train": 8, "val": 1,
    "test": 1}`` is a legal way to say 80/10/10. A sum that is neither 1 nor an
    obvious weighting is logged, because it is more often a typo than an intention.
    """
    if not ratios:
        raise ValueError("ratios must name at least one split")

    unknown = sorted(name for name in ratios if name not in SPLITS)
    if unknown:
        raise ValueError(
            f"unknown split name(s) {unknown}. The contract fixes the vocabulary: "
            f"{list(SPLITS)} (cdlib.data.contract.SPLITS)."
        )

    weights: dict[str, float] = {}
    for name, value in ratios.items():
        weight = float(value)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"ratios[{name!r}]={value!r} must be a finite, non-negative number")
        weights[name] = weight

    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError(f"ratios must contain at least one positive weight, got {dict(ratios)}")
    if abs(total - 1.0) > _RATIO_SUM_TOL:
        log.warning("ratios sum to %g, not 1 — normalising to %s", total, sorted(weights))
    return {name: weight / total for name, weight in weights.items()}


# --------------------------------------------------------------------------------------
# Grouping and stratification
# --------------------------------------------------------------------------------------


def _group_records(
    records: Sequence[PairRecord], group_key: str
) -> tuple[list[Hashable], dict[Hashable, list[int]]]:
    """Map each group id to the indices of its records, in first-appearance order."""
    order: list[Hashable] = []
    members: dict[Hashable, list[int]] = {}
    for idx, rec in enumerate(records):
        gid = getattr(rec, group_key)
        if gid not in members:
            members[gid] = []
            order.append(gid)
        members[gid].append(idx)
    return order, members


def _stratum_of(
    records: Sequence[PairRecord], indices: Sequence[int], stratify_by: StratifyBy
) -> Hashable:
    """One stratum label per *group*, by majority vote over its records.

    A group whose records disagree is a labelling bug upstream, not something to
    silently average away, so it is logged; the majority keeps the split usable.
    """
    if callable(stratify_by):
        values = [stratify_by(records[i]) for i in indices]
    else:
        values = [getattr(records[i], stratify_by) for i in indices]

    counts: dict[Hashable, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    if len(counts) > 1:
        log.warning(
            "group %r carries several stratify_by values %s — using the majority",
            getattr(records[indices[0]], "scene_id", "?"),
            sorted(counts, key=str),
        )
    return max(sorted(counts, key=str), key=lambda value: counts[value])


# --------------------------------------------------------------------------------------
# Allocation
# --------------------------------------------------------------------------------------


def _allocate_groups(
    group_order: Sequence[Hashable],
    strata: Mapping[Hashable, Hashable],
    ratios: Mapping[str, float],
    rng: random.Random,
) -> dict[Hashable, str]:
    """Assign every group to exactly one split, honouring ratios as closely as possible.

    Greedy largest-deficiency allocation: each group goes to whichever split is
    furthest below its running target. Deficiencies carry *across* strata, which is
    what stops a rare stratum of two or three groups from rounding entirely into the
    largest split — the failure mode plain per-stratum rounding has.
    """
    names = list(ratios)
    rank = {name: i for i, name in enumerate(names)}

    buckets: dict[Hashable, list[Hashable]] = {}
    for gid in group_order:
        buckets.setdefault(strata[gid], []).append(gid)

    target = dict.fromkeys(names, 0.0)
    assigned: dict[str, list[Hashable]] = {name: [] for name in names}

    for key in sorted(buckets, key=str):
        members = list(buckets[key])
        rng.shuffle(members)
        for gid in members:
            for name in names:
                target[name] += ratios[name]
            # Ties break towards the larger ratio, then declared order, so the result
            # depends only on (records, ratios, seed).
            pick = max(
                names,
                key=lambda name: (target[name] - len(assigned[name]), ratios[name], -rank[name]),
            )
            assigned[pick].append(gid)

    _fill_empty_splits(assigned, ratios, len(group_order))

    return {gid: name for name, gids in assigned.items() for gid in gids}


def _fill_empty_splits(
    assigned: dict[str, list[Hashable]], ratios: Mapping[str, float], n_groups: int
) -> None:
    """Give every requested split at least one group where the corpus allows it.

    With few groups, proportional rounding can leave ``test`` (or ``val``) empty,
    which does not fail loudly — it just produces a benchmark with nothing in it.
    One group is moved from the most over-allocated split that can spare one; when
    there are simply fewer groups than splits, the remainder stay empty and are
    named in a warning.
    """
    wanted = [name for name, ratio in ratios.items() if ratio > 0.0]
    rank = {name: i for i, name in enumerate(ratios)}

    for name in sorted((n for n in wanted if not assigned[n]), key=lambda n: (-ratios[n], rank[n])):
        donors = [d for d in wanted if len(assigned[d]) >= 2]
        if not donors:
            break
        donor = max(donors, key=lambda d: (len(assigned[d]) - ratios[d] * n_groups, -rank[d]))
        assigned[name].append(assigned[donor].pop())

    empty = [name for name in wanted if not assigned[name]]
    if empty:
        log.warning(
            "%d group(s) cannot fill %d split(s): %s left empty. Model selection and "
            "evaluation on an empty split are meaningless — get more scenes.",
            n_groups,
            len(wanted),
            empty,
        )


# --------------------------------------------------------------------------------------
# Temporal buffer
# --------------------------------------------------------------------------------------


def _buffer_violations(
    records: Sequence[PairRecord], assignment: Sequence[str], buffer: int
) -> set[int]:
    """Indices of records that come within ``buffer`` frames of a *different* split.

    Both sides of every offending pair are dropped: at a cut point the frames either
    side are the near-duplicates of each other, so keeping one of them keeps the
    leak. Frame indices are only comparable inside one ``source_video``, so the scan
    is per video.
    """
    by_video: dict[str, list[int]] = {}
    for idx, rec in enumerate(records):
        by_video.setdefault(rec.source_video, []).append(idx)

    dropped: set[int] = set()
    degenerate: list[str] = []
    for video, indices in by_video.items():
        if len(indices) < 2:
            continue

        # Every public loader emits a constant `frame_idx` — aerial tiles and
        # panoramas have no frame ordering to report. With no distinct indices, every
        # record sits zero frames from every other, so a naive scan would call every
        # cross-split pair a violation and drop the entire dataset: silently, and
        # totally. The group-level partition already guarantees rule 2; the buffer
        # only *adds* protection against within-video adjacency, and it cannot add
        # what the frame indices do not contain. So skip, and say so.
        if len({records[i].frame_idx for i in indices}) < 2:
            degenerate.append(video)
            continue

        ordered = sorted(indices, key=lambda i: (min(records[i].frame_idx), i))
        for position, i in enumerate(ordered):
            hi = max(records[i].frame_idx)
            for j in ordered[position + 1 :]:
                # ``ordered`` is sorted by start frame, so once the gap exceeds the
                # buffer for one j it exceeds it for every later j.
                if min(records[j].frame_idx) - hi > buffer:
                    break
                if assignment[i] != assignment[j]:
                    dropped.add(i)
                    dropped.add(j)

    if degenerate:
        log.warning(
            "temporal_buffer=%d was requested but %d source video(s) carry no distinct "
            "frame_idx values (e.g. %s), so no temporal adjacency can be computed for "
            "them and the buffer was not applied there. The scene/source-level split "
            "still holds. If these records really are temporally ordered, populate "
            "PairRecord.frame_idx; if they are not, pass temporal_buffer=0 to say so "
            "explicitly.",
            buffer,
            len(degenerate),
            sorted(degenerate)[:3],
        )
    return dropped


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def scene_disjoint_split(
    records: Iterable[PairRecord],
    ratios: Mapping[str, float],
    *,
    group_key: str = "scene_id",
    temporal_buffer: int = 0,
    seed: int = 0,
    stratify_by: StratifyBy | None = None,
) -> dict[str, list[PairRecord]]:
    """Partition ``records`` into splits that share no scene or source video.

    Args:
        records: The pair index to partition. Records are passed through, not copied,
            and their order within a split follows their order here.
        ratios: Split name -> weight, e.g. ``{"train": 0.7, "val": 0.1, "test": 0.2}``.
            Names are validated against :data:`cdlib.data.contract.SPLITS`. Weights are
            **normalised**, so integer weights such as ``{"train": 8, "val": 1,
            "test": 1}`` work; a sum far from 1 is logged in case it is a typo.
            Ratios apply to *groups*, so realised record proportions are approximate.
        group_key: ``PairRecord`` attribute to partition on — ``"scene_id"`` (default)
            or ``"source_video"``. ``"pair_id"`` and ``"frame_idx"`` are refused: they
            are the frame-level split root ``CLAUDE.md`` rule 2 forbids.
        temporal_buffer: Number of frames. Within one ``source_video``, records whose
            ``frame_idx`` comes within this many frames of a record in a *different*
            split are dropped — both sides of the pair, since each is the other's
            near-duplicate. ``0`` (default) disables the buffer entirely. It only
            bites when two splits are temporally adjacent inside one video, i.e. when
            ``group_key="scene_id"`` and a video contains several scenes.
        seed: Seed for the group shuffle. Same records, ratios and seed give byte-identical
            output; a private :class:`random.Random` is used, never global RNG state.
        stratify_by: Optional ``PairRecord`` attribute name or callable
            ``(PairRecord) -> Hashable``. Groups are bucketed by that key and each
            bucket is distributed across the splits in the given ratios, so a rare
            nuisance type does not end up concentrated in one split. A group whose
            records disagree takes the majority value.

    Returns:
        ``{split_name: [PairRecord, ...]}`` for every name in ``ratios``, **plus**
        :data:`DROPPED_KEY` holding the records the temporal buffer removed. The
        dropped key is always present, even when empty, and every input record
        appears in exactly one list — checked before returning, so records cannot go
        missing unnoticed. Callers that iterate the result must skip
        :data:`DROPPED_KEY`; it is not a split.

    Raises:
        ValueError: on an unknown split name, a non-positive or negative ratio set, a
            negative buffer, or a group key that would split at frame level.
        AssertionError: if the produced partition leaks a group (self-check).
    """
    records = list(records)
    _validate_group_key(group_key)
    weights = _normalised_ratios(ratios)
    if temporal_buffer < 0:
        raise ValueError(f"temporal_buffer must be >= 0 frames, got {temporal_buffer}")

    out: dict[str, list[PairRecord]] = {name: [] for name in weights}
    out[DROPPED_KEY] = []
    if not records:
        return out

    group_order, members = _group_records(records, group_key)
    if stratify_by is None:
        strata: dict[Hashable, Hashable] = dict.fromkeys(group_order, None)
    else:
        strata = {gid: _stratum_of(records, members[gid], stratify_by) for gid in group_order}

    group_split = _allocate_groups(group_order, strata, weights, random.Random(seed))

    assignment: list[str] = [""] * len(records)
    for gid, indices in members.items():
        for idx in indices:
            assignment[idx] = group_split[gid]

    dropped = (
        _buffer_violations(records, assignment, temporal_buffer) if temporal_buffer > 0 else set()
    )

    for idx, rec in enumerate(records):
        out[DROPPED_KEY if idx in dropped else assignment[idx]].append(rec)

    # Self-check, not decoration: this module exists to make these two properties
    # true, and a silent regression here is a silently invalid benchmark.
    assert_no_group_leakage(out, group_key)
    placed = sum(len(v) for v in out.values())
    if placed != len(records):
        raise RuntimeError(
            f"split lost records: {len(records)} in, {placed} out. This is a bug in "
            "cdlib.data.splits, not in your data."
        )
    if dropped:
        log.info(
            "temporal_buffer=%d dropped %d/%d records straddling a split boundary",
            temporal_buffer,
            len(dropped),
            len(records),
        )
    return out


def assert_no_group_leakage(
    splits: Mapping[str, Sequence[PairRecord]], group_key: str = "scene_id"
) -> None:
    """Assert that no group appears in two splits — root ``CLAUDE.md`` rule 2.

    Args:
        splits: A mapping as returned by :func:`scene_disjoint_split`. The
            :data:`DROPPED_KEY` bucket is skipped: dropped records share a group with
            the split that group was assigned to by construction, which is not a leak.
        group_key: ``PairRecord`` attribute that must not cross a split.

    Raises:
        AssertionError: naming every offending group and the two splits it crossed.
    """
    owner: dict[Hashable, str] = {}
    offences: dict[tuple[Hashable, str, str], None] = {}
    for name, recs in splits.items():
        if name == DROPPED_KEY:
            continue
        for rec in recs:
            gid = getattr(rec, group_key)
            held_by = owner.setdefault(gid, name)
            if held_by != name:
                offences[(gid, held_by, name)] = None

    assert not offences, (
        f"{group_key} leakage across splits (root CLAUDE.md rule 2 — adjacent frames "
        "are near-duplicates, so a group in two splits invalidates every number): "
        + "; ".join(f"{gid!r} in both {a!r} and {b!r}" for gid, a, b in sorted(offences, key=str))
    )


def split_summary(
    splits: Mapping[str, Sequence[PairRecord]], *, group_key: str = "scene_id"
) -> dict[str, dict[str, float]]:
    """Per-split record, group, scene and source-video counts, for DATA_CARD tables.

    ``n_groups`` counts ``group_key``; ``n_scenes`` and ``n_source_videos`` are always
    reported because the two together are what shows a reader that the split really is
    source-disjoint. The :data:`DROPPED_KEY` bucket is included — a data card that
    hides the discarded records is not a data card.

    Args:
        splits: A mapping as returned by :func:`scene_disjoint_split`.
        group_key: Attribute counted as ``n_groups``.

    Returns:
        ``{split_name: {"n_records", "n_groups", "n_scenes", "n_source_videos",
        "frac_records"}}``. ``frac_records`` is over all records in ``splits``,
        dropped ones included, and is ``0.0`` when there are none.
    """
    total = sum(len(recs) for recs in splits.values())
    summary: dict[str, dict[str, float]] = {}
    for name, recs in splits.items():
        summary[name] = {
            "n_records": len(recs),
            "n_groups": len({getattr(rec, group_key) for rec in recs}),
            "n_scenes": len({rec.scene_id for rec in recs}),
            "n_source_videos": len({rec.source_video for rec in recs}),
            "frac_records": (len(recs) / total) if total else 0.0,
        }
    return summary


__all__ = [
    "DROPPED_KEY",
    "GROUP_KEYS",
    "assert_no_group_leakage",
    "scene_disjoint_split",
    "split_summary",
]
