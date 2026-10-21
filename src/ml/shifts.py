"""Parameters whose value runs in groups over time, and which change on the same days.

A raw material or consumable lot, its certificate values, a vector lot and its
titre: over the batches in date order, each keeps one value for as long as a lot
lasts, then changes to another. A model can use such a parameter as a stand-in for
time. This module finds them per manufacturing site (time, and the lots, run per
plant rather than per data source), over the batches the missingness report counts
in its statistics, and groups the ones that change on the same days.
``ml task new --exclude-shifting`` leaves every one of them out of a task.

Dates count by the day: a day's batches have no order among themselves, and on the
day a lot runs out some of them can have the old lot and some the new one.

1. Every parameter (any type) with at least :data:`MIN_BATCHES` values and two
   distinct ones is tested, unless the screening register already excludes it at
   the site. Batches without a date are left out.
2. Groups. Over the days the parameter has a value (blanks skipped), a group is one
   value kept from one day to the next; values are compared exactly. One group
   carries on from a day to the next, and only when both days hold its value; any
   other value on a day starts a group. So the day a lot changes ends the old lot's
   group and starts the new one's, while two values used side by side start a group
   every day.
3. A parameter *runs in groups* when
   - its groups last :data:`MIN_DAYS` of its days or more on average (its days with
     a value / its groups), longer than a day: a value set per day, such as a date,
     does not; and
   - it has fewer groups than chance: in at most :data:`P_MAX` of
     :data:`PERMUTATIONS` shuffles of its values over its batches do as few groups
     come out (permutation p). A rare value scattered at random does not.
4. Its *shift days* are the days a group starts, but its first day. Two parameters
   are compared over the days both have a value, and need :data:`MIN_BATCHES`
   batches there: each shift day counts on the first of those days on or after it,
   and *shared shifts* is the share of their shift days they have in common
   (Jaccard: in common / of either).
5. Groups of parameters by complete linkage on 1 - shared shifts: every pair in a
   group shares at least :data:`THRESHOLD` of its shift days. Each group suggests a
   representative as in :mod:`ml.correlation`, for the screening register's
   *shifts with group*.

This is description only. Nothing here changes the batch table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .correlation import CorrelationGroup, grouped
from .logger import log

MODULE = "shifts"

#: Values a parameter needs, and batches in common a pair needs, to be compared.
MIN_BATCHES = 20
#: The days with a value a parameter's groups must last on average: more than one.
MIN_DAYS = 2.0
#: Shuffles of a parameter's values in the permutation test, and the p it must reach.
PERMUTATIONS = 200
P_MAX = 0.01
#: The shared shifts every pair in a group of parameters reaches.
THRESHOLD = 0.8


@dataclass
class ShiftReport:
    #: The parameters that run in groups, and how many were tested.
    analysed: list[str] = field(default_factory=list)
    tested: int = 0
    #: The data sources (register sites) the batches come from.
    sources: list[str] = field(default_factory=list)
    groups: list[CorrelationGroup] = field(default_factory=list)
    #: Per parameter, its groups in time order: [first day, last day, value, batches].
    segments: dict[str, list[list]] = field(default_factory=dict)
    cohort: str = ""
    left_out: list[str] = field(default_factory=list)


def count_groups(day: np.ndarray, code: np.ndarray) -> np.ndarray:
    """Groups along the last axis: distinct (day, value) pairs, less the days that carry one on.

    ``day`` numbers the parameter's days with a value 0, 1, … per batch; ``code`` is
    each batch's value as an integer, one row per arrangement of the values.
    """
    width = int(day.max()) + 2  # a value's next day is one up, another value further
    key = np.sort(code * width + day, axis=-1)
    step = np.diff(key, axis=-1)
    carried = np.sort(np.where(step == 1, key[..., 1:] % width, -1), axis=-1)
    days = ((carried >= 0) & (np.diff(carried, axis=-1, prepend=-1) != 0)).sum(axis=-1)
    return 1 + (step > 0).sum(axis=-1) - days


def segments(day: np.ndarray, code: np.ndarray) -> list[list[int]]:
    """The groups :func:`count_groups` counts, in order: [value, first day, last day, batches]."""
    pairs, counts = np.unique(np.c_[day, code], axis=0, return_counts=True)
    cut = np.flatnonzero(np.diff(pairs[:, 0])) + 1
    groups: list[list[int]] = []
    before: dict[int, int] = {}  # the day before's values, and their group
    carry = None  # the value whose group carried on into the day before
    for here, batches in zip(np.split(pairs, cut), np.split(counts, cut)):
        d = int(here[0, 0])
        shared = [int(c) for c in here[:, 1] if c in before]
        carry = carry if carry in shared else (shared[0] if shared else None)
        today = {}
        for c, k in zip(here[:, 1].tolist(), batches.tolist()):
            if c == carry:
                today[c] = before[c]
                group = groups[today[c]]
                group[2], group[3] = d, group[3] + k
            else:
                today[c] = len(groups)
                groups.append([c, d, d, k])
        before = today
    return groups


def shared_shifts(days_a, shifts_a, days_b, shifts_b) -> float:
    """The share of their shift days two parameters have in common, over their common days."""
    common = np.intersect1d(days_a, days_b)

    def moved(shifts):
        inside = shifts[(shifts > common[0]) & (shifts <= common[-1])]
        return set(common[np.searchsorted(common, inside)].tolist())

    a, b = moved(shifts_a), moved(shifts_b)
    return len(a & b) / len(a | b) if a | b else 0.0


def _text(value) -> str:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M").removesuffix(" 00:00")
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def for_build(
    frame: pd.DataFrame,
    id_column: str,
    order_column: str,
    analysed: pd.Series | None = None,
    screened_out: dict[str, str] | None = None,
    label: str = "",
) -> tuple[list[dict], ShiftReport]:
    """The parameters that run in groups, one row each, grouped by their shift days.

    ``analysed`` as for :func:`ml.correlation.for_build`; the parameters in
    ``screened_out`` are left out. ``label`` (the manufacturing site) ends each group's
    name, so names stay unique when one register site holds several plants.
    """
    if analysed is not None:
        frame = frame[analysed.reindex(frame.index).fillna(False).astype(bool)]
    dates = (
        pd.to_datetime(frame[order_column], errors="coerce").dt.normalize()
        if order_column in frame.columns
        else pd.Series(pd.NaT, index=frame.index)
    )
    frame, dates = frame[dates.notna()], dates[dates.notna()]
    report = ShiftReport(left_out=sorted(screened_out or {}))

    rng = np.random.default_rng(0)
    rows_of, days_of, shifts_of, present = {}, {}, {}, {}
    for column in frame.columns:
        if column in (id_column, order_column) or column in report.left_out:
            continue
        values = frame[column]
        has = values.notna().to_numpy()
        if has.sum() < MIN_BATCHES or values.nunique() < 2:
            continue
        report.tested += 1
        code, uniques = pd.factorize(values[has])
        day, days = pd.factorize(dates[has], sort=True)
        count = int(count_groups(day, code))
        if len(days) / count < MIN_DAYS:
            continue
        chance = count_groups(day, rng.permuted(np.tile(code, (PERMUTATIONS, 1)), axis=1))
        p = (1 + int((chance <= count).sum())) / (PERMUTATIONS + 1)
        if p > P_MAX:
            continue
        found = segments(day, code)
        days_of[column] = days.to_numpy()
        starts = [days_of[column][f] for _, f, _, _ in found if f > 0]
        shifts_of[column] = np.unique(np.array(starts, dtype=days_of[column].dtype))
        present[column] = has
        report.segments[str(column)] = [
            [f"{days[first]:%Y-%m-%d}", f"{days[last]:%Y-%m-%d}", _text(uniques[c]), batches]
            for c, first, last, batches in found
        ]
        rows_of[column] = {
            "Parameter": str(column),
            "Batches with a value": int(has.sum()),
            "Distinct values": len(uniques),
            "Groups": count,
            "Days per group": round(len(days) / count, 1),
            "Permutation p": round(p, 4),
        }
    report.analysed = list(rows_of)
    log.step(
        MODULE,
        f"{len(report.analysed):,} of {report.tested:,} parameters run in groups over "
        f"{len(frame):,} batches; comparing their shift days",
        f"{len(report.left_out):,} left out as screened out",
    )

    k = len(report.analysed)
    measured = np.array([present[c] for c in report.analysed], dtype=float).reshape(k, len(frame))
    n = measured @ measured.T
    similarity = np.eye(k)
    for i, a in enumerate(report.analysed):
        for j in range(i + 1, k):
            b = report.analysed[j]
            if n[i, j] >= MIN_BATCHES:
                similarity[i, j] = similarity[j, i] = shared_shifts(
                    days_of[a], shifts_of[a], days_of[b], shifts_of[b]
                )
    report.groups = grouped(
        similarity, n, np.full((k, k), np.nan), report.analysed, THRESHOLD, "Shifting", "complete"
    )
    for group in report.groups:
        group.name += f" @ {label}" if label else ""
    group_of = {m: g.name for g in report.groups for m in g.members}

    rows = []
    for i, column in enumerate(report.analysed):
        others = similarity[i].copy()
        others[i] = -1
        best = int(np.argmax(others)) if k > 1 else i
        found = best != i and others[best] > 0
        rows.append(
            {
                "Parameter": str(column),
                "Shift group": group_of.get(column, ""),
                **{key: v for key, v in rows_of[column].items() if key != "Parameter"},
                "Closest parameter": report.analysed[best] if found else "",
                "Shared shifts with it": round(float(others[best]), 2) if found else None,
            }
        )
    log.success(
        MODULE,
        f"{len(report.groups)} groups of parameters that change on the same days "
        f"(shared shifts ≥ {THRESHOLD})",
    )
    return rows, report
