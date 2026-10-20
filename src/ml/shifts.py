"""Parameters whose values fall into level groups over time, and which share those groups.

A raw material or consumable lot, its certificate values, a vector lot and its
titre: over the batches in date order, each holds (very nearly) one value for as
long as a lot lasts, then jumps to another. Its batches fall into *level groups*:
stretches of consecutive batches with very nearly the same value. Parameters
whose level groups cover the same batches carry the same information, which lot
a batch was made with. This module finds them, per site, over the same batches
as the missingness report:

1. Every parameter (any type) with at least :data:`MIN_BATCHES` values and two
   distinct ones is tested, unless the screening register excludes it at the
   site for another reason than :data:`REASON`.
2. Level groups. With the batches in date order and blanks skipped, a new group
   starts where the value moves by more than the tolerance from the batch before.
   For a number or a date, the tolerance is :data:`TOLERANCE` times the median
   change from one batch to the next: within a level group most changes are
   noise, so their median is the noise (0 when the values repeat exactly). For a
   category, any change starts a new group.
3. A parameter *moves in level groups* when all of these hold:
   - its groups average at least :data:`MIN_RUN` batches;
   - the values within a group are very close to the same: the median jump from
     one group's level (mean) to the next is at least :data:`SEPARATION` times the
     spread within the groups (pooled SD). A category's groups are exact.
     A drift is cut into groups no further apart than the drift within them;
   - the values do not drift within a group: the pooled within-group correlation
     of value and batch position is below :data:`MAX_DRIFT` in size (a lot's
     values scatter around its level; a date advances with every batch);
   - it has fewer groups than chance: in no more than :data:`P_MAX` of
     :data:`PERMUTATIONS` shuffles of its values over the batches do as few
     groups come out (permutation p).
   A trend fails the first or the second: with noise it breaks into short groups,
   and either way its groups are no further apart than the drift within them.
4. Two such parameters are compared over the batches where both have a value:
   the adjusted Rand index (ARI) of their level groups, 1 when they split those
   batches into the same groups, about 0 when no more alike than chance. A pair
   needs :data:`MIN_BATCHES` batches. q is the chance of their level groups
   starting on that many of the same batches at random (hypergeometric),
   Benjamini-Hochberg adjusted over every pair compared at the site.
5. Groups of parameters and their representative as in :mod:`ml.correlation`, on
   the ARI, joined at a mean of :data:`THRESHOLD`. A pair whose q is not below
   :data:`ALPHA` counts as unrelated: a group boundary or two in common is easily
   chance.

This is description only. Nothing here changes the batch table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .correlation import LEVELS, CorrelationGroup, _bh, grouped
from .logger import log
from .missingness import _ordered
from .screening import SHIFTED

MODULE = "shifts"

#: Values a parameter needs, and batches in common a pair needs, to be compared.
MIN_BATCHES = 20
#: How far a value may move from the batch before and stay in its level group:
#: this many times the median change from one batch to the next.
TOLERANCE = 3.0
#: The batches a level group must average.
MIN_RUN = 3
#: How many times the spread within the level groups the jump between them must
#: be: within a group, the values are very close to the same.
SEPARATION = 4.0
#: The within-group correlation of value and batch position at which a parameter drifts.
MAX_DRIFT = 0.5
#: Shuffles of a parameter's values in the permutation test, and the p it must reach.
PERMUTATIONS = 200
P_MAX = 0.01
#: Mean ARI at which two parameters (or groups of them) join.
THRESHOLD = 0.8
#: The q below which a pair's shared group boundaries count for grouping.
ALPHA = 0.05
#: The screening reason that leaves this analysis's own decisions in it.
REASON = SHIFTED


@dataclass
class ShiftReport:
    #: The parameters that move in level groups, and how many were tested.
    analysed: list[str] = field(default_factory=list)
    tested: int = 0
    groups: list[CorrelationGroup] = field(default_factory=list)
    #: The batches compared, in date order; per parameter, its level group's level
    #: per batch as one base-36 digit (the rank among its groups, "." for no value),
    #: and per level group its first batch (a position in ``batches``) and value.
    batches: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    levels: dict[str, str] = field(default_factory=dict)
    segments: dict[str, dict] = field(default_factory=dict)
    #: ARI × 100, rounded, between the analysed parameters.
    similarity: list[list[int | None]] = field(default_factory=list)
    cohort: str = ""
    screened: dict[str, dict] = field(default_factory=dict)
    left_out: list[str] = field(default_factory=list)


def level_groups(x: np.ndarray, tolerance: float) -> np.ndarray:
    """Each value's level group: a new one where it moves more than ``tolerance``."""
    return np.cumsum(np.r_[False, np.abs(np.diff(x)) > tolerance])


def ari(a: np.ndarray, b: np.ndarray) -> float:
    """The adjusted Rand index of two labellings of the same items."""
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    table = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(table, (a, b), 1)

    def pairs(x):
        return (x * (x - 1) / 2).sum()

    both, rows, cols, total = (
        pairs(table),
        pairs(table.sum(1)),
        pairs(table.sum(0)),
        pairs(np.array([len(a)])),
    )
    expected, best = rows * cols / total, (rows + cols) / 2
    return float((both - expected) / (best - expected)) if best != expected else 0.0


def _numbers(values: pd.Series) -> tuple[np.ndarray, float | None]:
    """The values as numbers (a date in days), and the tolerance; a category: codes, None."""
    if pd.api.types.is_datetime64_any_dtype(values):
        x = ((values - pd.Timestamp(0)) / pd.Timedelta(days=1)).to_numpy(dtype=float)
    elif pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        x = values.to_numpy(dtype=float)
    else:
        return pd.factorize(values)[0].astype(float), None
    return x, TOLERANCE * float(np.median(np.abs(np.diff(x))))


def _text(value) -> str:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def for_build(
    frame: pd.DataFrame,
    id_column: str,
    order_column: str,
    analysed: pd.Series | None = None,
    screened_out: dict[str, str] | None = None,
) -> tuple[list[dict], ShiftReport]:
    """Groups of parameters whose level groups cover the same batches; one row per parameter.

    ``analysed`` and ``screened_out`` as for :func:`ml.correlation.for_build`.
    """
    frame, _ = _ordered(frame, order_column)
    if analysed is not None:
        frame = frame[analysed.reindex(frame.index).fillna(False).astype(bool)]
    screened_out = screened_out or {}
    report = ShiftReport()
    report.left_out = sorted(c for c, why in screened_out.items() if why != REASON)
    report.batches = (
        frame[id_column].astype(str).tolist() if id_column in frame.columns else []
    ) or [f"row {i}" for i in frame.index]
    report.dates = (
        pd.to_datetime(frame[order_column], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
        if order_column in frame.columns
        else pd.Series("", index=frame.index)
    ).tolist()

    rng = np.random.default_rng(0)
    rows_of: dict[str, dict] = {}
    labels: dict[str, np.ndarray] = {}
    for column in frame.columns:
        if column in (id_column, order_column) or column in report.left_out:
            continue
        values = frame[column]
        present = values.notna().to_numpy()
        if present.sum() < MIN_BATCHES or values.nunique(dropna=True) < 2:
            continue
        report.tested += 1
        x, tolerance = _numbers(values[present])
        group = level_groups(x, tolerance or 0.0)
        count = int(group[-1]) + 1
        if len(x) / count < MIN_RUN:
            continue
        means = np.bincount(group, x) / np.bincount(group)
        within = np.sqrt(((x - means[group]) ** 2).sum() / max(len(x) - count, 1))
        within = 0.0 if within < 1e-9 * np.abs(x).max() else within  # rounding in the means
        jump = float(np.median(np.abs(np.diff(means)))) if count > 1 else 0.0
        position = np.arange(len(x), dtype=float)
        dx = x - means[group]
        dp = position - (np.bincount(group, position) / np.bincount(group))[group]
        scale = np.sqrt((dx**2).sum() * (dp**2).sum())
        drift = float((dx * dp).sum() / scale) if scale > 1e-12 else 0.0
        if count < 2 or jump < SEPARATION * within or abs(drift) >= MAX_DRIFT:
            continue
        shuffled = rng.permuted(np.tile(x, (PERMUTATIONS, 1)), axis=1)
        fewer = (np.abs(np.diff(shuffled, axis=1)) > (tolerance or 0.0)).sum(axis=1) <= count - 1
        p = (1 + int(fewer.sum())) / (PERMUTATIONS + 1)
        if p > P_MAX:
            continue
        full = np.full(len(values), -1)
        full[present] = group
        labels[column] = full
        starts = np.flatnonzero(np.r_[True, np.diff(group) > 0])
        # A level group is drawn by the rank of its level among the parameter's groups.
        rank = pd.Series(means).rank(method="dense").to_numpy() - 1
        shade = np.round(rank / max(rank.max(), 1) * (LEVELS - 1)).astype(int)
        digits = np.full(len(values), ".", dtype="<U1")
        digits[present] = [np.base_repr(int(v), 36).lower() for v in shade[group]]
        report.levels[column] = "".join(digits)
        kept = values[present]
        report.segments[column] = {
            "starts": np.flatnonzero(present)[starts].tolist(),
            "values": [
                _text(kept.iloc[s:e].median() if tolerance else kept.iloc[s])
                for s, e in zip(starts, [*starts[1:], len(kept)])
            ],
        }
        rows_of[column] = {
            "Parameter": str(column),
            "Batches with a value": int(present.sum()),
            "Distinct values": int(values.nunique()),
            "Level groups": count,
            "Mean group length": round(len(x) / count, 1),
            # In the parameter's own unit (a date: days); none for a category.
            "Median jump": None if tolerance is None else float(f"{jump:.4g}"),
            "SD within groups": None if tolerance is None else float(f"{within:.4g}"),
            "Drift within groups": round(drift, 2),
            "Permutation p": round(p, 4),
        }
    report.analysed = list(labels)
    log.step(
        MODULE,
        f"{len(report.analysed):,} of {report.tested:,} parameters move in level groups over "
        f"{len(frame):,} batches; comparing their groups",
        f"{len(report.left_out):,} left out as screened out",
    )

    p = len(report.analysed)
    matrix = np.array([labels[c] for c in report.analysed]).reshape(p, len(frame))
    present = (matrix >= 0).astype(float)
    n = present @ present.T
    similarity, p_value = np.eye(p), np.full((p, p), np.nan)
    for i in range(p):
        for j in range(i + 1, p):
            both = (matrix[i] >= 0) & (matrix[j] >= 0)
            if both.sum() < MIN_BATCHES:
                continue
            a, b = matrix[i][both], matrix[j][both]
            similarity[i, j] = similarity[j, i] = ari(a, b)
            shift_a, shift_b = a[1:] != a[:-1], b[1:] != b[:-1]
            shared = int((shift_a & shift_b).sum())
            p_value[i, j] = stats.hypergeom.sf(
                shared - 1, len(shift_a), int(shift_a.sum()), int(shift_b.sum())
            )
    q = np.full((p, p), np.nan)
    upper = np.triu_indices(p, k=1)
    pairs = ~np.isnan(p_value[upper])
    if pairs.any():
        q[tuple(i[pairs] for i in upper)] = _bh(p_value[upper][pairs])
        q = np.fmin(q, q.T)
    report.similarity = [[int(round(100 * v)) for v in row] for row in similarity]
    supported = np.where(np.eye(p, dtype=bool) | (q < ALPHA), similarity, 0.0)
    report.groups = grouped(supported.clip(min=0), n, q, report.analysed, THRESHOLD, "Shifting")
    group_of = {m: g for g in report.groups for m in g.members}

    rows = []
    for i, column in enumerate(report.analysed):
        others = similarity[i].copy()
        others[i] = -1
        best = int(np.argmax(others)) if p > 1 else i
        found = best != i and others[best] > 0
        rows.append(
            {
                **rows_of[column],
                "Closest parameter": report.analysed[best] if found else "",
                "ARI with it": round(float(similarity[i, best]), 3) if found else None,
                "Batches in common": int(n[i, best]) if found else None,
                "q": float(f"{q[i, best]:.3g}") if found and not np.isnan(q[i, best]) else None,
                "Group": getattr(group_of.get(column), "name", ""),
                "Suggested representative": getattr(group_of.get(column), "representative", ""),
            }
        )
    log.success(
        MODULE,
        f"{len(report.groups)} groups of parameters whose level groups cover the same batches "
        f"(mean ARI ≥ {THRESHOLD})",
    )
    return rows, report
