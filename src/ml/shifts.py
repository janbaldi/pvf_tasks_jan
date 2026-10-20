"""Parameters that hold one value for a run of batches, then shift to another, together.

A raw material or consumable lot, its certificate values, a vector lot and its
titre: each is the same on every batch made with that lot and changes when the
lot does. Such a parameter is a step function of time, not a trend, and every
parameter that steps on the same batches carries the same information: which
lot a batch was made with. This module finds them, per site, over the same
batches as the missingness report:

1. Every parameter (any type) with at least :data:`MIN_BATCHES` values and two
   distinct ones is tested, unless the screening register excludes it at the
   site for another reason than :data:`REASON`.
2. With the batches in date order and blanks skipped, a run is a stretch of
   consecutive batches with exactly the same value. The parameter *shifts in
   steps* when its runs average at least :data:`MIN_RUN` batches and there are
   fewer runs than chance would give for those values (the runs test for any
   number of categories, Barton and David, |z| > :data:`~ml.missingness.RUNS_Z`).
   A trend changes value on almost every batch, so it has no runs to speak of.
3. Two such parameters are compared over the batches where both have a value:
   the Jaccard similarity of their shifts, the share of batches where either
   changes value that both change on. A pair needs :data:`MIN_BATCHES` batches.
   The chance of sharing that many shifts at random is hypergeometric; q is that
   p-value, Benjamini-Hochberg adjusted over every pair compared at the site.
4. Groups and their representative as in :mod:`ml.correlation`, on the Jaccard
   similarity, joined at a mean of :data:`THRESHOLD`. A pair whose q is not below
   :data:`ALPHA` counts as unrelated: one shared shift is easily chance.

This is description only. Nothing here changes the batch table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .correlation import CorrelationGroup, _bh, grouped
from .logger import log
from .missingness import RUNS_Z, _ordered
from .screening import SHIFTED

MODULE = "shifts"

#: Values a parameter needs, and batches in common a pair needs, to be compared.
MIN_BATCHES = 20
#: The batches a run must average for a parameter to shift in steps.
MIN_RUN = 3
#: Mean Jaccard similarity of shifts at which two parameters (or groups) join.
THRESHOLD = 0.8
#: The q below which a pair's shared shifts count for grouping.
ALPHA = 0.05
#: The screening reason that leaves this analysis's own decisions in it.
REASON = SHIFTED


@dataclass
class ShiftReport:
    #: The parameters that shift in steps, and how many were tested.
    analysed: list[str] = field(default_factory=list)
    tested: int = 0
    groups: list[CorrelationGroup] = field(default_factory=list)
    #: The batches compared, in date order; per parameter, its runs drawn as
    #: alternating "0"/"1" ("." where it has no value), and each run's value.
    batches: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    runs: dict[str, str] = field(default_factory=dict)
    values: dict[str, list[str]] = field(default_factory=dict)
    #: Jaccard similarity × 100, rounded, between the analysed parameters.
    similarity: list[list[int | None]] = field(default_factory=list)
    cohort: str = ""
    screened: dict[str, dict] = field(default_factory=dict)
    left_out: list[str] = field(default_factory=list)


def runs_test(codes: np.ndarray) -> tuple[int, float]:
    """The runs of a sequence of category codes, and the z of their count against chance."""
    n = len(codes)
    runs = 1 + int((codes[1:] != codes[:-1]).sum())
    counts = np.unique(codes, return_counts=True)[1].astype(float)
    square, cube = (counts**2).sum(), (counts**3).sum()
    expected = (n * (n + 1) - square) / n
    variance = (square * (square + n * (n + 1)) - 2 * n * cube - n**3) / (n**2 * (n - 1))
    return runs, (runs - expected) / np.sqrt(variance) if variance > 0 else 0.0


def _text(value) -> str:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def for_build(
    frame: pd.DataFrame,
    id_column: str,
    order_column: str,
    analysed: pd.Series | None = None,
    screened_out: dict[str, str] | None = None,
) -> tuple[list[dict], ShiftReport]:
    """Groups of parameters that shift value on the same batches, and one row per such parameter.

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

    rows_of: dict[str, dict] = {}
    codes: dict[str, np.ndarray] = {}
    for column in frame.columns:
        if column in (id_column, order_column) or column in report.left_out:
            continue
        values = frame[column]
        present = values.notna().to_numpy()
        if present.sum() < MIN_BATCHES or values.nunique(dropna=True) < 2:
            continue
        report.tested += 1
        code = pd.factorize(values[present])[0]
        runs, z = runs_test(code)
        if len(code) / runs < MIN_RUN or z >= -RUNS_Z:
            continue
        full = np.full(len(values), -1)
        full[present] = code
        codes[column] = full
        starts = np.flatnonzero(np.r_[True, code[1:] != code[:-1]])
        parity = (np.cumsum(np.r_[True, code[1:] != code[:-1]]) - 1) % 2
        bits = np.full(len(values), ".", dtype="<U1")
        bits[present] = parity.astype(str)
        report.runs[column] = "".join(bits)
        report.values[column] = [_text(v) for v in values[present].iloc[starts]]
        rows_of[column] = {
            "Parameter": str(column),
            "Batches with a value": int(present.sum()),
            "Distinct values": int(values.nunique()),
            "Runs": runs,
            "Mean run length": round(len(code) / runs, 1),
            "Runs test z": round(float(z), 2),
        }
    report.analysed = list(codes)
    log.step(
        MODULE,
        f"{len(report.analysed):,} of {report.tested:,} parameters shift value in steps over "
        f"{len(frame):,} batches; comparing their shifts",
        f"{len(report.left_out):,} left out as screened out",
    )

    p = len(report.analysed)
    matrix = np.array([codes[c] for c in report.analysed]).reshape(p, len(frame))
    present = (matrix >= 0).astype(float)
    n = present @ present.T
    similarity, p_value = np.eye(p), np.full((p, p), np.nan)
    for i in range(p):
        for j in range(i + 1, p):
            both = (matrix[i] >= 0) & (matrix[j] >= 0)
            if both.sum() < MIN_BATCHES:
                continue
            a, b = matrix[i][both], matrix[j][both]
            shift_a, shift_b = a[1:] != a[:-1], b[1:] != b[:-1]
            shared, either = int((shift_a & shift_b).sum()), int((shift_a | shift_b).sum())
            similarity[i, j] = similarity[j, i] = shared / either if either else 0.0
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
    report.groups = grouped(supported, n, q, report.analysed, THRESHOLD, "Shifting")
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
                "Jaccard with it": round(float(similarity[i, best]), 3) if found else None,
                "Batches in common": int(n[i, best]) if found else None,
                "q": float(f"{q[i, best]:.3g}") if found and not np.isnan(q[i, best]) else None,
                "Group": getattr(group_of.get(column), "name", ""),
                "Suggested representative": getattr(group_of.get(column), "representative", ""),
            }
        )
    log.success(
        MODULE,
        f"{len(report.groups)} groups of parameters that shift value together "
        f"(mean Jaccard of shifts ≥ {THRESHOLD})",
    )
    return rows, report
