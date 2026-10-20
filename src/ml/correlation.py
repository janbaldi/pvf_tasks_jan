"""Numerical parameters that put the batches in the same order.

A decision tree splits on the order of a parameter's values, not on the values
themselves: two parameters that rank the batches alike offer it the same splits,
and it picks one of them by chance. This module finds such parameters, per site,
over the same batches as the missingness report:

1. Every numerical catalog parameter (``Value Type`` numeric, duration or ratio)
   with at least :data:`MIN_BATCHES` values and two distinct ones takes part,
   unless the screening register excludes it at the site for another reason
   than :data:`REASON`: what is already left out needs no representative.
2. Two parameters are compared by Spearman's rank correlation over the batches
   where both have a value; the pair needs :data:`MIN_BATCHES` of them. The sign
   does not matter to a tree, so the similarity is |ρ|.
3. Average linkage on ``1 - |ρ|``, cut at ``1 - THRESHOLD``, gives the groups: the
   average |ρ| between two merged groups reaches :data:`THRESHOLD`. A pair with
   too few batches in common counts as unrelated.
4. Per pair a p-value from ``t = ρ √((n-2)/(1-ρ²))`` on n-2 degrees of freedom,
   Benjamini-Hochberg adjusted over every pair compared at the site (q). A group
   reports its weakest pair: the lowest |ρ|, and the highest q.

This is description only. Nothing here changes the batch table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .logger import log
from .missingness import _ordered

MODULE = "correlation"

#: The catalog value types that are numbers with an order.
NUMERIC_TYPES = ("numeric", "duration", "ratio")
#: Average |ρ| at which two parameters (or groups) join.
THRESHOLD = 0.8
#: Values a parameter needs, and batches in common a pair needs, to be compared.
MIN_BATCHES = 20
#: The screening reason that leaves this analysis's own decisions in it.
REASON = "correlated with group"
#: Levels a value's percentile rank is drawn in: one base-36 digit per batch.
LEVELS = 36


@dataclass
class CorrelationGroup:
    name: str
    members: list[str]
    mean_rho: float
    min_rho: float
    min_batches: int
    max_q: float


@dataclass
class CorrelationReport:
    analysed: list[str] = field(default_factory=list)
    groups: list[CorrelationGroup] = field(default_factory=list)
    #: The batches compared, in date order, and what the viewer draws per parameter:
    #: its percentile rank per batch, one base-36 digit, "." where it has no value.
    batches: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    ranks: dict[str, str] = field(default_factory=dict)
    #: ρ × 100, rounded, between the analysed parameters, in ``analysed`` order.
    rho: list[list[int | None]] = field(default_factory=list)
    cohort: str = ""
    #: Parameter → the screening decision standing for it at this site.
    screened: dict[str, dict] = field(default_factory=dict)
    #: Parameters left out because the register already excludes them here.
    left_out: list[str] = field(default_factory=list)


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def _rank_text(values: pd.Series) -> str:
    """Percentile ranks as one base-36 digit per batch, "." where there is no value."""
    spread = (values.rank() - 1) / max(values.notna().sum() - 1, 1)
    level = (spread * (LEVELS - 1)).round()
    return "".join("." if pd.isna(v) else np.base_repr(int(v), 36).lower() for v in level)


def for_build(
    frame: pd.DataFrame,
    value_types: dict[str, str],
    id_column: str,
    order_column: str,
    analysed: pd.Series | None = None,
    screened_out: dict[str, str] | None = None,
) -> tuple[list[dict], CorrelationReport]:
    """Groups of numerical parameters that rank the batches alike, and one row per parameter.

    ``analysed`` marks the batches compared (default: all), as for the missingness
    report. ``screened_out`` maps a parameter to the reason the register excludes
    it at this site; it takes part only when that reason is :data:`REASON`.
    """
    frame, _ = _ordered(frame, order_column)
    if analysed is not None:
        frame = frame[analysed.reindex(frame.index).fillna(False).astype(bool)]
    screened_out = screened_out or {}
    report = CorrelationReport()
    report.left_out = sorted(c for c, why in screened_out.items() if why != REASON)
    numeric = [
        c
        for c in frame.columns
        if str(value_types.get(c, "")).strip() in NUMERIC_TYPES
        and c not in report.left_out
        and pd.api.types.is_numeric_dtype(frame[c])
        and not pd.api.types.is_bool_dtype(frame[c])
    ]
    values = frame[numeric].astype(float)
    report.analysed = [
        c
        for c in numeric
        if values[c].notna().sum() >= MIN_BATCHES and values[c].nunique(dropna=True) >= 2
    ]
    log.step(
        MODULE,
        f"Ranking {len(report.analysed):,} numerical parameters against each other over "
        f"{len(frame):,} batches",
        f"{len(report.left_out):,} left out as screened out",
    )
    report.batches = (
        frame[id_column].astype(str).tolist() if id_column in frame.columns else []
    ) or [f"row {i}" for i in frame.index]
    report.dates = (
        pd.to_datetime(frame[order_column], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
        if order_column in frame.columns
        else pd.Series("", index=frame.index)
    ).tolist()
    values = values[report.analysed]
    report.ranks = {c: _rank_text(values[c]) for c in report.analysed}

    p = len(report.analysed)
    rho = values.corr(method="spearman", min_periods=MIN_BATCHES).to_numpy()
    present = values.notna().to_numpy(dtype=float)
    n = present.T @ present
    report.rho = [[None if np.isnan(v) else int(round(100 * v)) for v in row] for row in rho]
    upper = np.triu_indices(p, k=1)
    pairs = ~np.isnan(rho[upper])
    r, df = np.abs(rho[upper][pairs]).clip(max=1 - 1e-12), n[upper][pairs] - 2
    q = np.full(rho.shape, np.nan)
    if pairs.any():
        q[tuple(i[pairs] for i in upper)] = _bh(2 * stats.t.sf(r * np.sqrt(df / (1 - r**2)), df))
        q = np.fmin(q, q.T)

    rows_of: dict[str, dict] = {}
    similarity = np.nan_to_num(np.abs(rho))
    np.fill_diagonal(similarity, 1.0)
    groups: list[list[int]] = []
    if p >= 2:
        distance = squareform(1.0 - similarity, checks=False)
        cut = fcluster(linkage(distance, method="average"), t=1.0 - THRESHOLD, criterion="distance")
        members_of: dict[int, list[int]] = {}
        for index, label in enumerate(cut):
            members_of.setdefault(int(label), []).append(index)
        groups = [idx for idx in members_of.values() if len(idx) >= 2]
    # Largest first, then the tightest; the name is the tie-break, as in missingness.
    groups.sort(
        key=lambda idx: (
            -len(idx),
            -similarity[np.ix_(idx, idx)].mean(),
            report.analysed[idx[0]],
        )
    )
    for number, idx in enumerate(groups, start=1):
        off = ~np.eye(len(idx), dtype=bool)
        sub, qs = similarity[np.ix_(idx, idx)][off], q[np.ix_(idx, idx)][off]
        group = CorrelationGroup(
            name=f"Correlated {number:02d}",
            members=[report.analysed[i] for i in idx],
            mean_rho=float(sub.mean()),
            min_rho=float(sub.min()),
            min_batches=int(n[np.ix_(idx, idx)][off].min()),
            max_q=float(np.nanmax(qs)) if np.isfinite(qs).any() else float("nan"),
        )
        report.groups.append(group)
        for i in idx:
            rows_of[report.analysed[i]] = {"Group": group.name}

    rows = []
    for i, column in enumerate(report.analysed):
        others = similarity[i].copy()
        others[i] = -1
        best = int(np.argmax(others)) if p > 1 else i
        rows.append(
            {
                "Parameter": column,
                "Batches with a value": int(n[i, i]),
                "Distinct values": int(values[column].nunique()),
                "Closest parameter": report.analysed[best] if best != i else "",
                "|ρ| with it": round(float(similarity[i, best]), 3) if best != i else None,
                "Batches in common": int(n[i, best]) if best != i else None,
                "q": float(f"{q[i, best]:.3g}") if best != i and not np.isnan(q[i, best]) else None,
                "Group": rows_of.get(column, {}).get("Group", ""),
            }
        )
    log.success(
        MODULE,
        f"{len(report.groups)} groups of numerical parameters that rank the batches alike "
        f"(average |ρ| ≥ {THRESHOLD})",
    )
    return rows, report
