"""Numerical parameters that put the batches in the same order.

A decision tree splits on the order of a parameter's values, not on the values
themselves: two parameters that rank the batches alike offer it the same splits,
and it picks one of them by chance. This module finds such parameters, per site,
over the same batches as the missingness report:

1. Every numerical catalog parameter (``Value Type`` numeric, duration or ratio)
   with at least :data:`MIN_BATCHES` values and two distinct ones takes part,
   whatever the screening register says of it: the register is shown alongside,
   never used to pick (:attr:`CorrelationReport.not_analysed` says why the others
   do not take part).
2. Two parameters are compared by Spearman's rank correlation over the batches
   where both have a value; the pair needs :data:`MIN_BATCHES` of them. The sign
   does not matter to a tree, so the similarity is |ρ|.
3. Complete linkage on ``1 - |ρ|``, cut at ``1 - threshold``, gives the groups:
   every pair of members has |ρ| of at least the threshold (``build.
   correlation_threshold``, default :data:`THRESHOLD`). A pair with
   too few batches in common counts as unrelated.
4. Per pair a p-value from ``t = ρ √((n-2)/(1-ρ²))`` on n-2 degrees of freedom,
   Benjamini-Hochberg adjusted over every pair compared at the site (q). A group
   reports its weakest pair: the lowest |ρ|, and the highest q.
5. Each group suggests the member that carries most of the others' information:
   the highest mean, over the other members, of |ρ| times the share of that
   member's batches the candidate has a value on too (a parameter only stands in
   for another where it was measured). Ties go to the most batches with a value.

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
from .screening import CORRELATED

MODULE = "correlation"

#: The catalog value types that are numbers with an order.
NUMERIC_TYPES = ("numeric", "duration", "ratio")
#: The |ρ| every pair in a group reaches, unless config.yaml sets another.
THRESHOLD = 0.9
#: Values a parameter needs, and batches in common a pair needs, to be compared.
MIN_BATCHES = 20
#: The screening reason of this report's own decisions.
REASON = CORRELATED


@dataclass
class CorrelationGroup:
    name: str
    members: list[str]
    #: The mean and the lowest similarity (|ρ|, or the shared shifts) between members.
    mean: float
    weakest: float
    min_batches: int
    max_q: float
    representative: str = ""
    #: The representative's mean |ρ| × shared coverage over the other members.
    score: float = 0.0
    #: Every member's score, as the representative's: the order to pick another in.
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class CorrelationReport:
    analysed: list[str] = field(default_factory=list)
    groups: list[CorrelationGroup] = field(default_factory=list)
    #: The batches compared, in date order, and each analysed parameter's value per
    #: batch (None where it has none): what the viewer's scatterplots draw.
    batches: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    values: dict[str, list[float | None]] = field(default_factory=dict)
    #: ρ × 100, rounded, between the analysed parameters, in ``analysed`` order.
    rho: list[list[int | None]] = field(default_factory=list)
    cohort: str = ""
    #: Set by the build: parameter → its standing exclusion at this site, made in this
    #: report (``screened``) or in the missingness report (``elsewhere``), and the
    #: representatives here that are excluded themselves.
    screened: dict[str, dict] = field(default_factory=dict)
    elsewhere: dict[str, dict] = field(default_factory=dict)
    conflicts: list[dict] = field(default_factory=list)
    #: Catalog parameters that do not take part, and why.
    not_analysed: dict[str, str] = field(default_factory=dict)
    #: The |ρ| every pair in a group reaches.
    threshold: float = THRESHOLD


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def grouped(
    similarity: np.ndarray,
    n: np.ndarray,
    q: np.ndarray,
    names: list[str],
    threshold: float,
    prefix: str,
    method: str = "average",
) -> list[CorrelationGroup]:
    """Hierarchical groups on ``1 - similarity``, cut at ``1 - threshold``, with their stats.

    ``method`` is the linkage: ``"complete"`` holds every pair to the threshold,
    ``"average"`` only the mean between the two groups it joins.

    ``n`` holds the batches each pair shares (its diagonal: each parameter's), ``q``
    each pair's adjusted p-value. Each group's suggested representative comes first.
    """
    groups: list[list[int]] = []
    if len(names) >= 2:
        distance = squareform(1.0 - similarity, checks=False)
        cut = fcluster(linkage(distance, method=method), t=1.0 - threshold, criterion="distance")
        members_of: dict[int, list[int]] = {}
        for index, label in enumerate(cut):
            members_of.setdefault(int(label), []).append(index)
        groups = [idx for idx in members_of.values() if len(idx) >= 2]
    # Largest first, then the tightest; the name is the tie-break, as in missingness.
    groups.sort(key=lambda idx: (-len(idx), -similarity[np.ix_(idx, idx)].mean(), names[idx[0]]))
    out = []
    for number, idx in enumerate(groups, start=1):
        sub_n = n[np.ix_(idx, idx)]
        carried = similarity[np.ix_(idx, idx)] * sub_n / np.diag(sub_n)[None, :]
        np.fill_diagonal(carried, 0.0)
        scores = carried.sum(axis=1) / (len(idx) - 1)
        best = max(range(len(idx)), key=lambda k: (scores[k], sub_n[k, k]))
        # The representative first: the viewer draws the group from it and labels each row to it.
        members = list(idx)
        idx = [idx[best]] + [i for k, i in enumerate(idx) if k != best]
        off = ~np.eye(len(idx), dtype=bool)
        sub, qs = similarity[np.ix_(idx, idx)][off], q[np.ix_(idx, idx)][off]
        out.append(
            CorrelationGroup(
                name=f"{prefix} {number:02d}",
                members=[names[i] for i in idx],
                mean=float(sub.mean()),
                weakest=float(sub.min()),
                min_batches=int(n[np.ix_(idx, idx)][off].min()),
                max_q=float(np.nanmax(qs)) if np.isfinite(qs).any() else float("nan"),
                representative=names[idx[0]],
                score=float(scores[best]),
                scores={names[i]: float(scores[k]) for k, i in enumerate(members)},
            )
        )
    return out


def for_build(
    frame: pd.DataFrame,
    value_types: dict[str, str],
    id_column: str,
    order_column: str,
    analysed: pd.Series | None = None,
    threshold: float = THRESHOLD,
) -> tuple[list[dict], CorrelationReport]:
    """Groups of numerical parameters that rank the batches alike, and one row per parameter.

    ``analysed`` marks the batches compared (default: all), as for the missingness
    report. ``threshold`` is the |ρ| every pair in a group must reach.
    """
    frame, _ = _ordered(frame, order_column)
    if analysed is not None:
        frame = frame[analysed.reindex(frame.index).fillna(False).astype(bool)]
    report = CorrelationReport(threshold=threshold)
    numeric = []
    for c in frame.columns:
        if c not in value_types:
            continue
        if str(value_types.get(c, "")).strip() not in NUMERIC_TYPES:
            report.not_analysed[c] = "not numerical in the catalog"
        elif not pd.api.types.is_numeric_dtype(frame[c]) or pd.api.types.is_bool_dtype(frame[c]):
            report.not_analysed[c] = "its values are not numbers"
        else:
            numeric.append(c)
    values = frame[numeric].astype(float)
    for c in numeric:
        if values[c].notna().sum() < MIN_BATCHES:
            report.not_analysed[c] = f"fewer than {MIN_BATCHES} values"
        elif values[c].nunique(dropna=True) < 2:
            report.not_analysed[c] = "one value only"
        else:
            report.analysed.append(c)
    log.step(
        MODULE,
        f"Ranking {len(report.analysed):,} numerical parameters against each other over "
        f"{len(frame):,} batches",
        f"{len(report.not_analysed):,} catalog parameters not compared",
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

    similarity = np.nan_to_num(np.abs(rho))
    np.fill_diagonal(similarity, 1.0)
    report.groups = grouped(
        similarity, n, q, report.analysed, threshold, "Correlated", method="complete"
    )
    report.values = {
        c: [None if pd.isna(v) else float(f"{v:.4g}") for v in values[c]] for c in report.analysed
    }
    rows_of = {
        m: {"Group": g.name, "Suggested representative": g.representative}
        for g in report.groups
        for m in g.members
    }

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
                "Suggested representative": rows_of.get(column, {}).get(
                    "Suggested representative", ""
                ),
            }
        )
    log.success(
        MODULE,
        f"{len(report.groups)} groups of numerical parameters that rank the batches alike "
        f"(every pair |ρ| ≥ {threshold})",
    )
    return rows, report
