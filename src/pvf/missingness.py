"""Parameters that go missing together, and what explains it.

Ported from the missingness analysis of the ``task_generator`` EDA tool, with its
defaults (Jaccard similarity, average linkage, the union pattern, normalised
mutual information), onto the task's cohort:

1. Every column becomes a missing/present vector over the cohort's batches.
   Columns missing in fewer than ``min_rate`` or more than ``max_rate`` of the
   batches are left out: a column that is nearly always there, or nearly never,
   has no pattern to share.
2. The Jaccard similarity of two columns is the share of batches missing either
   that miss both. Average linkage on ``1 - similarity``, cut at
   ``1 - threshold``, gives the groups; a group needs ``min_size`` members.
3. A group's pattern is the union: a batch counts as missing for the group when
   any member is missing.
4. Every categorical PTF parameter outside the group is scored as an explainer:
   the mutual information between its categories and the group's pattern,
   divided by the pattern's entropy, so 0 says nothing and 1 says the category
   alone decides whether the group is recorded. Batches where the candidate is
   itself blank are left out of its score. The EDA tool took the pattern's
   entropy over every batch and the joint over the non-blank ones; here both
   use the same batches, so the ratio stays within [0, 1] by construction.
5. For the best candidate: the missing rate per category, the share of the
   group's missing batches that fall in its worst category, and a flag when the
   candidate's own blanks track the pattern. A column that goes blank at the
   same time as the group has not explained anything.

This is description only. Nothing here changes what goes into the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .logger import log

MODULE = "missingness"

# Method constants carried over from the EDA tool. They define the method, not
# the task, so they are not task settings.
#: Categories rarer than this share of the scored batches are pooled as "other".
RARE_SHARE = 0.01
#: A candidate explainer needs a value in at least this share of the batches.
MIN_PRESENT = 0.5
#: A candidate needs this many scored batches per category, on average. Mutual
#: information rises with the number of categories: one with a batch or two in
#: each separates any pattern perfectly and would explain nothing. The EDA tool
#: had no such guard.
MIN_PER_CATEGORY = 5
#: |phi| between a candidate's own blanks and the pattern above which it is flagged.
SELF_CORRELATION = 0.3
#: How many runner-up explainers the report lists.
ALTERNATIVES = 4


@dataclass
class Explainer:
    column: str
    score: float
    #: category → share of its batches where the group is missing
    rates: dict[str, float] = field(default_factory=dict)
    #: category → batches with that category
    batches: dict[str, int] = field(default_factory=dict)
    coverage: float = 0.0
    self_correlated: bool = False


@dataclass
class MissingGroup:
    name: str
    members: list[str]
    member_rates: dict[str, float]
    group_rate: float
    cohesion: float
    explainer: Explainer | None
    strong: bool
    alternatives: list[tuple[str, float]]
    #: members × batches, 1 where the member is missing, batches in ``batch_labels`` order
    pattern: list[list[int]]
    batch_labels: list[str]


@dataclass
class MissingnessReport:
    enabled: bool = False
    rows: int = 0
    columns: int = 0
    analysed: list[str] = field(default_factory=list)
    too_complete: int = 0
    too_empty: int = 0
    groups: list[MissingGroup] = field(default_factory=list)
    singletons: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)
    order: str = ""
    settings: dict = field(default_factory=dict)


def _jaccard(matrix: np.ndarray) -> np.ndarray:
    m = matrix.astype(np.float64)
    both = m.T @ m
    counts = m.sum(axis=0)
    either = counts[:, None] + counts[None, :] - both
    with np.errstate(invalid="ignore", divide="ignore"):
        similarity = np.where(either == 0, 0.0, both / either)
    np.fill_diagonal(similarity, 1.0)
    return similarity


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _pooled(values: pd.Series) -> pd.Series:
    """Categories as text, with the rare ones pooled."""
    text = values.astype(str)
    share = text.value_counts(normalize=True)
    return text.where(~text.isin(share[share < RARE_SHARE].index), "other")


def _score(values: pd.Series, missing: np.ndarray) -> float:
    """Normalised mutual information between a category and a missing pattern."""
    present = values.notna().to_numpy()
    target = missing[present]
    if not len(target) or target.all() or not target.any():
        return 0.0
    joint = pd.crosstab(_pooled(values[present]).to_numpy(), target).to_numpy() / len(target)
    p_category = joint.sum(axis=1, keepdims=True)
    p_missing = joint.sum(axis=0, keepdims=True)
    nonzero = joint > 0
    mi = float((joint[nonzero] * np.log2(joint[nonzero] / (p_category @ p_missing)[nonzero])).sum())
    return float(np.clip(mi / _entropy(p_missing.ravel()), 0.0, 1.0))


def _phi(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _explain(values: pd.Series, missing: np.ndarray, column: str, score: float) -> Explainer:
    present = values.notna().to_numpy()
    frame = pd.DataFrame(
        {"category": _pooled(values[present]).to_numpy(), "missing": missing[present]}
    )
    grouped = frame.groupby("category")["missing"]
    rates = grouped.mean().sort_values(ascending=False)
    worst = rates.index[0] if len(rates) else None
    missed = int(frame["missing"].sum())
    coverage = (
        float(((frame["category"] == worst) & frame["missing"]).sum() / missed) if missed else 0.0
    )
    return Explainer(
        column=column,
        score=score,
        rates={str(k): float(v) for k, v in rates.items()},
        batches={str(k): int(v) for k, v in grouped.size().items()},
        coverage=coverage,
        self_correlated=abs(_phi(~present, missing)) > SELF_CORRELATION,
    )


def explainer_candidates(
    frame: pd.DataFrame, value_types: dict[str, str], max_categories: int, exclude: set[str]
) -> list[str]:
    """Categorical PTF parameters with few enough categories and enough values in each."""
    return [
        column
        for column in frame.columns
        if column not in exclude
        and (
            value_types.get(column) in ("categorical", "boolean")
            or str(value_types.get(column, "")).strip().startswith("[")
        )
        and 2 <= frame[column].nunique(dropna=True) <= max_categories
        and frame[column].notna().mean() >= MIN_PRESENT
        and frame[column].notna().sum() >= MIN_PER_CATEGORY * frame[column].nunique(dropna=True)
    ]


def analyse(
    frame: pd.DataFrame,
    value_types: dict[str, str],
    settings,
    id_column: str = "",
    order_column: str = "",
) -> MissingnessReport:
    """Find the groups of columns that go missing together, and score explainers.

    Parameters
    ----------
    frame
        The cohort: one row per batch, every PVF column.
    value_types
        PTF ``Value Type`` per parameter; decides which columns may explain.
    settings
        The task's ``missingness`` block (:class:`pvf.taskconfig.Missingness`).
    id_column
        Labels the batches in the pattern plot, and is never an explainer.
    order_column
        Batches are drawn in this column's date order, when it is given and present.
    """
    report = MissingnessReport(
        enabled=settings.enabled,
        rows=len(frame),
        columns=frame.shape[1],
        settings={
            "threshold": settings.threshold,
            "min_rate": settings.min_rate,
            "max_rate": settings.max_rate,
            "min_size": settings.min_size,
            "min_strength": settings.min_strength,
            "max_categories": settings.max_categories,
        },
    )
    if not settings.enabled or frame.empty:
        return report
    log.step(MODULE, f"Looking for parameters that go missing together over {len(frame)} batches")

    if order_column and order_column in frame.columns:
        frame = frame.iloc[
            np.argsort(
                pd.to_datetime(frame[order_column], errors="coerce").to_numpy(), kind="stable"
            )
        ]
        report.order = f"ordered by {order_column}"
    else:
        report.order = "in PVF order"
    labels = (
        frame[id_column].astype(str).tolist()
        if id_column and id_column in frame.columns
        else [f"row {i}" for i in frame.index]
    )

    blank = frame.isna()
    rates = blank.mean()
    report.too_complete = int((rates < settings.min_rate).sum())
    report.too_empty = int((rates > settings.max_rate).sum())
    report.analysed = [
        str(c) for c in frame.columns if settings.min_rate <= rates[c] <= settings.max_rate
    ]
    report.candidates = explainer_candidates(
        frame, value_types, settings.max_categories, {id_column} if id_column else set()
    )
    if len(report.analysed) < 2:
        log.info(MODULE, f"{len(report.analysed)} columns have a missing rate in range; no groups")
        report.singletons = list(report.analysed)
        return report

    matrix = blank[report.analysed].to_numpy()
    similarity = _jaccard(matrix)
    distance = squareform(np.clip(1.0 - similarity, 0.0, None), checks=False)
    cut = fcluster(
        linkage(distance, method="average"), t=1.0 - settings.threshold, criterion="distance"
    )

    members_of: dict[int, list[int]] = {}
    for index, label in enumerate(cut):
        members_of.setdefault(int(label), []).append(index)
    groups = [idx for idx in members_of.values() if len(idx) >= settings.min_size]
    report.singletons = [
        report.analysed[i]
        for idx in members_of.values()
        if len(idx) < settings.min_size
        for i in idx
    ]
    # Largest first, then the most often missing; the name is the tie-break, so
    # the numbering does not depend on the order linkage labelled them in.
    groups.sort(
        key=lambda idx: (-len(idx), -matrix[:, idx].any(axis=1).mean(), report.analysed[idx[0]])
    )

    for number, idx in enumerate(groups, start=1):
        members = [report.analysed[i] for i in idx]
        pattern = matrix[:, idx].any(axis=1)
        sub = similarity[np.ix_(idx, idx)]
        cohesion = float(sub[~np.eye(len(idx), dtype=bool)].mean())

        scored = sorted(
            (
                (column, _score(frame[column], pattern))
                for column in report.candidates
                if column not in members
            ),
            key=lambda item: (-item[1], item[0]),
        )
        best = _explain(frame[scored[0][0]], pattern, *scored[0]) if scored else None
        report.groups.append(
            MissingGroup(
                name=f"Missing {number:02d}",
                members=members,
                member_rates={m: float(rates[m]) for m in members},
                group_rate=float(pattern.mean()),
                cohesion=cohesion,
                explainer=best,
                strong=bool(best and best.score >= settings.min_strength),
                alternatives=scored[1 : 1 + ALTERNATIVES],
                pattern=matrix[:, idx].T.astype(int).tolist(),
                batch_labels=labels,
            )
        )

    explained = sum(group.strong for group in report.groups)
    log.success(
        MODULE,
        f"{len(report.groups)} groups of parameters missing together, "
        f"{explained} of them with an explainer scoring ≥ {settings.min_strength}",
        f"{len(report.analysed)} of {report.columns} columns in range; "
        f"{len(report.candidates)} categorical parameters scored as explainers",
    )
    return report


def rows(report: MissingnessReport) -> list[dict]:
    """One row per grouped column, then the columns that grouped with nothing."""
    out = [
        {
            "Group": group.name,
            "Parameter": member,
            "Missing rate": round(group.member_rates[member], 4),
            "Group missing rate": round(group.group_rate, 4),
            "Mean Jaccard within group": round(group.cohesion, 4),
            "Best explainer": group.explainer.column if group.explainer else "",
            "Explainer score": round(group.explainer.score, 4) if group.explainer else "",
            "Explainer passes min_strength": group.strong,
            "Explainer blanks track the group": (
                group.explainer.self_correlated if group.explainer else ""
            ),
        }
        for group in report.groups
        for member in group.members
    ]
    out += [
        {
            "Group": "",
            "Parameter": column,
            "Missing rate": "",
            "Group missing rate": "",
            "Mean Jaccard within group": "",
            "Best explainer": "",
            "Explainer score": "",
            "Explainer passes min_strength": "",
            "Explainer blanks track the group": "",
        }
        for column in report.singletons
    ]
    return out
