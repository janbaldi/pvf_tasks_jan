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
4. Every categorical catalog parameter outside the group is scored as an explainer:
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

The build runs the same grouping over the whole batch table (:func:`for_build`),
and adds what an audit of a column filter asks for, per parameter:

6. With the batches in date order, the gaps (runs of missing batches) are
   counted against what chance would give for that many missing batches
   (Wald-Wolfowitz runs test). Fewer, longer gaps than chance are clumped and
   named by where they sit: at the start (introduced late), at the end
   (retired), at both ends (measured in a window), in one stretch (gap in the
   middle). More gaps than chance is regular skipping. See :data:`PATTERNS`.
7. A depth-2 decision tree on the categorical parameters
   names the batches the parameter is missing on, e.g. "Site = B". Its leaves
   are scored like an explainer in 4. A category whose own blanks track the
   pattern is not offered to the tree, for the reason given in 5. For a group,
   the tree replaces the explainer of 4, so the group and its members are
   explained by one method.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.tree import DecisionTreeClassifier

from .logger import log
from .taskconfig import Missingness

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
#: |z| of the runs test beyond which the gaps are not random: two-sided p < 0.01.
RUNS_Z = 2.576
#: Share of a parameter's missing batches one gap (or the two end gaps) must
#: hold to name the pattern after it; the rest may be scattered single misses.
DOMINANT = 0.8
#: A decision tree scoring at least this is reported as the condition.
CONDITIONAL = 0.5
#: Two splits name conditions like "Site = B and Kit = X".
TREE_DEPTH = 2
#: The latest batches "Measured in last" counts.
RECENT = 10

#: Every pattern a parameter is given, and what it means. The order is the report's.
PATTERNS = {
    "Complete": "Never missing.",
    "Never recorded": "Missing on every batch.",
    "Introduced late": "Clumped, and most missing batches are the first ones: "
    "measured from some batch on.",
    "Retired": "Clumped, and most missing batches are the last ones: no longer measured.",
    "Measured in a window": "Clumped, and most missing batches are the first and the last.",
    "Gap in the middle": "Clumped, and most missing batches are one stretch in the middle.",
    "Clumped": "Fewer, longer gaps than chance, spread over several stretches.",
    "Random gaps": "As many gaps as chance would give: consistent with missing at random.",
    "Regular skipping": "More gaps than chance: missing batches avoid each other, "
    "as when every n-th batch is skipped.",
}


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
    #: Set by the build: the group pattern's :data:`PATTERNS` key and its tree condition.
    kind: str = ""
    condition: str = ""


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
    #: Set by the build, for the pattern viewer: the batches in order, and per
    #: column with any blank its missing (1) / present (0) string over them.
    batches: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    missing: dict[str, str] = field(default_factory=dict)


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
    return _nmi(_pooled(values[present]).to_numpy(), target)


def _nmi(categories: np.ndarray, target: np.ndarray) -> float:
    """Mutual information over the target's entropy; the target holds both classes."""
    joint = pd.crosstab(categories, target).to_numpy() / len(target)
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
    """Categorical catalog parameters with few enough categories and enough values in each."""
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


def _ordered(frame: pd.DataFrame, order_column: str) -> tuple[pd.DataFrame, str]:
    """The batches in date order, undated ones last; and how they were ordered."""
    if not order_column or order_column not in frame.columns:
        return frame, "in batch table order"
    dates = pd.to_datetime(frame[order_column], errors="coerce")
    return frame.iloc[np.argsort(dates.to_numpy(), kind="stable")], f"ordered by {order_column}"


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
        The cohort: one row per batch, every batch table column.
    value_types
        catalog ``Value Type`` per parameter; decides which columns may explain.
    settings
        The task's ``missingness`` block (:class:`ml.taskconfig.Missingness`).
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

    frame, report.order = _ordered(frame, order_column)
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


# ---------------------------------------------------------------------------
# The build: every parameter's gaps and condition, and the groups
# ---------------------------------------------------------------------------
def timeline(missing: np.ndarray) -> dict:
    """The gaps of one missing/present vector, in batch order, and the pattern they make."""
    missing = np.asarray(missing, dtype=bool)
    n, m = len(missing), int(missing.sum())
    k = n - m
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    lengths = ends - starts
    # Runs test, counting only the runs of missing batches.
    # shortcut: normal approximation, rough below ~10 missing batches; small
    # counts land on "Random gaps", which claims nothing.
    expected = m * (k + 1) / n if n else 0.0
    variance = m * (m - 1) * k * (k + 1) / (n * n * (n - 1)) if n > 1 else 0.0
    z = (len(starts) - expected) / np.sqrt(variance) if variance > 0 else 0.0
    leading = int(lengths[0]) if m and starts[0] == 0 else 0
    trailing = int(lengths[-1]) if m and ends[-1] == n else 0
    if m == 0:
        kind = "Complete"
    elif m == n:
        kind = "Never recorded"
    elif z > RUNS_Z:
        kind = "Regular skipping"
    elif z >= -RUNS_Z:
        kind = "Random gaps"
    elif leading >= DOMINANT * m:
        kind = "Introduced late"
    elif trailing >= DOMINANT * m:
        kind = "Retired"
    elif leading + trailing >= DOMINANT * m:
        kind = "Measured in a window"
    elif lengths.max() >= DOMINANT * m:
        kind = "Gap in the middle"
    else:
        kind = "Clumped"
    after = missing[1:][missing[:-1]]
    return {
        "Pattern": kind,
        "Missing %": round(100 * m / n, 1) if n else 0.0,
        "Gaps": len(starts),
        "Random would give": round(expected, 1),
        "Runs test z": round(float(z), 2),
        "Longest gap": int(lengths.max()) if m else 0,
        "Missing after missing %": round(100 * float(after.mean()), 1) if len(after) else None,
        f"Measured in last {RECENT} %": round(100 * float((~missing[-RECENT:]).mean()), 1)
        if n
        else None,
    }


@dataclass
class Condition:
    #: The tree's leaves where at least half the batches are missing, as one rule.
    rule: str
    score: float
    #: The column of the tree's first split.
    column: str
    #: Other columns that split the batches exactly as the first split does.
    aliases: list[str] = field(default_factory=list)


def _phis(blanks: np.ndarray, target: np.ndarray) -> np.ndarray:
    """phi between each column of ``blanks`` and ``target``; 0 for a constant column."""
    a = blanks - blanks.mean(axis=0)
    b = target - target.mean()
    scale = np.sqrt((a * a).sum(axis=0) * (b * b).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(scale > 0, (a * b[:, None]).sum(axis=0) / scale, 0.0)


class _Tree:
    """The categorical parameters one-hot, ready for a decision tree per pattern."""

    def __init__(self, frame: pd.DataFrame, candidates: list[str]):
        self.candidates = candidates
        columns, self.names, self.levels = [], [], {}
        for column in candidates:
            values = frame[column]
            text = _pooled(values[values.notna()]).reindex(values.index)
            self.levels[column] = sorted(text.dropna().unique())
            for category in self.levels[column]:
                columns.append((text == category).to_numpy())
                self.names.append((column, category))
        self.x = np.column_stack(columns) if columns else np.zeros((len(frame), 0), bool)
        self.source = np.array([candidates.index(column) for column, _ in self.names], dtype=int)
        # Features that split the batches identically (or as each other's complement)
        # share a twin id; the tree is offered the first, and the others are its aliases.
        _, self.twin = np.unique((self.x ^ self.x[:1]).T, axis=0, return_inverse=True)
        self.blanks = frame[candidates].isna().to_numpy(np.float64)

    def condition(self, missing: np.ndarray, exclude: set[str]) -> Condition | None:
        if missing.all() or not missing.any() or not self.names:
            return None
        tracks = np.abs(_phis(self.blanks, missing.astype(np.float64))) > SELF_CORRELATION
        usable = [
            not tracks[i] and self.candidates[i] not in exclude for i in range(len(self.candidates))
        ]
        offered = np.flatnonzero(np.array(usable)[self.source])
        if not len(offered):
            return None
        _, first = np.unique(self.twin[offered], return_index=True)
        keep = offered[np.sort(first)]
        x = self.x[:, keep]
        tree = DecisionTreeClassifier(
            max_depth=TREE_DEPTH, min_samples_leaf=MIN_PER_CATEGORY, random_state=0
        ).fit(x, missing)
        t = tree.tree_
        if t.node_count == 1:
            return None
        leaves = tree.apply(x)
        rules: list[str] = []

        def walk(node: int, path: list[str]) -> None:
            if t.children_left[node] == -1:
                batches = int((leaves == node).sum())
                share = float(missing[leaves == node].mean())
                if share >= 0.5:
                    rules.append(f"{' and '.join(path)} ({100 * share:.0f}% of {batches} missing)")
                return
            column, category = self.names[keep[t.feature[node]]]
            others = [level for level in self.levels[column] if level != category]
            # A two-valued column reads better as the other value (its blanks go there too).
            left = f"= {others[0]}" if len(others) == 1 else f"≠ {category}"
            walk(t.children_left[node], [*path, f"{column} {left}"])
            walk(t.children_right[node], [*path, f"{column} = {category}"])

        walk(0, [])
        root = keep[t.feature[0]]
        column = self.names[root][0]
        twins = offered[self.twin[offered] == self.twin[root]]
        aliases = sorted({self.names[i][0] for i in twins} - {column})
        return Condition(" or ".join(rules), _nmi(leaves, missing), column, aliases)


def for_build(
    frame: pd.DataFrame,
    value_types: dict[str, str],
    id_column: str,
    order_column: str,
) -> tuple[list[dict], MissingnessReport]:
    """Every batch table column's missingness, and the groups that go missing together.

    One row per column: its pattern and gap statistics over the batches in date
    order, the tree's condition, and its group. The groups use the task's
    default settings, with the tree as their explainer.
    """
    settings = Missingness()
    frame, order = _ordered(frame, order_column)
    report = analyse(frame, value_types, settings, id_column=id_column, order_column=order_column)
    undated = int(pd.to_datetime(frame.get(order_column), errors="coerce").isna().sum())
    if undated and order_column in frame.columns:
        report.order += f", the {undated:,} batches without one last"
    log.step(MODULE, f"Profiling the gaps of {frame.shape[1]:,} columns, {order}")

    tree = _Tree(frame, report.candidates)
    blank = frame.isna()
    dates = (
        pd.to_datetime(frame[order_column], errors="coerce").dt.strftime("%Y-%m-%d")
        if order_column in frame.columns
        else pd.Series("", index=frame.index)
    )
    group_of = {member: group.name for group in report.groups for member in group.members}
    report.batches = (
        frame[id_column].astype(str).tolist() if id_column in frame.columns else []
    ) or [f"row {i}" for i in frame.index]
    report.dates = dates.fillna("").tolist()
    rows = []
    for column in frame.columns:
        missing = blank[column].to_numpy()
        if missing.any():
            report.missing[str(column)] = "".join(missing.astype(int).astype(str))
        found = tree.condition(missing, {column})
        measured = np.flatnonzero(~missing)
        rows.append(
            {
                "Parameter": str(column),
                **timeline(missing),
                "Condition": found.rule if found and found.score >= CONDITIONAL else "",
                "Condition score": round(found.score, 2) if found else None,
                "Same split as": ", ".join(found.aliases) if found else "",
                "First measured": dates.iloc[measured[0]] if len(measured) else "",
                "Last measured": dates.iloc[measured[-1]] if len(measured) else "",
                "Group": group_of.get(column, ""),
            }
        )

    for group in report.groups:
        union = np.array(group.pattern, dtype=bool).any(axis=0)
        group.kind = timeline(union)["Pattern"]
        found = tree.condition(union, set(group.members))
        group.alternatives = []
        group.explainer = (
            _explain(frame[found.column], union, found.column, found.score) if found else None
        )
        group.strong = bool(found and found.score >= CONDITIONAL)
        group.condition = (
            found.rule + "".join(f"; {name} splits the same batches" for name in found.aliases)
            if group.strong
            else ""
        )
    report.settings["condition_score"] = CONDITIONAL
    log.success(
        MODULE,
        f"{sum(bool(row['Condition']) for row in rows):,} columns missing on a clear condition, "
        f"{len(report.groups)} groups missing together",
    )
    return rows, report


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
