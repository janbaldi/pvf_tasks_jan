"""Groups of parameters that move together, and what a task does about them.

Manufacturing parameters measure the same few underlying things several times
over: four cell counts along one day's processing move together, and a model
handed all four cannot say which of them mattered. This module finds those
groups from the data. What happens next is the task's decision, not this
module's:

``describe``     describe the groups and change nothing.
``keep_one``     keep one member of each group and drop the rest.
``residualise``  keep one member and replace the others by their residuals.
``summarise``    replace the group by one score: the mean of its z-scored members.
``off``          do not look.

Only a group that comes back in most subsamples of the batches is acted on
(``clustering.min_stability``); an unstable group is described and left alone.
When the cohort spans several sites, correlations are computed on site-centred
values, so a group means "moves together within a site", not "differs by site".

Two things this deliberately does not claim. Residualising does not preserve all
the information in a cluster, and it does not make the output independent:
regressions are fitted on the rows where every regressor is present, so columns
with different missingness patterns are only orthogonal on those rows, and
orthogonality in a least-squares sense is not zero rank correlation and not
independence.

A pair of parameters that share too few batches has an *unknown* relationship.
That is not the same as an uncorrelated one, and it never merges a cluster here.

Nothing here draws anything. It returns what it found, and :mod:`ml.plots`
turns that into figures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from .encode import Record, TaskParams, _record, near_constant, pairwise
from .logger import log

MODULE = "decorrelate"

RESIDUAL_SUFFIX = "_resid"
#: The actions that change what reaches the dataset.
CHANGING = ("keep_one", "residualise", "summarise")
#: Subsamples drawn to measure how often a group comes back, and their size.
STABILITY_RESAMPLES = 50
STABILITY_FRACTION = 0.8
#: The distance a pair with too little shared data is given. Above any threshold
#: cut, including a cut at |r| = 0, so unknown evidence can never merge a group.
UNKNOWN_DISTANCE = 2.0


@dataclass
class Fit:
    """One member against the regressors it was residualised on."""

    column: str
    model: str
    regressors: list[str] = field(default_factory=list)
    coefficients: list[float] = field(default_factory=list)
    intercept: float = float("nan")
    adj_r2: float = float("nan")
    rows: int = 0
    fallback: str = ""
    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    fitted: list[float] = field(default_factory=list)
    residual: list[float] = field(default_factory=list)

    @property
    def residualised(self) -> bool:
        return not self.fallback


@dataclass
class Cluster:
    """A group of parameters that move together, and what was done about it."""

    name: str
    members: list[str]
    representative: str
    action: str
    strategy: str = ""
    #: The catalog value every member shares (``clustering.name_by``), if any.
    label: str = ""
    #: Share of subsamples in which every member landed in one group again.
    stability: float = float("nan")
    #: For ``summarise``: the score column and how it is computed.
    summary: dict[str, Any] = field(default_factory=dict)
    why_representative: str = ""
    representative_rows: int = 0
    representative_score: float = float("nan")
    min_overlap: int = 0
    fits: list[Fit] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    records: list[Record] = field(default_factory=list)
    corr_before: float = float("nan")
    corr_after: float = float("nan")
    vif_before: float = float("nan")
    vif_after: float = float("nan")
    vif_rows_before: int = 0
    vif_rows_after: int = 0
    corr_matrix_before: list[list[float | None]] = field(default_factory=list)
    corr_matrix_after: list[list[float | None]] = field(default_factory=list)
    overlap_matrix: list[list[int]] = field(default_factory=list)
    labels_after: list[str] = field(default_factory=list)

    def state(self) -> dict[str, Any]:
        """The fitted part, as plain data the run writes next to the dataset."""
        return {
            "name": self.name,
            "members": list(self.members),
            "representative": self.representative,
            "action": self.action,
            "label": self.label,
            "stability": float(self.stability),
            "summary": self.summary,
            "dropped": list(self.dropped),
            "fits": [
                {
                    "column": fit.column,
                    "output": f"{fit.column}{RESIDUAL_SUFFIX}",
                    "regressors": list(fit.regressors),
                    "coefficients": [float(c) for c in fit.coefficients],
                    "intercept": float(fit.intercept),
                    "adj_r2": float(fit.adj_r2),
                    "rows": int(fit.rows),
                    "model": fit.model,
                    "fallback": fit.fallback,
                }
                for fit in self.fits
            ],
        }


@dataclass
class ClusterReport:
    """Every cluster, what was skipped, and on what evidence."""

    clusters: list[Cluster] = field(default_factory=list)
    candidates: int = 0
    usable: list[str] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    singletons: list[str] = field(default_factory=list)
    unsupported_pairs: int = 0
    method: str = ""
    threshold: float = 0.0
    min_overlap: int = 0
    action: str = "off"
    linkage: list[list[float]] = field(default_factory=list)
    linkage_labels: list[str] = field(default_factory=list)
    #: Correlations were computed on site-centred values.
    within_site: bool = False

    @property
    def members(self) -> list[str]:
        return [column for cluster in self.clusters for column in cluster.members]

    @property
    def changed(self) -> list[str]:
        """Members of the groups that were acted on, so are not passed through as they are."""
        return [c for k in self.clusters if k.action in CHANGING for c in k.members]

    @property
    def transforms(self) -> bool:
        return self.action in CHANGING

    def state(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "method": self.method,
            "threshold": self.threshold,
            "min_overlap": self.min_overlap,
            "clusters": [cluster.state() for cluster in self.clusters],
        }


# ---------------------------------------------------------------------------
# What is eligible at all
# ---------------------------------------------------------------------------
def eligible(
    df: pd.DataFrame, columns: list[str], params: TaskParams
) -> tuple[list[str], list[dict]]:
    """Numeric, non-constant parameters with enough values to correlate.

    Everything turned away is turned away for a stated reason: a parameter that
    quietly failed to be a number is the kind of thing that ends up in a dataset
    as a column of NaN.
    """
    usable: list[str] = []
    skipped: list[dict] = []
    for column in columns:
        if column not in df.columns:
            skipped.append({"Parameter": column, "Values": 0, "Reason": "not in the cohort"})
            continue
        values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        count = int(values.notna().sum())
        if count < params.cluster_min_overlap:
            skipped.append(
                {
                    "Parameter": column,
                    "Values": count,
                    "Reason": f"fewer than {params.cluster_min_overlap} usable values",
                }
            )
        elif near_constant(values, params.near_constant_tolerance):
            skipped.append({"Parameter": column, "Values": count, "Reason": "constant"})
        else:
            usable.append(column)
    if skipped:
        log.info(
            MODULE,
            f"{len(skipped)} parameters cannot take part in clustering",
            ", ".join(f"{s['Parameter']} ({s['Reason']})" for s in skipped[:10]),
        )
    return usable, skipped


def correlations(
    df: pd.DataFrame, columns: list[str], method: str, min_overlap: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Signed correlations and the batch counts behind them.

    A pair with fewer than ``min_overlap`` shared batches comes back as NaN —
    unknown — rather than as a coefficient computed from a handful of rows.
    """
    values = df[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    r = pd.DataFrame(np.nan, index=columns, columns=columns, dtype="float64")
    overlap = pd.DataFrame(0, index=columns, columns=columns, dtype="int64")
    for i, a in enumerate(columns):
        r.iat[i, i] = 1.0
        overlap.iat[i, i] = int(values[a].notna().sum())
        for j in range(i + 1, len(columns)):
            b = columns[j]
            coefficient, shared = pairwise(values[a], values[b], method=method)
            if shared < min_overlap:
                coefficient = float("nan")
            r.iat[i, j] = r.iat[j, i] = coefficient
            overlap.iat[i, j] = overlap.iat[j, i] = shared
    return r, overlap


def find_clusters(
    r: pd.DataFrame, params: TaskParams
) -> tuple[list[list[str]], list[list[float]], int]:
    """Group parameters by how strongly they move together.

    Correlation becomes a distance, complete linkage builds the tree, and the
    tree is cut where correlation falls below the threshold — so every member of
    a group is that correlated with every other, not merely with a chain of
    neighbours. A pair whose relationship is unknown is placed further apart than
    any cut, which is what keeps unsupported evidence from building a group.
    """
    columns = list(r.columns)
    if len(columns) < 2:
        return [], [], 0

    distance = 1.0 - r.abs().to_numpy(dtype="float64")
    unknown = ~np.isfinite(distance)
    unsupported = int(np.triu(unknown, k=1).sum())
    distance[unknown] = UNKNOWN_DISTANCE
    distance = np.clip((distance + distance.T) / 2.0, 0.0, None)
    np.fill_diagonal(distance, 0.0)

    tree = linkage(squareform(distance, checks=False), method="complete")
    labels = fcluster(tree, t=1.0 - params.cluster_corr_threshold, criterion="distance")

    grouped: dict[int, list[str]] = {}
    for column, label in zip(columns, labels):
        grouped.setdefault(int(label), []).append(column)
    clusters = sorted(
        (sorted(m) for m in grouped.values() if len(m) >= params.cluster_min_size),
        key=lambda members: (-len(members), members[0]),
    )
    return clusters, tree.tolist(), unsupported


# ---------------------------------------------------------------------------
# Which member represents the group
# ---------------------------------------------------------------------------
def choose_representative(
    df: pd.DataFrame, members: list[str], min_overlap: int, prefer: tuple[str, ...] = ()
) -> tuple[str, float, int, str]:
    """The member that stands for the group, chosen without looking at the target.

    Coverage first, then fit. A parameter measured on three batches can correlate
    perfectly with everything on those three and still be the worst column to
    keep, so candidates have to reach the group's median coverage before their
    mean pairwise R² is compared at all. Ties break on coverage, then on an
    explicit preference from the task, then on the name — so the choice does not
    move when the input columns are reordered.

    This is a description of the group. It is not evidence that the chosen
    parameter matters to the outcome.
    """
    scores: dict[str, float] = {}
    for candidate in members:
        fits = []
        for other in members:
            if other == candidate:
                continue
            r, shared = pairwise(df[candidate], df[other])
            if np.isfinite(r) and shared >= min_overlap:
                fits.append(r**2)
        scores[candidate] = float(np.mean(fits)) if fits else 0.0

    coverage = {column: int(df[column].notna().sum()) for column in members}
    floor = max(min_overlap, int(np.median(list(coverage.values()))))
    eligible_members = [m for m in members if coverage[m] >= floor] or list(members)
    best = min(
        eligible_members,
        key=lambda c: (-round(scores[c], 6), -coverage[c], 0 if c in prefer else 1, str(c)),
    )
    why = (
        f"mean pairwise R² {scores[best]:.3f} against the other members, over "
        f"{coverage[best]} batches with a value (candidates needed at least {floor})"
    )
    if best in prefer:
        why += "; named in the task's clustering.prefer list"
    return best, scores[best], coverage[best], why


# ---------------------------------------------------------------------------
# How much overlap is left
# ---------------------------------------------------------------------------
def max_correlation(block: pd.DataFrame, method: str, min_overlap: int) -> float:
    """The strongest correlation between any two columns, over supported pairs."""
    columns = list(block.columns)
    if len(columns) < 2:
        return float("nan")
    worst = float("nan")
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            r, shared = pairwise(block[a], block[b], method=method)
            if shared < min_overlap or not np.isfinite(r):
                continue
            worst = abs(r) if not np.isfinite(worst) else max(worst, abs(r))
    return worst


def max_vif(block: pd.DataFrame) -> tuple[float, int]:
    """The worst variance inflation in a block, and the rows it was computed on.

    NaN means undefined — too few complete rows, or nothing left to regress on —
    which is a different answer from infinite, and infinite means one column is
    an exact combination of the others on those rows.
    """
    clean = block.replace([np.inf, -np.inf], np.nan).apply(pd.to_numeric, errors="coerce").dropna()
    if clean.shape[1] < 2 or clean.shape[0] <= clean.shape[1] + 1:
        return float("nan"), int(clean.shape[0])
    clean = clean.loc[:, clean.std(axis=0) > 0]
    if clean.shape[1] < 2:
        return float("nan"), int(clean.shape[0])

    worst = 0.0
    for column in clean.columns:
        others = clean.drop(columns=[column]).to_numpy(dtype="float64")
        target = clean[column].to_numpy(dtype="float64")
        r2 = LinearRegression().fit(others, target).score(others, target)
        worst = max(worst, np.inf if r2 >= 1 - 1e-12 else 1.0 / (1.0 - r2))
    return float(worst), int(clean.shape[0])


def _matrix(r: pd.DataFrame) -> list[list[float | None]]:
    """A correlation matrix for the report, with unknown left as null."""
    return [
        [None if not np.isfinite(value) else round(float(value), 3) for value in row]
        for row in r.to_numpy(dtype="float64")
    ]


# ---------------------------------------------------------------------------
# Removing the overlap
# ---------------------------------------------------------------------------
def _adjusted_r2(y: np.ndarray, predicted: np.ndarray, parameters: int) -> float:
    n = len(y)
    r2 = r2_score(y, predicted)
    if n <= parameters + 1:
        return float(r2)
    return float(1 - (1 - r2) * (n - 1) / (n - parameters - 1))


def _sequential_linear(
    df: pd.DataFrame, members: list[str], representative: str, params: TaskParams
) -> tuple[pd.DataFrame, list[Fit]]:
    """Each member against every member kept before it: Gram-Schmidt, in effect.

    The regressors are recorded because they are what the residual means: the
    third member is what is left of it once the representative *and* the second
    member are accounted for, which is not the same statement as "beyond the
    anchor". Where there are too few rows for that many regressors, the parameter
    is kept as it is and the fallback is recorded — a sparse series that has only
    been centred is not a residual.
    """
    # Most complete first, then most strongly related. A sparse parameter added
    # early becomes a regressor for every later one, and then the complete
    # parameters have only its handful of rows to be fitted on.
    others = sorted(
        (c for c in members if c != representative),
        key=lambda c: (
            -int(df[c].notna().sum()),
            -abs(pairwise(df[representative], df[c], params.cluster_corr_method)[0] or 0.0),
            str(c),
        ),
    )
    frame = pd.DataFrame({representative: df[representative]}, index=df.index)
    fits: list[Fit] = []
    basis = [representative]

    for column in others:
        mask = df[basis + [column]].notna().all(axis=1)
        n, p = int(mask.sum()), len(basis)
        if n < max(params.cluster_min_overlap, p + 2):
            frame[column] = df[column]
            fits.append(
                Fit(
                    column=column,
                    model=f"kept as recorded ({n} rows with every regressor present)",
                    regressors=list(basis),
                    rows=n,
                    fallback=(
                        f"{n} batches have all of {', '.join(basis)} and this parameter, "
                        f"which is below the {max(params.cluster_min_overlap, p + 2)} needed "
                        "to fit that many regressors"
                    ),
                )
            )
            log.warn(
                MODULE,
                f"{column}: {n} complete batches against {p} regressors — kept as recorded",
            )
            basis.append(column)
            continue

        name = f"{column}{RESIDUAL_SUFFIX}"
        x = df.loc[mask, basis].to_numpy(dtype="float64")
        y = df.loc[mask, column].to_numpy(dtype="float64")
        model = LinearRegression().fit(x, y)
        predicted = model.predict(x)
        frame[name] = np.nan
        frame.loc[mask, name] = y - predicted
        adj = _adjusted_r2(y, predicted, p)
        fits.append(
            Fit(
                column=column,
                model=f"OLS on {p} retained parameter{'s' if p > 1 else ''}",
                regressors=list(basis),
                coefficients=[float(c) for c in np.ravel(model.coef_)],
                intercept=float(model.intercept_),
                adj_r2=adj,
                rows=n,
                x=df.loc[mask, representative].to_numpy(dtype="float64").tolist(),
                y=y.tolist(),
                fitted=predicted.tolist(),
                residual=(y - predicted).tolist(),
            )
        )
        log.info(MODULE, f"{column}: residualised on {p} parameters, adjusted R² {adj:.3f}")
        basis.append(column)

    return frame, fits


def stability(
    values: pd.DataFrame, groups: list[list[str]], params: TaskParams
) -> list[float]:
    """How often each group comes back when the clustering is redone on subsamples.

    A group found on 21 batches can hang on two or three of them. Redrawing 80%
    of the batches and clustering again shows whether it is a property of the
    process or of this particular handful.
    """
    rng = np.random.default_rng(params.seed)
    hits = [0] * len(groups)
    size = max(params.cluster_min_overlap, int(round(STABILITY_FRACTION * len(values))))
    for _ in range(STABILITY_RESAMPLES):
        rows = rng.choice(len(values), size=min(size, len(values)), replace=False)
        # shortcut: pandas' corr instead of encode.pairwise, for speed; same
        # coefficient, same minimum overlap.
        r = values.iloc[rows].corr(
            method=params.cluster_corr_method, min_periods=params.cluster_min_overlap
        )
        found = [set(g) for g in find_clusters(r, params)[0]]
        for i, group in enumerate(groups):
            hits[i] += any(set(group) <= other for other in found)
    return [h / STABILITY_RESAMPLES for h in hits]


def summary(numeric: pd.DataFrame, members: list[str], representative: str, method: str) -> dict:
    """How a group becomes one score: z-score each member, align its sign, average."""
    out: dict[str, Any] = {"means": {}, "scales": {}, "signs": {}}
    for column in members:
        values = numeric[column]
        r = pairwise(numeric[representative], values, method)[0]
        out["means"][column] = float(values.mean())
        out["scales"][column] = float(values.std()) or 1.0
        out["signs"][column] = -1.0 if np.isfinite(r) and r < 0 else 1.0
    return out


def score(df: pd.DataFrame, fitted: dict) -> pd.Series:
    """A group's score for any rows: the mean over the members each row has."""
    parts = [
        fitted["signs"][c] * (pd.to_numeric(df[c], errors="coerce") - fitted["means"][c])
        / fitted["scales"][c]
        for c in fitted["means"]
        if c in df.columns
    ]
    return pd.concat(parts, axis=1).mean(axis=1) if parts else pd.Series(np.nan, index=df.index)


def fit(
    df: pd.DataFrame,
    columns: list[str],
    params: TaskParams,
    action: str = "describe",
    prefer: tuple[str, ...] = (),
    sites: pd.Series | None = None,
    names: dict[str, str] | None = None,
    fixed: list[list[str]] | None = None,
    min_stability: float = 0.0,
) -> ClusterReport:
    """Find the groups, choose their representatives, and carry out the action.

    ``sites`` centres every parameter within its site before correlating, when
    there is more than one. ``names`` gives each parameter its catalog value for
    ``clustering.name_by``. ``fixed`` uses these groups instead of finding them
    (so every cross-validation fold explains the same groups). A group found on
    the data is only acted on when its stability reaches ``min_stability``.
    """
    report = ClusterReport(
        candidates=len(columns),
        method=params.cluster_corr_method,
        threshold=params.cluster_corr_threshold,
        min_overlap=params.cluster_min_overlap,
        action=action,
    )
    if action == "off":
        log.info(MODULE, "Clustering is off — no groups were looked for")
        return report

    log.step(MODULE, "Looking for groups of parameters that move together")
    usable, skipped = eligible(df, columns, params)
    report.usable, report.skipped = usable, skipped
    if len(usable) < 2:
        return report

    numeric = df[usable].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    centred = numeric
    if sites is not None and sites.loc[numeric.index].nunique() > 1:
        # Two sites at different levels would otherwise look like a correlation.
        centred = numeric - numeric.groupby(sites.loc[numeric.index]).transform("mean")
        report.within_site = True
        log.info(MODULE, "Correlations computed within site: each site's mean is removed first")
    r, overlap = correlations(
        centred, usable, params.cluster_corr_method, params.cluster_min_overlap
    )
    groups, tree, unsupported = find_clusters(r, params)
    if fixed is not None:
        groups = [m for m in ([c for c in g if c in usable] for g in fixed) if len(m) >= 2]
    stable = [1.0] * len(groups) if fixed is not None else stability(centred, groups, params)
    report.unsupported_pairs = unsupported
    report.linkage, report.linkage_labels = tree, list(usable)
    grouped = {column for members in groups for column in members}
    report.singletons = [column for column in usable if column not in grouped]

    if unsupported:
        log.info(
            MODULE,
            f"{unsupported} parameter pairs share fewer than {params.cluster_min_overlap} "
            "batches — their relationship is unknown and they were not merged",
        )
    if not groups:
        log.info(
            MODULE,
            f"No group of {params.cluster_min_size} or more parameters correlates above "
            f"{params.cluster_corr_threshold} on enough shared batches",
        )
        return report

    outputs: set[str] = set()
    for number, members in enumerate(groups, start=1):
        representative, fit_score, rows, why = choose_representative(
            numeric, members, params.cluster_min_overlap, prefer
        )
        labels = {(names or {}).get(c, "") for c in members}
        label = labels.pop() if len(labels) == 1 else ""
        acted = action if stable[number - 1] >= min_stability else "describe"
        cluster = Cluster(
            name=f"Cluster {number:02d}" + (f" ({label})" if label else ""),
            members=members,
            representative=representative,
            action=acted,
            label=label,
            stability=stable[number - 1],
            why_representative=why,
            representative_rows=rows,
            representative_score=fit_score,
            min_overlap=int(overlap.loc[members, members].to_numpy().min()),
        )

        if acted == "describe":
            cluster.strategy = (
                "reported only; every member stays as it is"
                if action == "describe"
                else f"left alone: found in {stable[number - 1]:.0%} of subsamples, "
                f"below clustering.min_stability ({min_stability:.0%})"
            )
            cluster.frame = pd.DataFrame(index=df.index)
        elif acted == "summarise":
            fitted = summary(numeric, members, representative, params.cluster_corr_method)
            output = f"{label or representative} group"
            if output in outputs:
                output = f"{label or representative} {number:02d} group"
            outputs.add(output)
            fitted["output"] = output
            cluster.summary = fitted
            cluster.strategy = "one score: the mean of the z-scored members, signs aligned"
            cluster.frame = pd.DataFrame({output: score(numeric, fitted)}, index=df.index)
            flipped = [c for c in members if fitted["signs"][c] < 0]
            cluster.records.append(
                _record(
                    output,
                    representative,
                    "cluster summary (summarise)",
                    f"mean of the z-scored {', '.join(members)}"
                    + (f"; sign flipped for {', '.join(flipped)}" if flipped else ""),
                )
            )
        elif acted == "keep_one":
            cluster.strategy = "one member kept, the rest dropped"
            cluster.dropped = [c for c in members if c != representative]
            cluster.frame = numeric[[representative]].copy()
            cluster.records.append(
                _record(
                    representative,
                    representative,
                    "cluster representative",
                    f"kept for its group of {len(members)}; {why}",
                )
            )
        else:
            cluster.strategy = (
                "sequential linear residuals (each member against those retained before it)"
            )
            cluster.frame, cluster.fits = _sequential_linear(
                numeric, members, representative, params
            )
            cluster.records.append(
                _record(
                    representative,
                    representative,
                    f"cluster representative ({action})",
                    f"kept in its own units; the rest of the group is measured against it. {why}",
                )
            )
            for fit_ in cluster.fits:
                if fit_.residualised:
                    cluster.records.append(
                        _record(
                            f"{fit_.column}{RESIDUAL_SUFFIX}",
                            fit_.column,
                            f"cluster residual ({action})",
                            f"what '{fit_.column}' still says once "
                            f"{', '.join(fit_.regressors)} are accounted for "
                            f"({fit_.model}, {fit_.rows} batches)",
                        )
                    )
                else:
                    cluster.records.append(
                        _record(
                            fit_.column,
                            fit_.column,
                            f"cluster member, not residualised ({action})",
                            fit_.fallback,
                        )
                    )

        before = numeric[members]
        cluster.corr_before = max_correlation(
            before, params.cluster_corr_method, params.cluster_min_overlap
        )
        cluster.vif_before, cluster.vif_rows_before = max_vif(before)
        cluster.corr_matrix_before = _matrix(r.loc[members, members])
        cluster.overlap_matrix = overlap.loc[members, members].to_numpy(dtype="int64").tolist()
        if not cluster.frame.empty and cluster.frame.shape[1] > 1:
            after_r, _ = correlations(
                cluster.frame,
                list(cluster.frame.columns),
                params.cluster_corr_method,
                params.cluster_min_overlap,
            )
            cluster.corr_after = max_correlation(
                cluster.frame, params.cluster_corr_method, params.cluster_min_overlap
            )
            cluster.vif_after, cluster.vif_rows_after = max_vif(cluster.frame)
            cluster.corr_matrix_after = _matrix(after_r)
        cluster.labels_after = list(cluster.frame.columns)

        log.info(
            MODULE,
            f"{cluster.name}: {len(members)} parameters, represented by "
            f"'{representative}' ({acted}, stability {cluster.stability:.0%})",
        )
        report.clusters.append(cluster)

    log.success(
        MODULE,
        f"{len(report.clusters)} clusters covering {len(report.members)} parameters ({action})",
    )
    return report


def transform(df: pd.DataFrame, state: dict[str, Any]) -> pd.DataFrame:
    """Apply fitted clusters to rows that were not part of the fit.

    Only the recorded coefficients are used: nothing is refitted, so a validation
    row is residualised by what the training rows said, which is the only version
    of this that means anything.
    """
    out = pd.DataFrame(index=df.index)
    for cluster in state.get("clusters", []):
        if cluster.get("action", state.get("action")) not in CHANGING:
            continue
        if cluster.get("summary"):
            out[cluster["summary"]["output"]] = score(df, cluster["summary"])
            continue
        representative = cluster["representative"]
        if representative in df.columns:
            out[representative] = pd.to_numeric(df[representative], errors="coerce")
        for fit_ in cluster["fits"]:
            column = fit_["column"]
            if column not in df.columns:
                continue
            if fit_["fallback"] or not fit_["coefficients"]:
                out[column] = pd.to_numeric(df[column], errors="coerce")
                continue
            regressors = fit_["regressors"]
            if any(name not in df.columns for name in regressors):
                continue
            x = df[regressors].apply(pd.to_numeric, errors="coerce")
            y = pd.to_numeric(df[column], errors="coerce")
            predicted = x.mul(fit_["coefficients"], axis=1).sum(axis=1) + fit_["intercept"]
            residual = y - predicted
            out[fit_["output"]] = residual.where(x.notna().all(axis=1) & y.notna())
    return out


def rows(report: ClusterReport, retained: set[str]) -> list[dict]:
    """Cluster membership, singletons and skips — the whole population, not part.

    ``retained`` is the recipe's output columns, so each row says whether its
    parameter reached the dataset.
    """
    rows: list[dict] = []
    for cluster in report.clusters:
        for member in cluster.members:
            fit = next((f for f in cluster.fits if f.column == member), None)
            representative = member == cluster.representative
            rows.append(
                {
                    "Cluster": cluster.name,
                    "Parameter": member,
                    "Representative": cluster.representative,
                    "Is representative": representative,
                    "Action": cluster.action,
                    "Stability": round(cluster.stability, 2),
                    "Outcome": (
                        "summarised into the group score"
                        if cluster.summary
                        else "kept as the representative"
                        if representative and cluster.action in CHANGING
                        else "dropped"
                        if member in cluster.dropped
                        else fit.model
                        if fit
                        else "unchanged"
                    ),
                    "Output column": (
                        cluster.summary["output"]
                        if cluster.summary
                        else member
                        if representative or not fit or fit.fallback
                        else f"{member}{RESIDUAL_SUFFIX}"
                    ),
                    "Retained in dataset": (
                        member in retained
                        or f"{member}{RESIDUAL_SUFFIX}" in retained
                        or cluster.summary.get("output") in retained
                    ),
                    "Rows fitted": fit.rows if fit else cluster.representative_rows,
                    "Adjusted R2": fit.adj_r2 if fit else "",
                    "Regressors": ", ".join(fit.regressors) if fit else "",
                    "Smallest shared batches in cluster": cluster.min_overlap,
                }
            )
    for column in report.singletons:
        rows.append(
            {
                "Cluster": "",
                "Parameter": column,
                "Representative": column,
                "Is representative": True,
                "Action": "none",
                "Outcome": "correlates with nothing above the threshold",
                "Output column": column,
                "Retained in dataset": column in retained,
                "Rows fitted": "",
                "Adjusted R2": "",
                "Regressors": "",
                "Smallest shared batches in cluster": "",
            }
        )
    for skipped in report.skipped:
        rows.append(
            {
                "Cluster": "",
                "Parameter": skipped["Parameter"],
                "Representative": "",
                "Is representative": False,
                "Action": "skipped",
                "Outcome": skipped["Reason"],
                "Output column": "",
                "Retained in dataset": skipped["Parameter"] in retained,
                "Rows fitted": skipped["Values"],
                "Adjusted R2": "",
                "Regressors": "",
                "Smallest shared batches in cluster": "",
            }
        )
    return rows
