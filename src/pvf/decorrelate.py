"""Groups of parameters that move together, and what a task does about them.

Manufacturing parameters measure the same few underlying things several times
over: four cell counts along one day's processing move together, and a model
handed all four cannot say which of them mattered. This module finds those
groups from the data. What happens next is the task's decision, not this
module's:

``report_only``     describe the groups and change nothing.
``representative``  keep one member of each group and drop the rest.
``linear``          keep one member and replace the others by their residuals.
``nonlinear``       the same, by whichever curve fits the pair — experimental.
``off``             do not look.

Two things this deliberately does not claim. Residualising does not preserve all
the information in a cluster, and it does not make the output independent:
regressions are fitted on the rows where every regressor is present, so columns
with different missingness patterns are only orthogonal on those rows, and
orthogonality in a least-squares sense is not zero rank correlation and not
independence.

A pair of parameters that share too few batches has an *unknown* relationship.
That is not the same as an uncorrelated one, and it never merges a cluster here.

Nothing here draws anything. It returns what it found, and :mod:`pvf.plots`
turns that into figures.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.spatial.distance import squareform
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from .encode import Record, TaskParams, _record, near_constant, pairwise
from .logger import log

MODULE = "decorrelate"

RESIDUAL_SUFFIX = "_resid"
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
    #: A curve is fitted on scaled inputs, so scoring new rows needs the scales.
    x_scale: float = 1.0
    y_scale: float = 1.0
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

    @property
    def anchor(self) -> str:  # the name the reports used before there were actions
        return self.representative

    def state(self) -> dict[str, Any]:
        """The fitted part, as plain data the run writes next to the dataset."""
        return {
            "name": self.name,
            "members": list(self.members),
            "representative": self.representative,
            "action": self.action,
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
                    "x_scale": float(fit.x_scale),
                    "y_scale": float(fit.y_scale),
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

    @property
    def members(self) -> list[str]:
        return [column for cluster in self.clusters for column in cluster.members]

    @property
    def transforms(self) -> bool:
        return self.action in ("representative", "linear", "nonlinear")

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


def _curve(name: str):
    """The shapes the experimental non-linear action may try, and their domains."""
    return {
        "Exponential": (lambda t, a, b, c: a * np.exp(b * t) + c, 3, lambda t: True),
        "Inverse": (lambda t, a, c: a / (t + 1e-8) + c, 2, lambda t: np.all(np.abs(t) > 1e-6)),
        "Logarithmic": (
            lambda t, a, c: a * np.log(np.abs(t) + 1e-8) + c,
            2,
            lambda t: np.all(np.abs(t) > 1e-6),
        ),
    }[name]


def _anchor_fit(
    df: pd.DataFrame, members: list[str], representative: str, params: TaskParams
) -> tuple[pd.DataFrame, list[Fit]]:
    """Every member against the representative alone, by whichever shape fits it.

    Kept for compatibility with tasks that were set up this way. In-sample
    adjusted R² is descriptive: a curve that fits these batches better has not
    been validated on any others, which is why a curve has to beat the straight
    line by a margin before it is used at all.
    """
    frame = pd.DataFrame({representative: df[representative]}, index=df.index)
    fits: list[Fit] = []

    for column in (c for c in members if c != representative):
        mask = df[representative].notna() & df[column].notna()
        n = int(mask.sum())
        if n < max(params.cluster_min_overlap, 6):
            frame[column] = df[column]
            fits.append(
                Fit(
                    column=column,
                    model=f"kept as recorded ({n} shared batches)",
                    regressors=[representative],
                    rows=n,
                    fallback=f"{n} shared batches is too few to fit a curve to",
                )
            )
            continue

        x = df.loc[mask, [representative]]
        y = df.loc[mask, column]
        values_x = x.to_numpy(dtype="float64").ravel()
        values_y = y.to_numpy(dtype="float64")

        linear = LinearRegression().fit(x, y)
        predicted = linear.predict(x)
        best = (
            "Linear",
            _adjusted_r2(values_y, predicted, 1),
            predicted,
            [float(c) for c in np.ravel(linear.coef_)],
            float(linear.intercept_),
        )

        x_scale = np.max(np.abs(values_x)) or 1.0
        y_scale = np.max(np.abs(values_y)) or 1.0
        scaled_x, scaled_y = values_x / x_scale, values_y / y_scale
        direction = 1.0 if np.corrcoef(values_x, values_y)[0, 1] > 0 else -1.0
        starts = {
            "Exponential": [1.0, direction, float(np.min(scaled_y))],
            "Inverse": [1.0, float(np.min(scaled_y))],
            "Logarithmic": [1.0, float(np.mean(scaled_y))],
        }
        for shape, start in starts.items():
            function, count, domain = _curve(shape)
            inputs = values_x if shape == "Inverse" else scaled_x
            if not domain(inputs):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                try:
                    coefficients, _ = curve_fit(function, inputs, scaled_y, p0=start, maxfev=10000)
                except (RuntimeError, ValueError):
                    continue
            curve_predicted = function(inputs, *coefficients) * y_scale
            if not np.all(np.isfinite(curve_predicted)):
                continue
            adj = _adjusted_r2(values_y, curve_predicted, count)
            if adj > best[1] + 0.02:
                best = (shape, adj, curve_predicted, [float(c) for c in coefficients], float("nan"))

        shape, adj, predicted, coefficients, intercept = best
        name = f"{column}{RESIDUAL_SUFFIX}"
        frame[name] = np.nan
        frame.loc[mask, name] = values_y - np.asarray(predicted, dtype="float64")
        fits.append(
            Fit(
                column=column,
                model=shape,
                regressors=[representative],
                coefficients=coefficients,
                intercept=intercept,
                adj_r2=adj,
                rows=n,
                x_scale=1.0 if shape == "Linear" else x_scale,
                y_scale=1.0 if shape == "Linear" else y_scale,
                x=values_x.tolist(),
                y=values_y.tolist(),
                fitted=np.asarray(predicted, dtype="float64").tolist(),
                residual=(values_y - np.asarray(predicted, dtype="float64")).tolist(),
            )
        )
        log.info(MODULE, f"{column}: residualised against {representative} by {shape.lower()} fit")

    return frame, fits


# ---------------------------------------------------------------------------
# Fitting and applying
# ---------------------------------------------------------------------------
def fit(
    df: pd.DataFrame,
    columns: list[str],
    params: TaskParams,
    action: str = "report_only",
    prefer: tuple[str, ...] = (),
) -> ClusterReport:
    """Find the groups, choose their representatives, and carry out the action."""
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
    r, overlap = correlations(
        numeric, usable, params.cluster_corr_method, params.cluster_min_overlap
    )
    groups, tree, unsupported = find_clusters(r, params)
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

    for number, members in enumerate(groups, start=1):
        representative, score, rows, why = choose_representative(
            numeric, members, params.cluster_min_overlap, prefer
        )
        cluster = Cluster(
            name=f"Cluster {number:02d}",
            members=members,
            representative=representative,
            action=action,
            why_representative=why,
            representative_rows=rows,
            representative_score=score,
            min_overlap=int(overlap.loc[members, members].to_numpy().min()),
        )

        if action == "report_only":
            cluster.strategy = "reported only; every member stays as it is"
            cluster.frame = pd.DataFrame(index=df.index)
        elif action == "representative":
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
            builder = _sequential_linear if action == "linear" else _anchor_fit
            cluster.strategy = (
                "sequential linear residuals (each member against those retained before it)"
                if action == "linear"
                else "anchor fit, straight or curved — experimental"
            )
            cluster.frame, cluster.fits = builder(numeric, members, representative, params)
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
            f"'{representative}' ({action})",
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
    action = state.get("action", "off")
    out = pd.DataFrame(index=df.index)
    if action in ("off", "report_only"):
        return out

    for cluster in state.get("clusters", []):
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
            model = fit_.get("model", "")
            if model in ("Exponential", "Inverse", "Logarithmic"):
                function, _, domain = _curve(model)
                scaled = x.iloc[:, 0] / (fit_.get("x_scale") or 1.0)
                inputs = x.iloc[:, 0] if model == "Inverse" else scaled
                predicted = pd.Series(
                    function(inputs.to_numpy(dtype="float64"), *fit_["coefficients"]),
                    index=df.index,
                ) * (fit_.get("y_scale") or 1.0)
                predicted = predicted.where(np.isfinite(predicted))
            else:
                predicted = x.mul(fit_["coefficients"], axis=1).sum(axis=1) + fit_["intercept"]
            residual = y - predicted
            out[fit_["output"]] = residual.where(x.notna().all(axis=1) & y.notna())
    return out
