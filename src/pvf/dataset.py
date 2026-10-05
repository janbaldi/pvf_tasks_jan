"""From the PVF to the table one task asks for.

The PVF is the record of what happened to every batch. A task is narrower: one
cohort, one target, one declared set of roles, and every parameter that is
allowed to be a predictor turned into numbers.

Three things this module is careful about, because each of them is a way to get
a number that looks better than the process deserves:

*Roles come from the task, not from the data.* The target, the identifier and
the grouping column are removed from the predictor candidates before anything is
encoded, clustered or de-duplicated. A column is not treated as an identifier
because it happens to have many categories — it is one because the task says so.

*What the target produced is not a predictor.* The feature registry records what
each derived feature was computed from, so the descendants of the target can be
walked and excluded, with the chain that got them there.

*What is learned is learned on training rows.* Category vocabularies, target
means, duplicate selection, clusters and residual regressions are fitted on one
population and applied unchanged to any other. Where a task asks for no split,
that population is the whole cohort and the transformed table is labelled
exploratory, because a table preprocessed over every row is not a table you can
honestly cross-validate on afterwards.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import decorrelate, encode, features
from .encode import Record, TaskParams, sanitise
from .logger import log
from .taskconfig import STATUS_ALIASES, TaskSpec

MODULE = "dataset"

#: Roles a column can have in the exported table.
ID, GROUP, TARGET, PREDICTOR = "identifier", "group", "target", "predictor"


class TaskRefused(ValueError):
    """The task cannot produce an honest dataset, and says why."""


# ---------------------------------------------------------------------------
# Records the package writes out
# ---------------------------------------------------------------------------
@dataclass
class Decision:
    """One parameter, what happened to it, and why."""

    parameter: str
    decision: str
    reason: str
    stage: str

    def row(self) -> dict[str, str]:
        return {
            "Parameter": self.parameter,
            "Decision": self.decision,
            "Reason": self.reason,
            "Stage": self.stage,
        }


@dataclass
class CohortResult:
    raw_rows: int = 0
    rows: int = 0
    funnel: list[dict] = field(default_factory=list)
    exclusions: list[dict] = field(default_factory=list)
    #: Row index → (reason, stage), for the rows that did not make it.
    excluded_by: dict = field(default_factory=dict)
    rejected: dict[str, int] = field(default_factory=dict)
    examples: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Everything one run decided, for the report and the package to read."""

    spec: TaskSpec
    target_column: str = ""
    target_classes: dict[str, str] | None = None
    target_kind: str = "numeric"
    cohort: CohortResult = field(default_factory=CohortResult)

    candidates: list[str] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    missing_from_pvf: list[str] = field(default_factory=list)
    duplicate_ptf_parameters: list[str] = field(default_factory=list)

    routes: list[encode.Route] = field(default_factory=list)
    one_hot_tail: list[dict] = field(default_factory=list)
    target_encoding: dict[str, Any] = field(default_factory=dict)
    hashing_summary: list[dict] = field(default_factory=list)
    ordinal_unexpected: list[dict] = field(default_factory=list)
    numeric_notes: list[dict] = field(default_factory=list)
    nonpositive_durations: dict[str, int] = field(default_factory=dict)
    duplicate_decisions: list[dict] = field(default_factory=list)
    dropped_sparse: list[dict] = field(default_factory=list)
    clusters: decorrelate.ClusterReport = field(default_factory=decorrelate.ClusterReport)

    dictionary: list[Record] = field(default_factory=list)
    columns: list[dict] = field(default_factory=list)
    split: dict[str, Any] = field(default_factory=dict)
    recipe: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    rows: int = 0
    feature_columns: int = 0
    #: How many columns the recipe produces, which is not the raw predictor count.
    transformed_columns: int = 0
    selected_raw: list[str] = field(default_factory=list)
    dataset_role: str = "exploratory"
    fitted_on: str = ""

    def meaning(self, column: str) -> Record | None:
        return next((r for r in self.dictionary if r["output"] == column), None)

    def note(self, message: str) -> None:
        self.warnings.append(message)
        log.warn(MODULE, message)


# ---------------------------------------------------------------------------
# Cohort
# ---------------------------------------------------------------------------
def normalise_status(series: pd.Series) -> pd.Series:
    """Status text with its spelling flattened: case, spacing and known aliases.

    The original values stay in the frame — this is only what a filter compares
    against — so the audit trail still shows what the source actually said.
    """
    text = series.astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    return text.replace(STATUS_ALIASES)


def _matches(values: pd.Series, rule) -> pd.Series:
    """The rows one filter keeps, missing values aside."""
    if rule.op == "equals":
        return values == _comparable(rule.value, rule)
    if rule.op == "not_equals":
        return values != _comparable(rule.value, rule)
    if rule.op == "in":
        return values.isin([_comparable(v, rule) for v in rule.value])
    if rule.op == "not_in":
        return ~values.isin([_comparable(v, rule) for v in rule.value])
    if rule.op in ("min", "max"):
        numbers = pd.to_numeric(values, errors="coerce")
        bound = pd.to_numeric(pd.Series([rule.value]), errors="coerce").iloc[0]
        if pd.isna(bound):  # a date bound, then
            numbers = pd.to_datetime(values, errors="coerce")
            bound = pd.to_datetime(rule.value, errors="coerce")
        return numbers >= bound if rule.op == "min" else numbers <= bound
    if rule.op == "is_null":
        return values.isna()
    if rule.op == "not_null":
        return values.notna()
    raise TaskRefused(f"Unknown filter operator '{rule.op}'")


def _comparable(value: Any, rule) -> Any:
    """A YAML value in the same shape as the values it is compared against.

    Only a filter that asked for normalisation gets its own values flattened, so
    a rule written ``Withdrawal`` matches a batch recorded as ``withdrawn``
    without every other rule quietly becoming case-insensitive too.
    """
    if not rule.normalise:
        return value
    text = str(value).strip().lower()
    return STATUS_ALIASES.get(text, text)


def select_cohort(pvf: pd.DataFrame, spec: TaskSpec, result: TaskResult) -> pd.DataFrame:
    """The batches this task is about, one declared filter at a time.

    Every step is recorded with what it removed, and every excluded batch keeps
    the first rule that excluded it — "the dataset has 26 rows" means nothing
    without where the others went.
    """
    log.step(MODULE, f"Selecting the cohort for '{spec.name}'")
    cohort = pvf
    result.cohort.raw_rows = len(pvf)
    excluded_by: dict[Any, tuple[str, str]] = {}

    def step(label: str, keep: pd.Series, stage: str) -> pd.DataFrame:
        nonlocal cohort
        before = len(cohort)
        dropped = cohort.index[~keep.reindex(cohort.index).fillna(False)]
        for index in dropped:
            excluded_by.setdefault(index, (label, stage))
        cohort = cohort.loc[keep.reindex(cohort.index).fillna(False)]
        result.cohort.funnel.append(
            {
                "Step": stage,
                "Filter": label,
                "Batches before": before,
                "Removed": before - len(cohort),
                "Batches left": len(cohort),
            }
        )
        log.info(MODULE, f"{label}: {len(cohort)} of {before} batches left")
        return cohort

    if spec.cohort.sites:
        if spec.cohort.site_column not in pvf.columns:
            raise TaskRefused(
                f"cohort.site_column: '{spec.cohort.site_column}' is not a column of the PVF, "
                "so the cohort cannot be narrowed to a site"
            )
        step(
            f"manufactured at {', '.join(spec.cohort.sites)}",
            pvf[spec.cohort.site_column].isin(list(spec.cohort.sites)),
            "site",
        )

    for number, rule in enumerate(spec.cohort.filters, start=1):
        if rule.column not in pvf.columns:
            raise TaskRefused(
                f"cohort.filters[{number - 1}]: '{rule.column}' is not a column of the PVF. "
                "A filter that cannot run would quietly widen the cohort, so the task stops "
                "here — remove the filter or fix the column name"
            )
        values = pvf.loc[cohort.index, rule.column]
        compared = normalise_status(values) if rule.normalise else values
        keep = _matches(compared, rule)
        missing = (
            compared.isna()
            if rule.op not in ("is_null", "not_null")
            else pd.Series(False, index=compared.index)
        )
        keep = keep.where(~missing, rule.missing == "include")
        step(rule.description, keep.astype(bool), f"filter {number}")

    if spec.cohort.date_column:
        if spec.cohort.date_column not in pvf.columns:
            raise TaskRefused(
                f"cohort.date_column: '{spec.cohort.date_column}' is not a column of the PVF"
            )
        stamps = pd.to_datetime(pvf.loc[cohort.index, spec.cohort.date_column], errors="coerce")
        for bound, label in ((spec.cohort.date_from, "from"), (spec.cohort.date_to, "to")):
            if not bound:
                continue
            edge = pd.to_datetime(bound, errors="coerce")
            if pd.isna(edge):
                raise TaskRefused(f"cohort.date_{label}: '{bound}' is not a date")
            keep = stamps >= edge if label == "from" else stamps <= edge
            step(f"{spec.cohort.date_column} {label} {bound}", keep.fillna(False), f"date {label}")

    result.cohort.rows = len(cohort)
    result.cohort.excluded_by = excluded_by
    log.success(MODULE, f"Cohort: {len(cohort)} of {result.cohort.raw_rows} batches")
    if cohort.empty:
        raise TaskRefused(
            "No batches are left after the cohort filters. The funnel above says which "
            "filter emptied it"
        )
    return cohort.copy()


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
def record_exclusions(
    pvf: pd.DataFrame, spec: TaskSpec, result: TaskResult, kept: pd.Index
) -> None:
    """One row per batch in the PVF: in or out, and the first rule that decided.

    Written once the target step has run too, so "included" here means included
    in the dataset rather than surviving the filters and then dropping out of it.
    """
    identifier = spec.roles.id if spec.roles.id in pvf.columns else ""
    inside = set(kept)
    result.cohort.exclusions = [
        {
            "Batch": str(pvf.at[index, identifier]) if identifier else f"row {index}",
            "Included": index in inside,
            "Reason": ""
            if index in inside
            else result.cohort.excluded_by.get(index, ("unknown", ""))[0],
            "Stage": ""
            if index in inside
            else result.cohort.excluded_by.get(index, ("", "unknown"))[1],
        }
        for index in pvf.index
    ]


def prepare_target(
    cohort: pd.DataFrame, spec: TaskSpec, result: TaskResult
) -> tuple[pd.Series, pd.Index]:
    """The column to predict, as numbers, with the rows it cannot be read for.

    A binary target needs its positive class declared: choosing the rarer class
    would make the same task mean different things on different cohorts, and the
    class anyone cares about is not always the rare one.
    """
    column = spec.target.column
    if column not in cohort.columns:
        raise TaskRefused(f"target.column: '{column}' is not a column of the PVF")

    values = cohort[column]
    name = sanitise(column)

    if spec.target.type == "numeric":
        numbers = pd.to_numeric(values, errors="coerce")
        unreadable = int((values.notna() & numbers.isna()).sum())
        infinite = int(np.isinf(numbers.to_numpy(dtype="float64", na_value=np.nan)).sum())
        numbers = numbers.replace([np.inf, -np.inf], np.nan)
        if unreadable:
            result.cohort.rejected["target not a number"] = unreadable
        if infinite:
            result.cohort.rejected["target not finite"] = infinite
        result.target_kind = "numeric"
        target = numbers.rename(name)
    else:
        text = values.astype("string").str.strip()
        observed = set(text.dropna().unique())
        positive = spec.target.positive_class
        negative = spec.target.negative_class
        if not negative:
            others = sorted(observed - {positive})
            if len(others) != 1:
                raise TaskRefused(
                    f"target: '{column}' holds {sorted(observed)[:6]}. A binary target needs "
                    "exactly two classes; name the other one in target.negative_class, or "
                    "filter the cohort down to two"
                )
            negative = others[0]
        unexpected = sorted(observed - {positive, negative})
        if unexpected:
            raise TaskRefused(
                f"target: '{column}' also holds {unexpected[:6]}, which is neither "
                f"'{positive}' nor '{negative}'. Multiclass targets are out of scope"
            )
        if positive not in observed:
            raise TaskRefused(
                f"target.positive_class: no batch in this cohort has '{positive}' in "
                f"'{column}' (observed: {sorted(observed)[:6]})"
            )
        name = f"{name}_is_{sanitise(positive)}"
        target = text.map({negative: 0, positive: 1}).astype("Int64").rename(name)
        result.target_kind = "binary"
        result.target_classes = {"0": negative, "1": positive}
        log.info(
            MODULE,
            f"Target '{column}': '{positive}' → 1 ({int((text == positive).sum())} batches), "
            f"'{negative}' → 0 ({int((text == negative).sum())})",
        )

    result.target_column = name
    missing = target.isna()
    if missing.any():
        identifier = spec.roles.id if spec.roles.id in cohort.columns else ""
        result.cohort.rejected["target missing"] = int(missing.sum()) - sum(
            result.cohort.rejected.get(key, 0)
            for key in ("target not a number", "target not finite")
        )
        result.cohort.examples["target missing"] = (
            cohort.loc[missing, identifier].astype(str).tolist()[:50] if identifier else []
        )
        result.cohort.funnel.append(
            {
                "Step": "target",
                "Filter": f"{spec.target.column} was measured",
                "Batches before": len(cohort),
                "Removed": int(missing.sum()),
                "Batches left": int((~missing).sum()),
            }
        )
        log.warn(
            MODULE, f"{int(missing.sum())} batches have no usable target and are not in the dataset"
        )

    for index in cohort.index[missing]:
        result.cohort.excluded_by.setdefault(
            index, (f"{spec.target.column} was never measured", "target")
        )

    labelled = cohort.index[~missing]
    if not len(labelled):
        raise TaskRefused(
            f"No batch in the cohort has a usable value for '{column}'. There is nothing "
            "to predict and no dataset was written"
        )
    return target, labelled


# ---------------------------------------------------------------------------
# Which parameters may be predictors
# ---------------------------------------------------------------------------
def _requirements(feature: features.Feature) -> list[tuple[str, ...]]:
    return [(r,) if isinstance(r, str) else tuple(r) for r in feature.requires]


def target_descendants(target: str, params: features.Params) -> dict[str, str]:
    """Every derived column the target went into, and the chain that got it there.

    Conservative on alternatives: a feature that could have been computed from
    the target *or* from something else is treated as descended from it, because
    which input a given site actually used is not recorded in the PVF.
    """
    tainted: dict[str, str] = {target: target}
    lineage = {f.name: _requirements(f) for f in features.registry(params)}
    lineage.update(
        {
            name: [(source,) for source in sources]
            for name, sources in features.DERIVED_SOURCES.items()
        }
    )

    changed = True
    while changed:
        changed = False
        for name, requires in lineage.items():
            if name in tainted:
                continue
            for requirement in requires:
                hit = next((r for r in requirement if r in tainted), None)
                if hit is None:
                    continue
                chain = f"{tainted[hit]} → {name}"
                if len(requirement) > 1:
                    chain += f" (one of {', '.join(requirement)})"
                tainted[name] = chain
                changed = True
                break
    tainted.pop(target, None)
    return tainted


def candidate_predictors(
    cohort: pd.DataFrame, ptf: pd.DataFrame, spec: TaskSpec, result: TaskResult
) -> list[str]:
    """The one predictor set every later step uses.

    Built once, here, so the encoders, the duplicate sweep and the clustering
    cannot each work from a slightly different idea of what a predictor is.
    """
    log.step(MODULE, "Choosing the predictors")
    declared = [str(p) for p in ptf["Parameter"]]
    seen: dict[str, int] = {}
    for parameter in declared:
        seen[parameter] = seen.get(parameter, 0) + 1
    result.duplicate_ptf_parameters = sorted(name for name, count in seen.items() if count > 1)
    if result.duplicate_ptf_parameters:
        result.note(
            f"The PTF lists {len(result.duplicate_ptf_parameters)} parameters more than once; "
            "each is used once here: " + ", ".join(result.duplicate_ptf_parameters[:10])
        )

    parameters = list(dict.fromkeys(declared))
    result.missing_from_pvf = [p for p in parameters if p not in cohort.columns]
    available = [p for p in parameters if p in cohort.columns]

    def refuse(parameter: str, reason: str, stage: str) -> None:
        result.decisions.append(Decision(parameter, "excluded", reason, stage))

    roles = {
        spec.target.column: "the target",
        spec.roles.id: "the identifier",
        spec.roles.group: "the grouping column",
    }
    roles.pop("", None)
    descendants = target_descendants(spec.target.column, features.Params())
    unavailable = dict(spec.availability.unavailable)
    overrides = set(spec.availability.available)

    kept: list[str] = []
    for parameter in available:
        if parameter in roles:
            refuse(parameter, f"{roles[parameter]}, kept as metadata rather than encoded", "role")
        elif parameter == result.target_column:
            refuse(parameter, "collides with the target's output name", "role")
        elif parameter in dict(spec.roles.exclude):
            refuse(parameter, dict(spec.roles.exclude)[parameter], "task exclusion")
        elif parameter in descendants and parameter not in overrides:
            refuse(
                parameter, f"computed from the target: {descendants[parameter]}", "target lineage"
            )
        elif parameter in unavailable and parameter not in overrides:
            refuse(
                parameter,
                f"not available at the prediction point ({spec.availability.cutoff or 'declared'})"
                f": {unavailable[parameter]}",
                "availability",
            )
        elif spec.predictive and parameter in features.SITE_RELATIVE and parameter not in overrides:
            refuse(
                parameter,
                "standardised within its own site's batches, so its value depends on which "
                "batches were processed with it — exploratory only",
                "site-relative",
            )
        elif spec.roles.include and parameter not in spec.roles.include:
            refuse(parameter, "not in the task's columns.include list", "include list")
        else:
            kept.append(parameter)

    if spec.roles.include:
        for parameter in spec.roles.include:
            if parameter not in cohort.columns:
                raise TaskRefused(
                    f"columns.include: '{parameter}' is not in the PVF, so the predictors "
                    "this task asked for cannot all be supplied"
                )
            if parameter not in kept:
                raise TaskRefused(
                    f"columns.include: '{parameter}' was asked for as a predictor but is "
                    "excluded for another reason — see decisions.csv"
                )

    if spec.predictive:
        late = [p for p in kept if p in unavailable]
        if late:
            result.note(f"{len(late)} declared-unavailable parameters survived selection")
    if not kept:
        raise TaskRefused(
            "No parameter is left to predict with once roles, target lineage and the task's "
            "own exclusions are applied"
        )

    result.candidates = kept
    log.success(
        MODULE,
        f"{len(kept)} predictors from {len(available)} PVF parameters "
        f"({len(available) - len(kept)} excluded, {len(result.missing_from_pvf)} not in the PVF)",
    )
    return kept


# ---------------------------------------------------------------------------
# The preprocessing recipe
# ---------------------------------------------------------------------------
@dataclass
class Preprocessor:
    """What was learned from the training rows, and how to apply it elsewhere.

    Plain data throughout: dictionaries of category vocabularies, means, widths,
    retained parameters and regression coefficients. It is written next to the
    dataset as ``recipe.json``, so applying it again needs this module, not this
    process.
    """

    params: TaskParams
    fitted_on: str = ""
    rows_fitted: int = 0
    routes: list[encode.Route] = field(default_factory=list)
    binary: dict = field(default_factory=dict)
    one_hot: dict = field(default_factory=dict)
    target_encoding: dict = field(default_factory=dict)
    hashing: dict = field(default_factory=dict)
    ordinal: dict = field(default_factory=dict)
    durations: list[str] = field(default_factory=list)
    numeric: list[str] = field(default_factory=list)
    retained_numeric: list[str] = field(default_factory=list)
    clusters: dict = field(default_factory=dict)
    passthrough: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    output_columns: list[str] = field(default_factory=list)
    dictionary: list[Record] = field(default_factory=list)
    #: How the training rows' own target encoding was cross-fitted, for the report.
    crossfit: dict = field(default_factory=dict)

    def state(self) -> dict[str, Any]:
        return {
            "fitted_on": self.fitted_on,
            "rows_fitted": self.rows_fitted,
            "settings": {
                name: getattr(self.params, name)
                for name in (
                    "one_hot",
                    "target_encoding",
                    "feature_hashing",
                    "one_hot_top_x",
                    "onehot_max_categories",
                    "target_max_categories",
                    "max_cardinality_ratio",
                    "target_encoding_smoothing",
                    "target_encoding_folds",
                    "missing_category",
                    "min_non_missing",
                    "duplicate_r2_threshold",
                    "duplicate_min_overlap",
                    "cluster_corr_method",
                    "cluster_corr_threshold",
                    "cluster_min_overlap",
                )
            },
            "binary": self.binary,
            "one_hot": self.one_hot,
            "target_encoding": self.target_encoding,
            "hashing": self.hashing,
            "ordinal": self.ordinal,
            "durations": self.durations,
            "numeric": self.numeric,
            "retained_numeric": self.retained_numeric,
            "clusters": self.clusters,
            "passthrough": self.passthrough,
            "dropped_for_sparsity": self.dropped,
            "output_columns": self.output_columns,
        }

    # -- applying -----------------------------------------------------------
    def transform(
        self,
        df: pd.DataFrame,
        target: pd.Series | None = None,
        crossfit: bool = False,
        groups: pd.Series | None = None,
        order: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Encode any rows with what was fitted. Nothing here learns anything.

        ``crossfit`` is for the training rows themselves: their target encoding
        is computed fold by fold, each fold using only its own training part.
        """
        parts: list[pd.DataFrame] = []
        parts.append(encode.apply_binary(df, self.binary, self.params)[0])
        parts.append(encode.apply_one_hot(df, self.one_hot, self.params)[0])
        if crossfit and target is not None:
            encoded, info = encode.crossfit_target(
                df, self.target_encoding, target, self.params, groups=groups, order=order
            )
            self.crossfit = info
            parts.append(encoded)
        else:
            parts.append(encode.apply_target(df, self.target_encoding, self.params)[0])
        parts.append(encode.apply_hashed(df, self.hashing, self.params)[0])
        parts.append(encode.apply_ordinal(df, self.ordinal)[0])

        numbers = self._numbers(df)
        if self.clusters.get("clusters"):
            parts.append(decorrelate.transform(numbers, self.clusters))
        present = [c for c in self.passthrough if c in numbers.columns]
        if present:
            parts.append(numbers[present])

        frame = pd.concat([p for p in parts if not p.empty], axis=1)
        for column in self.output_columns:
            if column not in frame.columns:
                frame[column] = np.nan
        return frame[self.output_columns]

    def _numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        columns = [c for c in self.numeric if c in df.columns]
        frame, _ = encode.to_numeric(df, columns)
        encode.drop_nonpositive(frame, [c for c in self.durations if c in frame.columns])
        return frame


def _unit(parameter: str) -> str:
    """The unit a parameter's own name declares, where it declares one."""
    match = re.search(r"\(([^()]{1,24})\)\s*$", str(parameter))
    if not match:
        return ""
    text = match.group(1).strip()
    return text if any(char.isalpha() or char == "%" for char in text) else ""


def fit_preprocessing(
    train: pd.DataFrame,
    target: pd.Series,
    predictors: list[str],
    ptf: pd.DataFrame,
    spec: TaskSpec,
    result: TaskResult,
    reserved: set[str],
) -> Preprocessor:
    """Learn every data-derived decision from one population of rows."""
    params = spec.params
    if params.min_non_missing > len(train):
        raise TaskRefused(
            f"quality.min_non_missing is {params.min_non_missing} but only {len(train)} rows "
            "are being fitted on, so every column would be dropped. Lower it, or widen the "
            "cohort, or ask for a smaller validation fraction"
        )
    pre = Preprocessor(params=params, rows_fitted=len(train))
    taken = set(reserved)

    value_type = dict(zip(ptf["Parameter"].astype(str), ptf["Value Type"].astype(str)))
    categorical = [p for p in predictors if value_type.get(p) == "categorical"]
    ordinal_names = [p for p in predictors if encode.category_order(value_type.get(p)) is not None]
    categorical = [p for p in categorical if p not in ordinal_names]
    numeric = [
        p
        for p in predictors
        if value_type.get(p) in ("numeric", "ratio", "duration") and p not in ordinal_names
    ]
    durations = [p for p in predictors if value_type.get(p) == "duration"]

    # ── Categorical ───────────────────────────────────────────────────────
    log.step(MODULE, "Encoding categorical parameters")
    pre.routes = encode.route_categoricals(train, categorical, params)
    result.routes = pre.routes
    by_strategy = {
        strategy: [r.column for r in pre.routes if r.strategy == strategy]
        for strategy in ("binary", "one_hot", "target", "hashing", "skip")
    }
    for route in pre.routes:
        if route.strategy == "skip":
            result.decisions.append(
                Decision(route.column, "excluded", route.reason, "categorical routing")
            )

    pre.binary = encode.fit_binary(train, by_strategy["binary"], params, taken)
    pre.one_hot, result.one_hot_tail = encode.fit_one_hot(
        train, by_strategy["one_hot"], params, taken
    )

    hashing = list(by_strategy["hashing"])
    if by_strategy["target"]:
        pre.target_encoding, failed = encode.fit_target(
            train, by_strategy["target"], target, params, taken
        )
        if failed:
            if params.target_encoding_fallback == "stop":
                raise TaskRefused(
                    f"Target encoding could not run for {len(failed)} parameters "
                    f"({pre.target_encoding['batches']} batches with a target), and "
                    "encoding.target_encoding_fallback is 'stop'"
                )
            if params.feature_hashing:
                hashing += failed
                for column in failed:
                    result.decisions.append(
                        Decision(
                            column,
                            "changed",
                            "too few labelled batches for target encoding — hashed instead",
                            "encoding",
                        )
                    )
            else:
                for column in failed:
                    result.decisions.append(
                        Decision(
                            column,
                            "excluded",
                            "too few labelled batches for target encoding, and hashing is off",
                            "encoding",
                        )
                    )
    pre.hashing, result.hashing_summary = encode.fit_hashed(train, hashing, params, taken)
    pre.ordinal = encode.fit_ordinal(ptf, ordinal_names, taken)

    # ── Numeric ───────────────────────────────────────────────────────────
    log.step(MODULE, "Handling numeric parameters")
    pre.numeric, pre.durations = numeric, durations
    numbers, result.numeric_notes = encode.to_numeric(train, numeric)
    result.nonpositive_durations = encode.drop_nonpositive(numbers, durations)

    usable = [c for c in numeric if numbers[c].notna().any()]
    pre.retained_numeric, result.duplicate_decisions = encode.duplicate_selection(
        numbers, usable, params, prefer=spec.clustering.prefer
    )
    for entry in result.duplicate_decisions:
        result.decisions.append(
            Decision(
                entry["Dropped"],
                "excluded",
                f"duplicates '{entry['Kept']}' (R²={entry['R2']} over "
                f"{entry['Shared batches']} shared batches)",
                "near-duplicate",
            )
        )

    # ── Clusters ──────────────────────────────────────────────────────────
    result.clusters = decorrelate.fit(
        numbers,
        pre.retained_numeric,
        params,
        action=spec.clustering.action,
        prefer=spec.clustering.prefer,
    )
    pre.clusters = result.clusters.state() if result.clusters.transforms else {}
    clustered = set(result.clusters.members) if result.clusters.transforms else set()
    pre.passthrough = [c for c in pre.retained_numeric if c not in clustered]
    for cluster in result.clusters.clusters:
        for dropped in cluster.dropped:
            result.decisions.append(
                Decision(
                    dropped,
                    "excluded",
                    f"represented by '{cluster.representative}' in {cluster.name}",
                    "clustering",
                )
            )

    # ── The dictionary, and what the output columns are ───────────────────
    records: list[Record] = []
    records += encode.apply_binary(train, pre.binary, params)[1]
    records += encode.apply_one_hot(train, pre.one_hot, params)[1]
    records += encode.apply_target(train, pre.target_encoding, params)[1]
    records += encode.apply_hashed(train, pre.hashing, params)[1]
    ordinal_frame, unexpected, ordinal_records = encode.apply_ordinal(train, pre.ordinal)
    result.ordinal_unexpected = unexpected.to_dict("records")
    records += ordinal_records
    for cluster in result.clusters.clusters:
        records += cluster.records
    for column in pre.passthrough:
        records.append(
            encode._record(
                column,
                column,
                "duration" if column in durations else "numeric",
                "used as recorded, with anything at or below zero treated as missing"
                if column in durations
                else "used as recorded",
            )
        )

    pre.dictionary = records
    pre.output_columns = [r["output"] for r in records]
    duplicated = [
        c for c, count in pd.Series(pre.output_columns).value_counts().items() if count > 1
    ]
    if duplicated:
        raise TaskRefused(f"Two encoders produced the same output name: {duplicated[:5]}")

    # ── Sparsity, decided on the fitted rows ──────────────────────────────
    fitted_frame = pre.transform(train, target=target, crossfit=False)
    counts = fitted_frame.notna().sum()
    thin = [c for c in pre.output_columns if counts.get(c, 0) < params.min_non_missing]
    for column in thin:
        record = next((r for r in records if r["output"] == column), None)
        result.dropped_sparse.append(
            {
                "Feature": column,
                "Batches with a value": int(counts.get(column, 0)),
                "Produced by": record["strategy"] if record else "unknown",
            }
        )
        result.decisions.append(
            Decision(
                record["source"] if record else column,
                "excluded",
                f"'{column}' has {int(counts.get(column, 0))} values, below the "
                f"{params.min_non_missing} this task requires",
                "sparsity",
            )
        )
    if thin:
        pre.dropped = thin
        pre.output_columns = [c for c in pre.output_columns if c not in set(thin)]
        pre.dictionary = [r for r in pre.dictionary if r["output"] not in set(thin)]
        log.info(MODULE, f"{len(thin)} columns have too few values and were dropped")

    if not pre.output_columns:
        raise TaskRefused(
            "Every candidate predictor was dropped before the dataset was assembled. "
            "decisions.csv says why each one went"
        )
    return pre


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def assign_split(
    cohort: pd.DataFrame, labelled: pd.Index, spec: TaskSpec, result: TaskResult
) -> tuple[pd.Index, pd.Index]:
    """Training and validation rows, by the strategy the task declared.

    No split is invented: a task that asks for none gets none, and its
    transformed table is labelled exploratory instead.
    """
    if not spec.split.requested:
        return labelled, pd.Index([])

    fraction = spec.split.validation_fraction
    if spec.split.strategy == "grouped":
        groups = cohort.loc[labelled, spec.roles.group].astype("string").fillna("(missing)")
        distinct = list(dict.fromkeys(groups.tolist()))
        if len(distinct) < 2:
            raise TaskRefused(
                f"split.strategy: a grouped split needs at least two distinct values of "
                f"'{spec.roles.group}'; this cohort has {len(distinct)}"
            )
        # Whole groups, largest first, until the validation side is big enough.
        sizes = groups.value_counts()
        wanted = max(1, int(round(fraction * len(labelled))))
        validation_groups: list[str] = []
        taken = 0
        for group, size in sizes.sort_values(ascending=True).items():
            if taken >= wanted or len(validation_groups) == len(distinct) - 1:
                break
            validation_groups.append(group)
            taken += int(size)
        validation = labelled[groups.isin(validation_groups).to_numpy()]
        detail = f"whole groups of {spec.roles.group}: {', '.join(map(str, validation_groups))}"
    else:
        order_column = spec.split.order_column
        if order_column not in cohort.columns:
            raise TaskRefused(f"split.order_column: '{order_column}' is not a column of the PVF")
        stamps = pd.to_datetime(cohort.loc[labelled, order_column], errors="coerce")
        if stamps.isna().all():
            stamps = pd.to_numeric(cohort.loc[labelled, order_column], errors="coerce")
        if stamps.isna().any():
            raise TaskRefused(
                f"split.order_column: '{order_column}' cannot be ordered for "
                f"{int(stamps.isna().sum())} batches, so a chronological split would be a guess"
            )
        ordered = stamps.sort_values(kind="stable").index
        cut = len(ordered) - max(1, int(round(fraction * len(ordered))))
        validation = pd.Index(ordered[cut:])
        detail = f"the last {len(validation)} batches by {order_column}"

    training = labelled.difference(validation)
    if not len(training) or not len(validation):
        raise TaskRefused(
            "The requested split leaves one side empty; change split.validation_fraction"
        )
    result.split = {
        "strategy": spec.split.strategy,
        "detail": detail,
        "training": int(len(training)),
        "validation": int(len(validation)),
    }
    log.info(
        MODULE,
        f"Split ({spec.split.strategy}): {len(training)} training, "
        f"{len(validation)} validation — {detail}",
    )
    return training, validation


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------
def build(
    pvf: pd.DataFrame, ptf: pd.DataFrame, spec: TaskSpec
) -> tuple[pd.DataFrame, pd.DataFrame, TaskResult]:
    """Run one task.

    Returns the table the task is for, the selected raw predictors beside their
    metadata, and everything that was decided. For an exploratory task the first
    is the transformed table; for a predictive one it is the raw modelling input,
    and the transformed view is what the recipe produces.
    """
    result = TaskResult(spec=spec)
    cohort = select_cohort(pvf, spec, result)
    target, labelled = prepare_target(cohort, spec, result)
    cohort = cohort.loc[labelled]
    target = target.loc[labelled]
    result.cohort.rows = len(cohort)
    record_exclusions(pvf, spec, result, labelled)

    predictors = candidate_predictors(cohort, ptf, spec, result)
    training, validation = assign_split(cohort, labelled, spec, result)

    reserved = {
        name for name in (spec.roles.id, spec.roles.group, result.target_column, "split") if name
    }
    pre = fit_preprocessing(
        cohort.loc[training], target.loc[training], predictors, ptf, spec, result, reserved
    )
    result.fitted_on = (
        f"{len(training)} training batches ({result.split['detail']})"
        if spec.split.requested
        else f"all {len(training)} batches in the cohort"
    )
    pre.fitted_on = result.fitted_on

    groups = (
        cohort.loc[training, spec.roles.group]
        if spec.roles.group
        and spec.roles.group in cohort.columns
        and spec.split.strategy != "chronological"
        else None
    )
    order = (
        pd.to_datetime(cohort.loc[training, spec.split.order_column], errors="coerce")
        if spec.split.strategy == "chronological"
        else None
    )
    transformed = pre.transform(
        cohort.loc[training], target=target.loc[training], crossfit=True, groups=groups, order=order
    )
    if len(validation):
        transformed = pd.concat(
            [transformed, pre.transform(cohort.loc[validation], target=None)]
        ).loc[cohort.index]

    result.target_encoding = {
        "columns": {
            column: fitted["name"]
            for column, fitted in pre.target_encoding.get("columns", {}).items()
        },
        "batches": pre.crossfit.get("batches", pre.target_encoding.get("batches", 0)),
        "folds": pre.target_encoding.get("folds"),
        "smoothing": pre.target_encoding.get("smoothing"),
        "prior": pre.target_encoding.get("prior"),
        "strategy": pre.crossfit.get("strategy", ""),
        "note": pre.crossfit.get("note", ""),
        "unseen": pre.crossfit.get("unseen", {}),
    }

    # The raw table is the parameters the recipe actually uses — not every
    # candidate. A parameter the encoders then skipped is not a modelling input,
    # and exporting it as one would overstate what this dataset offers.
    result.selected_raw = [
        column
        for column in dict.fromkeys(record["source"] for record in pre.dictionary)
        if column in cohort.columns
    ]
    result.transformed_columns = len(pre.output_columns)
    raw = cohort[result.selected_raw].copy()
    metadata = _metadata(cohort, spec, result, training, validation)
    raw_table = pd.concat([metadata, raw, target.rename(result.target_column)], axis=1)
    transformed_table = pd.concat(
        [metadata, transformed, target.rename(result.target_column)], axis=1
    )

    if spec.predictive:
        final = raw_table
        result.dataset_role = (
            "raw modelling inputs; the transformed view is produced by recipe.json, "
            f"fitted on {result.fitted_on}"
        )
    else:
        final = transformed_table
        result.dataset_role = (
            "exploratory transformed data, preprocessed over "
            f"{result.fitted_on} — not a table to cross-validate on as it stands"
        )

    result.dictionary = list(pre.dictionary)
    result.recipe = pre.state()
    result.rows = len(final)
    result.feature_columns = final.shape[1] - metadata.shape[1] - 1
    result.columns = _column_table(final, spec, result, pre, ptf)
    for parameter in pre.output_columns:
        record = next((r for r in pre.dictionary if r["output"] == parameter), None)
        if record:
            result.decisions.append(
                Decision(record["source"], "included", record["strategy"], "encoding")
            )

    log.success(
        MODULE,
        f"Task '{spec.name}': {result.rows} batches × {result.feature_columns} columns, "
        f"predicting '{result.target_column}'",
    )
    return final, raw_table, result


def _metadata(
    cohort: pd.DataFrame,
    spec: TaskSpec,
    result: TaskResult,
    training: pd.Index,
    validation: pd.Index,
) -> pd.DataFrame:
    """The identifier, the grouping column and the split, each exactly once."""
    frame = pd.DataFrame(index=cohort.index)
    for column in (spec.roles.id, spec.roles.group):
        if column and column in cohort.columns:
            frame[column] = cohort[column]
        elif column:
            result.note(f"'{column}' is not in the PVF, so the dataset carries no such column")
    if len(validation):
        frame["split"] = np.where(frame.index.isin(validation), "validation", "training")
    return frame


def _column_table(
    final: pd.DataFrame, spec: TaskSpec, result: TaskResult, pre: Preprocessor, ptf: pd.DataFrame
) -> list[dict]:
    """One row per exported column, in output order. Nothing is left unexplained."""
    value_type = dict(zip(ptf["Parameter"].astype(str), ptf["Value Type"].astype(str)))
    rows: list[dict] = []
    for column in final.columns:
        if column == spec.roles.id:
            role, source, transformation, meaning = ID, column, "none", "which batch this row is"
        elif column == spec.roles.group:
            role, source, transformation = GROUP, column, "none"
            meaning = "grouping metadata, for grouped cross-validation; not a predictor"
        elif column == "split":
            role, source, transformation = "split", "assigned by this run", spec.split.strategy
            meaning = result.split.get("detail", "")
        elif column == result.target_column:
            role, source, transformation = TARGET, spec.target.column, result.target_kind
            meaning = (
                f"1 = {result.target_classes['1']}, 0 = {result.target_classes['0']}"
                if result.target_classes
                else "the value this task predicts"
            )
        else:
            record = next((r for r in pre.dictionary if r["output"] == column), None)
            if record:
                role, source = PREDICTOR, record["source"]
                transformation, meaning = record["strategy"], record["detail"]
            else:
                feeds = [r for r in pre.dictionary if r["source"] == column]
                role, source = PREDICTOR, column
                transformation = "none — raw input, recipe.json says what it becomes"
                meaning = (
                    "feeds " + ", ".join(f"{r['output']} ({r['strategy']})" for r in feeds[:4])
                    if feeds
                    else "carried through unchanged"
                )
        rows.append(
            {
                "Column": column,
                "Role": role,
                "Source": source,
                "Type": str(final[column].dtype),
                "PTF value type": value_type.get(source, ""),
                "Unit": _unit(source),
                "Transformation": transformation,
                "Meaning": meaning,
            }
        )
    return rows
