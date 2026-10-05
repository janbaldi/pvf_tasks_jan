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

import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

from . import decorrelate, encode, features
from . import missingness as missing_analysis
from .clean import CENSORED_SUFFIX
from .encode import Record, TaskParams, sanitise
from .io import STAGE_COLUMN
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
    dropped_constant: list[dict] = field(default_factory=list)
    clusters: decorrelate.ClusterReport = field(default_factory=decorrelate.ClusterReport)
    #: Parameter → the process stage it exists from, where the PTF says.
    stages: dict[str, str] = field(default_factory=dict)
    #: Predictors that track the target closely on the fitted rows.
    proxies: list[dict] = field(default_factory=list)
    missingness: Any = None
    #: Row index → cross-validation fold (1..k), for the rows that have one.
    folds: dict = field(default_factory=dict)
    #: The transformed table, whatever the dataset is; exported beside it.
    transformed: pd.DataFrame | None = None

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


def lineage(names: list[str] | tuple[str, ...] = ()) -> dict[str, list[tuple[str, ...]]]:
    """What every derived column was computed from.

    The feature registry, the certificate columns the loader derives, and the
    ``<parameter> censored`` indicators cleaning can add for any of ``names``.
    """
    table = {f.name: _requirements(f) for f in features.registry(features.Params())}
    table.update(
        {
            name: [(source,) for source in sources]
            for name, sources in features.DERIVED_SOURCES.items()
        }
    )
    for name in names:
        if str(name).endswith(CENSORED_SUFFIX):
            table.setdefault(str(name), [(str(name)[: -len(CENSORED_SUFFIX)],)])
    return table


def target_descendants(
    target: str, params: features.Params | None = None, names: list[str] | tuple[str, ...] = ()
) -> dict[str, str]:
    """Every derived column the target went into, and the chain that got it there.

    Conservative on alternatives: a feature that could have been computed from
    the target *or* from something else is treated as descended from it, because
    which input a given site actually used is not recorded in the PVF.
    """
    tainted: dict[str, str] = {target: target}
    lineage_table = lineage([*names, f"{target}{CENSORED_SUFFIX}"])

    changed = True
    while changed:
        changed = False
        for name, requires in lineage_table.items():
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


def parameter_stages(
    ptf: pd.DataFrame, spec: TaskSpec, names: list[str]
) -> tuple[dict[str, int | None], list[str]]:
    """The process stage from which each parameter exists, as a rank.

    The PTF's ``Available At`` says it for a parameter a source records. A
    derived column cannot exist before its latest input, so its stage is the
    latest of its own and its inputs' — walked through the same lineage the
    target check uses. Unknown stays unknown (``None``). Also returns the PTF
    stage values that are not process stages, for the report.
    """
    if STAGE_COLUMN not in ptf.columns:
        return {}, []
    availability = spec.availability
    declared: dict[str, int | None] = {}
    unknown_values: set[str] = set()
    for parameter, value in zip(ptf["Parameter"].astype(str), ptf[STAGE_COLUMN]):
        rank = availability.rank(value)
        if rank is None and pd.notna(value) and str(value).strip():
            unknown_values.add(str(value).strip())
        declared[parameter] = rank

    table = lineage(names)
    ranks = dict(declared)
    changed = True
    while changed:
        changed = False
        for name, requires in table.items():
            inputs: list[int | None] = []
            for requirement in requires:
                known = [ranks.get(r) for r in requirement if ranks.get(r) is not None]
                # Any one alternative will do, but which one was used is not
                # recorded: take the latest, so a stage is never understated.
                inputs.append(max(known) if known else None)
            own = declared.get(name)
            if inputs and all(rank is not None for rank in inputs):
                derived = max(inputs) if own is None else max(own, *inputs)
            else:
                derived = own
            if derived != ranks.get(name):
                ranks[name] = derived
                changed = True
    return ranks, sorted(unknown_values)


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
        spec.roles.patient: "the patient column",
    }
    roles.pop("", None)
    descendants = target_descendants(spec.target.column, names=list(cohort.columns))
    unavailable = dict(spec.availability.unavailable)
    overrides = set(spec.availability.available)

    # When the PTF says from which stage each parameter exists, a predictive
    # task's cutoff is checked mechanically: later or unstaged is out.
    ranks, odd_values = parameter_stages(ptf, spec, list(cohort.columns))
    cutoff = spec.availability.cutoff_rank
    staged = bool(ranks) and spec.predictive and cutoff is not None
    result.stages = {
        name: (spec.availability.stages[rank] if rank is not None else "")
        for name, rank in ranks.items()
    }
    if odd_values:
        result.note(
            f"The PTF's '{STAGE_COLUMN}' column holds values that are not process stages "
            f"({', '.join(odd_values[:6])}); those parameters count as unstaged"
        )
    if spec.predictive and not spec.roles.include:
        if ranks and cutoff is None:
            result.note(
                f"availability.cutoff '{spec.availability.cutoff}' is free text, so the PTF's "
                f"'{STAGE_COLUMN}' stages were not used; availability rests on the declared list"
            )
        elif not ranks:
            if not spec.availability.reviewed:
                raise TaskRefused(
                    f"availability.cutoff: the PTF has no '{STAGE_COLUMN}' column, so a cutoff "
                    "cannot be checked against it. Add the column to the PTF, or list the late "
                    "parameters under availability.unavailable and set availability.reviewed: true"
                )
            result.note(
                f"The PTF has no '{STAGE_COLUMN}' column, so availability rests on the "
                "hand-written availability.unavailable list alone"
            )

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
        elif staged and parameter not in overrides and ranks.get(parameter) is None:
            refuse(
                parameter,
                f"the PTF gives it no '{STAGE_COLUMN}' stage, so it cannot be shown to exist "
                f"by the prediction point ({spec.availability.cutoff})",
                "availability",
            )
        elif staged and parameter not in overrides and ranks[parameter] > cutoff:
            refuse(
                parameter,
                f"available at {spec.availability.stages[ranks[parameter]]}, after the "
                f"prediction point ({spec.availability.cutoff})",
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
#: Bumped whenever recipe.json changes shape; a recipe says which version wrote it.
RECIPE_VERSION = 2
#: The suffix of a missing-value indicator column.
MISSING_SUFFIX = "__missing"


@dataclass
class Preprocessor:
    """What was learned from the training rows, and how to apply it elsewhere.

    Plain data throughout: dictionaries of category vocabularies, means, widths,
    retained parameters and regression coefficients. It is written next to the
    dataset as ``recipe.json`` and read back with :meth:`from_state`, so applying
    it to new batches needs this package — ``pvf apply`` — not this process.
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
    #: The encoded columns kept after the sparsity and near-constant checks.
    encoded_columns: list[str] = field(default_factory=list)
    #: Encoded columns that get a 0/1 missing-value indicator.
    indicators: list[str] = field(default_factory=list)
    #: Encoded column → the fitted median its gaps are filled with.
    medians: dict[str, float] = field(default_factory=dict)
    output_columns: list[str] = field(default_factory=list)
    dictionary: list[Record] = field(default_factory=list)

    def state(self) -> dict[str, Any]:
        return {
            "recipe_version": RECIPE_VERSION,
            "fitted_on": self.fitted_on,
            "rows_fitted": self.rows_fitted,
            "settings": asdict(self.params),
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
            "encoded_columns": self.encoded_columns,
            "indicators": self.indicators,
            "medians": self.medians,
            "output_columns": self.output_columns,
            "dictionary": self.dictionary,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> Preprocessor:
        """The recipe a run wrote, ready to apply to any rows."""
        known = {f.name for f in fields(TaskParams)}
        settings = {k: v for k, v in (state.get("settings") or {}).items() if k in known}
        output = list(state.get("output_columns") or [])
        return cls(
            params=TaskParams(**settings),
            fitted_on=state.get("fitted_on", ""),
            rows_fitted=int(state.get("rows_fitted") or 0),
            binary=state.get("binary") or {},
            one_hot=state.get("one_hot") or {},
            target_encoding=state.get("target_encoding") or {},
            hashing=state.get("hashing") or {},
            ordinal=state.get("ordinal") or {},
            durations=list(state.get("durations") or []),
            numeric=list(state.get("numeric") or []),
            retained_numeric=list(state.get("retained_numeric") or []),
            clusters=state.get("clusters") or {},
            passthrough=list(state.get("passthrough") or []),
            dropped=list(state.get("dropped_for_sparsity") or []),
            encoded_columns=list(state.get("encoded_columns") or output),
            indicators=list(state.get("indicators") or []),
            medians=dict(state.get("medians") or {}),
            output_columns=output,
            dictionary=list(state.get("dictionary") or []),
        )

    @classmethod
    def load(cls, path: str | Path) -> Preprocessor:
        return cls.from_state(json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def inputs(self) -> list[str]:
        """The PVF parameters a row needs for this recipe to encode it."""
        return list(dict.fromkeys(record["source"] for record in self.dictionary))

    # -- applying -----------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode any rows with what was fitted. Nothing here learns anything."""
        target_encoded = encode.apply_target(df, self.target_encoding, self.params)[0]
        return self._finish(self._encode(df, target_encoded))

    def crossfit_transform(
        self,
        df: pd.DataFrame,
        target: pd.Series,
        groups: pd.Series | None = None,
        order: pd.Series | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """The training rows themselves, their target encoding computed fold by fold.

        Each fold's encoding uses only that fold's training part, so no row is
        encoded with a mean its own target is inside. Returns what the folds did.
        """
        encoded, info = encode.crossfit_target(
            df, self.target_encoding, target, self.params, groups=groups, order=order
        )
        return self._finish(self._encode(df, encoded)), info

    def _encode(self, df: pd.DataFrame, target_encoded: pd.DataFrame) -> pd.DataFrame:
        parts: list[pd.DataFrame] = [
            encode.apply_binary(df, self.binary, self.params)[0],
            encode.apply_one_hot(df, self.one_hot, self.params)[0],
            target_encoded,
            encode.apply_hashed(df, self.hashing, self.params)[0],
            encode.apply_ordinal(df, self.ordinal)[0],
        ]
        numbers = self._numbers(df)
        if self.clusters.get("clusters"):
            parts.append(decorrelate.transform(numbers, self.clusters))
        present = [c for c in self.passthrough if c in numbers.columns]
        if present:
            parts.append(numbers[present])
        frame = pd.concat([p for p in parts if not p.empty], axis=1)
        frame = frame.loc[:, ~frame.columns.duplicated()]
        wanted = self.encoded_columns or self.output_columns
        return frame.reindex(columns=wanted)

    def _finish(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Indicators, then imputation — in that order, so an indicator sees the gap."""
        indicators = {
            f"{column}{MISSING_SUFFIX}": frame[column].isna().astype("Int64")
            for column in self.indicators
            if column in frame.columns
        }
        if self.medians:
            fill = {c: v for c, v in self.medians.items() if c in frame.columns}
            frame = frame.astype({c: "float64" for c in fill}).fillna(fill)
        if indicators:
            frame = pd.concat([frame, pd.DataFrame(indicators, index=frame.index)], axis=1)
        return frame.reindex(columns=self.output_columns)

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
    # A yes/no parameter is a category with two levels.
    categorical = [p for p in predictors if value_type.get(p) in ("categorical", "boolean")]
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

    pre.encoded_columns = list(pre.output_columns)

    # ── Sparsity, decided on the fitted rows ──────────────────────────────
    fitted_frame = pre.transform(train)
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
        log.info(MODULE, f"{len(thin)} columns have too few values and were dropped")

    # ── Constant to measurement precision, on the fitted rows ─────────────
    # An empty one-hot tail, a hash bucket nobody landed in, or a ratio that
    # only differs in its seventh digit says nothing, and scaled to unit
    # variance the last of those becomes the largest number in the table.
    flat = [
        c
        for c in pre.output_columns
        if c not in set(thin)
        and encode.near_constant(fitted_frame[c], params.near_constant_tolerance)
    ]
    for column in flat:
        record = next((r for r in records if r["output"] == column), None)
        values = pd.to_numeric(fitted_frame[column], errors="coerce").dropna()
        result.dropped_constant.append(
            {
                "Feature": column,
                "Produced by": record["strategy"] if record else "unknown",
                "Value": round(float(values.iloc[0]), 6) if len(values) else "",
            }
        )
        result.decisions.append(
            Decision(
                record["source"] if record else column,
                "excluded",
                f"'{column}' is constant on the fitted rows, to measurement precision",
                "near-constant",
            )
        )
    if flat:
        log.info(MODULE, f"{len(flat)} columns are constant on the fitted rows and were dropped")

    gone = set(thin) | set(flat)
    pre.dropped = thin
    pre.encoded_columns = [c for c in pre.output_columns if c not in gone]
    pre.dictionary = [r for r in pre.dictionary if r["output"] not in gone]
    if not pre.encoded_columns:
        raise TaskRefused(
            "Every candidate predictor was dropped before the dataset was assembled. "
            "decisions.csv says why each one went"
        )
    kept = fitted_frame[pre.encoded_columns]

    # ── Gaps: indicators and imputation, both learned here ────────────────
    if params.missing_indicators:
        pre.indicators = [c for c in pre.encoded_columns if kept[c].isna().any()]
        sources = {r["output"]: r["source"] for r in pre.dictionary}
        pre.dictionary += [
            encode._record(
                f"{c}{MISSING_SUFFIX}",
                sources.get(c, c),
                "missing indicator",
                f"1 where '{c}' has no value, 0 where it has one",
            )
            for c in pre.indicators
        ]
    if params.impute == "median":
        medians = kept.apply(pd.to_numeric, errors="coerce").median()
        pre.medians = {c: float(v) for c, v in medians.items() if pd.notna(v)}
        for record in pre.dictionary:
            if record["output"] in pre.medians:
                record["detail"] += (
                    f"; a gap is filled with the training median {pre.medians[record['output']]:g}"
                )
    pre.output_columns = pre.encoded_columns + [f"{c}{MISSING_SUFFIX}" for c in pre.indicators]

    # ── Predictors that look like the target ──────────────────────────────
    if params.proxy_correlation > 0 and len(train) >= 8:
        y = pd.to_numeric(target.reindex(train.index), errors="coerce")
        sources = {r["output"]: r["source"] for r in pre.dictionary}
        for column in pre.encoded_columns:
            r, shared = encode.pairwise(kept[column].astype("float64"), y)
            if shared >= 8 and np.isfinite(r) and abs(r) >= params.proxy_correlation:
                result.proxies.append(
                    {"Feature": column, "Parameter": sources.get(column, column),
                     "r": round(r, 3), "Shared batches": shared}
                )
        for proxy in result.proxies:
            result.note(
                f"'{proxy['Parameter']}' correlates r={proxy['r']} with the target over "
                f"{proxy['Shared batches']} fitted batches — check it is not the target "
                "measured another way, which would be a leak"
            )
    return pre


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def split_groups(cohort: pd.DataFrame, labelled: pd.Index, spec: TaskSpec) -> pd.Series | None:
    """The units a split must keep whole: the grouping column, joined up by patient.

    Two batches belong together when they share a group (a vector lot) or a
    patient — directly or through a chain of batches. A re-manufactured batch
    shares its patient's starting material with the first one, so it cannot be
    allowed to sit on the other side of a split from it.
    """
    columns = [c for c in (spec.roles.group, spec.roles.patient) if c and c in cohort.columns]
    if not columns:
        return None
    rows = list(labelled)
    parent = list(range(len(rows)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for column in columns:
        values = cohort.loc[labelled, column].astype("string")
        if column == spec.roles.group:
            values = values.fillna("(missing)")
        first: dict[str, int] = {}
        for position, value in enumerate(values.tolist()):
            if value is pd.NA or value is None:
                continue
            if value in first:
                parent[root(position)] = root(first[value])
            else:
                first[value] = position
    label_column = spec.roles.group if spec.roles.group in cohort.columns else columns[0]
    names = cohort.loc[labelled, label_column].astype("string").fillna("(missing)").tolist()
    members: dict[int, list[str]] = {}
    for position, name in enumerate(names):
        members.setdefault(root(position), []).append(str(name))
    labels = {r: "+".join(sorted(set(m))) for r, m in members.items()}
    return pd.Series([labels[root(i)] for i in range(len(rows))], index=labelled)


def _splitter(n_splits: int, grouped: bool, stratified: bool, seed: int):
    if grouped and stratified:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if grouped:
        return GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if stratified:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _stratify(spec: TaskSpec, target: pd.Series | None, rows: pd.Index) -> pd.Series | None:
    if target is None or spec.target.type != "binary" or not spec.split.stratify:
        return None
    labels = target.loc[rows]
    return labels if labels.value_counts().min() >= 2 else None


def assign_split(
    cohort: pd.DataFrame,
    labelled: pd.Index,
    spec: TaskSpec,
    result: TaskResult,
    target: pd.Series | None = None,
) -> tuple[pd.Index, pd.Index]:
    """Training and validation rows, by the strategy the task declared.

    No split is invented: a task that asks for none gets none, and its
    transformed table is labelled exploratory instead. A grouped split shuffles
    whole groups with the task's seed — keeping a binary target's classes
    balanced where it can — rather than always holding out the same kind of
    group. ``split.folds`` adds cross-validation folds over the training rows.
    """
    if not spec.split.requested and not spec.split.folds:
        return labelled, pd.Index([])

    fraction = spec.split.validation_fraction
    groups = split_groups(cohort, labelled, spec)
    detail = ""
    validation = pd.Index([])
    if spec.split.strategy == "grouped":
        if groups is None:
            raise TaskRefused("split.strategy: a grouped split needs columns.group")
        distinct = groups.nunique()
        if distinct < 2:
            raise TaskRefused(
                f"split.strategy: a grouped split needs at least two distinct values of "
                f"'{spec.roles.group}'; this cohort has {distinct}"
            )
        n_splits = int(min(distinct, max(2, round(1 / fraction))))
        strata = _stratify(spec, target, labelled)
        splitter = _splitter(n_splits, True, strata is not None, spec.seed)
        _, test = next(
            splitter.split(np.zeros(len(labelled)), strata if strata is not None else None, groups)
        )
        validation = labelled[test]
        held = sorted(set(groups.loc[validation]))
        detail = (
            f"whole groups of {spec.roles.group}"
            + (f" (joined by {spec.roles.patient})" if spec.roles.patient else "")
            + f", drawn with seed {spec.seed}"
            + (", classes balanced" if strata is not None else "")
            + f": {', '.join(held)}"
        )
    elif spec.split.strategy == "chronological":
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

    training = labelled.difference(validation, sort=False)
    if spec.split.requested and (not len(training) or not len(validation)):
        raise TaskRefused(
            "The requested split leaves one side empty; change split.validation_fraction"
        )
    if groups is not None and len(validation):
        shared = set(groups.loc[training]) & set(groups.loc[validation])
        if shared:  # cannot happen with whole groups; a chronological split can
            result.note(
                f"{len(shared)} group(s) of {spec.roles.group} have batches on both sides of "
                "the split; a model can learn them from their siblings"
            )

    result.split = {
        "strategy": spec.split.strategy or "none",
        "detail": detail or "no validation rows; folds only",
        "training": int(len(training)),
        "validation": int(len(validation)),
        "seed": spec.seed,
    }
    if spec.split.folds:
        _assign_folds(cohort, training, spec, result, target, groups)
    log.info(
        MODULE,
        f"Split ({result.split['strategy']}): {len(training)} training, "
        f"{len(validation)} validation — {result.split['detail']}"
        + (f"; {result.split['folds']} folds over the training rows" if spec.split.folds else ""),
    )
    return training, validation


def _assign_folds(
    cohort: pd.DataFrame,
    training: pd.Index,
    spec: TaskSpec,
    result: TaskResult,
    target: pd.Series | None,
    groups: pd.Series | None,
) -> None:
    """Cross-validation folds over the training rows, written to splits.csv.

    Grouped (and class-balanced) where the task has groups; in time order for a
    chronological task, where fold k is the k-th block and is only ever to be
    predicted from the blocks before it.
    """
    wanted = spec.split.folds
    if spec.split.strategy == "chronological":
        stamps = pd.to_datetime(cohort.loc[training, spec.split.order_column], errors="coerce")
        ordered = stamps.sort_values(kind="stable").index
        blocks = np.array_split(np.arange(len(ordered)), min(wanted, len(ordered)))
        for number, block in enumerate(blocks, start=1):
            for position in block:
                result.folds[ordered[position]] = number
        result.split.update({"folds": len(blocks), "fold_strategy": "chronological blocks"})
        return
    train_groups = groups.loc[training] if groups is not None else None
    available = int(train_groups.nunique()) if train_groups is not None else len(training)
    n_splits = min(wanted, available)
    if n_splits < 2:
        result.note(f"split.folds: only {n_splits} group(s) in the training rows, so no folds")
        return
    strata = _stratify(spec, target, training)
    splitter = _splitter(n_splits, train_groups is not None, strata is not None, spec.seed)
    for number, (_, test) in enumerate(
        splitter.split(np.zeros(len(training)), strata, train_groups), start=1
    ):
        for position in test:
            result.folds[training[position]] = number
    result.split.update(
        {
            "folds": n_splits,
            "fold_strategy": ("grouped" if train_groups is not None else "shuffled")
            + (", class-balanced" if strata is not None else ""),
        }
    )
    if n_splits < wanted:
        result.note(f"split.folds: {wanted} folds asked for, {n_splits} groups allow {n_splits}")


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
    and the transformed view — fitted on the training rows, applied to the rest —
    is ``result.transformed``.
    """
    result = TaskResult(spec=spec)
    cohort = select_cohort(pvf, spec, result)
    target, labelled = prepare_target(cohort, spec, result)
    cohort = cohort.loc[labelled]
    target = target.loc[labelled]
    result.cohort.rows = len(cohort)
    record_exclusions(pvf, spec, result, labelled)
    _check_unique_ids(cohort, spec)

    if spec.missingness.enabled:
        value_types = dict(zip(ptf["Parameter"].astype(str), ptf["Value Type"].astype(str)))
        result.missingness = missing_analysis.analyse(
            cohort,
            value_types,
            spec.missingness,
            id_column=spec.roles.id,
            order_column=spec.split.order_column or spec.cohort.date_column,
        )

    predictors = candidate_predictors(cohort, ptf, spec, result)
    training, validation = assign_split(cohort, labelled, spec, result, target)

    reserved = {
        name
        for name in (
            spec.roles.id,
            spec.roles.group,
            spec.roles.patient,
            result.target_column,
            "split",
            "fold",
        )
        if name
    }
    pre = fit_preprocessing(
        cohort.loc[training], target.loc[training], predictors, ptf, spec, result, reserved
    )
    result.fitted_on = (
        f"{len(training)} training batches ({result.split['detail']})"
        if len(validation)
        else f"all {len(training)} batches in the cohort"
    )
    pre.fitted_on = result.fitted_on

    chronological = spec.split.strategy == "chronological"
    groups = None if chronological else split_groups(cohort, training, spec)
    order = (
        pd.to_datetime(cohort.loc[training, spec.split.order_column], errors="coerce")
        if chronological
        else None
    )
    transformed, crossfit = pre.crossfit_transform(
        cohort.loc[training], target.loc[training], groups=groups, order=order
    )
    if len(validation):
        transformed = pd.concat([transformed, pre.transform(cohort.loc[validation])]).loc[
            cohort.index
        ]

    result.target_encoding = {
        "columns": {
            column: fitted["name"]
            for column, fitted in pre.target_encoding.get("columns", {}).items()
        },
        "batches": crossfit.get("batches", pre.target_encoding.get("batches", 0)),
        "folds": pre.target_encoding.get("folds"),
        "smoothing": pre.target_encoding.get("smoothing"),
        "prior": pre.target_encoding.get("prior"),
        "strategy": crossfit.get("strategy", ""),
        "note": crossfit.get("note", ""),
        "unseen": crossfit.get("unseen", {}),
    }

    # The raw table is the parameters the recipe actually uses — not every
    # candidate. A parameter the encoders then skipped is not a modelling input,
    # and exporting it as one would overstate what this dataset offers.
    result.selected_raw = [column for column in pre.inputs if column in cohort.columns]
    result.transformed_columns = len(pre.output_columns)
    raw = cohort[result.selected_raw].copy()
    metadata = _metadata(cohort, spec, result, training, validation)
    raw_table = pd.concat([metadata, raw, target.rename(result.target_column)], axis=1)
    transformed_table = pd.concat(
        [metadata, transformed, target.rename(result.target_column)], axis=1
    )
    result.transformed = transformed_table

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
    _sample_size_notes(cohort, target, training, spec, result)

    log.success(
        MODULE,
        f"Task '{spec.name}': {result.rows} batches × {result.feature_columns} columns, "
        f"predicting '{result.target_column}'",
    )
    return final, raw_table, result


def _check_unique_ids(cohort: pd.DataFrame, spec: TaskSpec) -> None:
    """A batch twice in one dataset is counted twice and can sit on both sides of a split."""
    column = spec.roles.id
    if not column or column not in cohort.columns:
        return
    ids = cohort[column]
    repeated = ids[ids.notna() & ids.duplicated(keep=False)]
    if len(repeated):
        raise TaskRefused(
            f"columns.id: '{column}' is not unique in this cohort — "
            f"{', '.join(sorted(map(str, set(repeated)))[:5])} appear more than once. "
            "Fix the source, or narrow the cohort so each batch is in it once"
        )


def _sample_size_notes(
    cohort: pd.DataFrame,
    target: pd.Series,
    training: pd.Index,
    spec: TaskSpec,
    result: TaskResult,
) -> None:
    """How much evidence the table really holds, said where a reader will see it."""
    fitted = len(training)
    columns = result.transformed_columns
    if spec.roles.group and spec.roles.group in cohort.columns:
        distinct = int(cohort[spec.roles.group].nunique())
        if distinct < 10:
            result.note(
                f"Only {distinct} distinct {spec.roles.group} values: any grouped validation "
                f"rests on {distinct} groups, not on {len(cohort)} batches"
            )
    if fitted and columns > fitted:
        result.note(
            f"{columns} predictor columns for {fitted} fitted batches (p/n = "
            f"{columns / fitted:.1f}): a model needs strong regularisation, and its "
            "performance has to be estimated by cross-validation, not in-sample"
        )
    if spec.target.type == "binary" and columns:
        minority = int(target.loc[training].value_counts().min())
        epv = minority / columns
        if epv < 10:
            result.note(
                f"{minority} batches in the smaller class for {columns} predictor columns "
                f"({epv:.2f} events per variable, against the usual 10 or more)"
            )


def _metadata(
    cohort: pd.DataFrame,
    spec: TaskSpec,
    result: TaskResult,
    training: pd.Index,
    validation: pd.Index,
) -> pd.DataFrame:
    """The identifier, the grouping columns, the split and the fold, each exactly once."""
    frame = pd.DataFrame(index=cohort.index)
    for column in (spec.roles.id, spec.roles.group, spec.roles.patient):
        if column and column in cohort.columns:
            frame[column] = cohort[column]
        elif column:
            result.note(f"'{column}' is not in the PVF, so the dataset carries no such column")
    if len(validation):
        frame["split"] = np.where(frame.index.isin(validation), "validation", "training")
    if result.folds:
        frame["fold"] = pd.Series(result.folds, dtype="Int64").reindex(frame.index)
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
        elif column == spec.roles.patient:
            role, source, transformation = GROUP, column, "none"
            meaning = "the patient; keeps a patient's batches on one side of a split"
        elif column == "fold":
            role, source, transformation = "split", "assigned by this run", "cross-validation"
            meaning = (
                "cross-validation fold over the training rows "
                f"({result.split.get('fold_strategy')}); empty for validation rows"
            )
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
