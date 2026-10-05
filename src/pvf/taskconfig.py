"""The YAML that defines a task, and the validation it has to pass first.

A task is a question asked of the PVF: one cohort, one target, one set of
predictors, one thing to do about correlated parameters, one destination. All of
that is written in YAML and nothing here reads Python from it — the filters are a
fixed set of operators, applied in the order they are declared.

Everything is validated before the run touches the output directory, and every
complaint names the YAML field and what to do about it. A task that cannot be
run is better refused at the boundary than half written to disk.

Relative paths resolve against the directory the YAML file itself is in, so a
task folder can be moved without editing it. Absolute paths are left alone.

A task can live anywhere. Its PVF, PTF and output folder may be written out under
``inputs``/``output``; or left out, in which case they come from the workspace
config — the one named by ``workspace:``, or else the one the run was started
with (``--config``, ``$PVF_CONFIG`` or the nearest ``pvf.yaml``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from .encode import TaskParams, sanitise

#: Filter operators a cohort may use. No expressions, no eval: a fixed set.
OPERATORS = (
    "equals",
    "not_equals",
    "in",
    "not_in",
    "min",
    "max",
    "is_null",
    "not_null",
)
PURPOSES = ("exploratory", "predictive")
TARGET_TYPES = ("numeric", "binary")
MISSING_POLICIES = ("exclude", "include")
#: What may be done about a group of parameters that move together.
CLUSTER_ACTIONS = ("off", "report_only", "representative", "linear", "nonlinear")
SPLIT_STRATEGIES = ("grouped", "chronological")
IMPUTATION = ("none", "median")

#: Process stages, earliest first. A PTF ``Available At`` value names one of
#: these, and a predictive task's ``availability.cutoff`` names the last stage
#: whose parameters it may use. A task can replace the list (availability.stages).
DEFAULT_STAGES = (
    "D0",
    "D1",
    "D3",
    "D6",
    "D8",
    "D10",
    "Harvest",
    "Wash",
    "Formulation",
    "Final product",
    "Post-thaw",
    "Release",
    "Post-release",
)
CORRELATION_METHODS = ("spearman", "pearson")

#: Status spellings that mean the same thing, flattened before a filter runs.
STATUS_ALIASES = {
    "withdrawn": "withdrawal",
    "withdrawl": "withdrawal",
    "terminated": "termination",
}


class TaskConfigError(ValueError):
    """One or more problems with a task YAML, each naming its field."""


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Filter:
    """One cohort rule, applied where it is declared.

    ``missing`` says what happens to a row whose value is absent: ``exclude``
    drops it, ``include`` keeps it. There is no third option, because a rule that
    silently does neither is how a cohort quietly broadens.
    """

    column: str
    op: str
    value: Any = None
    missing: str = "exclude"
    normalise: bool = False
    reason: str = ""

    @property
    def description(self) -> str:
        if self.reason:
            return self.reason
        if self.op in ("is_null", "not_null"):
            return f"{self.column} {self.op.replace('_', ' ')}"
        return f"{self.column} {self.op.replace('_', ' ')} {self.value}"


@dataclass(frozen=True)
class Cohort:
    sites: tuple[str, ...] = ()
    site_column: str = "Site Merged"
    filters: tuple[Filter, ...] = ()
    date_column: str = ""
    date_from: str = ""
    date_to: str = ""


@dataclass(frozen=True)
class Target:
    column: str = ""
    type: str = "numeric"
    positive_class: str = ""
    negative_class: str = ""
    missing: str = "drop"


@dataclass(frozen=True)
class Roles:
    """Which column is what. These are authoritative; nothing guesses them."""

    id: str = ""
    group: str = ""
    #: Batches of one patient (a re-manufacture, say) share biology, so they are
    #: kept on one side of a split together with their group.
    patient: str = ""
    include: tuple[str, ...] = ()
    exclude: tuple[tuple[str, str], ...] = ()

    @property
    def excluded_columns(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.exclude)


@dataclass(frozen=True)
class Availability:
    """What is known at the moment a prediction would be made.

    Nothing is inferred from a column's name. Where the PTF has an
    ``Available At`` column, ``cutoff`` names the last process stage a
    prediction may draw on and everything later — or unstaged — is left out
    mechanically. Without that column, the task has to declare which parameters
    come too late and state that the predictor list was reviewed.
    """

    cutoff: str = ""
    reviewed: bool = False
    unavailable: tuple[tuple[str, str], ...] = ()
    available: tuple[str, ...] = ()
    stages: tuple[str, ...] = DEFAULT_STAGES

    @property
    def unavailable_columns(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.unavailable)

    def rank(self, stage: object) -> int | None:
        """Where a stage falls in the process, or ``None`` when it is not one."""
        if stage is None or (isinstance(stage, float) and stage != stage):
            return None
        wanted = str(stage).strip().lower()
        for position, name in enumerate(self.stages):
            if name.lower() == wanted:
                return position
        return None

    @property
    def cutoff_rank(self) -> int | None:
        return self.rank(self.cutoff)


@dataclass(frozen=True)
class Clustering:
    """Discovery is one setting; what is done about what was found is another."""

    action: str = "report_only"
    method: str = "spearman"
    threshold: float = 0.7
    min_size: int = 2
    min_overlap: int = 10
    prefer: tuple[str, ...] = ()

    @property
    def discovers(self) -> bool:
        return self.action != "off"

    @property
    def transforms(self) -> bool:
        return self.action in ("representative", "linear", "nonlinear")


@dataclass(frozen=True)
class Split:
    strategy: str = ""
    validation_fraction: float = 0.25
    order_column: str = ""
    #: Cross-validation folds over the training rows, written to splits.csv. 0: none.
    folds: int = 0
    #: Keep the classes of a binary target balanced across the split and folds.
    stratify: bool = True

    @property
    def requested(self) -> bool:
        return bool(self.strategy)


@dataclass(frozen=True)
class Missingness:
    """Parameters that go missing together, and what explains it (description only)."""

    enabled: bool = True
    threshold: float = 0.9
    min_rate: float = 0.05
    max_rate: float = 0.95
    min_size: int = 2
    min_strength: float = 0.1
    max_categories: int = 20


@dataclass(frozen=True)
class Reporting:
    preview_rows: int = 20
    max_table_rows: int = 500


@dataclass(frozen=True)
class TaskSpec:
    """One task, fully specified. Nothing downstream reads YAML again."""

    name: str = "task"
    question: str = ""
    purpose: str = "exploratory"
    seed: int = 0
    pvf: Path = Path("PVF.xlsx")
    ptf: Path = Path("PTF.xlsx")
    output_root: Path = Path("outputs/tasks")
    cohort: Cohort = field(default_factory=Cohort)
    target: Target = field(default_factory=Target)
    roles: Roles = field(default_factory=Roles)
    availability: Availability = field(default_factory=Availability)
    encoding: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    clustering: Clustering = field(default_factory=Clustering)
    split: Split = field(default_factory=Split)
    missingness: Missingness = field(default_factory=Missingness)
    report: Reporting = field(default_factory=Reporting)
    source: str = ""
    workspace: str = ""

    @property
    def predictive(self) -> bool:
        return self.purpose == "predictive"

    @property
    def params(self) -> TaskParams:
        """The encoder and clustering knobs, as the encoders want them."""
        return TaskParams(
            seed=self.seed,
            cluster_corr_method=self.clustering.method,
            cluster_corr_threshold=self.clustering.threshold,
            cluster_min_size=self.clustering.min_size,
            cluster_min_overlap=self.clustering.min_overlap,
            **self.encoding,
            **self.quality,
        )

    def as_dict(self) -> dict[str, Any]:
        """The effective settings, defaults included, for ``task.yaml``."""
        return {
            "task": {
                "name": self.name,
                "question": self.question,
                "purpose": self.purpose,
                "seed": self.seed,
            },
            "inputs": {"pvf": str(self.pvf), "ptf": str(self.ptf)},
            "output": {"root": str(self.output_root)},
            "cohort": {
                "sites": list(self.cohort.sites),
                "site_column": self.cohort.site_column,
                "filters": [
                    {
                        "column": f.column,
                        "op": f.op,
                        "value": f.value,
                        "missing": f.missing,
                        "normalise": f.normalise,
                        "reason": f.reason,
                    }
                    for f in self.cohort.filters
                ],
                "date_column": self.cohort.date_column,
                "date_from": self.cohort.date_from,
                "date_to": self.cohort.date_to,
            },
            "target": {
                "column": self.target.column,
                "type": self.target.type,
                "positive_class": self.target.positive_class,
                "negative_class": self.target.negative_class,
                "missing": self.target.missing,
            },
            "columns": {
                "id": self.roles.id,
                "group": self.roles.group,
                "patient": self.roles.patient,
                "include": list(self.roles.include),
                "exclude": [{"column": c, "reason": r} for c, r in self.roles.exclude],
            },
            "availability": {
                "cutoff": self.availability.cutoff,
                "reviewed": self.availability.reviewed,
                "unavailable": [
                    {"column": c, "reason": r} for c, r in self.availability.unavailable
                ],
                "available": list(self.availability.available),
                "stages": list(self.availability.stages),
            },
            "encoding": dict(self.encoding),
            "quality": dict(self.quality),
            "clustering": {
                "action": self.clustering.action,
                "method": self.clustering.method,
                "threshold": self.clustering.threshold,
                "min_size": self.clustering.min_size,
                "min_overlap": self.clustering.min_overlap,
                "prefer": list(self.clustering.prefer),
            },
            "split": {
                "strategy": self.split.strategy,
                "validation_fraction": self.split.validation_fraction,
                "order_column": self.split.order_column,
                "folds": self.split.folds,
                "stratify": self.split.stratify,
            },
            "missingness": {
                "enabled": self.missingness.enabled,
                "threshold": self.missingness.threshold,
                "min_rate": self.missingness.min_rate,
                "max_rate": self.missingness.max_rate,
                "min_size": self.missingness.min_size,
                "min_strength": self.missingness.min_strength,
                "max_categories": self.missingness.max_categories,
            },
            "report": {
                "preview_rows": self.report.preview_rows,
                "max_table_rows": self.report.max_table_rows,
            },
        }


# ---------------------------------------------------------------------------
# Reading and validating
# ---------------------------------------------------------------------------
def _known(data: Any, allowed: tuple[str, ...], where: str, errors: list[str]) -> dict:
    """A mapping with no keys nobody asked for."""
    if data is None:
        return {}
    if not isinstance(data, dict):
        errors.append(f"{where}: expected a mapping, found {type(data).__name__}")
        return {}
    for key in data:
        if key not in allowed:
            errors.append(f"{where}.{key}: unknown setting (allowed: {', '.join(allowed)})")
    return data


def _text(data: dict, key: str, where: str, errors: list[str], default: str = "") -> str:
    value = data.get(key, default)
    if value is None:
        return default
    if isinstance(value, (str, int, float)):
        return str(value)
    errors.append(f"{where}.{key}: expected text, found {type(value).__name__}")
    return default


def _choice(
    data: dict, key: str, options: tuple[str, ...], where: str, errors: list[str], default: str
) -> str:
    value = _text(data, key, where, errors, default)
    if value and value not in options:
        errors.append(f"{where}.{key}: '{value}' is not one of {', '.join(options)}")
        return default
    return value


def _number(
    data: dict,
    key: str,
    where: str,
    errors: list[str],
    default: float,
    low: float | None = None,
    high: float | None = None,
    integer: bool = False,
) -> Any:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{where}.{key}: expected a number, found {value!r}")
        return default
    if value != value or value in (float("inf"), float("-inf")):
        errors.append(f"{where}.{key}: must be a finite number, found {value!r}")
        return default
    if low is not None and value < low:
        errors.append(f"{where}.{key}: {value} is below the smallest sensible value {low}")
        return default
    if high is not None and value > high:
        errors.append(f"{where}.{key}: {value} is above the largest sensible value {high}")
        return default
    return int(value) if integer else float(value)


def _flag(data: dict, key: str, where: str, errors: list[str], default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        errors.append(f"{where}.{key}: expected true or false, found {value!r}")
        return default
    return value


def _names(data: dict, key: str, where: str, errors: list[str]) -> tuple[str, ...]:
    value = data.get(key) or []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        errors.append(f"{where}.{key}: expected a list of column names")
        return ()
    return tuple(str(item) for item in value)


def _reasoned(data: dict, key: str, where: str, errors: list[str]) -> tuple[tuple[str, str], ...]:
    """A list of ``{column, reason}`` entries — a decision and why it was taken."""
    value = data.get(key) or []
    if not isinstance(value, list):
        errors.append(f"{where}.{key}: expected a list of column/reason entries")
        return ()
    out: list[tuple[str, str]] = []
    for index, item in enumerate(value):
        spot = f"{where}.{key}[{index}]"
        if isinstance(item, str):
            out.append((item, "no reason given"))
            continue
        if not isinstance(item, dict) or "column" not in item:
            errors.append(f"{spot}: expected {{column: ..., reason: ...}}")
            continue
        _known(item, ("column", "reason"), spot, errors)
        out.append((str(item["column"]), str(item.get("reason") or "no reason given")))
    return tuple(out)


def _filters(data: dict, where: str, errors: list[str]) -> tuple[Filter, ...]:
    raw = data.get("filters") or []
    if not isinstance(raw, list):
        errors.append(f"{where}.filters: expected a list of filters")
        return ()

    out: list[Filter] = []
    for index, item in enumerate(raw):
        spot = f"{where}.filters[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{spot}: expected a mapping with column and op")
            continue
        _known(item, ("column", "op", "value", "missing", "normalise", "reason"), spot, errors)
        column = _text(item, "column", spot, errors)
        op = _choice(item, "op", OPERATORS, spot, errors, "")
        if not column or not op:
            errors.append(f"{spot}: both column and op are required")
            continue
        value = item.get("value")
        if op in ("in", "not_in"):
            if isinstance(value, (str, int, float)):
                value = [value]
            if not isinstance(value, list) or not value:
                errors.append(f"{spot}.value: '{op}' needs a non-empty list of values")
                continue
        elif op in ("min", "max"):
            if value is None:
                errors.append(f"{spot}.value: '{op}' needs a bound")
                continue
        elif op in ("equals", "not_equals") and value is None:
            errors.append(f"{spot}.value: '{op}' needs a value")
            continue
        out.append(
            Filter(
                column=column,
                op=op,
                value=value,
                missing=_choice(item, "missing", MISSING_POLICIES, spot, errors, "exclude"),
                normalise=_flag(item, "normalise", spot, errors, False),
                reason=_text(item, "reason", spot, errors),
            )
        )
    return tuple(out)


#: Encoder settings, with the defaults that decide results rather than display.
ENCODING_FIELDS = {
    "one_hot": (bool, True),
    "target_encoding": (bool, True),
    "feature_hashing": (bool, True),
    "one_hot_top_x": (int, 10),
    "onehot_max_categories": (int, 15),
    "target_max_categories": (int, 50),
    "max_cardinality_ratio": (float, 0.5),
    "target_encoding_smoothing": (float, 10.0),
    "target_encoding_folds": (int, 5),
    "target_encoding_fallback": (str, "hashing"),
    "hashing_min_buckets": (int, 8),
    "hashing_max_buckets": (int, 32),
    "hashing_signed": (bool, True),
    "missing_category": (str, "missing"),
    "target_min_rows_per_category": (float, 5.0),
    "hashing_min_rows": (int, 100),
    "missing_indicators": (bool, False),
    "impute": (str, "none"),
}
QUALITY_FIELDS = {
    "min_non_missing": (int, 20),
    "duplicate_r2_threshold": (float, 0.99),
    "duplicate_min_overlap": (int, 10),
    "near_constant_tolerance": (float, 1e-4),
    "proxy_correlation": (float, 0.95),
}
_RANGES = {
    "one_hot_top_x": (1, 200),
    "onehot_max_categories": (2, 200),
    "target_max_categories": (2, 1000),
    "max_cardinality_ratio": (0.0, 1.0),
    "target_encoding_smoothing": (0.0, 1e6),
    "target_encoding_folds": (2, 50),
    "hashing_min_buckets": (2, 1024),
    "hashing_max_buckets": (2, 4096),
    "min_non_missing": (1, 1_000_000),
    "duplicate_r2_threshold": (0.0, 1.0),
    "duplicate_min_overlap": (2, 1_000_000),
    "target_min_rows_per_category": (0.0, 1_000.0),
    "hashing_min_rows": (0, 10_000_000),
    "near_constant_tolerance": (0.0, 0.5),
    "proxy_correlation": (0.0, 1.0),
}


def _knobs(
    data: dict, fields: dict[str, tuple[type, Any]], where: str, errors: list[str]
) -> dict[str, Any]:
    _known(data, tuple(fields), where, errors)
    out: dict[str, Any] = {}
    for name, (kind, default) in fields.items():
        if kind is bool:
            out[name] = _flag(data, name, where, errors, default)
        elif kind is str:
            out[name] = _text(data, name, where, errors, default)
        else:
            low, high = _RANGES.get(name, (None, None))
            out[name] = _number(
                data, name, where, errors, default, low=low, high=high, integer=kind is int
            )
    if "target_encoding_fallback" in out and out["target_encoding_fallback"] not in (
        "hashing",
        "stop",
    ):
        errors.append(f"{where}.target_encoding_fallback: expected 'hashing' or 'stop'")
    if "missing_category" in out and out["missing_category"] not in ("missing", "category"):
        errors.append(
            f"{where}.missing_category: expected 'missing' (leave the row empty) "
            "or 'category' (treat absence as a category of its own)"
        )
    if "impute" in out and out["impute"] not in IMPUTATION:
        errors.append(f"{where}.impute: expected one of {', '.join(IMPUTATION)}")
    if "hashing_min_buckets" in out and out["hashing_min_buckets"] > out["hashing_max_buckets"]:
        errors.append(f"{where}.hashing_min_buckets: larger than hashing_max_buckets")
    return out


def _resolve(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return Path(os.path.normpath(path if path.is_absolute() else (base / path)))


def parse(
    raw: dict[str, Any],
    base: Path,
    source: str = "",
    defaults: dict[str, str] | None = None,
) -> TaskSpec:
    """Validate a task mapping and resolve its paths. Raises on any problem.

    ``defaults`` holds the workspace's ``pvf``, ``ptf`` and ``tasks`` paths, used
    for whatever the task file itself leaves out.
    """
    errors: list[str] = []
    defaults = defaults or {}
    _known(
        raw,
        (
            "workspace",
            "task",
            "inputs",
            "output",
            "cohort",
            "target",
            "columns",
            "availability",
            "encoding",
            "quality",
            "clustering",
            "split",
            "missingness",
            "report",
        ),
        "(root)",
        errors,
    )

    meta = _known(raw.get("task"), ("name", "question", "purpose", "seed"), "task", errors)
    name = _text(meta, "name", "task", errors, "task")
    purpose = _choice(meta, "purpose", PURPOSES, "task", errors, "exploratory")
    seed = _number(meta, "seed", "task", errors, 0, low=0, integer=True)

    inputs = _known(raw.get("inputs"), ("pvf", "ptf"), "inputs", errors)
    output = _known(raw.get("output"), ("root",), "output", errors)

    cohort_raw = _known(
        raw.get("cohort"),
        ("sites", "site_column", "filters", "date_column", "date_from", "date_to"),
        "cohort",
        errors,
    )
    cohort = Cohort(
        sites=_names(cohort_raw, "sites", "cohort", errors),
        site_column=_text(cohort_raw, "site_column", "cohort", errors, "Site Merged"),
        filters=_filters(cohort_raw, "cohort", errors),
        date_column=_text(cohort_raw, "date_column", "cohort", errors),
        date_from=_text(cohort_raw, "date_from", "cohort", errors),
        date_to=_text(cohort_raw, "date_to", "cohort", errors),
    )
    if (cohort.date_from or cohort.date_to) and not cohort.date_column:
        errors.append("cohort.date_column: a date bound needs the column it applies to")

    target_raw = _known(
        raw.get("target"),
        ("column", "type", "positive_class", "negative_class", "missing"),
        "target",
        errors,
    )
    target = Target(
        column=_text(target_raw, "column", "target", errors),
        type=_choice(target_raw, "type", TARGET_TYPES, "target", errors, "numeric"),
        positive_class=_text(target_raw, "positive_class", "target", errors),
        negative_class=_text(target_raw, "negative_class", "target", errors),
        missing=_choice(target_raw, "missing", ("drop",), "target", errors, "drop"),
    )
    if not target.column:
        errors.append("target.column: a task has to name the column it predicts")
    if target.type == "binary" and not target.positive_class:
        errors.append(
            "target.positive_class: a binary target has to say which class is 1. "
            "The pipeline will not pick the rarer class for you"
        )
    if target.type == "numeric" and target.positive_class:
        errors.append("target.positive_class: only a binary target has one")

    roles_raw = _known(
        raw.get("columns"), ("id", "group", "patient", "include", "exclude"), "columns", errors
    )
    roles = Roles(
        id=_text(roles_raw, "id", "columns", errors),
        group=_text(roles_raw, "group", "columns", errors),
        patient=_text(roles_raw, "patient", "columns", errors),
        include=_names(roles_raw, "include", "columns", errors),
        exclude=_reasoned(roles_raw, "exclude", "columns", errors),
    )

    availability_raw = _known(
        raw.get("availability"),
        ("cutoff", "reviewed", "unavailable", "available", "stages"),
        "availability",
        errors,
    )
    stages = _names(availability_raw, "stages", "availability", errors) or DEFAULT_STAGES
    if len({stage.lower() for stage in stages}) != len(stages):
        errors.append("availability.stages: a stage is listed twice")
    availability = Availability(
        cutoff=_text(availability_raw, "cutoff", "availability", errors),
        reviewed=_flag(availability_raw, "reviewed", "availability", errors, False),
        unavailable=_reasoned(availability_raw, "unavailable", "availability", errors),
        available=_names(availability_raw, "available", "availability", errors),
        stages=tuple(stages),
    )

    encoding = _knobs(raw.get("encoding") or {}, ENCODING_FIELDS, "encoding", errors)
    quality = _knobs(raw.get("quality") or {}, QUALITY_FIELDS, "quality", errors)

    cluster_raw = _known(
        raw.get("clustering"),
        ("action", "method", "threshold", "min_size", "min_overlap", "prefer"),
        "clustering",
        errors,
    )
    clustering = Clustering(
        action=_choice(cluster_raw, "action", CLUSTER_ACTIONS, "clustering", errors, "report_only"),
        method=_choice(
            cluster_raw, "method", CORRELATION_METHODS, "clustering", errors, "spearman"
        ),
        threshold=_number(cluster_raw, "threshold", "clustering", errors, 0.7, low=0.0, high=1.0),
        min_size=_number(cluster_raw, "min_size", "clustering", errors, 2, low=2, integer=True),
        min_overlap=_number(
            cluster_raw, "min_overlap", "clustering", errors, 10, low=3, integer=True
        ),
        prefer=_names(cluster_raw, "prefer", "clustering", errors),
    )

    split_raw = _known(
        raw.get("split"),
        ("strategy", "validation_fraction", "order_column", "folds", "stratify"),
        "split",
        errors,
    )
    split = Split(
        strategy=_choice(split_raw, "strategy", SPLIT_STRATEGIES, "split", errors, ""),
        validation_fraction=_number(
            split_raw, "validation_fraction", "split", errors, 0.25, low=0.05, high=0.9
        ),
        order_column=_text(split_raw, "order_column", "split", errors),
        folds=_number(split_raw, "folds", "split", errors, 0, low=0, high=50, integer=True),
        stratify=_flag(split_raw, "stratify", "split", errors, True),
    )
    if split.folds == 1:
        errors.append("split.folds: 0 for none, or at least 2")
    if split.strategy == "grouped" and not roles.group:
        errors.append("split.strategy: a grouped split needs columns.group")
    if split.strategy == "chronological" and not split.order_column:
        errors.append("split.order_column: a chronological split needs the column it orders by")

    missing_raw = _known(
        raw.get("missingness"),
        tuple(f for f in Missingness.__dataclass_fields__),
        "missingness",
        errors,
    )
    missingness = Missingness(
        enabled=_flag(missing_raw, "enabled", "missingness", errors, True),
        threshold=_number(missing_raw, "threshold", "missingness", errors, 0.9, low=0.0, high=1.0),
        min_rate=_number(missing_raw, "min_rate", "missingness", errors, 0.05, low=0.0, high=1.0),
        max_rate=_number(missing_raw, "max_rate", "missingness", errors, 0.95, low=0.0, high=1.0),
        min_size=_number(missing_raw, "min_size", "missingness", errors, 2, low=2, integer=True),
        min_strength=_number(
            missing_raw, "min_strength", "missingness", errors, 0.1, low=0.0, high=1.0
        ),
        max_categories=_number(
            missing_raw, "max_categories", "missingness", errors, 20, low=2, integer=True
        ),
    )

    report_raw = _known(raw.get("report"), ("preview_rows", "max_table_rows"), "report", errors)
    report = Reporting(
        preview_rows=_number(report_raw, "preview_rows", "report", errors, 20, low=0, integer=True),
        max_table_rows=_number(
            report_raw, "max_table_rows", "report", errors, 500, low=10, integer=True
        ),
    )

    # ── Roles cannot overlap, and a predictor cannot also be metadata ──────
    roles_by_column: dict[str, list[str]] = {}
    for label, column in (
        ("target", target.column),
        ("id", roles.id),
        ("group", roles.group),
        ("patient", roles.patient),
    ):
        if column:
            roles_by_column.setdefault(column, []).append(label)
    for column, labels in roles_by_column.items():
        if len(labels) > 1:
            errors.append(f"columns: '{column}' is declared as {' and '.join(labels)} at once")
    for column in roles.include:
        if column in roles_by_column:
            errors.append(
                f"columns.include: '{column}' is already the "
                f"{roles_by_column[column][0]} column and cannot also be a predictor"
            )
        if column in roles.excluded_columns:
            errors.append(f"columns.include: '{column}' is in columns.exclude as well")

    if purpose == "predictive" and not (
        roles.include or availability.reviewed or availability.cutoff
    ):
        errors.append(
            "availability.cutoff: a predictive task has to say what is known at the "
            "prediction point — name the last process stage it may use in "
            f"availability.cutoff (one of {', '.join(availability.stages)}), give an "
            "explicit columns.include list, or state availability.reviewed: true"
        )
    if availability.cutoff and availability.cutoff_rank is None and not availability.reviewed:
        errors.append(
            f"availability.cutoff: '{availability.cutoff}' is not one of the process stages "
            f"({', '.join(availability.stages)}); use one of them, or set "
            "availability.reviewed: true to keep a free-text cutoff checked by hand"
        )

    def located(section: dict, key: str, where: str, fallback: str, what: str) -> Path:
        text = _text(section, key, where, errors).strip()
        if text:
            return _resolve(text, base)
        if fallback:
            return Path(fallback)
        errors.append(
            f"{where}.{key}: {what} — name it here, set workspace: <path to pvf.yaml>, "
            "or run from inside a workspace"
        )
        return Path(f"<{key}>")

    pvf = located(inputs, "pvf", "inputs", defaults.get("pvf", ""), "the PVF this task reads")
    ptf = located(inputs, "ptf", "inputs", defaults.get("ptf", ""), "the PTF this task reads")
    root = (
        _resolve(_text(output, "root", "output", errors), base)
        if str(output.get("root") or "").strip()
        else Path(defaults["tasks"])
        if defaults.get("tasks")
        else base / "outputs" / "tasks"
    )
    if pvf == ptf:
        errors.append("inputs: the PVF and the PTF cannot be the same file")
    if root in (pvf.parent / pvf.name, ptf):
        errors.append("output.root: the destination collides with an input file")
    if not name.strip():
        errors.append("task.name: a task needs a name; it is the folder the run is written to")
    if any(char in name for char in '/\\:*?"<>|'):
        errors.append(f"task.name: '{name}' has characters that cannot be a folder name")

    if errors:
        raise TaskConfigError(
            f"{source or 'task config'} cannot be run:\n  - " + "\n  - ".join(errors)
        )

    return TaskSpec(
        name=name,
        question=_text(meta, "question", "task", errors),
        purpose=purpose,
        seed=seed,
        pvf=pvf,
        ptf=ptf,
        output_root=root,
        cohort=cohort,
        target=target,
        roles=roles,
        availability=availability,
        encoding=encoding,
        quality=quality,
        clustering=clustering,
        split=split,
        missingness=missingness,
        report=report,
        source=source or str(base),
        workspace=defaults.get("config", ""),
    )


def load(path: str | Path, config: dict[str, Any] | None = None) -> TaskSpec:
    """Read a task YAML and validate it. Paths resolve against the file's folder.

    ``config`` is the workspace config the run was started with; it supplies the
    inputs and output folder a task file leaves out, unless the task names its
    own workspace.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise TaskConfigError(f"{path} does not exist")
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TaskConfigError(f"{path} is not a mapping")
    if "tasks" in raw and "task" not in raw:
        return from_legacy(raw, path)
    workspace = raw.get("workspace")
    if workspace:
        if not isinstance(workspace, str):
            raise TaskConfigError(f"{path}: workspace: expected the path to a pvf.yaml")
        config = _workspace(_resolve(workspace, path.parent))
    return parse(raw, path.parent, source=str(path), defaults=workspace_defaults(config))


def _workspace(path: Path) -> dict[str, Any]:
    from . import config as workspace_config

    if path.is_dir():
        path = workspace_config.find(start=path)
    try:
        return workspace_config.load(path)
    except workspace_config.ConfigError as exc:
        raise TaskConfigError(f"workspace: {exc}") from exc


def workspace_defaults(config: dict[str, Any] | None) -> dict[str, str]:
    """The paths a task inherits from a workspace config."""
    if not config:
        return {}
    paths = config.get("paths") or {}
    return {
        "pvf": str(paths.get("pvf") or ""),
        "ptf": str(paths.get("ptf") or ""),
        "tasks": str(paths.get("tasks") or ""),
        "config": str(config.get("__path__") or ""),
    }


# ---------------------------------------------------------------------------
# The configuration this package used to take
# ---------------------------------------------------------------------------
#: The cohort the old `tasks:` section applied without saying so.
LEGACY_FILTERS = [
    {
        "column": "Type",
        "op": "equals",
        "value": "Commercial",
        "reason": "manufactured commercially",
    },
    {
        "column": "Manufacturing and Release Testing Completed? (Y/N)",
        "op": "equals",
        "value": "Y",
        "reason": "manufacturing and testing finished",
    },
    {
        "column": "Non-Conformance Type",
        "op": "not_in",
        "value": ["termination", "withdrawal"],
        "normalise": True,
        "missing": "include",
        "reason": "not terminated or withdrawn",
    },
]
#: What the old code excluded by hand, in dataset.EXCLUDED.
LEGACY_EXCLUDE = [
    {
        "column": "Target CAR+ Viable Cells per dose (cells per dose)",
        "reason": "already carried by Subject Weight (kg)",
    }
]


def from_legacy(config: dict[str, Any], config_path: str | Path) -> TaskSpec:
    """Translate the old ``paths:`` + ``tasks:`` config into one task.

    Kept so existing commands keep working. The cohort rules that used to be
    written into ``dataset.py`` become declared filters here, which is the only
    way they can be seen or changed.
    """
    path = Path(config_path).expanduser().resolve()
    paths = config.get("paths") or {}
    tasks = dict(config.get("tasks") or {})

    site = tasks.pop("site", "")
    decorrelation = str(tasks.pop("decorrelation", "report_only")).lower()
    clustering = {
        "action": {"off": "off", "false": "off", "none": "off"}.get(decorrelation, decorrelation),
        "method": tasks.pop("cluster_corr_method", "spearman"),
        "threshold": tasks.pop("cluster_corr_threshold", 0.7),
        "min_size": tasks.pop("cluster_min_size", 2),
    }
    if "cluster_min_overlap" in tasks:
        clustering["min_overlap"] = tasks.pop("cluster_min_overlap")
    tasks.pop("cluster_r2_improvement_margin", None)

    target_column = tasks.pop("target", "")
    identifier = tasks.pop("id_column", "") or ""
    group = tasks.pop("group_column", "") or ""
    quality = {key: tasks.pop(key) for key in list(QUALITY_FIELDS) if key in tasks}

    raw = {
        "task": {
            # A folder name, so it is a slug rather than the target's own
            # punctuation: "FP Flow CAR+ (%)" is a column name, not a directory.
            "name": sanitise(f"{site or 'cohort'}_{target_column}")[:60] or "task",
            "question": f"What does {target_column} look like for the {site} cohort?",
            "purpose": "exploratory",
            "seed": config.get("seed", 0),
        },
        "inputs": {"pvf": paths.get("pvf", ""), "ptf": paths.get("ptf", "")},
        "output": {"root": paths.get("tasks", "outputs/tasks")},
        "cohort": {"sites": [site] if site else [], "filters": LEGACY_FILTERS},
        "target": {"column": target_column},
        "columns": {"id": identifier, "group": group, "exclude": LEGACY_EXCLUDE},
        "encoding": {k: v for k, v in tasks.items() if k in ENCODING_FIELDS},
        "quality": quality,
        "clustering": clustering,
    }
    unknown = [k for k in tasks if k not in ENCODING_FIELDS]
    if unknown:
        raise TaskConfigError(f"{path}: tasks.{', tasks.'.join(unknown)} is not a known setting")
    # One rule for every config this package reads: a relative path is relative
    # to the file it is written in.
    spec = parse(raw, path.parent, source=str(path))
    return replace(spec, source=str(path))
