"""The YAML that defines a task, and the validation it has to pass first.

A task is a question asked of the batch table: one cohort, one target, one set of
predictors, one thing to do about correlated parameters, one destination. All of
that is written in YAML and nothing here reads Python from it — the filters are a
fixed set of operators, applied in the order they are declared.

Everything is validated before the run touches the output directory, and every
complaint names the YAML field and what to do about it. A task that cannot be
run is better refused at the boundary than half written to disk.

Each task is a workspace: a folder holding its ``task.yaml``, with every run
written into ``runs/`` beside it. Relative paths resolve against that folder.
The batch table and catalog come from ``config.yaml`` — the one the task names under
``config:``, or ``--config``, ``$ML_CONFIG`` or the nearest ``config.yaml`` —
unless the task names its own under ``inputs``.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import ConfigError
from .config import load as load_config
from .encode import TaskParams

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
TARGET_TYPES = ("numeric", "binary")
MISSING_POLICIES = ("exclude", "include")
#: What may be done about a group of parameters that move together.
CLUSTER_ACTIONS = ("off", "describe", "keep_one", "residualise")
IMPUTATION = ("none", "median")

CORRELATION_METHODS = ("spearman", "pearson")

#: Status spellings that mean the same thing, flattened before a filter runs.
STATUS_ALIASES = {
    "withdrawn": "withdrawal",
    "withdrawl": "withdrawal",
    "terminated": "termination",
}


#: The file that makes a folder a task workspace.
TASK_FILE = "task.yaml"


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
    #: kept together with their group when the target encoding is cross-fitted.
    patient: str = ""


@dataclass(frozen=True)
class Predictors:
    """Which parameters may be predictors, beyond what the roles already rule out."""

    include: tuple[str, ...] = ()
    exclude: tuple[tuple[str, str], ...] = ()
    #: Catalog column → categories of it whose parameters are kept; the rest are left out.
    include_by_catalog: tuple[tuple[str, tuple[str, ...]], ...] = ()
    #: Catalog column → categories of it whose parameters are left out.
    exclude_by_catalog: tuple[tuple[str, tuple[str, ...]], ...] = ()


@dataclass(frozen=True)
class Clustering:
    """Discovery is one setting; what is done about what was found is another."""

    action: str = "describe"
    method: str = "spearman"
    threshold: float = 0.7
    min_size: int = 2
    min_overlap: int = 10
    prefer: tuple[str, ...] = ()


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
    seed: int = 0
    batch_table: Path = Path("batch_table.xlsx")
    catalog: Path = Path("catalog.xlsx")
    output_root: Path = Path("runs")
    cohort: Cohort = field(default_factory=Cohort)
    target: Target = field(default_factory=Target)
    roles: Roles = field(default_factory=Roles)
    predictors: Predictors = field(default_factory=Predictors)
    encoding: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    clustering: Clustering = field(default_factory=Clustering)
    missingness: Missingness = field(default_factory=Missingness)
    report: Reporting = field(default_factory=Reporting)
    source: str = ""
    workspace: str = ""

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
        """The effective settings, defaults included, in the shape of ``task.yaml``."""
        d = _plain(asdict(self))
        predictors = d["predictors"]
        predictors["exclude"] = [{"column": c, "reason": r} for c, r in predictors["exclude"]]
        for key in ("include_by_catalog", "exclude_by_catalog"):
            predictors[key] = dict(predictors[key])
        return {
            "task": {key: d[key] for key in ("name", "question", "seed")},
            "inputs": {"batch_table": str(self.batch_table), "catalog": str(self.catalog)},
            "cohort": d["cohort"],
            "target": d["target"],
            "roles": d["roles"],
            "predictors": predictors,
            **{key: d[key] for key in ("encoding", "quality", "clustering", "missingness")},
            "report": d["report"],
        }


def _plain(value: Any) -> Any:
    """Tuples as lists, all the way down, so the settings can be written as YAML."""
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


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


def _categories(
    data: dict, key: str, where: str, errors: list[str]
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """``{catalog column: [category, ...]}`` — categories whose parameters are kept or left out."""
    value = data.get(key) or {}
    if not isinstance(value, dict):
        errors.append(f"{where}.{key}: expected {{catalog column: [categories]}}")
        return ()
    out = []
    for column, categories in value.items():
        if isinstance(categories, (str, int, float)):
            categories = [categories]
        if not isinstance(categories, list) or not categories:
            errors.append(f"{where}.{key}.{column}: expected a non-empty list of categories")
            continue
        out.append((str(column), tuple(str(c) for c in categories)))
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
MISSINGNESS_FIELDS = {
    "enabled": (bool, True),
    "threshold": (float, 0.9),
    "min_rate": (float, 0.05),
    "max_rate": (float, 0.95),
    "min_size": (int, 2),
    "min_strength": (float, 0.1),
    "max_categories": (int, 20),
}
REPORT_FIELDS = {"preview_rows": (int, 20), "max_table_rows": (int, 500)}
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
    "threshold": (0.0, 1.0),
    "min_rate": (0.0, 1.0),
    "max_rate": (0.0, 1.0),
    "min_size": (2, None),
    "min_strength": (0.0, 1.0),
    "max_categories": (2, None),
    "preview_rows": (0, None),
    "max_table_rows": (10, None),
}


def _knobs(
    data: dict, fields: dict[str, tuple[type, Any]], where: str, errors: list[str]
) -> dict[str, Any]:
    data = _known(data, tuple(fields), where, errors)
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

    ``defaults`` holds the ``batch_table`` and ``catalog`` paths from ``config.yaml``, used
    for whatever the task file itself leaves out.
    """
    errors: list[str] = []
    defaults = defaults or {}
    _known(
        raw,
        (
            "config",
            "task",
            "inputs",
            "cohort",
            "target",
            "roles",
            "predictors",
            "encoding",
            "quality",
            "clustering",
            "missingness",
            "report",
        ),
        "(root)",
        errors,
    )

    meta = _known(raw.get("task"), ("name", "question", "seed"), "task", errors)
    name = _text(meta, "name", "task", errors, "task")
    seed = _number(meta, "seed", "task", errors, 0, low=0, integer=True)

    inputs = _known(raw.get("inputs"), ("batch_table", "catalog"), "inputs", errors)

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

    roles_raw = _known(raw.get("roles"), ("id", "group", "patient"), "roles", errors)
    predictors_raw = _known(
        raw.get("predictors"),
        ("include", "exclude", "include_by_catalog", "exclude_by_catalog"),
        "predictors",
        errors,
    )
    roles = Roles(
        id=_text(roles_raw, "id", "roles", errors),
        group=_text(roles_raw, "group", "roles", errors),
        patient=_text(roles_raw, "patient", "roles", errors),
    )
    predictors = Predictors(
        include=_names(predictors_raw, "include", "predictors", errors),
        exclude=_reasoned(predictors_raw, "exclude", "predictors", errors),
        include_by_catalog=_categories(predictors_raw, "include_by_catalog", "predictors", errors),
        exclude_by_catalog=_categories(predictors_raw, "exclude_by_catalog", "predictors", errors),
    )

    encoding = _knobs(raw.get("encoding"), ENCODING_FIELDS, "encoding", errors)
    quality = _knobs(raw.get("quality"), QUALITY_FIELDS, "quality", errors)

    cluster_raw = _known(
        raw.get("clustering"),
        ("action", "method", "threshold", "min_size", "min_overlap", "prefer"),
        "clustering",
        errors,
    )
    clustering = Clustering(
        action=_choice(cluster_raw, "action", CLUSTER_ACTIONS, "clustering", errors, "describe"),
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

    missingness = Missingness(
        **_knobs(raw.get("missingness"), MISSINGNESS_FIELDS, "missingness", errors)
    )
    report = Reporting(**_knobs(raw.get("report"), REPORT_FIELDS, "report", errors))

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
            errors.append(f"roles: '{column}' is declared as {' and '.join(labels)} at once")
    for column in predictors.include:
        if column in roles_by_column:
            errors.append(
                f"predictors.include: '{column}' is already the "
                f"{roles_by_column[column][0]} column and cannot also be a predictor"
            )
        if column in dict(predictors.exclude):
            errors.append(f"predictors.include: '{column}' is in predictors.exclude as well")

    def located(section: dict, key: str, where: str, fallback: str, what: str) -> Path:
        text = _text(section, key, where, errors).strip()
        if text:
            return _resolve(text, base)
        if fallback:
            return Path(fallback)
        errors.append(
            f"{where}.{key}: {what} — set paths.{key} in config.yaml (found by --config, "
            "$ML_CONFIG or the nearest config.yaml), or name it here"
        )
        return Path(f"<{key}>")

    batch_table = located(
        inputs,
        "batch_table",
        "inputs",
        defaults.get("batch_table", ""),
        "the batch table this task reads",
    )
    catalog = located(
        inputs, "catalog", "inputs", defaults.get("catalog", ""), "the catalog this task reads"
    )
    if batch_table == catalog:
        errors.append("inputs: the batch table and the catalog cannot be the same file")
    if not name.strip():
        errors.append("task.name: a task needs a name")

    if errors:
        raise TaskConfigError(
            f"{source or 'task config'} cannot be run:\n  - " + "\n  - ".join(errors)
        )

    return TaskSpec(
        name=name,
        question=_text(meta, "question", "task", errors),
        seed=seed,
        batch_table=batch_table,
        catalog=catalog,
        output_root=base / "runs",
        cohort=cohort,
        target=target,
        roles=roles,
        predictors=predictors,
        encoding=encoding,
        quality=quality,
        clustering=clustering,
        missingness=missingness,
        report=report,
        source=source or str(base),
        workspace=str(base),
    )


def load(path: str | Path, config: dict[str, Any] | None = None) -> TaskSpec:
    """Read a task workspace's ``task.yaml`` (or a task file) and validate it.

    ``config`` is the ``config.yaml`` the run was started with, or else the one
    the task names under ``config:``; it supplies the batch table and catalog a task leaves out.
    """
    path, raw = _read(path)
    named = _config_path(path, raw)
    if config is None and named:
        try:
            config = load_config(named)
        except ConfigError as exc:
            raise TaskConfigError(f"{path}: config: {exc}") from exc
    return parse(raw, path.parent, source=str(path), defaults=config_defaults(config))


def config_file(task: str | Path | None) -> Path | None:
    """The ``config.yaml`` a task names under ``config:``, if ``task`` is a task at all."""
    if not task or not (Path(task).is_file() or (Path(task) / TASK_FILE).is_file()):
        return None
    return _config_path(*_read(task))


def _read(path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / TASK_FILE
    if not path.is_file():
        raise TaskConfigError(f"{path} does not exist")
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TaskConfigError(f"{path} is not a mapping")
    return path, raw


def _config_path(path: Path, raw: dict[str, Any]) -> Path | None:
    value = raw.get("config")
    if not value:
        return None
    if not isinstance(value, str):
        raise TaskConfigError(f"{path}: config: expected the path to a config.yaml")
    return _resolve(value, path.parent)


def config_defaults(config: dict[str, Any] | None) -> dict[str, str]:
    """The paths a task inherits from ``config.yaml``."""
    if not config:
        return {}
    paths = config.get("paths") or {}
    return {
        "batch_table": str(paths.get("batch_table") or ""),
        "catalog": str(paths.get("catalog") or ""),
    }
