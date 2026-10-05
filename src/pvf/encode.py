"""Turning PVF parameters into numbers a model can take.

Every encoder here is split in two: a ``fit`` that learns from a training
population and returns plain data, and an ``apply`` that uses what was learned on
any rows at all. That split is the whole point. A category mean learned from the
rows it is then used to score is not an encoding, it is the answer key — so the
learned part is fitted once, on training rows, and frozen before it touches a
validation row or a new batch.

Nothing here decides *which* encoding a column gets — :func:`route_categoricals`
does that, and :mod:`pvf.dataset` applies it — so an encoder can be read on its
own.

The rule the routing follows: a parameter with two categories is 0/1, a
parameter with few categories becomes one column per category, a parameter with
many becomes the target's mean per category, and a parameter with very many is
hashed into a fixed number of buckets. A parameter with nearly as many
categories as there are batches is an identifier wearing a category's clothes,
and is left out.

Every state a fit returns is JSON-encodable, because the run writes it out next
to the dataset: a recipe someone can read and apply again is worth more than a
pickled object nobody can open.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from .logger import log

MODULE = "encode"

#: Values that mean "yes" when a two-category column has to pick which side is 1.
YES = frozenset({"yes", "y", "true"})

#: The label a missing value is encoded under when the task asks for that.
MISSING_LABEL = "(missing)"

#: One row of the feature dictionary: what a column is, in the report's terms.
Record = dict[str, str]


@dataclass(frozen=True)
class TaskParams:
    """The encoder, quality and clustering knobs, all of them from the task YAML.

    Defaults here are the ones the YAML validation fills in, so a setting left
    out of a task file still has one value and one meaning rather than two.
    """

    one_hot: bool = True
    target_encoding: bool = True
    feature_hashing: bool = True
    one_hot_top_x: int = 10
    onehot_max_categories: int = 15
    target_max_categories: int = 50
    max_cardinality_ratio: float = 0.5
    target_encoding_smoothing: float = 10.0
    target_encoding_folds: int = 5
    #: What happens where target encoding cannot run: "hashing" or "stop".
    target_encoding_fallback: str = "hashing"
    hashing_min_buckets: int = 8
    hashing_max_buckets: int = 32
    hashing_signed: bool = True
    #: "missing" leaves an absent value absent; "category" encodes it as one.
    missing_category: str = "missing"

    cluster_corr_method: str = "spearman"
    cluster_corr_threshold: float = 0.7
    cluster_min_size: int = 2
    cluster_min_overlap: int = 10

    min_non_missing: int = 20
    duplicate_r2_threshold: float = 0.99
    duplicate_min_overlap: int = 10
    seed: int = 0

    @property
    def enabled_encoders(self) -> dict[str, bool]:
        return {
            "one_hot": self.one_hot,
            "target": self.target_encoding,
            "hashing": self.feature_hashing,
        }

    @property
    def encodes_missing(self) -> bool:
        return self.missing_category == "category"


def sanitise(name: str) -> str:
    """A column name safe to build other names out of."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_") or "col"


def unique_name(base: str, taken: set[str]) -> str:
    """``base``, or the first ``base__2``, ``base__3`` … nothing has claimed yet.

    Two parameters can sanitise to the same thing, a category can be called
    ``Other``, and a one-hot prefix can collide with a real column name. Every
    output name in the dataset goes through here, so a collision costs a suffix
    rather than a silently overwritten column.
    """
    if base not in taken:
        taken.add(base)
        return base
    for suffix in range(2, 1000):
        candidate = f"{base}__{suffix}"
        if candidate not in taken:
            taken.add(candidate)
            log.info(MODULE, f"'{base}' was already taken — this one is '{candidate}'")
            return candidate
    raise ValueError(f"Cannot find a free name for '{base}'")


def _record(output: str, source: str, strategy: str, detail: str) -> Record:
    return {"output": output, "source": source, "strategy": strategy, "detail": detail}


def _labels(series: pd.Series, params: TaskParams) -> pd.Series:
    """One column's values as the strings the encoders key on.

    Missing stays missing unless the task asked for absence to be a category of
    its own, in which case it gets one label and keeps it everywhere.
    """
    text = series.astype("string").str.strip()
    if params.encodes_missing:
        return text.fillna(MISSING_LABEL)
    return text


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
@dataclass
class Route:
    """Which encoder a categorical parameter was sent to, and why."""

    column: str
    categories: int
    ratio: float
    strategy: str
    reason: str


def route_categoricals(df: pd.DataFrame, columns: list[str], params: TaskParams) -> list[Route]:
    """Send each categorical parameter to exactly one encoder.

    Cardinality decides, and the ratio of categories to rows vetoes: a column
    with a category for nearly every batch carries the batch's identity rather
    than a property of it. Identifiers are declared in the task rather than
    detected here; this catches the ones nobody declared.
    """
    rows = len(df)
    enabled = params.enabled_encoders
    routes: list[Route] = []

    for column in columns:
        values = _labels(df[column], params)
        k = int(values.nunique(dropna=True))
        ratio = k / rows if rows else 0.0

        if k <= 1:
            routes.append(Route(column, k, ratio, "skip", "one category, or none"))
            continue
        if k == 2:
            routes.append(Route(column, k, ratio, "binary", "two categories"))
            continue
        if ratio > params.max_cardinality_ratio:
            routes.append(
                Route(
                    column,
                    k,
                    ratio,
                    "skip",
                    f"{k} categories over {rows} batches (k/n {ratio:.2f} > "
                    f"{params.max_cardinality_ratio}) — an identifier, not a category",
                )
            )
            continue

        if k <= params.onehot_max_categories:
            preference, band = ["one_hot", "target", "hashing"], "few"
        elif k <= params.target_max_categories:
            preference, band = ["target", "hashing", "one_hot"], "several"
        else:
            preference, band = ["hashing", "target"], "many"

        chosen = next((name for name in preference if enabled.get(name)), None)
        if chosen is None:
            routes.append(
                Route(
                    column,
                    k,
                    ratio,
                    "skip",
                    f"{band} categories (k={k}), and every encoder that suits them is switched off",
                )
            )
        else:
            routes.append(
                Route(
                    column,
                    k,
                    ratio,
                    chosen,
                    f"{band} categories (k={k}); preference {' > '.join(preference)}",
                )
            )
    return routes


# ---------------------------------------------------------------------------
# Binary
# ---------------------------------------------------------------------------
def fit_binary(
    df: pd.DataFrame, columns: list[str], params: TaskParams, taken: set[str]
) -> dict[str, dict]:
    """Which of a two-category parameter's values is 1, decided on training rows."""
    state: dict[str, dict] = {}
    for column in columns:
        values = sorted(_labels(df[column], params).dropna().unique(), key=str)
        if len(values) < 2:
            continue
        positive = next((v for v in values if str(v).lower() in YES), values[1])
        negative = next(v for v in values if v != positive)
        state[column] = {
            "name": unique_name(f"{column}_{positive}", taken),
            "positive": str(positive),
            "negative": str(negative),
        }
        log.info(MODULE, f"{column}: '{positive}' → 1, '{negative}' → 0")
    return state


def apply_binary(
    df: pd.DataFrame, state: dict[str, dict], params: TaskParams
) -> tuple[pd.DataFrame, list[Record]]:
    """Apply a fitted binary mapping. A value neither side names is left missing."""
    encoded: dict[str, pd.Series] = {}
    records: list[Record] = []
    for column, fitted in state.items():
        if column not in df.columns:
            continue
        mapping = {fitted["negative"]: 0, fitted["positive"]: 1}
        encoded[fitted["name"]] = _labels(df[column], params).map(mapping).astype("Int64")
        records.append(
            _record(
                fitted["name"],
                column,
                "binary",
                f"1 where '{fitted['positive']}', 0 where '{fitted['negative']}'; "
                "any other value is left missing",
            )
        )
    return pd.DataFrame(encoded, index=df.index), records


# ---------------------------------------------------------------------------
# One-hot
# ---------------------------------------------------------------------------
def fit_one_hot(
    df: pd.DataFrame, columns: list[str], params: TaskParams, taken: set[str]
) -> tuple[dict[str, dict], list[dict]]:
    """The categories that get a column each, and the label the tail goes under.

    The vocabulary is fixed here and never grows afterwards: a category that only
    appears in validation rows falls into the tail, which is what an unseen
    category is.
    """
    state: dict[str, dict] = {}
    tail: list[dict] = []

    for column in columns:
        values = _labels(df[column], params)
        counts = values.value_counts()
        kept = list(counts.nlargest(params.one_hot_top_x).index)
        # A real category called "Other" would be indistinguishable from the
        # tail, so the tail moves out of its way rather than merging with it.
        other = "Other"
        while other in set(counts.index):
            other += " (rare)"
        in_tail = (~values.isin(kept)) & values.notna()
        outputs = {
            category: unique_name(f"{column}_{category}", taken) for category in [*kept, other]
        }
        state[column] = {"categories": [str(c) for c in kept], "other": other, "outputs": outputs}
        tail.append(
            {
                "Parameter": column,
                "Categories kept": len(kept),
                "Batches in Other": int(in_tail.sum()),
                "Other %": round(100 * in_tail.sum() / len(df), 1) if len(df) else 0.0,
            }
        )
    return state, tail


def apply_one_hot(
    df: pd.DataFrame, state: dict[str, dict], params: TaskParams
) -> tuple[pd.DataFrame, list[Record]]:
    encoded: dict[str, pd.Series] = {}
    records: list[Record] = []

    for column, fitted in state.items():
        if column not in df.columns:
            continue
        values = _labels(df[column], params)
        present = values.notna()
        for category in fitted["categories"]:
            name = fitted["outputs"][category]
            encoded[name] = ((values == category) & present).astype(int)
            records.append(_record(name, column, "one-hot", f"1 where '{column}' is '{category}'"))
        other_name = fitted["outputs"][fitted["other"]]
        encoded[other_name] = (~values.isin(fitted["categories"]) & present).astype(int)
        records.append(
            _record(
                other_name,
                column,
                "one-hot",
                "1 where the value is outside the categories kept from the training rows, "
                "which includes a category first seen here",
            )
        )
    return pd.DataFrame(encoded, index=df.index), records


# ---------------------------------------------------------------------------
# Target encoding
# ---------------------------------------------------------------------------
def _smoothed_means(
    categories: pd.Series, y: pd.Series, prior: float, m: float
) -> tuple[pd.Series, pd.Series]:
    stats = pd.DataFrame({"c": categories, "y": y}).dropna()
    grouped = stats.groupby("c")["y"].agg(["count", "mean"])
    means = (grouped["count"] * grouped["mean"] + m * prior) / (grouped["count"] + m)
    return means, grouped["count"]


def folds(
    index: pd.Index,
    k: int,
    seed: int,
    groups: pd.Series | None = None,
    order: pd.Series | None = None,
) -> tuple[list[tuple[pd.Index, pd.Index]], str, str]:
    """The (training rows, rows to encode) pairs one cross-fitted encoding uses.

    Three strategies, and the task picks which by what it declared. Grouped folds
    keep every row of a group on one side, so a category that is really the group
    cannot be learned from its own siblings. Chronological folds only ever look
    backwards: the earliest block has nothing before it and is left unencoded
    rather than encoded from its own future. Otherwise, plain shuffled folds.

    Returns the pairs, the strategy used, and a note about anything it had to
    give up.
    """
    n = len(index)
    if groups is not None:
        distinct = int(groups.nunique(dropna=True))
        if distinct >= 2:
            usable = min(k, distinct)
            note = "" if usable == k else f"{distinct} groups, so {usable} folds rather than {k}"
            splitter = GroupKFold(n_splits=usable)
            codes = groups.astype("string").fillna(MISSING_LABEL)
            pairs = [
                (index[train], index[test])
                for train, test in splitter.split(np.arange(n), groups=codes)
            ]
            return pairs, "grouped", note
        return (
            [],
            "grouped",
            f"only {distinct} distinct group(s) — grouped folds need at least two",
        )

    if order is not None:
        ranks = order.rank(method="first").to_numpy()
        blocks = np.array_split(np.argsort(ranks), min(k, n))
        pairs = []
        seen: list[int] = []
        for position, block in enumerate(blocks):
            if position and seen:
                pairs.append((index[np.asarray(seen)], index[block]))
            seen.extend(block.tolist())
        note = "the earliest block has nothing before it and is left unencoded"
        return pairs, "chronological", note

    splitter = KFold(n_splits=min(k, n), shuffle=True, random_state=seed)
    pairs = [(index[train], index[test]) for train, test in splitter.split(np.arange(n))]
    return pairs, "shuffled", ""


def fit_target(
    df: pd.DataFrame,
    columns: list[str],
    target: pd.Series,
    params: TaskParams,
    taken: set[str],
) -> tuple[dict[str, Any], list[str]]:
    """The category means the *training* rows support, and the prior behind them.

    These are what a validation row or a new batch is encoded with. The training
    rows themselves are encoded by :func:`crossfit_target`, which refits this on
    each fold's own training part — being encoded by a mean one is inside is the
    leak this whole module is arranged against.
    """
    y = pd.to_numeric(target.reindex(df.index), errors="coerce")
    usable = y.notna()
    if int(usable.sum()) < max(params.target_encoding_folds, 2):
        return (
            {
                "batches": int(usable.sum()),
                "folds": params.target_encoding_folds,
                "smoothing": params.target_encoding_smoothing,
                "prior": float("nan"),
                "columns": {},
            },
            list(columns),
        )

    prior = float(y[usable].mean())
    state: dict[str, Any] = {
        "batches": int(usable.sum()),
        "folds": params.target_encoding_folds,
        "smoothing": params.target_encoding_smoothing,
        "prior": prior,
        "columns": {},
    }
    for column in columns:
        means, counts = _smoothed_means(
            _labels(df.loc[usable, column], params),
            y[usable],
            prior,
            params.target_encoding_smoothing,
        )
        state["columns"][column] = {
            "name": unique_name(f"{sanitise(column)}_target_enc", taken),
            "means": {str(k): float(v) for k, v in means.items()},
            "counts": {str(k): int(v) for k, v in counts.items()},
        }
        log.info(MODULE, f"{column} → {state['columns'][column]['name']}")
    return state, []


def apply_target(
    df: pd.DataFrame, state: dict[str, Any], params: TaskParams
) -> tuple[pd.DataFrame, list[Record]]:
    """Encode any rows with the fitted means. Unseen categories get the prior."""
    encoded: dict[str, pd.Series] = {}
    records: list[Record] = []
    prior = state.get("prior", float("nan"))

    for column, fitted in state.get("columns", {}).items():
        if column not in df.columns:
            continue
        values = _labels(df[column], params)
        out = values.map(fitted["means"]).astype("float64")
        out = out.where(values.isna(), out.fillna(prior))
        encoded[fitted["name"]] = out
        records.append(
            _record(
                fitted["name"],
                column,
                "target encoding",
                f"mean of the target per category, smoothed with m="
                f"{state['smoothing']:g} towards the training prior {prior:.3f}; "
                "a category the training rows never held gets that prior, and a "
                "missing value stays missing",
            )
        )
    return pd.DataFrame(encoded, index=df.index), records


def crossfit_target(
    df: pd.DataFrame,
    state: dict[str, Any],
    target: pd.Series,
    params: TaskParams,
    groups: pd.Series | None = None,
    order: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The training rows' own encoding, each fold computed without its own labels.

    Every fold's prior and category means come from that fold's training rows
    only, which is the part the old implementation got wrong: a prior taken from
    the whole cohort carries the held-out rows' targets into their own encoding.
    """
    columns = state.get("columns", {})
    out = pd.DataFrame(
        {fitted["name"]: pd.Series(np.nan, index=df.index) for fitted in columns.values()},
        index=df.index,
    )
    y = pd.to_numeric(target.reindex(df.index), errors="coerce")
    labelled = y.index[y.notna()]
    info = {
        "strategy": "",
        "note": "",
        "batches": int(len(labelled)),
        "folds": state.get("folds"),
        "smoothing": state.get("smoothing"),
        "prior": state.get("prior"),
        # How often a fold had never seen the category it was asked to encode.
        # High here means the category is confounded with whatever the folds
        # respect — a column that only ever gets the prior carries nothing.
        "unseen": {},
    }
    if not len(columns) or not len(labelled):
        return out, info

    pairs, strategy, note = folds(
        labelled,
        int(state.get("folds") or params.target_encoding_folds),
        params.seed,
        groups=groups.loc[labelled] if groups is not None else None,
        order=order.loc[labelled] if order is not None else None,
    )
    info["strategy"], info["note"] = strategy, note
    if not pairs:
        info["note"] = note or "no usable folds"
        return out, info

    unseen = {fitted["name"]: 0 for fitted in columns.values()}
    for train_idx, test_idx in pairs:
        fold_prior = float(y.loc[train_idx].mean())
        for column, fitted in columns.items():
            means, _ = _smoothed_means(
                _labels(df.loc[train_idx, column], params),
                y.loc[train_idx],
                fold_prior,
                params.target_encoding_smoothing,
            )
            values = _labels(df.loc[test_idx, column], params)
            mapped = values.map(means).astype("float64")
            unseen[fitted["name"]] += int((mapped.isna() & values.notna()).sum())
            out.loc[test_idx, fitted["name"]] = mapped.where(
                values.isna(), mapped.fillna(fold_prior)
            )
    info["unseen"] = unseen
    for name, count in unseen.items():
        if count >= 0.5 * len(labelled):
            log.warn(
                MODULE,
                f"{name}: {count} of {len(labelled)} rows were encoded with their fold's prior "
                "because the category was not in that fold's training rows — the category is "
                "largely confounded with what the folds respect",
            )
    return out, info


# ---------------------------------------------------------------------------
# Feature hashing
# ---------------------------------------------------------------------------
def _bucket_and_sign(value: str, column: str, buckets: int) -> tuple[int, int]:
    """Deterministic, and deliberately not seeded: the same category has to land
    in the same bucket in every run, including a run that scores new batches."""
    digest = hashlib.md5(f"{column}\x00{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % buckets, 1 if digest[8] % 2 == 0 else -1


def fit_hashed(
    df: pd.DataFrame, columns: list[str], params: TaskParams, taken: set[str]
) -> tuple[dict[str, dict], list[dict]]:
    """How wide each hashed column is. The bucket a value lands in needs no fit."""
    state: dict[str, dict] = {}
    summary: list[dict] = []
    for column in columns:
        k = int(_labels(df[column], params).nunique(dropna=True))
        width = int(
            min(
                params.hashing_max_buckets,
                max(params.hashing_min_buckets, 1 << max(k - 1, 0).bit_length()),
            )
        )
        base = sanitise(column)
        state[column] = {
            "width": width,
            "categories": k,
            "outputs": [unique_name(f"{base}_hash_{i:02d}", taken) for i in range(width)],
        }
        summary.append({"Parameter": column, "Categories": k, "Buckets": width})
        log.info(MODULE, f"{column}: {k} categories → {width} buckets")
    return state, summary


def apply_hashed(
    df: pd.DataFrame, state: dict[str, dict], params: TaskParams
) -> tuple[pd.DataFrame, list[Record]]:
    """Very many categories into a fixed number of buckets.

    Two categories can share a bucket, and that is the trade: a fixed width
    whatever the cardinality, at the price of occasionally adding two categories
    together. Signed hashing makes those collisions cancel rather than accumulate.
    """
    frames, records = [], []
    for column, fitted in state.items():
        if column not in df.columns:
            continue
        width = fitted["width"]
        matrix = np.zeros((len(df), width))
        values = _labels(df[column], params)
        for row, value in enumerate(values.to_numpy(dtype=object)):
            if pd.isna(value):
                continue
            bucket, sign = _bucket_and_sign(str(value), column, width)
            matrix[row, bucket] = sign if params.hashing_signed else 1.0

        frame = pd.DataFrame(matrix, index=df.index, columns=fitted["outputs"])
        frames.append(frame)
        for name in fitted["outputs"]:
            records.append(
                _record(
                    name,
                    column,
                    "feature hashing",
                    f"{'signed ' if params.hashing_signed else ''}MD5 hash of '{column}' into "
                    f"{width} buckets (k={fitted['categories']}); two categories may share a "
                    "bucket, and a missing value is all zeroes",
                )
            )
    encoded = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=df.index)
    return encoded, records


# ---------------------------------------------------------------------------
# Ordinal
# ---------------------------------------------------------------------------
def category_order(value: object) -> list[str] | None:
    """The ordered categories a PTF value type spells out, or ``None``.

    An ordinal parameter's value type is a bracketed list — ``[Low, Medium,
    High]`` — and the order in it is the whole point: it is what makes the
    encoding an integer rather than a set of columns.
    """
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list) and parsed:
            return [str(item).strip() for item in parsed]
    except (SyntaxError, ValueError):
        pass
    items = [re.sub(r"^['\"]|['\"]$", "", part.strip()) for part in text[1:-1].split(",")]
    kept = [item for item in items if item]
    if not kept:
        log.warn(MODULE, f"Value type {text!r} looks ordinal but lists nothing")
    return kept or None


def fit_ordinal(ptf: pd.DataFrame, columns: list[str], taken: set[str]) -> dict[str, dict]:
    """The PTF's own category order, which is not learned from the data at all."""
    state: dict[str, dict] = {}
    for _, row in ptf.iterrows():
        parameter = str(row.get("Parameter"))
        categories = category_order(row.get("Value Type"))
        if categories is None or parameter not in columns:
            continue
        state[parameter] = {
            "categories": categories,
            "name": unique_name(parameter, taken),
        }
    return state


def apply_ordinal(
    df: pd.DataFrame, state: dict[str, dict]
) -> tuple[pd.DataFrame, pd.DataFrame, list[Record]]:
    """Ordered categories as their rank, with anything unlisted left missing."""
    encoded: dict[str, pd.Series] = {}
    unexpected: list[dict] = []
    records: list[Record] = []

    for parameter, fitted in state.items():
        if parameter not in df.columns:
            continue
        categories = fitted["categories"]
        values = df[parameter].astype("string").str.strip()
        outside = values.dropna()[~values.dropna().isin(categories)]
        for value, count in outside.value_counts().items():
            unexpected.append({"Parameter": parameter, "Value": value, "Batches": int(count)})
            log.warn(
                MODULE,
                f"{parameter}: '{value}' is not one of its PTF categories "
                f"({count} batches) — left missing",
            )
        ranks = {category: rank for rank, category in enumerate(categories, start=1)}
        encoded[fitted["name"]] = values.map(ranks).astype("Int64")
        records.append(
            _record(
                fitted["name"],
                parameter,
                "ordinal",
                f"rank 1..{len(categories)} in the PTF's order: " + ", ".join(categories),
            )
        )

    return (
        pd.DataFrame(encoded, index=df.index),
        pd.DataFrame(unexpected, columns=["Parameter", "Value", "Batches"]),
        records,
    )


# ---------------------------------------------------------------------------
# Numeric hygiene
# ---------------------------------------------------------------------------
def to_numeric(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    """Numeric parameters as floats, with what could not be read reported.

    Infinities go with them: an infinity is a division that should have been
    missing, and it survives a model fit and a plot axis as though it were a
    number.
    """
    converted: dict[str, pd.Series] = {}
    notes: list[dict] = []
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce")
        failed = int((df[column].notna() & values.isna()).sum())
        infinite = int(np.isinf(values.to_numpy(dtype="float64", na_value=np.nan)).sum())
        converted[column] = values.replace([np.inf, -np.inf], np.nan)
        if failed or infinite:
            notes.append({"Parameter": column, "Unreadable": failed, "Infinite": infinite})
            log.info(
                MODULE,
                f"{column}: {failed} values could not be read as numbers, "
                f"{infinite} were infinite — all missing now",
            )
    frame = pd.DataFrame(converted, index=df.index) if converted else pd.DataFrame(index=df.index)
    return frame, notes


def drop_nonpositive(df: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    """A duration of zero or less is a recording error, not a short step."""
    dropped: dict[str, int] = {}
    for column in columns:
        mask = df[column] <= 0
        if int(mask.sum()):
            df.loc[mask, column] = np.nan
            dropped[column] = int(mask.sum())
            log.info(MODULE, f"{column}: {int(mask.sum())} values at or below zero → missing")
    return dropped


def pairwise(a: pd.Series, b: pd.Series, method: str = "pearson") -> tuple[float, int]:
    """One correlation and the number of rows both columns actually have.

    The count is half the answer. Two columns that overlap on four batches have
    no established relationship whatever the coefficient says, and a caller that
    only reads the coefficient cannot tell that apart from a real zero.
    """
    mask = a.notna() & b.notna()
    overlap = int(mask.sum())
    if overlap < 3:
        return float("nan"), overlap
    x, y = a[mask], b[mask]
    if x.nunique() < 2 or y.nunique() < 2:
        return float("nan"), overlap
    return float(x.corr(y, method=method)), overlap


def duplicate_selection(
    df: pd.DataFrame, columns: list[str], params: TaskParams, prefer: tuple[str, ...] = ()
) -> tuple[list[str], list[dict]]:
    """Which of the parameters that say the same thing is the one to keep.

    A deterministic sweep rather than a list of pairs. Candidates are considered
    most-complete first, with a task's own preference and then the name breaking
    ties, and each is either kept or dropped against something already kept — so
    a dropped parameter always names a representative that is still in the
    dataset, and C is never dropped for resembling B when B itself was dropped.
    """
    coverage = {column: int(df[column].notna().sum()) for column in columns}
    order = sorted(columns, key=lambda c: (-coverage[c], 0 if c in prefer else 1, str(c)))

    kept: list[str] = []
    decisions: list[dict] = []
    for candidate in order:
        match = None
        for representative in kept:
            r, overlap = pairwise(df[representative], df[candidate])
            if overlap < params.duplicate_min_overlap or not np.isfinite(r):
                continue
            if r**2 >= params.duplicate_r2_threshold:
                match = (representative, r**2, overlap)
                break
        if match is None:
            kept.append(candidate)
            continue
        representative, r2, overlap = match
        decisions.append(
            {
                "Dropped": candidate,
                "Kept": representative,
                "R2": round(r2, 4),
                "Shared batches": overlap,
                "Coverage": coverage[candidate],
            }
        )
        log.info(
            MODULE,
            f"{candidate} duplicates {representative} (R²={r2:.4f} over {overlap} batches) "
            "— dropped",
        )
    return kept, decisions
