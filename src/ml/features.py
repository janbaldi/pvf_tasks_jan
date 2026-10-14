"""Features calculated from parameters the batch table already holds.

A *parameter* arrives from a source and is cleaned or harmonised on the way in;
that work lives in :mod:`ml.io`, :mod:`ml.clean` and :mod:`ml.enrich`. A
*feature* is arithmetic over one or more parameters, and every one of them lives
here, in a single ordered registry.

One entry per feature. To add one, append a :class:`Feature` — no other file
changes. Entries run in list order, so a feature may require a column an earlier
entry produced, which is how the normalised harvest lactate reaches the growth
profile and the per-cell metabolite readings reach their D6-D10 slope without a
dependency graph.

No column is named here: every name is a key (``col("vector_titer")``) that the
vocabulary resolves (:mod:`ml.names`), and so are the feature names themselves.

A feature is computed only when its name is a catalog parameter. A feature that is
not in the catalog is reported, not created: the catalog is the schema, and a column the
schema does not know about has nowhere to go. The report lists those entries with
everything the catalog needs to accept them.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .logger import log
from .names import col, cols, values

MODULE = "features"

#: A requirement is a column name, or a tuple of names of which any one will do.
Requirement = str | tuple[str, ...]


@dataclass(frozen=True)
class Feature:
    """One derived column: what it needs, how it is computed, and why it exists."""

    name: str
    requires: tuple[Requirement, ...]
    compute: Callable[[pd.DataFrame], Any]
    group: str
    process_day: str
    value_type: str
    rationale: str


@dataclass(frozen=True)
class Params:
    """The numbers the features are calibrated against, all from config.

    Setpoints and thresholds are process decisions, not properties of the data —
    they belong in ``config/config.yaml`` where a process change can move them
    without touching this file.
    """

    co2_target: float = 5.0
    temp_target: float = 37.0
    contact_target: float = 120.0
    contact_tolerance: float = 30.0
    seeding_tolerance: float = 5.0
    cpdl_low: float = 4.0
    cpdl_high: float = 5.0
    lactate_low: float = 12.5
    lactate_high: float = 30.0
    cd4_low: float = 40.0
    cd4_high: float = 60.0


@dataclass
class FeatureReport:
    """What the registry did to one site's frame.

    ``entries`` is every registry entry with the status it ended in, which is the
    single table the report reads. The counts are derived from it rather than
    tallied separately, so they cannot disagree with it.
    """

    site: str = ""
    rows: int = 0
    entries: list[dict] = field(default_factory=list)
    profiles: dict[str, dict] = field(default_factory=dict)

    def count(self, status: str) -> int:
        return sum(1 for e in self.entries if e["status"] == status)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _num(series: Any) -> pd.Series:
    """Coerce to float, with anything unparseable becoming NaN."""
    return pd.to_numeric(series, errors="coerce")


def safe_div(numerator: Any, denominator: Any) -> pd.Series:
    """Divide, with division by zero giving NaN rather than an infinity.

    An infinity survives into a model fit and a plot axis as a number; NaN is
    handled as the missing value it actually is.
    """
    out = _num(numerator) / _num(denominator)
    return out.replace([np.inf, -np.inf], np.nan)


def _first(df: pd.DataFrame, *candidates: str) -> pd.Series:
    """The first of these columns that the frame has, as numbers."""
    for name in candidates:
        if name in df.columns:
            return _num(df[name])
    return pd.Series(np.nan, index=df.index)


def _present(df: pd.DataFrame, names: Sequence[str]) -> list[str]:
    return [name for name in names if name in df.columns]


def _row_sum(df: pd.DataFrame, names: Sequence[str]) -> pd.Series:
    """Sum across columns, counting a row that has at least one value.

    A batch seeded into one G-Rex leaves the second vessel's column empty. Plain
    addition would make its pooled total missing, which reads as "not measured"
    rather than "there was no second vessel".
    """
    columns = _present(df, names)
    if not columns:
        return pd.Series(np.nan, index=df.index)
    return df[columns].apply(_num).sum(axis=1, min_count=1)


def yn_to_bool(series: pd.Series) -> pd.Series:
    """Yes/no-ish text as 1.0 / 0.0.

    The negatives are enumerated and everything else that is present counts as a
    positive, because these columns record an observation ("no clumps", "few
    small") rather than a checkbox.
    """
    negative = {"n", "no", "false", "f", "0", "none", "no clumps", "absent"}

    def convert(value: Any) -> float:
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float, np.number)):
            return 1.0 if value != 0 else 0.0
        return 0.0 if str(value).strip().lower() in negative else 1.0

    return series.apply(convert)


def zscore(series: pd.Series) -> pd.Series:
    """Standardise within the frame this is called on — one site, not the batch table."""
    values = _num(series)
    spread = values.std(ddof=0)
    if not spread:
        return pd.Series(np.nan, index=series.index)
    return (values - values.mean()) / spread


def _readings(
    df: pd.DataFrame, days: Sequence[float], columns: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """The day-ordered readings each batch has, and the days they were taken on.

    A day whose column the site does not record drops out with its column. A site
    that measures two of the three days still has a slope through two points; it
    is the pairing that has to survive, not the full set.
    """
    pairs = [(day, column) for day, column in zip(days, columns) if column in df.columns]
    if not pairs:
        return np.empty((len(df), 0)), np.empty(0)
    kept_days, kept_columns = zip(*pairs)
    return (
        df[list(kept_columns)].apply(_num).to_numpy(dtype=float),
        np.asarray(kept_days, dtype=float),
    )


def slope(df: pd.DataFrame, days: Sequence[float], columns: Sequence[str]) -> pd.Series:
    """Least-squares slope per row over the days that have a value.

    Rows with fewer than two readings have no slope, and NaN says so.
    """
    values, x = _readings(df, days, columns)
    if x.size < 2:
        return pd.Series(np.nan, index=df.index)
    mask = ~np.isnan(values)
    n = mask.sum(axis=1)
    xs = np.where(mask, x, np.nan)
    x_mean = np.nanmean(xs, axis=1)
    y_mean = np.nanmean(values, axis=1)
    dx = xs - x_mean[:, None]
    dy = values - y_mean[:, None]
    variance = np.nansum(dx * dx, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.nansum(dx * dy, axis=1) / variance
    return pd.Series(np.where((n >= 2) & (variance > 0), out, np.nan), index=df.index)


def auc(df: pd.DataFrame, days: Sequence[float], columns: Sequence[str]) -> pd.Series:
    """Trapezoidal area per row, over the days that have a value.

    A gap in the middle is bridged rather than treated as zero: the trapezium
    simply spans from the last reading to the next one.
    """
    values, x = _readings(df, days, columns)
    if x.size < 2:
        return pd.Series(np.nan, index=df.index)
    totals = np.full(len(df), np.nan)
    for i, row in enumerate(values):
        mask = ~np.isnan(row)
        if mask.sum() >= 2:
            xs, ys = x[mask], row[mask]
            totals[i] = float(np.sum(np.diff(xs) * (ys[:-1] + ys[1:]) / 2))
    return pd.Series(totals, index=df.index)


def process_day_of(column: str) -> str:
    """The process day a column's name announces, or "Context" when it names none."""
    match = re.search(r"\bday\s*(\d+)", column, flags=re.IGNORECASE)
    if match:
        return f"D{match.group(1)}"
    return next((tag for tag in ("D0", "D1", "D3", "D6", "D8", "D10") if tag in column), "Context")


# ---------------------------------------------------------------------------
# Column names used by more than one feature
# ---------------------------------------------------------------------------
def _coa(key: str) -> str:
    """A certificate column as the batch table carries it, with its prefix."""
    return col("coa_prefix") + col(key)


def _titer() -> tuple[str, str]:
    """The measured vector titer: the certificate's, else the batch record's."""
    return (_coa("coa_titer"), col("vector_titer"))


def _impurities() -> tuple[str, ...]:
    return tuple(col("coa_prefix") + name for name in cols("coa_impurities"))


METABOLITE_DAYS = (6.0, 8.0, 10.0)


def _pooled_vector_volume(df: pd.DataFrame) -> pd.Series:
    return _row_sum(df, (col("vector_volume_a"), col("vector_volume_b")))


def _pooled_seeded_cells(df: pd.DataFrame) -> pd.Series:
    """Cells the vessels were seeded with: the recorded total, else the vessels."""
    if col("cells_available") in df.columns:
        return _num(df[col("cells_available")])
    return _row_sum(df, (col("cells_seeded_a"), col("cells_seeded_b")))


# ---------------------------------------------------------------------------
# Feature families — written once, expanded into entries
# ---------------------------------------------------------------------------
def _spread_features(key: str, column_a: str, column_b: str, what: str) -> list[Feature]:
    """Mean, extremes, spread and CV of one quantity across the two vessels.

    Five entries from one description, because a vessel pair is measured the same
    way whatever is in it.
    """
    pair = (column_a, column_b)

    def values(df: pd.DataFrame) -> pd.DataFrame:
        return df[list(pair)].apply(_num)

    statistics: list[tuple[str, Callable[[pd.DataFrame], pd.Series], str]] = [
        ("mean", lambda d: values(d).mean(axis=1), f"Mean {what} across vessels A and B"),
        ("min", lambda d: values(d).min(axis=1), f"Lower of the two vessels' {what}"),
        ("max", lambda d: values(d).max(axis=1), f"Higher of the two vessels' {what}"),
        ("std", lambda d: values(d).std(axis=1, ddof=0), f"Spread of {what} between vessels"),
        (
            "cv",
            lambda d: safe_div(
                values(d).std(axis=1, ddof=0), values(d).mean(axis=1).replace(0, np.nan)
            ),
            f"Spread of {what} relative to its mean, so vessel imbalance is "
            "comparable across batches of different size",
        ),
    ]
    return [
        Feature(
            name=col(key, statistic=statistic),
            requires=pair,
            compute=compute,
            group="Vessel balance",
            process_day="D3",
            value_type="numeric",
            rationale=rationale,
        )
        for statistic, compute, rationale in statistics
    ]


def _setpoint_features(columns: Sequence[str], params: Params) -> list[Feature]:
    """One deviation feature per environmental column the frame happens to carry.

    The batch records name a CO2 or temperature column per reading and per vessel,
    and the set differs between sites, so these are read off the columns (by the
    vocabulary's markers) rather than listed.
    """
    features: list[Feature] = []
    for column in columns:
        if col("co2_marker") in column:
            target, suffix, unit = params.co2_target, f"dev_from_{params.co2_target}pct", "CO2"
        elif col("temperature_marker") in column:
            target, suffix, unit = (
                params.temp_target,
                f"dev_from_{params.temp_target}C",
                "temperature",
            )
        else:
            continue
        features.append(
            Feature(
                name=f"{column} {suffix}",
                requires=(column,),
                compute=lambda d, c=column, t=target: (_num(d[c]) - t).abs(),
                group="Environment",
                process_day=process_day_of(column),
                value_type="numeric",
                rationale=f"Distance from the {unit} setpoint, as a measure of how "
                "closely the incubator held it",
            )
        )
    return features


def _contact_time_features(column: str, which: int, params: Params) -> list[Feature]:
    """Deviation from, and compliance with, the transduction contact time.

    ``which`` is 0 for the batch's own reading, 1 and 2 for vessels A and B: the
    position of the feature names in the ``f_contact_deviation`` and
    ``f_contact_within`` lists.
    """
    low = params.contact_target - params.contact_tolerance
    high = params.contact_target + params.contact_tolerance
    label = ("", "vessel A", "vessel B")[which]
    return [
        Feature(
            name=cols("f_contact_deviation")[which],
            requires=(column,),
            compute=lambda d, c=column: (_num(d[c]) - params.contact_target).abs(),
            group="Contact time",
            process_day="D3",
            value_type="numeric",
            rationale=f"Minutes away from the {params.contact_target:.0f} min "
            f"transduction contact time{' for ' + label if label else ''}",
        ),
        Feature(
            name=cols("f_contact_within")[which],
            requires=(column,),
            compute=lambda d, c=column: _num(d[c]).between(low, high).astype(float),
            group="Contact time",
            process_day="D3",
            value_type="boolean",
            rationale=f"Whether contact time stayed inside {params.contact_target:.0f}"
            f"±{params.contact_tolerance:.0f} min",
        ),
    ]


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
def registry(params: Params, columns: Sequence[str] = ()) -> list[Feature]:
    """Every derived feature, in the order they are computed.

    ``columns`` is the frame the features will be applied to, and is read only by
    the families that expand over whatever environmental columns a site records.
    """
    entries: list[Feature] = [
        # ── Flow and phenotype ────────────────────────────────────────────
        Feature(
            col("f_log_pt_cd4_cd8"),
            (col("pt_cd4_pct"), col("pt_cd8_pct")),
            lambda d: np.log10(d[col("pt_cd4_pct")] / d[col("pt_cd8_pct")]),
            "Flow and phenotype",
            "D0",
            "numeric",
            "CD4 to CD8 balance of the incoming material, on a log scale so a "
            "two-fold shift reads the same in either direction",
        ),
        Feature(
            col("f_log_pw_cd4_cd8"),
            (col("pw_cd4_pct"), col("pw_cd8_pct")),
            lambda d: np.log10(d[col("pw_cd4_pct")] / d[col("pw_cd8_pct")]),
            "Flow and phenotype",
            "D0",
            "numeric",
            "CD4 to CD8 balance after enrichment, on a log scale",
        ),
        Feature(
            col("f_pt_cd3_cells"),
            (col("pt_viable_cells"), col("pt_cd3_pct")),
            lambda d: d[col("pt_viable_cells")] * d[col("pt_cd3_pct")],
            "Flow and phenotype",
            "D0",
            "numeric",
            "Absolute CD3+ cells post thaw — the population the process actually works with",
        ),
        Feature(
            col("f_pt_cd4_share"),
            (col("pt_cd4_pct"), col("pt_cd3_pct")),
            lambda d: d[col("pt_cd4_pct")] * d[col("pt_cd3_pct")] / 100,
            "Flow and phenotype",
            "D0",
            "numeric",
            "CD4+ as a share of all cells rather than of the CD3+ gate",
        ),
        Feature(
            col("f_pt_cd4_cells"),
            (col("f_pt_cd4_share"), col("pt_viable_cells")),
            lambda d: d[col("f_pt_cd4_share")] * d[col("pt_viable_cells")],
            "Flow and phenotype",
            "D0",
            "numeric",
            "Absolute CD3+CD4+ cells post thaw",
        ),
        Feature(
            col("f_pw_cd4_share"),
            (col("pw_cd4_pct"), col("pw_cd3_pct")),
            lambda d: d[col("pw_cd4_pct")] * d[col("pw_cd3_pct")] / 100,
            "Flow and phenotype",
            "D0",
            "numeric",
            "CD4+ as a share of all cells after enrichment",
        ),
        Feature(
            col("f_pw_cd8_share"),
            (col("pw_cd8_pct"), col("pw_cd3_pct")),
            lambda d: d[col("pw_cd8_pct")] * d[col("pw_cd3_pct")] / 100,
            "Flow and phenotype",
            "D0",
            "numeric",
            "CD8+ as a share of all cells after enrichment",
        ),
        # ── Metabolites ───────────────────────────────────────────────────
        Feature(
            col("f_harvest_lactate_norm"),
            (col("vessel_count"), col("harvest_lactate"), col("vsvg_cells")),
            lambda d: (
                (d[col("harvest_lactate")] * 1_000_000_000)
                * d[col("vessel_count")]
                / d[col("vsvg_cells")]
            ),
            "Metabolites",
            "D10",
            "numeric",
            "Harvest lactate per cell seeded, so batches seeded into one or two "
            "G-Rex can be compared",
        ),
        Feature(
            cols("f_lactate_per_cell")[0],
            (col("vessel_count"), col("d6_lactate_a"), col("d6_lactate_b"), col("vsvg_cells")),
            lambda d: np.where(
                d[col("vessel_count")] == 1,
                d[col("d6_lactate_a")] * 1e9 / d[col("vsvg_cells")],
                (d[col("d6_lactate_a")] + d[col("d6_lactate_b")]) * 1e9 / d[col("vsvg_cells")],
            ),
            "Metabolites",
            "D6",
            "numeric",
            "Lactate produced per million cells seeded, summed over the vessels in use",
        ),
        Feature(
            cols("f_lactate_per_cell")[1],
            (col("vessel_count"), col("d8_lactate_a"), col("d8_lactate_b"), col("vsvg_cells")),
            lambda d: np.where(
                d[col("vessel_count")] == 1,
                d[col("d8_lactate_a")] * 1e9 / d[col("vsvg_cells")],
                (d[col("d8_lactate_a")] + d[col("d8_lactate_b")]) * 1e9 / d[col("vsvg_cells")],
            ),
            "Metabolites",
            "D8",
            "numeric",
            "Lactate produced per million cells seeded, summed over the vessels in use",
        ),
        Feature(
            cols("f_lactate_per_cell")[2],
            (col("vessel_count"), col("harvest_lactate"), col("vsvg_cells")),
            lambda d: np.where(
                d[col("vessel_count")] == 1,
                d[col("harvest_lactate")] * 1e9 / d[col("vsvg_cells")],
                d[col("harvest_lactate")] * 2e9 / d[col("vsvg_cells")],
            ),
            "Metabolites",
            "D10",
            "numeric",
            "Harvest lactate per million cells seeded; the second vessel is "
            "sampled once and counted twice",
        ),
        Feature(
            cols("f_glucose_per_cell")[0],
            (col("vessel_count"), col("d6_glucose_a"), col("d6_glucose_b"), col("vsvg_cells")),
            lambda d: np.where(
                d[col("vessel_count")] == 1,
                d[col("d6_glucose_a")] * 1e9 / d[col("vsvg_cells")],
                (d[col("d6_glucose_a")] + d[col("d6_glucose_b")]) * 1e9 / d[col("vsvg_cells")],
            ),
            "Metabolites",
            "D6",
            "numeric",
            "Glucose remaining per million cells seeded, summed over the vessels in use",
        ),
        Feature(
            cols("f_glucose_per_cell")[1],
            (col("vessel_count"), col("d8_glucose_a"), col("d8_glucose_b"), col("vsvg_cells")),
            lambda d: np.where(
                d[col("vessel_count")] == 1,
                d[col("d8_glucose_a")] * 1e9 / d[col("vsvg_cells")],
                (d[col("d8_glucose_a")] + d[col("d8_glucose_b")]) * 1e9 / d[col("vsvg_cells")],
            ),
            "Metabolites",
            "D8",
            "numeric",
            "Glucose remaining per million cells seeded, summed over the vessels in use",
        ),
        Feature(
            cols("f_glucose_per_cell")[2],
            (col("vessel_count"), col("harvest_glucose"), col("vsvg_cells")),
            lambda d: np.where(
                d[col("vessel_count")] == 1,
                d[col("harvest_glucose")] * 1e9 / d[col("vsvg_cells")],
                d[col("harvest_glucose")] * 2 * 1e9 / d[col("vsvg_cells")],
            ),
            "Metabolites",
            "D10",
            "numeric",
            "Harvest glucose per million cells seeded",
        ),
        Feature(
            col("f_lactate_slope"),
            (cols("f_lactate_per_cell"),),
            lambda d: slope(d, METABOLITE_DAYS, cols("f_lactate_per_cell")),
            "Metabolites",
            "D10",
            "numeric",
            "How fast normalised lactate climbs across expansion, as one number "
            "instead of three readings",
        ),
        Feature(
            col("f_lactate_auc"),
            (cols("f_lactate_per_cell"),),
            lambda d: auc(d, METABOLITE_DAYS, cols("f_lactate_per_cell")),
            "Metabolites",
            "D10",
            "numeric",
            "Total normalised lactate exposure across expansion",
        ),
        Feature(
            col("f_glucose_slope"),
            (cols("f_glucose_per_cell"),),
            lambda d: slope(d, METABOLITE_DAYS, cols("f_glucose_per_cell")),
            "Metabolites",
            "D10",
            "numeric",
            "How fast normalised glucose falls across expansion",
        ),
        Feature(
            col("f_glucose_auc"),
            (cols("f_glucose_per_cell"),),
            lambda d: auc(d, METABOLITE_DAYS, cols("f_glucose_per_cell")),
            "Metabolites",
            "D10",
            "numeric",
            "Total normalised glucose exposure across expansion",
        ),
        *[
            Feature(
                col("f_lactate_glucose_ratio", day=int(day)),
                (lactate, glucose),
                lambda d, lact=lactate, gluc=glucose: safe_div(d[lact], d[gluc]),
                "Metabolites",
                f"D{int(day)}",
                "numeric",
                "Lactate against glucose on the same day, which separates cells "
                "burning sugar fast from cells simply being fed more",
            )
            for day, lactate, glucose in zip(
                METABOLITE_DAYS, cols("f_lactate_per_cell"), cols("f_glucose_per_cell")
            )
        ],
        # ── Profiles ──────────────────────────────────────────────────────
        Feature(
            col("f_growth_profile"),
            (col("cpdl"), col("f_harvest_lactate_norm")),
            lambda d: np.select(
                [
                    (d[col("cpdl")] < params.cpdl_low)
                    | (d[col("f_harvest_lactate_norm")] < params.lactate_low),
                    (d[col("cpdl")] > params.cpdl_high)
                    & (d[col("f_harvest_lactate_norm")] > params.lactate_high),
                ],
                list(values("growth_profile_classes")[:2]),
                default=values("growth_profile_classes")[2],
            ),
            "Profiles",
            "D10",
            "categorical",
            "Below, within or above the expected growth, from doublings and normalised "
            "lactate together; thresholds are in config",
        ),
        Feature(
            col("f_cd4_profile"),
            (col("pt_cd4_pct"),),
            lambda d: pd.Series(
                np.select(
                    [
                        d[col("pt_cd4_pct")] <= params.cd4_low,
                        d[col("pt_cd4_pct")] >= params.cd4_high,
                    ],
                    list(values("cd4_profile_classes")[:2]),
                    default=values("cd4_profile_classes")[2],
                ),
                index=d.index,
            ).where(d[col("pt_cd4_pct")].notna()),
            "Profiles",
            "D0",
            "categorical",
            "The CD4 share of the incoming material in three bands; a batch with "
            "no reading gets no profile rather than the middle one",
        ),
        # ── Viability ─────────────────────────────────────────────────────
        Feature(
            col("f_viability_drop_rel"),
            (col("d3_viability"), col("pw_viability")),
            lambda d: (
                (d[col("d3_viability")] - d[col("pw_viability")]) / d[col("pw_viability")] * 100
            ),
            "Viability",
            "D3",
            "numeric",
            "Viability change over activation, relative to where it started",
        ),
        Feature(
            col("f_viability_drop_pts"),
            (col("d3_viability"), col("pw_viability")),
            lambda d: _num(d[col("d3_viability")]) - _num(d[col("pw_viability")]),
            "Viability",
            "D3",
            "numeric",
            "The same change in percentage points rather than as a relative "
            "change, which keeps it readable when the starting viability is low",
        ),
        Feature(
            col("f_viability_d3_d10_rel"),
            (col("d3_viability"), col("harvest_viability")),
            lambda d: (
                (d[col("d3_viability")] - d[col("harvest_viability")])
                / d[col("d3_viability")]
                * 100
            ),
            "Viability",
            "D10",
            "numeric",
            "Viability lost between activation and harvest, relative to D3",
        ),
        Feature(
            col("f_viability_expansion"),
            (col("harvest_viability"), col("d3_viability")),
            lambda d: (
                (d[col("harvest_viability")] - d[col("d3_viability")])
                / d[col("d3_viability")]
                * 100
            ),
            "Viability",
            "D10",
            "numeric",
            "The same comparison signed the other way, as the Power Query investigation reports it",
        ),
        Feature(
            col("f_viability_d3_d10_pts"),
            (col("d3_viability"), col("washed_viability")),
            lambda d: _num(d[col("washed_viability")]) - _num(d[col("d3_viability")]),
            "Viability",
            "D10",
            "numeric",
            "Net viability change from activation to the washed harvest, in percentage points",
        ),
        Feature(
            col("f_viability_wash"),
            (col("washed_viability"), col("harvest_viability")),
            lambda d: (
                (d[col("washed_viability")] - d[col("harvest_viability")])
                / d[col("harvest_viability")]
                * 100
            ),
            "Viability",
            "D10",
            "numeric",
            "What the LOVO wash did to viability",
        ),
        # ── Seeding and vessel balance ────────────────────────────────────
        Feature(
            col("f_seeding_pct"),
            (col("vessel_count"), col("vsvg_cells")),
            lambda d: np.select(
                [d[col("vessel_count")] == 1, d[col("vessel_count")] == 2],
                [d[col("vsvg_cells")] / 500_000, d[col("vsvg_cells")] / 1_000_000],
                default=np.nan,
            ),
            "Seeding",
            "D3",
            "numeric",
            "Cells seeded against the target for the number of vessels used",
        ),
        Feature(
            col("f_seeding_error"),
            (col("f_seeding_pct"),),
            lambda d: (_num(d[col("f_seeding_pct")]) - 100.0).abs(),
            "Seeding",
            "D3",
            "numeric",
            "Distance from target seeding in either direction, so over- and "
            "under-seeding count the same",
        ),
        Feature(
            col("f_seeding_within"),
            (col("f_seeding_pct"),),
            lambda d: (
                (_num(d[col("f_seeding_pct")]) - 100.0).abs() <= params.seeding_tolerance
            ).astype(float),
            "Seeding",
            "D3",
            "boolean",
            f"Whether seeding landed within ±{params.seeding_tolerance:.0f}% of target",
        ),
        Feature(
            col("f_cells_per_bag"),
            (col("cells_available"), col("culture_bags")),
            lambda d: safe_div(d[col("cells_available")], d[col("culture_bags")]),
            "Seeding",
            "D3",
            "numeric",
            "Cells per culture bag, which is the density the cells actually see",
        ),
        *_spread_features(
            "f_seeded_spread", col("cells_seeded_a"), col("cells_seeded_b"), "seeded viable cells"
        ),
        *_spread_features(
            "f_vector_volume_spread",
            col("vector_volume_a"),
            col("vector_volume_b"),
            "vector volume added",
        ),
        # ── MOI ───────────────────────────────────────────────────────────
        Feature(
            col("f_moi_a"),
            (_titer(), col("vector_volume_a"), col("cells_seeded_a")),
            lambda d: (
                _first(d, *_titer()) * safe_div(d[col("vector_volume_a")], d[col("cells_seeded_a")])
            ),
            "MOI",
            "D3",
            "numeric",
            "Infectious units per seeded cell in vessel A, from the measured "
            "titer rather than the nominal one",
        ),
        Feature(
            col("f_log_moi_a"),
            (col("f_moi_a"),),
            lambda d: np.log1p(_num(d[col("f_moi_a")])),
            "MOI",
            "D3",
            "numeric",
            "Vessel A MOI on a log scale, where its spread is even enough to model",
        ),
        Feature(
            col("f_moi_b"),
            (_titer(), col("vector_volume_b"), col("cells_seeded_b")),
            lambda d: (
                _first(d, *_titer()) * safe_div(d[col("vector_volume_b")], d[col("cells_seeded_b")])
            ),
            "MOI",
            "D3",
            "numeric",
            "Infectious units per seeded cell in vessel B",
        ),
        Feature(
            col("f_log_moi_b"),
            (col("f_moi_b"),),
            lambda d: np.log1p(_num(d[col("f_moi_b")])),
            "MOI",
            "D3",
            "numeric",
            "Vessel B MOI on a log scale",
        ),
        Feature(
            col("f_moi_pool"),
            (
                _titer(),
                col("vector_volume_a"),
                col("vector_volume_b"),
                (col("cells_available"), col("cells_seeded_a")),
            ),
            lambda d: (
                _first(d, *_titer()) * safe_div(_pooled_vector_volume(d), _pooled_seeded_cells(d))
            ),
            "MOI",
            "D3",
            "numeric",
            "Infectious units per seeded cell over the whole batch; a vessel that "
            "was not used contributes nothing rather than making the batch missing",
        ),
        Feature(
            col("f_log_moi_pool"),
            (col("f_moi_pool"),),
            lambda d: np.log1p(_num(d[col("f_moi_pool")])),
            "MOI",
            "D3",
            "numeric",
            "Pooled MOI on a log scale",
        ),
        Feature(
            col("f_moi_quality"),
            (col("f_moi_pool"), _coa("coa_pi_ratio")),
            lambda d: _num(d[col("f_moi_pool")]) * _num(d[_coa("coa_pi_ratio")]),
            "MOI",
            "D3",
            "numeric",
            "Pooled MOI weighted by how many particles in the lot are infectious, "
            "so two lots at the same titer are not treated as the same vector",
        ),
        Feature(
            col("f_moi_per_cd3"),
            (_titer(), col("vector_volume_a"), col("vector_volume_b"), col("f_pt_cd3_cells")),
            lambda d: (
                _first(d, *_titer()) * safe_div(_pooled_vector_volume(d), d[col("f_pt_cd3_cells")])
            ),
            "MOI",
            "D3",
            "numeric",
            "Infectious units per CD3+ cell rather than per cell of any kind — "
            "only T cells are the target",
        ),
        # ── Contact time ──────────────────────────────────────────────────
        *_contact_time_features(col("contact_time"), 0, params),
        *_contact_time_features(col("contact_time_a"), 1, params),
        *_contact_time_features(col("contact_time_b"), 2, params),
        # ── Incubation ────────────────────────────────────────────────────
        Feature(
            col("f_incubation_d0_d1"),
            (
                col("incubator_out_d1"),
                col("incubator_in_d0"),
            ),
            lambda d: (
                (
                    pd.to_datetime(d[col("incubator_out_d1")])
                    - pd.to_datetime(d[col("incubator_in_d0")])
                ).dt.total_seconds()
                / 60
            ),
            "Incubation",
            "D1",
            "numeric",
            "Minutes the cells spent in the incubator between day 0 and day 1",
        ),
        Feature(
            col("f_incubation_d1_d3"),
            (
                col("incubator_out_d3"),
                col("incubator_in_d1"),
            ),
            lambda d: (
                (
                    pd.to_datetime(d[col("incubator_out_d3")])
                    - pd.to_datetime(d[col("incubator_in_d1")])
                ).dt.total_seconds()
                / 60
            ),
            "Incubation",
            "D3",
            "numeric",
            "Minutes in the incubator between day 1 and day 3",
        ),
        Feature(
            col("f_incubation_d3_d6"),
            (
                col("incubator_out_d6"),
                col("incubator_back_d3"),
            ),
            lambda d: (
                (
                    pd.to_datetime(d[col("incubator_out_d6")])
                    - pd.to_datetime(d[col("incubator_back_d3")])
                ).dt.total_seconds()
                / 60
            ),
            "Incubation",
            "D6",
            "numeric",
            "Minutes in the incubator between day 3 and day 6",
        ),
        Feature(
            col("f_incubation_d6_d8"),
            (
                col("incubator_out_d8"),
                col("incubator_back_d6"),
            ),
            lambda d: (
                (
                    pd.to_datetime(d[col("incubator_out_d8")])
                    - pd.to_datetime(d[col("incubator_back_d6")])
                ).dt.total_seconds()
                / 60
            ),
            "Incubation",
            "D8",
            "numeric",
            "Minutes in the incubator between day 6 and day 8",
        ),
        Feature(
            col("f_incubation_d8_d10"),
            (
                col("incubator_out_d10"),
                col("incubator_back_d8"),
            ),
            lambda d: (
                (
                    pd.to_datetime(d[col("incubator_out_d10")])
                    - pd.to_datetime(d[col("incubator_back_d8")])
                ).dt.total_seconds()
                / 60
            ),
            "Incubation",
            "D10",
            "numeric",
            "Minutes in the incubator between day 8 and day 10",
        ),
        Feature(
            col("f_incubation_total"),
            (col("f_incubation_d3_d6"), col("f_incubation_d6_d8"), col("f_incubation_d8_d10")),
            lambda d: (
                _num(d[col("f_incubation_d3_d6")])
                + _num(d[col("f_incubation_d6_d8")])
                + _num(d[col("f_incubation_d8_d10")])
            ),
            "Incubation",
            "D10",
            "numeric",
            "Total time in the incubator across expansion; a batch handled slowly "
            "at one step can still end up with a normal total",
        ),
        # ── Clumps ────────────────────────────────────────────────────────
        Feature(
            col("f_clump_count_reduction"),
            (col("clumps_pre_count"), col("clumps_post_count")),
            lambda d: 1.0 - safe_div(d[col("clumps_post_count")], d[col("clumps_pre_count")]),
            "Clumps",
            "D1",
            "numeric",
            "Share of the clumps that massaging removed, which measures the "
            "intervention rather than the starting material",
        ),
        Feature(
            col("f_clump_size_reduction"),
            (col("clumps_pre_size"), col("clumps_post_size")),
            lambda d: 1.0 - safe_div(d[col("clumps_post_size")], d[col("clumps_pre_size")]),
            "Clumps",
            "D1",
            "numeric",
            "Share of the clump size that massaging removed",
        ),
        Feature(
            col("f_clump_burden_pre"),
            (col("clumps_pre_count"), col("clumps_pre_size")),
            lambda d: _num(d[col("clumps_pre_count")]) * _num(d[col("clumps_pre_size")]),
            "Clumps",
            "D1",
            "numeric",
            "Count and size together, so many small clumps and one large one are "
            "not recorded as the same burden",
        ),
        Feature(
            col("f_clump_burden_post"),
            (col("clumps_post_count"), col("clumps_post_size")),
            lambda d: _num(d[col("clumps_post_count")]) * _num(d[col("clumps_post_size")]),
            "Clumps",
            "D1",
            "numeric",
            "Size-weighted clump burden left after massaging",
        ),
        Feature(
            col("f_clump_burden_mix"),
            (col("clumps_mix_count"), col("clumps_mix_size")),
            lambda d: _num(d[col("clumps_mix_count")]) * _num(d[col("clumps_mix_size")]),
            "Clumps",
            "D1",
            "numeric",
            "Size-weighted clump burden left after mixing",
        ),
        Feature(
            col("f_mixing_x_clumps"),
            (col("mixing_cycles"), col("clumps_pre_count")),
            lambda d: _num(d[col("mixing_cycles")]) * _num(d[col("clumps_pre_count")]),
            "Clumps",
            "D1",
            "numeric",
            "Mixing effort against the burden it was applied to; neither number "
            "means much without the other",
        ),
        Feature(
            col("f_d3_clumps"),
            (col("d3_clump_pattern"),),
            lambda d: yn_to_bool(d[col("d3_clump_pattern")]),
            "Clumps",
            "D3",
            "boolean",
            "Whether clumps were still recorded at day 3",
        ),
        Feature(
            col("f_clump_bag_count"),
            (cols("clump_bags"),),
            lambda d: sum(yn_to_bool(d[c]) for c in _present(d, cols("clump_bags"))),
            "Clumps",
            "D10",
            "numeric",
            "How many harvest bags had clumps, counted over the bags the batch actually filled",
        ),
        # ── Dose and CAR ──────────────────────────────────────────────────
        Feature(
            col("f_dose_all"),
            (col("dose_per_kg"), col("dose_cells")),
            lambda d: np.select(
                [
                    d[col("dose_per_kg")].notna(),
                    d[col("dose_cells")].notna(),
                ],
                [
                    d[col("dose_per_kg")] / 1_000_000,
                    d[col("dose_cells")] / 100_000_000,
                ],
                default=np.nan,
            ),
            "Dose and CAR",
            "D10",
            "numeric",
            "One dose column for every batch, whether the weight-adjusted dose or "
            "only the absolute cell count was recorded",
        ),
        Feature(
            col("f_post_formulation_dose"),
            (
                col("formulated_concentration"),
                col("bag_volume"),
                col("harvest_car_pct"),
                col("subject_weight"),
            ),
            lambda d: (
                d[col("formulated_concentration")]
                / 1_000_000
                * d[col("bag_volume")]
                * d[col("harvest_car_pct")]
                / 100
                / d[col("subject_weight")]
            ),
            "Dose and CAR",
            "D10",
            "numeric",
            "Dose per kilogram implied by what actually went into the bag",
        ),
        Feature(
            col("f_achievable_dose"),
            (
                col("harvest_car_pct"),
                col("washed_cells"),
                col("subject_weight"),
            ),
            lambda d: (
                d[col("harvest_car_pct")]
                / 100
                * d[col("washed_cells")]
                / (d[col("subject_weight")] * 1_000_000)
            ),
            "Dose and CAR",
            "D10",
            "numeric",
            "The dose the harvest could have supported, against the dose that was given",
        ),
        Feature(
            col("f_car_cells"),
            (
                col("harvest_car_pct"),
                col("washed_cells"),
            ),
            lambda d: d[col("harvest_car_pct")] * d[col("washed_cells")] / 100,
            "Dose and CAR",
            "D10",
            "numeric",
            "Absolute CAR+ cells at harvest",
        ),
        Feature(
            col("f_car_yield"),
            (col("harvest_car_pct"), col("cpdl")),
            lambda d: d[col("harvest_car_pct")] / 100 * np.power(2, d[col("cpdl")]),
            "Dose and CAR",
            "D10",
            "numeric",
            "CAR+ cells returned per cell seeded, which folds expansion and "
            "transduction into one yield",
        ),
        Feature(
            col("f_vcn_per_cell"),
            (col("fp_car_pct"), col("vcn")),
            lambda d: d[col("fp_car_pct")] * d[col("vcn")] / 100,
            "Dose and CAR",
            "Final product",
            "numeric",
            "Copies per cell across the whole product rather than per transduced cell",
        ),
        Feature(
            col("f_flow_accuracy"),
            (col("fp_car_pct"), col("harvest_car_pct")),
            lambda d: d[col("fp_car_pct")] / d[col("harvest_car_pct")],
            "Dose and CAR",
            "Final product",
            "numeric",
            "Final product CAR+ against the harvest reading — a ratio far from one "
            "points at the assay, not the process",
        ),
        # ── Formulation and wash ──────────────────────────────────────────
        Feature(
            col("f_formulated_total_concentration"),
            (
                col("formulated_concentration"),
                col("formulated_viability"),
            ),
            lambda d: (
                d[col("formulated_concentration")]
                / (d[col("formulated_viability")] / 100)
                / 1_000_000
            ),
            "Formulation and wash",
            "D10",
            "numeric",
            "Total cells in the formulated suspension, dead ones included",
        ),
        Feature(
            col("f_cells_into_formulation"),
            (
                col("washed_concentration"),
                col("formulation_cell_volume"),
            ),
            lambda d: d[col("washed_concentration")] * d[col("formulation_cell_volume")],
            "Formulation and wash",
            "D10",
            "numeric",
            "Viable cells that went into formulation",
        ),
        Feature(
            col("f_cells_out_of_formulation"),
            (
                col("formulated_concentration"),
                col("formulation_medium_volume"),
            ),
            lambda d: d[col("formulated_concentration")] * d[col("formulation_medium_volume")],
            "Formulation and wash",
            "D10",
            "numeric",
            "Viable cells that came out of formulation",
        ),
        Feature(
            col("f_cells_lost_formulation"),
            (
                col("f_cells_into_formulation"),
                col("f_cells_out_of_formulation"),
            ),
            lambda d: d[col("f_cells_into_formulation")] - d[col("f_cells_out_of_formulation")],
            "Formulation and wash",
            "D10",
            "numeric",
            "The difference between those two, as cells",
        ),
        Feature(
            col("f_pct_lost_formulation"),
            (
                col("f_cells_lost_formulation"),
                col("f_cells_into_formulation"),
            ),
            lambda d: d[col("f_cells_lost_formulation")] / d[col("f_cells_into_formulation")] * 100,
            "Formulation and wash",
            "D10",
            "numeric",
            "The same loss as a share, which is comparable across batch sizes",
        ),
        Feature(
            col("f_cells_lost_wash"),
            (
                col("harvest_cells"),
                col("washed_cells"),
            ),
            lambda d: d[col("harvest_cells")] - d[col("washed_cells")],
            "Formulation and wash",
            "D10",
            "numeric",
            "Cells the wash did not return",
        ),
        # ── Vector CoA ────────────────────────────────────────────────────
        Feature(
            col("f_coa_impurity_index"),
            (_impurities(),),
            lambda d: sum(zscore(d[c]) for c in _present(d, _impurities())),
            "Vector CoA",
            "Vector CoA",
            "numeric",
            "Residual impurities on one scale, summed. Standardised within the "
            "site being processed, not across the merged batch table, so the two sites' "
            "values are not directly comparable",
        ),
        # ── IL-2 ──────────────────────────────────────────────────────────
        Feature(
            col("f_il2_d6"),
            (col("il2_activity_d6"), col("il2_content_d6")),
            lambda d: _num(d[col("il2_activity_d6")]) * _num(d[col("il2_content_d6")]) / 1000.0,
            "IL-2",
            "D6",
            "numeric",
            "IL-2 units fed at day 6: activity and protein mass mean little apart",
        ),
        Feature(
            col("f_il2_d8"),
            (col("il2_activity_d8"), col("il2_content_d8")),
            lambda d: _num(d[col("il2_activity_d8")]) * _num(d[col("il2_content_d8")]) / 1000.0,
            "IL-2",
            "D8",
            "numeric",
            "IL-2 units fed at day 8",
        ),
        # ── Context ───────────────────────────────────────────────────────
        Feature(
            col("f_shift"),
            (col("thaw_time"),),
            lambda d: np.where(
                pd.to_datetime(d[col("thaw_time")].astype(str)).dt.time
                < pd.to_datetime("11:59:00").time(),
                "Morning",
                "Afternoon",
            ),
            "Context",
            "D0",
            "categorical",
            "Which shift started the batch, taken from the thaw time",
        ),
    ]
    entries.extend(_setpoint_features(columns, params))
    return entries


# ---------------------------------------------------------------------------
# Derived columns this registry does not build
# ---------------------------------------------------------------------------
def coa_measured() -> tuple[str, ...]:
    """Every certificate column the loader coerces, and therefore may flag as absent."""
    donors = []
    for metric in (col("coa_car_metric"), col("coa_ifng_metric")):
        donors += [
            col("coa_donor", metric=metric, donor=1),
            col("coa_donor", metric=metric, donor=2),
            col("coa_donor_average", metric=metric),
        ]
    impurities = cols("coa_impurities")
    return (
        col("coa_titer"),
        col("coa_p24"),
        col("coa_pi_ratio"),
        *donors,
        col("coa_ph"),
        col("coa_osmolality"),
        col("coa_volume"),
        col("coa_dna_size"),
        *impurities,
        *(col("coa_calc_prefix") + name for name in impurities),
    )


def coa_lineage() -> dict[str, tuple[str, ...]]:
    """What every certificate-derived column is computed from.

    :func:`ml.io.load_lv_coa` builds these while parsing the workbook, which is
    where the parsing belongs, but the tasks stage needs to know what they came
    from — a column whose lineage nobody recorded cannot be checked for descent
    from the target. The loader checks what it produced against this table, so
    the two cannot drift apart in silence.
    """
    prefix = col("coa_prefix")
    titer, raw = _coa("coa_titer"), _coa("coa_titer_unadjusted")
    impurities = cols("coa_impurities")
    lineage: dict[str, tuple[str, ...]] = {
        _coa("coa_delivered_dose"): (titer, raw, _coa("coa_volume")),
        _coa("coa_delivered_dose_fallback"): (raw,),
        _coa("coa_titer_adjustment"): (titer, raw),
        _coa("coa_specific_infectivity"): (titer, _coa("coa_p24")),
        _coa("coa_last_dna_fraction"): tuple(prefix + name for name in impurities[1:]),
        f"{_coa('coa_ph')} Abs Deviation": (_coa("coa_ph"),),
        f"{_coa('coa_osmolality')} Abs Deviation": (_coa("coa_osmolality"),),
    }
    for metric in (col("coa_car_metric"), col("coa_ifng_metric")):
        donors = (
            prefix + col("coa_donor", metric=metric, donor=1),
            prefix + col("coa_donor", metric=metric, donor=2),
        )
        for statistic in ("MAX", "MIN", "DIFF", "DIFF Rel", "AVG"):
            lineage[f"{prefix}{metric} Donor {statistic}"] = donors
    for name in impurities:
        lineage[f"{prefix}{name} below LOQ"] = (f"{prefix}{name}",)
    for name in coa_measured():
        lineage[f"{prefix}{name} missing"] = (f"{prefix}{name}",)
    return lineage


def derived_sources() -> dict[str, tuple[str, ...]]:
    """Every column that is arithmetic over another, and what it was computed from.

    The registry's own entries are added by :func:`ml.dataset` from the registry.
    """
    return coa_lineage()


def site_relative() -> tuple[str, ...]:
    """Columns whose value depends on the frame they were computed over.

    A z-score is relative to the site's own batches, so it is not a fixed
    measurement and it cannot be recomputed for one new batch. Exploratory only.
    """
    return (col("f_coa_impurity_index"),)


# ---------------------------------------------------------------------------
# Running the registry
# ---------------------------------------------------------------------------
#: Statuses an entry can end in. "present" means a source already supplies the
#: column, so the feature stands aside rather than overwriting it.
CREATED, PRESENT, NOT_IN_CATALOG, BLOCKED, ERRORED = (
    "created",
    "present",
    "not in catalog",
    "blocked",
    "errored",
)


def profile_columns() -> tuple[str, str]:
    return col("f_growth_profile"), col("f_cd4_profile")


def _requirement_text(requirement: Requirement) -> str:
    return requirement if isinstance(requirement, str) else " or ".join(requirement)


def _missing(df: pd.DataFrame, requires: Sequence[Requirement]) -> list[str]:
    """The requirements this frame cannot satisfy, named as the feature states them."""
    unmet = []
    for requirement in requires:
        names = (requirement,) if isinstance(requirement, str) else requirement
        if not any(name in df.columns for name in names):
            unmet.append(_requirement_text(requirement))
    return unmet


def apply(
    df: pd.DataFrame,
    catalog_cols: Sequence[str],
    params: Params,
    site: str = "",
) -> FeatureReport:
    """Compute every feature this frame can support, in registry order.

    The frame is modified in place. Every entry ends in exactly one status, and
    the returned report carries all of them — a feature that produced nothing is
    as much a result as one that did.

    Parameters
    ----------
    df
        One site's batch data, already cleaned and enriched.
    catalog_cols
        The catalog parameter names. A feature whose name is not among them is
        reported and not computed: the catalog is the schema of the batch table.
    params
        Setpoints and thresholds from config.
    site
        Name used in the log lines and carried on the report.

    Returns
    -------
    FeatureReport
        One entry per registry feature, plus the class counts of the profile
        columns.
    """
    log.step(MODULE, f"[{site}] Computing derived features")
    catalog = set(catalog_cols)
    report = FeatureReport(site=site, rows=len(df))

    for feature in registry(params, df.columns):
        entry = {
            "name": feature.name,
            "group": feature.group,
            "process_day": feature.process_day,
            "value_type": feature.value_type,
            "rationale": feature.rationale,
            "inputs": ", ".join(_requirement_text(r) for r in feature.requires),
            "status": "",
            "note": "",
            "non_null": None,
        }
        report.entries.append(entry)

        if feature.name not in catalog:
            entry["status"] = NOT_IN_CATALOG
            entry["note"] = "add to the catalog to have this computed"
            continue
        if feature.name in df.columns:
            entry["status"] = PRESENT
            entry["note"] = "a source already supplies this column"
            continue

        unmet = _missing(df, feature.requires)
        if unmet:
            entry["status"] = BLOCKED
            entry["note"] = "missing: " + ", ".join(unmet)
            continue

        try:
            df[feature.name] = feature.compute(df)
        except Exception as exc:  # a bad column beats the whole registry otherwise
            entry["status"] = ERRORED
            entry["note"] = str(exc)
            log.warn(MODULE, f"[{site}] '{feature.name}' failed", str(exc))
            continue

        entry["status"] = CREATED
        entry["non_null"] = int(df[feature.name].notna().sum())

    report.profiles = {
        column: df[column].value_counts().to_dict()
        for column in profile_columns()
        if column in df.columns
    }

    log.info(
        MODULE,
        f"[{site}] features — {report.count(CREATED)} created, "
        f"{report.count(PRESENT)} already present, {report.count(BLOCKED)} blocked, "
        f"{report.count(NOT_IN_CATALOG)} not in catalog, {report.count(ERRORED)} failed",
    )
    for entry in report.entries:
        if entry["status"] == BLOCKED:
            log.warn(MODULE, f"[{site}] {entry['name']} blocked", entry["note"])
    if report.count(NOT_IN_CATALOG):
        log.warn(
            MODULE,
            f"[{site}] {report.count(NOT_IN_CATALOG)} features are not catalog parameters "
            "and were not computed",
            ", ".join(e["name"] for e in report.entries if e["status"] == NOT_IN_CATALOG),
        )

    log.success(MODULE, f"[{site}] Derived features complete")
    return report
