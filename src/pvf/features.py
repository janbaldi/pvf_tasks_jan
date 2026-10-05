"""Features calculated from parameters the PVF already holds.

A *parameter* arrives from a source and is cleaned or harmonised on the way in;
that work lives in :mod:`pvf.io`, :mod:`pvf.clean` and :mod:`pvf.enrich`. A
*feature* is arithmetic over one or more parameters, and every one of them lives
here, in a single ordered registry.

One entry per feature. To add one, append a :class:`Feature` — no other file
changes. Entries run in list order, so a feature may require a column an earlier
entry produced, which is how ``Harvest Lactate Normalized`` reaches
``growth profile`` and ``D6 Lact / E6 cells`` reaches the D6-D10 slope without a
dependency graph.

A feature is computed only when its name is a PTF parameter. A feature that is
not in the PTF is reported, not created: the PTF is the schema, and a column the
schema does not know about has nowhere to go. The report lists those entries with
everything the PTF needs to accept them.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .logger import log

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
    """Standardise within the frame this is called on — one site, not the PVF."""
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
TITER = ("LV CoA LV Titer", "Vector Titer (IU/mL)")
VOL_A = "Volume Vector Added to G-Rex A (mL)"
VOL_B = "Volume Vector Added to G-Rex B (mL)"
CELLS_A = "Day 3 Actual Viable Cells Seeded G-Rex A"
CELLS_B = "Day 3 Actual Viable Cells Seeded G-Rex B"
CELLS_AVAILABLE = "Day 3 Total Viable Cells Available for Seeding G-Rex (cells)"
VSVG_CELLS = "Total Viable Cells for Expansion after VSVg Sampling (cells)"
GREX_COUNT = "Day 3 Number of G-Rex to seed"
CLUMPS_PRE_COUNT = "Pre-Activation Clumps: # of Clumps before massage"
CLUMPS_PRE_SIZE = "Pre-Activation Clumps: Size of Clumps before massage"
CLUMPS_POST_COUNT = "Post-Massaging: # of Clumps"
CLUMPS_POST_SIZE = "Post-Massaging: Size of Clumps"
CLUMPS_MIX_COUNT = "Post-Mixing Clumps: # of Clumps"
CLUMPS_MIX_SIZE = "Post-Mixing Clumps: Size of Clumps"
CLUMP_BAGS = tuple(f"Day 10 Clump Present Bag {i}" for i in range(1, 7))
IMPURITIES = ("LV CoA HCP", "LV CoA pDNA", "LV CoA cDNA", "LV CoA E1a cDNA", "LV CoA E1b cDNA")
GLUCOSE_PER_E6 = ("D6 Glucose / E6 cells", "D8 Glucose / E6 cells", "D10 Glucose / E6 cells")
LACTATE_PER_E6 = ("D6 Lact / E6 cells", "D8 Lact / E6 cells", "D10 Lact / E6 cells")
METABOLITE_DAYS = (6.0, 8.0, 10.0)


def _pooled_vector_volume(df: pd.DataFrame) -> pd.Series:
    return _row_sum(df, (VOL_A, VOL_B))


def _pooled_seeded_cells(df: pd.DataFrame) -> pd.Series:
    """Cells the vessels were seeded with: the recorded total, else the vessels."""
    if CELLS_AVAILABLE in df.columns:
        return _num(df[CELLS_AVAILABLE])
    return _row_sum(df, (CELLS_A, CELLS_B))


# ---------------------------------------------------------------------------
# Feature families — written once, expanded into entries
# ---------------------------------------------------------------------------
def _spread_features(
    name: str, unit: str, column_a: str, column_b: str, what: str
) -> list[Feature]:
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
            name=f"{name}_{statistic}_AB{unit}",
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

    The PHF names a CO2 or temperature column per reading and per vessel, and the
    set differs between sites, so these are read off the columns rather than
    listed.
    """
    features: list[Feature] = []
    for column in columns:
        if "CO2 Saturation (%)" in column:
            target, suffix, unit = params.co2_target, f"dev_from_{params.co2_target}pct", "CO2"
        elif "Temperature (°C)" in column:
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


def _contact_time_features(column: str, label: str, params: Params) -> list[Feature]:
    """Deviation from, and compliance with, the transduction contact time."""
    low = params.contact_target - params.contact_tolerance
    high = params.contact_target + params.contact_tolerance
    suffix = f"_{label}" if label else ""
    return [
        Feature(
            name=f"D3_contact_time_deviation{suffix}",
            requires=(column,),
            compute=lambda d, c=column: (_num(d[c]) - params.contact_target).abs(),
            group="Contact time",
            process_day="D3",
            value_type="numeric",
            rationale=f"Minutes away from the {params.contact_target:.0f} min "
            f"transduction contact time{' for ' + label.replace('_', ' ') if label else ''}",
        ),
        Feature(
            name=f"D3_contact_time_within_window{suffix}",
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
            "Log (PT CD4:CD8)",
            ("Post Thaw FLOW CD4+ (%)", "Post Thaw FLOW CD8+ (%)"),
            lambda d: np.log10(d["Post Thaw FLOW CD4+ (%)"] / d["Post Thaw FLOW CD8+ (%)"]),
            "Flow and phenotype",
            "D0",
            "numeric",
            "CD4 to CD8 balance of the incoming material, on a log scale so a "
            "two-fold shift reads the same in either direction",
        ),
        Feature(
            "Log (PE CD4:CD8)",
            ("Post Wash POS FLOW CD4+ (%)", "Post Wash POS FLOW CD8+ (%)"),
            lambda d: np.log10(d["Post Wash POS FLOW CD4+ (%)"] / d["Post Wash POS FLOW CD8+ (%)"]),
            "Flow and phenotype",
            "D0",
            "numeric",
            "CD4 to CD8 balance after enrichment, on a log scale",
        ),
        Feature(
            "PT Total  CD3+ (Viable Cells)",
            ("Post Thaw Viable Cell Count after sampling(cells)", "Post Thaw FLOW CD3+ (%)"),
            lambda d: (
                d["Post Thaw Viable Cell Count after sampling(cells)"]
                * d["Post Thaw FLOW CD3+ (%)"]
            ),
            "Flow and phenotype",
            "D0",
            "numeric",
            "Absolute CD3+ cells post thaw — the population the process actually works with",
        ),
        Feature(
            "PT CD3+CD4+ (%)",
            ("Post Thaw FLOW CD4+ (%)", "Post Thaw FLOW CD3+ (%)"),
            lambda d: d["Post Thaw FLOW CD4+ (%)"] * d["Post Thaw FLOW CD3+ (%)"] / 100,
            "Flow and phenotype",
            "D0",
            "numeric",
            "CD4+ as a share of all cells rather than of the CD3+ gate",
        ),
        Feature(
            "PT Total CD4+ (Viable Cells)",
            ("PT CD3+CD4+ (%)", "Post Thaw Viable Cell Count after sampling(cells)"),
            lambda d: d["PT CD3+CD4+ (%)"] * d["Post Thaw Viable Cell Count after sampling(cells)"],
            "Flow and phenotype",
            "D0",
            "numeric",
            "Absolute CD3+CD4+ cells post thaw",
        ),
        Feature(
            "PE CD3+CD4+ (%)",
            ("Post Wash POS FLOW CD4+ (%)", "Post Wash POS FLOW CD3+ (%)"),
            lambda d: d["Post Wash POS FLOW CD4+ (%)"] * d["Post Wash POS FLOW CD3+ (%)"] / 100,
            "Flow and phenotype",
            "D0",
            "numeric",
            "CD4+ as a share of all cells after enrichment",
        ),
        Feature(
            "PE CD3+CD8+ (%)",
            ("Post Wash POS FLOW CD8+ (%)", "Post Wash POS FLOW CD3+ (%)"),
            lambda d: d["Post Wash POS FLOW CD8+ (%)"] * d["Post Wash POS FLOW CD3+ (%)"] / 100,
            "Flow and phenotype",
            "D0",
            "numeric",
            "CD8+ as a share of all cells after enrichment",
        ),
        # ── Metabolites ───────────────────────────────────────────────────
        Feature(
            "Harvest Lactate Normalized",
            (GREX_COUNT, "Harvest Lactate (g/L)", VSVG_CELLS),
            lambda d: (d["Harvest Lactate (g/L)"] * 1_000_000_000) * d[GREX_COUNT] / d[VSVG_CELLS],
            "Metabolites",
            "D10",
            "numeric",
            "Harvest lactate per cell seeded, so batches seeded into one or two "
            "G-Rex can be compared",
        ),
        Feature(
            "D6 Lact / E6 cells",
            (GREX_COUNT, "Day 6 Lactate G-Rex A (g/L)", "Day 6 Lactate G-Rex B (g/L)", VSVG_CELLS),
            lambda d: np.where(
                d[GREX_COUNT] == 1,
                d["Day 6 Lactate G-Rex A (g/L)"] * 1e9 / d[VSVG_CELLS],
                (d["Day 6 Lactate G-Rex A (g/L)"] + d["Day 6 Lactate G-Rex B (g/L)"])
                * 1e9
                / d[VSVG_CELLS],
            ),
            "Metabolites",
            "D6",
            "numeric",
            "Lactate produced per million cells seeded, summed over the vessels in use",
        ),
        Feature(
            "D8 Lact / E6 cells",
            (GREX_COUNT, "Day 8 Lactate G-Rex A (g/L)", "Day 8 Lactate G-Rex B (g/L)", VSVG_CELLS),
            lambda d: np.where(
                d[GREX_COUNT] == 1,
                d["Day 8 Lactate G-Rex A (g/L)"] * 1e9 / d[VSVG_CELLS],
                (d["Day 8 Lactate G-Rex A (g/L)"] + d["Day 8 Lactate G-Rex B (g/L)"])
                * 1e9
                / d[VSVG_CELLS],
            ),
            "Metabolites",
            "D8",
            "numeric",
            "Lactate produced per million cells seeded, summed over the vessels in use",
        ),
        Feature(
            "D10 Lact / E6 cells",
            (GREX_COUNT, "Harvest Lactate (g/L)", VSVG_CELLS),
            lambda d: np.where(
                d[GREX_COUNT] == 1,
                d["Harvest Lactate (g/L)"] * 1e9 / d[VSVG_CELLS],
                d["Harvest Lactate (g/L)"] * 2e9 / d[VSVG_CELLS],
            ),
            "Metabolites",
            "D10",
            "numeric",
            "Harvest lactate per million cells seeded; the second vessel is "
            "sampled once and counted twice",
        ),
        Feature(
            "D6 Glucose / E6 cells",
            (GREX_COUNT, "Day 6 Glucose G-Rex A (g/L)", "Day 6 Glucose G-Rex B (g/L)", VSVG_CELLS),
            lambda d: np.where(
                d[GREX_COUNT] == 1,
                d["Day 6 Glucose G-Rex A (g/L)"] * 1e9 / d[VSVG_CELLS],
                (d["Day 6 Glucose G-Rex A (g/L)"] + d["Day 6 Glucose G-Rex B (g/L)"])
                * 1e9
                / d[VSVG_CELLS],
            ),
            "Metabolites",
            "D6",
            "numeric",
            "Glucose remaining per million cells seeded, summed over the vessels in use",
        ),
        Feature(
            "D8 Glucose / E6 cells",
            (GREX_COUNT, "Day 8 Glucose G-Rex A (g/L)", "Day 8 Glucose G-Rex B (g/L)", VSVG_CELLS),
            lambda d: np.where(
                d[GREX_COUNT] == 1,
                d["Day 8 Glucose G-Rex A (g/L)"] * 1e9 / d[VSVG_CELLS],
                (d["Day 8 Glucose G-Rex A (g/L)"] + d["Day 8 Glucose G-Rex B (g/L)"])
                * 1e9
                / d[VSVG_CELLS],
            ),
            "Metabolites",
            "D8",
            "numeric",
            "Glucose remaining per million cells seeded, summed over the vessels in use",
        ),
        Feature(
            "D10 Glucose / E6 cells",
            (GREX_COUNT, "Harvest Glucose (g/L)", VSVG_CELLS),
            lambda d: np.where(
                d[GREX_COUNT] == 1,
                d["Harvest Glucose (g/L)"] * 1e9 / d[VSVG_CELLS],
                d["Harvest Glucose (g/L)"] * 2 * 1e9 / d[VSVG_CELLS],
            ),
            "Metabolites",
            "D10",
            "numeric",
            "Harvest glucose per million cells seeded",
        ),
        Feature(
            "Lactate_perE6_slope_D6_D10",
            (LACTATE_PER_E6,),
            lambda d: slope(d, METABOLITE_DAYS, LACTATE_PER_E6),
            "Metabolites",
            "D10",
            "numeric",
            "How fast normalised lactate climbs across expansion, as one number "
            "instead of three readings",
        ),
        Feature(
            "Lactate_perE6_AUC_D6_D10",
            (LACTATE_PER_E6,),
            lambda d: auc(d, METABOLITE_DAYS, LACTATE_PER_E6),
            "Metabolites",
            "D10",
            "numeric",
            "Total normalised lactate exposure across expansion",
        ),
        Feature(
            "Glucose_perE6_slope_D6_D10",
            (GLUCOSE_PER_E6,),
            lambda d: slope(d, METABOLITE_DAYS, GLUCOSE_PER_E6),
            "Metabolites",
            "D10",
            "numeric",
            "How fast normalised glucose falls across expansion",
        ),
        Feature(
            "Glucose_perE6_AUC_D6_D10",
            (GLUCOSE_PER_E6,),
            lambda d: auc(d, METABOLITE_DAYS, GLUCOSE_PER_E6),
            "Metabolites",
            "D10",
            "numeric",
            "Total normalised glucose exposure across expansion",
        ),
        *[
            Feature(
                f"Lactate_to_Glucose_perE6_ratio_D{int(day)}",
                (lactate, glucose),
                lambda d, lact=lactate, gluc=glucose: safe_div(d[lact], d[gluc]),
                "Metabolites",
                f"D{int(day)}",
                "numeric",
                "Lactate against glucose on the same day, which separates cells "
                "burning sugar fast from cells simply being fed more",
            )
            for day, lactate, glucose in zip(METABOLITE_DAYS, LACTATE_PER_E6, GLUCOSE_PER_E6)
        ],
        # ── Growth and CD4 profile ────────────────────────────────────────
        Feature(
            "growth profile",
            ("cPDL (D3 to D10)", "Harvest Lactate Normalized"),
            lambda d: np.select(
                [
                    (d["cPDL (D3 to D10)"] < params.cpdl_low)
                    | (d["Harvest Lactate Normalized"] < params.lactate_low),
                    (d["cPDL (D3 to D10)"] > params.cpdl_high)
                    & (d["Harvest Lactate Normalized"] > params.lactate_high),
                ],
                ["undergrowth", "overgrowth"],
                default="normal",
            ),
            "Profiles",
            "D10",
            "categorical",
            "Undergrowth, normal or overgrowth, from doublings and normalised "
            "lactate together; thresholds are in config",
        ),
        Feature(
            "CD4 profile",
            ("Post Thaw FLOW CD4+ (%)",),
            lambda d: pd.Series(
                np.select(
                    [
                        d["Post Thaw FLOW CD4+ (%)"] <= params.cd4_low,
                        d["Post Thaw FLOW CD4+ (%)"] >= params.cd4_high,
                    ],
                    ["low CD4", "high CD4"],
                    default="medium CD4",
                ),
                index=d.index,
            ).where(d["Post Thaw FLOW CD4+ (%)"].notna()),
            "Profiles",
            "D0",
            "categorical",
            "Low, medium or high CD4 share of the incoming material; a batch with "
            "no reading gets no profile rather than the middle one",
        ),
        # ── Viability ─────────────────────────────────────────────────────
        Feature(
            "Drop viability D3 vs D0 (%)",
            ("Day 3 Average of Pooled Viability (%)", "Post Wash POS Viability (%)"),
            lambda d: (
                (d["Day 3 Average of Pooled Viability (%)"] - d["Post Wash POS Viability (%)"])
                / d["Post Wash POS Viability (%)"]
                * 100
            ),
            "Viability",
            "D3",
            "numeric",
            "Viability change over activation, relative to where it started",
        ),
        Feature(
            "Viability_drop_D3_vs_D0_pct",
            ("Day 3 Average of Pooled Viability (%)", "Post Wash POS Viability (%)"),
            lambda d: (
                _num(d["Day 3 Average of Pooled Viability (%)"])
                - _num(d["Post Wash POS Viability (%)"])
            ),
            "Viability",
            "D3",
            "numeric",
            "The same change in percentage points rather than as a relative "
            "change, which keeps it readable when the starting viability is low",
        ),
        Feature(
            "D3-D10 viability change (%)",
            ("Day 3 Average of Pooled Viability (%)", "Harvest Pre-Wash Cell Viability (%)"),
            lambda d: (
                (
                    d["Day 3 Average of Pooled Viability (%)"]
                    - d["Harvest Pre-Wash Cell Viability (%)"]
                )
                / d["Day 3 Average of Pooled Viability (%)"]
                * 100
            ),
            "Viability",
            "D10",
            "numeric",
            "Viability lost between activation and harvest, relative to D3",
        ),
        Feature(
            "Viability change during expansion",
            ("Harvest Pre-Wash Cell Viability (%)", "Day 3 Average of Pooled Viability (%)"),
            lambda d: (
                (
                    d["Harvest Pre-Wash Cell Viability (%)"]
                    - d["Day 3 Average of Pooled Viability (%)"]
                )
                / d["Day 3 Average of Pooled Viability (%)"]
                * 100
            ),
            "Viability",
            "D10",
            "numeric",
            "The same comparison signed the other way, as the Power Query investigation reports it",
        ),
        Feature(
            "Viability_change_D3_to_D10_pct",
            ("Day 3 Average of Pooled Viability (%)", "Harvest Post Wash Viability Average (%)"),
            lambda d: (
                _num(d["Harvest Post Wash Viability Average (%)"])
                - _num(d["Day 3 Average of Pooled Viability (%)"])
            ),
            "Viability",
            "D10",
            "numeric",
            "Net viability change from activation to the washed harvest, in percentage points",
        ),
        Feature(
            "D10 harvest - D10 wash viability change (%)",
            ("Harvest Post Wash Viability Average (%)", "Harvest Pre-Wash Cell Viability (%)"),
            lambda d: (
                (
                    d["Harvest Post Wash Viability Average (%)"]
                    - d["Harvest Pre-Wash Cell Viability (%)"]
                )
                / d["Harvest Pre-Wash Cell Viability (%)"]
                * 100
            ),
            "Viability",
            "D10",
            "numeric",
            "What the LOVO wash did to viability",
        ),
        # ── Seeding and vessel balance ────────────────────────────────────
        Feature(
            "D3 % cells seeded / target",
            (GREX_COUNT, VSVG_CELLS),
            lambda d: np.select(
                [d[GREX_COUNT] == 1, d[GREX_COUNT] == 2],
                [d[VSVG_CELLS] / 500_000, d[VSVG_CELLS] / 1_000_000],
                default=np.nan,
            ),
            "Seeding",
            "D3",
            "numeric",
            "Cells seeded against the target for the number of vessels used",
        ),
        Feature(
            "D3_seeding_error_pct",
            ("D3 % cells seeded / target",),
            lambda d: (_num(d["D3 % cells seeded / target"]) - 100.0).abs(),
            "Seeding",
            "D3",
            "numeric",
            "Distance from target seeding in either direction, so over- and "
            "under-seeding count the same",
        ),
        Feature(
            "D3_seeding_within_5pct",
            ("D3 % cells seeded / target",),
            lambda d: (
                (_num(d["D3 % cells seeded / target"]) - 100.0).abs() <= params.seeding_tolerance
            ).astype(float),
            "Seeding",
            "D3",
            "boolean",
            f"Whether seeding landed within ±{params.seeding_tolerance:.0f}% of target",
        ),
        Feature(
            "Avg_cells_per_bag_D3",
            (CELLS_AVAILABLE, "# of Culture Bags"),
            lambda d: safe_div(d[CELLS_AVAILABLE], d["# of Culture Bags"]),
            "Seeding",
            "D3",
            "numeric",
            "Cells per culture bag, which is the density the cells actually see",
        ),
        *_spread_features("Seeded_cells", "", CELLS_A, CELLS_B, "seeded viable cells"),
        *_spread_features("Vector_volume", "_mL", VOL_A, VOL_B, "vector volume added"),
        # ── MOI ───────────────────────────────────────────────────────────
        Feature(
            "MOI_eff_G-Rex_A",
            (TITER, VOL_A, CELLS_A),
            lambda d: _first(d, *TITER) * safe_div(d[VOL_A], d[CELLS_A]),
            "MOI",
            "D3",
            "numeric",
            "Infectious units per seeded cell in vessel A, from the measured "
            "titer rather than the nominal one",
        ),
        Feature(
            "log_MOI_eff_G-Rex_A",
            ("MOI_eff_G-Rex_A",),
            lambda d: np.log1p(_num(d["MOI_eff_G-Rex_A"])),
            "MOI",
            "D3",
            "numeric",
            "Vessel A MOI on a log scale, where its spread is even enough to model",
        ),
        Feature(
            "MOI_eff_G-Rex_B",
            (TITER, VOL_B, CELLS_B),
            lambda d: _first(d, *TITER) * safe_div(d[VOL_B], d[CELLS_B]),
            "MOI",
            "D3",
            "numeric",
            "Infectious units per seeded cell in vessel B",
        ),
        Feature(
            "log_MOI_eff_G-Rex_B",
            ("MOI_eff_G-Rex_B",),
            lambda d: np.log1p(_num(d["MOI_eff_G-Rex_B"])),
            "MOI",
            "D3",
            "numeric",
            "Vessel B MOI on a log scale",
        ),
        Feature(
            "MOI_eff_pool",
            (TITER, VOL_A, VOL_B, (CELLS_AVAILABLE, CELLS_A)),
            lambda d: (
                _first(d, *TITER) * safe_div(_pooled_vector_volume(d), _pooled_seeded_cells(d))
            ),
            "MOI",
            "D3",
            "numeric",
            "Infectious units per seeded cell over the whole batch; a vessel that "
            "was not used contributes nothing rather than making the batch missing",
        ),
        Feature(
            "log_MOI_eff_pool",
            ("MOI_eff_pool",),
            lambda d: np.log1p(_num(d["MOI_eff_pool"])),
            "MOI",
            "D3",
            "numeric",
            "Pooled MOI on a log scale",
        ),
        Feature(
            "MOI_eff_quality_adjusted",
            ("MOI_eff_pool", "LV CoA PI Ratio"),
            lambda d: _num(d["MOI_eff_pool"]) * _num(d["LV CoA PI Ratio"]),
            "MOI",
            "D3",
            "numeric",
            "Pooled MOI weighted by how many particles in the lot are infectious, "
            "so two lots at the same titer are not treated as the same vector",
        ),
        Feature(
            "MOI_per_CD3_pool",
            (TITER, VOL_A, VOL_B, "PT Total  CD3+ (Viable Cells)"),
            lambda d: (
                _first(d, *TITER)
                * safe_div(_pooled_vector_volume(d), d["PT Total  CD3+ (Viable Cells)"])
            ),
            "MOI",
            "D3",
            "numeric",
            "Infectious units per CD3+ cell rather than per cell of any kind — "
            "only T cells are the target",
        ),
        # ── Contact time ──────────────────────────────────────────────────
        *_contact_time_features("Day 3 Incubation Time (120min+/-30 min)", "", params),
        *_contact_time_features(
            "Amount of Time G-Rex A in Incubator (120min+/-30 min)", "G-Rex_A", params
        ),
        *_contact_time_features(
            "Amount of Time G-Rex B in Incubator (120min+/-30 min)", "G-Rex_B", params
        ),
        # ── Incubation ────────────────────────────────────────────────────
        Feature(
            "Incubation D0-D1 (min)",
            (
                "Date and Time Cells Removed From Incubator Day 1",
                "Date and Time Cells placed in Incubator Day 0",
            ),
            lambda d: (
                (
                    pd.to_datetime(d["Date and Time Cells Removed From Incubator Day 1"])
                    - pd.to_datetime(d["Date and Time Cells placed in Incubator Day 0"])
                ).dt.total_seconds()
                / 60
            ),
            "Incubation",
            "D1",
            "numeric",
            "Minutes the cells spent in the incubator between day 0 and day 1",
        ),
        Feature(
            "Incubation D1-D3",
            (
                "Date and Time Cells Removed From Incubator Day 3",
                "Date and Time Cells placed in Incubator Day 1",
            ),
            lambda d: (
                (
                    pd.to_datetime(d["Date and Time Cells Removed From Incubator Day 3"])
                    - pd.to_datetime(d["Date and Time Cells placed in Incubator Day 1"])
                ).dt.total_seconds()
                / 60
            ),
            "Incubation",
            "D3",
            "numeric",
            "Minutes in the incubator between day 1 and day 3",
        ),
        Feature(
            "Incubation D3-D6 (min)",
            (
                "Date and Time Cells Removed From Incubator Day 6",
                "Day 3 Date and Time G-Rex A Returned to Incubator",
            ),
            lambda d: (
                (
                    pd.to_datetime(d["Date and Time Cells Removed From Incubator Day 6"])
                    - pd.to_datetime(d["Day 3 Date and Time G-Rex A Returned to Incubator"])
                ).dt.total_seconds()
                / 60
            ),
            "Incubation",
            "D6",
            "numeric",
            "Minutes in the incubator between day 3 and day 6",
        ),
        Feature(
            "Incubation D6-D8 (min)",
            (
                "Date and Time Cells Removed From Incubator Day 8",
                "Day 6 Date and Time G-Rex A Returned to Incubator",
            ),
            lambda d: (
                (
                    pd.to_datetime(d["Date and Time Cells Removed From Incubator Day 8"])
                    - pd.to_datetime(d["Day 6 Date and Time G-Rex A Returned to Incubator"])
                ).dt.total_seconds()
                / 60
            ),
            "Incubation",
            "D8",
            "numeric",
            "Minutes in the incubator between day 6 and day 8",
        ),
        Feature(
            "Incubation D8-D10 (min)",
            (
                "Date and Time Cells Removed From Incubator Day 10",
                "Day 8 Date and Time G-Rex A Returned to Incubator",
            ),
            lambda d: (
                (
                    pd.to_datetime(d["Date and Time Cells Removed From Incubator Day 10"])
                    - pd.to_datetime(d["Day 8 Date and Time G-Rex A Returned to Incubator"])
                ).dt.total_seconds()
                / 60
            ),
            "Incubation",
            "D10",
            "numeric",
            "Minutes in the incubator between day 8 and day 10",
        ),
        Feature(
            "Incubation_total_D3_to_D10_min",
            ("Incubation D3-D6 (min)", "Incubation D6-D8 (min)", "Incubation D8-D10 (min)"),
            lambda d: (
                _num(d["Incubation D3-D6 (min)"])
                + _num(d["Incubation D6-D8 (min)"])
                + _num(d["Incubation D8-D10 (min)"])
            ),
            "Incubation",
            "D10",
            "numeric",
            "Total time in the incubator across expansion; a batch handled slowly "
            "at one step can still end up with a normal total",
        ),
        # ── Clumps ────────────────────────────────────────────────────────
        Feature(
            "Clump_count_reduction_fraction",
            (CLUMPS_PRE_COUNT, CLUMPS_POST_COUNT),
            lambda d: 1.0 - safe_div(d[CLUMPS_POST_COUNT], d[CLUMPS_PRE_COUNT]),
            "Clumps",
            "D1",
            "numeric",
            "Share of the clumps that massaging removed, which measures the "
            "intervention rather than the starting material",
        ),
        Feature(
            "Clump_size_reduction_fraction",
            (CLUMPS_PRE_SIZE, CLUMPS_POST_SIZE),
            lambda d: 1.0 - safe_div(d[CLUMPS_POST_SIZE], d[CLUMPS_PRE_SIZE]),
            "Clumps",
            "D1",
            "numeric",
            "Share of the clump size that massaging removed",
        ),
        Feature(
            "Clump_burden_weighted_pre",
            (CLUMPS_PRE_COUNT, CLUMPS_PRE_SIZE),
            lambda d: _num(d[CLUMPS_PRE_COUNT]) * _num(d[CLUMPS_PRE_SIZE]),
            "Clumps",
            "D1",
            "numeric",
            "Count and size together, so many small clumps and one large one are "
            "not recorded as the same burden",
        ),
        Feature(
            "Clump_burden_weighted_post_massage",
            (CLUMPS_POST_COUNT, CLUMPS_POST_SIZE),
            lambda d: _num(d[CLUMPS_POST_COUNT]) * _num(d[CLUMPS_POST_SIZE]),
            "Clumps",
            "D1",
            "numeric",
            "Size-weighted clump burden left after massaging",
        ),
        Feature(
            "Clump_burden_weighted_post_mixing",
            (CLUMPS_MIX_COUNT, CLUMPS_MIX_SIZE),
            lambda d: _num(d[CLUMPS_MIX_COUNT]) * _num(d[CLUMPS_MIX_SIZE]),
            "Clumps",
            "D1",
            "numeric",
            "Size-weighted clump burden left after mixing",
        ),
        Feature(
            "Mixing_cycles_x_pre_activation_clump_count",
            ("Day 1 Number of Mixing Cycles Post Activation", CLUMPS_PRE_COUNT),
            lambda d: (
                _num(d["Day 1 Number of Mixing Cycles Post Activation"]) * _num(d[CLUMPS_PRE_COUNT])
            ),
            "Clumps",
            "D1",
            "numeric",
            "Mixing effort against the burden it was applied to; neither number "
            "means much without the other",
        ),
        Feature(
            "D3_clumps_flag",
            ("day 3 clump pattern",),
            lambda d: yn_to_bool(d["day 3 clump pattern"]),
            "Clumps",
            "D3",
            "boolean",
            "Whether clumps were still recorded at day 3",
        ),
        Feature(
            "D10_clump_bag_count",
            (CLUMP_BAGS,),
            lambda d: sum(yn_to_bool(d[c]) for c in _present(d, CLUMP_BAGS)),
            "Clumps",
            "D10",
            "numeric",
            "How many harvest bags had clumps, counted over the bags the batch actually filled",
        ),
        # ── Dose and CAR ──────────────────────────────────────────────────
        Feature(
            "Dose (CAR+ viable cells/kg) all weights",
            ("Dose (CAR+ viable cells/kg)", "Dose: Number of CAR+ Viable T-Cells (cells)"),
            lambda d: np.select(
                [
                    d["Dose (CAR+ viable cells/kg)"].notna(),
                    d["Dose: Number of CAR+ Viable T-Cells (cells)"].notna(),
                ],
                [
                    d["Dose (CAR+ viable cells/kg)"] / 1_000_000,
                    d["Dose: Number of CAR+ Viable T-Cells (cells)"] / 100_000_000,
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
            "Post formulation dose",
            (
                "Final Formulation CS5 Viable Cell Concentration Average (cells/mL)",
                "Volume per Bag (mL)",
                "Harvest Pre-Wash Flow CAR+ Expression (%)",
                "Subject Weight (kg)",
            ),
            lambda d: (
                d["Final Formulation CS5 Viable Cell Concentration Average (cells/mL)"]
                / 1_000_000
                * d["Volume per Bag (mL)"]
                * d["Harvest Pre-Wash Flow CAR+ Expression (%)"]
                / 100
                / d["Subject Weight (kg)"]
            ),
            "Dose and CAR",
            "D10",
            "numeric",
            "Dose per kilogram implied by what actually went into the bag",
        ),
        Feature(
            "D10 Achievable Dose (E6 CAR+ viable T-cells)",
            (
                "Harvest Pre-Wash Flow CAR+ Expression (%)",
                "Harvest Post Wash Viable Cell Count (cells)",
                "Subject Weight (kg)",
            ),
            lambda d: (
                d["Harvest Pre-Wash Flow CAR+ Expression (%)"]
                / 100
                * d["Harvest Post Wash Viable Cell Count (cells)"]
                / (d["Subject Weight (kg)"] * 1_000_000)
            ),
            "Dose and CAR",
            "D10",
            "numeric",
            "The dose the harvest could have supported, against the dose that was given",
        ),
        Feature(
            "D10 Total CAR+ viable T-cells achieved",
            (
                "Harvest Pre-Wash Flow CAR+ Expression (%)",
                "Harvest Post Wash Viable Cell Count (cells)",
            ),
            lambda d: (
                d["Harvest Pre-Wash Flow CAR+ Expression (%)"]
                * d["Harvest Post Wash Viable Cell Count (cells)"]
                / 100
            ),
            "Dose and CAR",
            "D10",
            "numeric",
            "Absolute CAR+ cells at harvest",
        ),
        Feature(
            "CAR T produced per cell engaged D3",
            ("Harvest Pre-Wash Flow CAR+ Expression (%)", "cPDL (D3 to D10)"),
            lambda d: (
                d["Harvest Pre-Wash Flow CAR+ Expression (%)"]
                / 100
                * np.power(2, d["cPDL (D3 to D10)"])
            ),
            "Dose and CAR",
            "D10",
            "numeric",
            "CAR+ cells returned per cell seeded, which folds expansion and "
            "transduction into one yield",
        ),
        Feature(
            "VCN/cell",
            ("FP Flow CAR+ (%)", "Provirus Vector Copy Number (copies/transduced cell)"),
            lambda d: (
                d["FP Flow CAR+ (%)"]
                * d["Provirus Vector Copy Number (copies/transduced cell)"]
                / 100
            ),
            "Dose and CAR",
            "Final product",
            "numeric",
            "Copies per cell across the whole product rather than per transduced cell",
        ),
        Feature(
            "Flow accuracy (effective)",
            ("FP Flow CAR+ (%)", "Harvest Pre-Wash Flow CAR+ Expression (%)"),
            lambda d: d["FP Flow CAR+ (%)"] / d["Harvest Pre-Wash Flow CAR+ Expression (%)"],
            "Dose and CAR",
            "Final product",
            "numeric",
            "Final product CAR+ against the harvest reading — a ratio far from one "
            "points at the assay, not the process",
        ),
        # ── Formulation and wash ──────────────────────────────────────────
        Feature(
            "Total CS5 cell concentration (E6 cells/mL)",
            (
                "Final Formulation CS5 Viable Cell Concentration Average (cells/mL)",
                "Final Formulation CS5 Viability Average (%)",
            ),
            lambda d: (
                d["Final Formulation CS5 Viable Cell Concentration Average (cells/mL)"]
                / (d["Final Formulation CS5 Viability Average (%)"] / 100)
                / 1_000_000
            ),
            "Formulation and wash",
            "D10",
            "numeric",
            "Total cells in the formulated suspension, dead ones included",
        ),
        Feature(
            "total viable cells used for formulation",
            (
                "Harvest Post Wash Viable Cell Concentration Average (cells/mL)",
                "Volume of Cells Used for Formulation (mL)",
            ),
            lambda d: (
                d["Harvest Post Wash Viable Cell Concentration Average (cells/mL)"]
                * d["Volume of Cells Used for Formulation (mL)"]
            ),
            "Formulation and wash",
            "D10",
            "numeric",
            "Viable cells that went into formulation",
        ),
        Feature(
            "total viable cells in formulated suspension",
            (
                "Final Formulation CS5 Viable Cell Concentration Average (cells/mL)",
                "Volume CS5 Used for Formulation (mL)",
            ),
            lambda d: (
                d["Final Formulation CS5 Viable Cell Concentration Average (cells/mL)"]
                * d["Volume CS5 Used for Formulation (mL)"]
            ),
            "Formulation and wash",
            "D10",
            "numeric",
            "Viable cells that came out of formulation",
        ),
        Feature(
            "number of viable cells lost during formulation",
            (
                "total viable cells used for formulation",
                "total viable cells in formulated suspension",
            ),
            lambda d: (
                d["total viable cells used for formulation"]
                - d["total viable cells in formulated suspension"]
            ),
            "Formulation and wash",
            "D10",
            "numeric",
            "The difference between those two, as cells",
        ),
        Feature(
            "% cells lost during formulation",
            (
                "number of viable cells lost during formulation",
                "total viable cells used for formulation",
            ),
            lambda d: (
                d["number of viable cells lost during formulation"]
                / d["total viable cells used for formulation"]
                * 100
            ),
            "Formulation and wash",
            "D10",
            "numeric",
            "The same loss as a share, which is comparable across batch sizes",
        ),
        Feature(
            "total cells lost during LOVO wash (cells)",
            (
                "Harvest Pre-Wash Viable Cell Count after Sampling (cells)",
                "Harvest Post Wash Viable Cell Count (cells)",
            ),
            lambda d: (
                d["Harvest Pre-Wash Viable Cell Count after Sampling (cells)"]
                - d["Harvest Post Wash Viable Cell Count (cells)"]
            ),
            "Formulation and wash",
            "D10",
            "numeric",
            "Cells the wash did not return",
        ),
        # ── Vector CoA ────────────────────────────────────────────────────
        Feature(
            "LVCoA_Impurity_Index_zsum",
            (IMPURITIES,),
            lambda d: sum(zscore(d[c]) for c in _present(d, IMPURITIES)),
            "Vector CoA",
            "Vector CoA",
            "numeric",
            "Residual impurities on one scale, summed. Standardised within the "
            "site being processed, not across the merged PVF, so the two sites' "
            "values are not directly comparable",
        ),
        # ── IL-2 ──────────────────────────────────────────────────────────
        Feature(
            "IL2_IU_total_D6",
            ("Day 6 Activity of IL-2 (IU/mg)", "Day 6 IL-2 Protein Content (µg/vial)"),
            lambda d: (
                _num(d["Day 6 Activity of IL-2 (IU/mg)"])
                * _num(d["Day 6 IL-2 Protein Content (µg/vial)"])
                / 1000.0
            ),
            "IL-2",
            "D6",
            "numeric",
            "IL-2 units fed at day 6: activity and protein mass mean little apart",
        ),
        Feature(
            "IL2_IU_total_D8",
            ("Day 8 Activity of IL-2 (IU/mg)", "Day 8 IL-2 Protein Content (µg/vial)"),
            lambda d: (
                _num(d["Day 8 Activity of IL-2 (IU/mg)"])
                * _num(d["Day 8 IL-2 Protein Content (µg/vial)"])
                / 1000.0
            ),
            "IL-2",
            "D8",
            "numeric",
            "IL-2 units fed at day 8",
        ),
        # ── Context ───────────────────────────────────────────────────────
        Feature(
            "Shift",
            ("Thaw Start Time",),
            lambda d: np.where(
                pd.to_datetime(d["Thaw Start Time"].astype(str)).dt.time
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
#: The impurity concentration columns the certificate loader censors and flags.
_COA_IMPURITIES = ("HCP", "pDNA", "cDNA", "E1a cDNA", "E1b cDNA")
#: Every certificate column the loader coerces, and therefore may flag as absent.
_COA_MEASURED = (
    "LV Titer",
    "Physical Titer p24 ELISA",
    "PI Ratio",
    "CAR% - Donor 1",
    "CAR% - Donor 2",
    "CAR% - Average",
    "IFNg - Donor 1",
    "IFNg - Donor 2",
    "IFNg - Average",
    "pH",
    "Osmolality",
    "Volume",
    "DNA size",
    *_COA_IMPURITIES,
    *(f"Calc {name}" for name in _COA_IMPURITIES),
)


def coa_lineage() -> dict[str, tuple[str, ...]]:
    """What every certificate-derived column is computed from.

    :func:`pvf.io.load_lv_coa` builds these while parsing the workbook, which is
    where the parsing belongs, but the tasks stage needs to know what they came
    from — a column whose lineage nobody recorded cannot be checked for descent
    from the target. The loader checks what it produced against this table, so
    the two cannot drift apart in silence.
    """
    prefix = "LV CoA "
    lineage: dict[str, tuple[str, ...]] = {
        f"{prefix}Delivered Dose per GRex": (
            f"{prefix}LV Titer",
            f"{prefix}LV Titer (unadjusted)",
            f"{prefix}Volume",
        ),
        f"{prefix}Delivered Dose used adj titer": (f"{prefix}LV Titer (unadjusted)",),
        f"{prefix}Titer Adjustment Factor": (
            f"{prefix}LV Titer",
            f"{prefix}LV Titer (unadjusted)",
        ),
        f"{prefix}Specific Infectivity": (
            f"{prefix}LV Titer",
            f"{prefix}Physical Titer p24 ELISA",
        ),
        f"{prefix}E1b Fraction of Total DNA": tuple(
            f"{prefix}{name}" for name in ("pDNA", "cDNA", "E1a cDNA", "E1b cDNA")
        ),
        f"{prefix}pH Abs Deviation": (f"{prefix}pH",),
        f"{prefix}Osmolality Abs Deviation": (f"{prefix}Osmolality",),
    }
    for metric in ("CAR%", "IFNg"):
        donors = (f"{prefix}{metric} - Donor 1", f"{prefix}{metric} - Donor 2")
        for statistic in ("MAX", "MIN", "DIFF", "DIFF Rel", "AVG"):
            lineage[f"{prefix}{metric} Donor {statistic}"] = donors
    for name in _COA_IMPURITIES:
        lineage[f"{prefix}{name} below LOQ"] = (f"{prefix}{name}",)
    for name in _COA_MEASURED:
        lineage[f"{prefix}{name} missing"] = (f"{prefix}{name}",)
    return lineage


#: Every column that is arithmetic over another, and what it was computed from.
#: The registry's own entries are added by :func:`derived_sources`.
DERIVED_SOURCES: dict[str, tuple[str, ...]] = coa_lineage()

#: Columns whose value depends on the frame they were computed over — a z-score
#: is relative to the site's own batches, so it is not a fixed measurement and it
#: cannot be recomputed for one new batch. Exploratory only.
SITE_RELATIVE = ("LVCoA_Impurity_Index_zsum",)


# ---------------------------------------------------------------------------
# Running the registry
# ---------------------------------------------------------------------------
#: Statuses an entry can end in. "present" means a source already supplies the
#: column, so the feature stands aside rather than overwriting it.
CREATED, PRESENT, NOT_IN_PTF, BLOCKED, ERRORED = (
    "created",
    "present",
    "not in PTF",
    "blocked",
    "errored",
)

PROFILE_COLUMNS = ("growth profile", "CD4 profile")


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
    ptf_cols: Sequence[str],
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
    ptf_cols
        The PTF parameter names. A feature whose name is not among them is
        reported and not computed: the PTF is the schema of the PVF.
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
    ptf = set(ptf_cols)
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

        if feature.name not in ptf:
            entry["status"] = NOT_IN_PTF
            entry["note"] = "add to the PTF to have this computed"
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
        for column in PROFILE_COLUMNS
        if column in df.columns
    }

    log.info(
        MODULE,
        f"[{site}] features — {report.count(CREATED)} created, "
        f"{report.count(PRESENT)} already present, {report.count(BLOCKED)} blocked, "
        f"{report.count(NOT_IN_PTF)} not in PTF, {report.count(ERRORED)} failed",
    )
    for entry in report.entries:
        if entry["status"] == BLOCKED:
            log.warn(MODULE, f"[{site}] {entry['name']} blocked", entry["note"])
    if report.count(NOT_IN_PTF):
        log.warn(
            MODULE,
            f"[{site}] {report.count(NOT_IN_PTF)} features are not PTF parameters "
            "and were not computed",
            ", ".join(e["name"] for e in report.entries if e["status"] == NOT_IN_PTF),
        )

    log.success(MODULE, f"[{site}] Derived features complete")
    return report
