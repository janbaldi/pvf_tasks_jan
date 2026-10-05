"""Making the two sites' values mean the same thing.

The passes, in the order they run:

    1. String hygiene  (strip whitespace, \xa0, empty → NaN)
    2. Integrity corrections  (known bad values, OOS-type strings, etc.)
    3. Type coercion   (numeric, datetime, duration, ratio)
    4. Yes/No filling, where a task has said what a blank means there

Two things are settings rather than facts about the world. What a censored value
such as ``<0.5`` becomes is a policy — the operator carries information that
dropping it destroys, so the choice is recorded with the count of values it
touched. And a blank in a yes/no column is *unknown*, not "No": it is filled only
for the columns a config names, with the reason it names.

Returns a CleaningReport consumed by the reporter.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from .logger import log

MODULE = "clean"

#: What a value recorded as "< x" becomes. The operator is information: under the
#: limit of quantitation is low, not unknown, and not exactly the limit either.
CENSORED_POLICIES = {
    "limit": "the limit itself (0.5 for '<0.5')",
    "half": "half the limit, the usual substitution for a below-LOQ value",
    "missing": "missing, because a censored value is not a measurement",
}


# ── Report artefact ────────────────────────────────────────────────────────────


@dataclass
class CleaningReport:
    string_fixes: dict[str, int] = field(default_factory=dict)
    numeric_coercions: list[dict] = field(default_factory=list)
    datetime_failures: list[dict] = field(default_factory=list)
    duration_failures: list[dict] = field(default_factory=list)
    integrity_changes: list[dict] = field(default_factory=list)
    yesno_imputations: list[dict] = field(default_factory=list)
    categorical_diffs: list[dict] = field(default_factory=list)
    pct_out_of_range: list[dict] = field(default_factory=list)
    censored: list[dict] = field(default_factory=list)
    settings: dict = field(default_factory=dict)


# ── 1. String hygiene ──────────────────────────────────────────────────────────


def strip_strings(phf: dict[str, pd.DataFrame], ptf_cols: list[str]) -> CleaningReport:
    """Strip leading/trailing whitespace and \\xa0 from object columns."""
    log.step(MODULE, "Stripping whitespace and \\xa0 characters")
    report = CleaningReport()

    total = 0
    for col in ptf_cols:
        for site, df in phf.items():
            # Text arrives as object or, on newer pandas, as a string dtype.
            # Testing only for object silently skips every column on those
            # versions, which is a cleaning pass that reports having done nothing.
            if col in df.columns and (
                df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col])
            ):
                before = df[col].copy()
                df[col] = df[col].apply(
                    lambda x: str(x).strip().replace("\xa0", "") if pd.notna(x) else x
                )
                changed = (before != df[col]).sum()
                if changed:
                    total += changed

    # Empty strings → NaN across all sites
    for site in phf:
        phf[site].replace("", np.nan, inplace=True)

    report.string_fixes["cells_modified"] = total
    log.success(MODULE, f"String hygiene done — {total:,} cells modified")
    return report


# ── 2a. Integrity corrections ─────────────────────────────────────────────────


def apply_integrity_corrections(
    phf: dict[str, pd.DataFrame], settings: dict | None = None
) -> list[dict]:
    """The known data-quality fixes, each with how many rows it touched.

    These are business corrections someone established against the source
    systems, not general truths. The one that rewrites every Raritan batch's
    ``Type`` is the clearest case, so it is a setting: a task that is not about
    the commercial cohort should not have the column silently overwritten for it.
    """
    settings = settings or {}
    log.step(MODULE, "Applying integrity corrections")
    changes: list[dict] = []

    def record(site, col, description, rows=None, why=""):
        changes.append(
            {
                "site": site,
                "column": col,
                "correction": description,
                "rows": int(rows) if rows is not None else "",
                "why": why,
            }
        )
        log.info(MODULE, f"[{site}] {col}: {description}" + (f" ({rows} rows)" if rows else ""))

    r = phf["Raritan"]
    g = phf["Ghent"]

    # A clump count that is really a date Excel mangled. Compared as text,
    # because whether the column arrives as text or as a number is decided by
    # Excel's type inference, not by the value being wrong.
    mask = r["Post-Mixing Clumps: # of Clumps"].astype(str).str.strip() == "45691"
    if mask.any():
        r.loc[mask, "Post-Mixing Clumps: # of Clumps"] = np.nan
        record(
            "Raritan",
            "Post-Mixing Clumps: # of Clumps",
            "replaced '45691' with NaN",
            mask.sum(),
            "a clump count Excel turned into a date serial",
        )

    # Wrong country
    mask = r["Clinical Site"] == "107306"
    if mask.any():
        r.loc[mask, "Country"] = "Israel"
        record(
            "Raritan",
            "Country",
            "set to 'Israel' for clinical site 107306",
            mask.sum(),
            "the site is in Israel; the export has the wrong country",
        )

    if settings.get("raritan_type_commercial", True):
        wrong = int((r["Type"].astype("string") != "Commercial").sum())
        r["Type"] = "Commercial"
        record(
            "Raritan",
            "Type",
            "set every row to 'Commercial'",
            wrong,
            settings.get(
                "raritan_type_commercial_reason",
                "the Raritan export does not fill Type, and everything it holds is "
                "commercial manufacturing",
            ),
        )
    else:
        log.info(MODULE, "[Raritan] Type is left as the source recorded it")

    # OOS Type string normalisation — Raritan
    oos_replacements_raritan = [
        ("VIability", "Viability"),
        ("Low Dose", "Dose"),
        ("Phenotype (%CD3+, %NK)", "CD3%,NK Cell%"),
        ("Phenotype (NK%)", "NK Cell%"),
        ("Concentration", "Viable Cell Concentration"),
        ("CAR", "CAR%"),
        ("CAR%%", "CAR%"),
        ("Phenotype", "CD3%"),
        ("Potency", "IFN Gamma Potency"),
        ("IFN Gamma Potency", "IFN Gamma"),
    ]
    for old, new in oos_replacements_raritan:
        r["OOS Type"] = r["OOS Type"].str.replace(old, new, regex=False)
    record("Raritan", "OOS Type", f"{len(oos_replacements_raritan)} string normalisations applied")

    # OOS Type string normalisation — Ghent
    g["OOS Type"] = g["OOS Type"].str.replace("IFN Gamma Potency", "IFN Gamma", regex=False)
    record("Ghent", "OOS Type", "normalised 'IFN Gamma Potency' → 'IFN Gamma'")

    # Ghent known bad cell
    mask = g["Patient Lot/Batch #"] == "QCGS03V"
    if mask.any():
        g.loc[mask, "Pre-Activation Clumps: Y/N?"] = "No"
        record("Ghent", "Pre-Activation Clumps: Y/N?", "set to 'No' for lot QCGS03V")

    # Mycoplasma string fix
    # Assigned back rather than replaced in place: an in-place call on a column
    # selected out of a frame updates a copy and leaves the frame untouched.
    g["Mycoplasma"] = g["Mycoplasma"].replace("mycoplasma not detected", "not detected")
    record("Ghent", "Mycoplasma", "normalised 'mycoplasma not detected' → 'not detected'")

    # Raritan NC type
    r["Non-Conformance Type"] = r["Non-Conformance Type"].replace("withdrawn", "withdrawal")
    record("Raritan", "Non-Conformance Type", "normalised 'withdrawn' → 'withdrawal'")

    log.success(MODULE, f"{len(changes)} integrity corrections applied")
    return changes


# ── 2b. Numeric coercion ──────────────────────────────────────────────────────


def coerce_numeric(
    phf: dict[str, pd.DataFrame],
    ptf_cols: list[str],
    ptf_mapping: dict[str, str],
    censored: str = "limit",
    report: "CleaningReport | None" = None,
) -> list[dict]:
    """Make every column the PTF calls numeric a number, and say what that cost.

    A value written ``<0.5`` is censored: the measurement is below the assay's
    limit, which is not the same as being 0.5 and not the same as being missing.
    Which of the three it becomes is ``cleaning.censored`` in config, and the
    count of affected values is recorded either way — silently dropping the
    operator changes what the number means.

    A column that is mostly text still gets coerced. The old behaviour left it as
    object strings, which then broke every correlation and comparison downstream
    while reporting nothing but a warning.
    """
    log.step(MODULE, f"Coercing numeric columns (censored values → {censored})")
    if censored not in CENSORED_POLICIES:
        raise ValueError(f"cleaning.censored: expected one of {', '.join(CENSORED_POLICIES)}")

    coercions: list[dict] = []
    censoring = re.compile(r"^\s*(?P<operator>[<>])\s*(?P<value>-?\d+(?:[.,]\d+)?)\s*$")

    # Endotoxin is recorded with its unit in the cell.
    col_endo = "Endotoxin (EU/mL)"
    for site, df in phf.items():
        if col_endo in df.columns:
            df[col_endo] = df[col_endo].apply(
                lambda x: str(x).replace(" EU/mL", "") if pd.notna(x) else x
            )

    for col in ptf_cols:
        if ptf_mapping.get(col) != "numeric":
            continue
        for site, df in phf.items():
            if col not in df.columns:
                continue

            text = df[col].astype("string").str.strip()
            marks = text.str.extract(censoring)
            flagged = marks["value"].notna()
            values = pd.to_numeric(text.where(~flagged), errors="coerce")

            if flagged.any():
                limits = pd.to_numeric(
                    marks.loc[flagged, "value"].str.replace(",", ".", regex=False),
                    errors="coerce",
                )
                if censored == "half":
                    limits = limits / 2.0
                if censored != "missing":
                    values.loc[flagged] = limits
                entry = {
                    "site": site,
                    "column": col,
                    "values": int(flagged.sum()),
                    "policy": censored,
                    "meaning": CENSORED_POLICIES[censored],
                    "examples": text[flagged].dropna().unique()[:3].tolist(),
                }
                if report is not None:
                    report.censored.append(entry)
                coercions.append(
                    {
                        "site": site,
                        "column": col,
                        "action": f"{int(flagged.sum())} censored values → "
                        f"{CENSORED_POLICIES[censored]}",
                    }
                )
                log.warn(
                    MODULE,
                    f"[{site}] {col}: {int(flagged.sum())} censored values → "
                    f"{CENSORED_POLICIES[censored]}",
                    ", ".join(map(str, entry["examples"])),
                )

            unreadable = text.notna() & ~flagged & values.isna()
            if int(unreadable.sum()):
                coercions.append(
                    {
                        "site": site,
                        "column": col,
                        "action": f"cast to float; {int(unreadable.sum())} non-numeric "
                        "values → NaN",
                    }
                )
                log.warn(
                    MODULE,
                    f"[{site}] {col}: {int(unreadable.sum())} non-numeric values coerced → NaN",
                    f"{text[unreadable].dropna().unique().tolist()[:10]}",
                )
            df[col] = values.astype("float64")

    log.success(MODULE, f"Numeric coercion done — {len(coercions)} column actions")
    return coercions


# ── 2c. Datetime coercion ─────────────────────────────────────────────────────

_DATE_FORMATS = [
    "%Y-%m-%d",
    "%H:%M",
    "%H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
]


def _parse_date(s) -> datetime | type(pd.NaT):
    if pd.isna(s):
        return pd.NaT
    s_str = str(s).strip()
    if not s_str:
        return pd.NaT
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s_str, fmt)
        except ValueError:
            continue
    return pd.NaT


def coerce_datetimes(
    phf: dict[str, pd.DataFrame],
    ptf_cols: list[str],
    ptf_mapping: dict[str, str],
) -> list[dict]:
    """Parse datetime PTF columns; report any values that failed to parse."""
    log.step(MODULE, "Coercing datetime columns")
    failures: list[dict] = []

    for col in ptf_cols:
        if ptf_mapping.get(col) != "datetime":
            continue
        for site, df in phf.items():
            if col not in df.columns:
                continue
            parsed = df[col].apply(_parse_date)
            failed = df[col][~df[col].isna() & parsed.isna()]
            if len(failed):
                failures.append(
                    {
                        "site": site,
                        "column": col,
                        "count": len(failed),
                        "sample": failed.tolist()[:5],
                    }
                )
                log.warn(
                    MODULE,
                    f"[{site}] {col}: {len(failed)} values failed datetime parse",
                    str(failed.tolist()[:5]),
                )
            df[col] = pd.to_datetime(parsed, errors="coerce")

    log.success(MODULE, f"Datetime coercion done — {len(failures)} columns had parse failures")
    return failures


# ── 2d. Duration coercion ─────────────────────────────────────────────────────


def _parse_duration_to_minutes(s, numeric_unit: str = "minutes") -> float:
    """A duration in minutes, from the two spellings the sources use.

    ``HH:MM`` and ``HH:MM:SS`` are clock times and convert exactly. A bare number
    has no unit in the cell, so the unit is a setting (``cleaning.duration_unit``)
    rather than an assumption: an Excel column that holds fractions of a day and
    one that holds minutes look identical here. Anything else is not a duration
    and comes back missing, to be reported as a parse failure.
    """
    if pd.isna(s):
        return float("nan")
    s_str = str(s).strip()
    if not s_str:
        return float("nan")
    parts = s_str.split(":")
    if len(parts) in (2, 3):
        try:
            minutes = int(parts[0]) * 60 + int(parts[1])
            return minutes + (int(parts[2]) / 60 if len(parts) == 3 else 0)
        except ValueError:
            return float("nan")
    try:
        value = float(s_str.replace(",", "."))
    except ValueError:
        return float("nan")
    factor = {"minutes": 1.0, "hours": 60.0, "days": 24 * 60.0, "seconds": 1 / 60.0}
    if numeric_unit not in factor:
        raise ValueError(f"cleaning.duration_unit: expected one of {', '.join(factor)}")
    return value * factor[numeric_unit]


def coerce_durations(
    phf: dict[str, pd.DataFrame],
    ptf_cols: list[str],
    ptf_mapping: dict[str, str],
    numeric_unit: str = "minutes",
) -> list[dict]:
    log.step(MODULE, f"Coercing duration columns (a bare number means {numeric_unit})")
    failures: list[dict] = []

    for col in ptf_cols:
        if ptf_mapping.get(col) != "duration":
            continue
        for site, df in phf.items():
            if col not in df.columns:
                continue
            parsed = df[col].apply(_parse_duration_to_minutes, numeric_unit=numeric_unit)
            failed = df[col][~df[col].isna() & parsed.isna()]
            if len(failed):
                failures.append(
                    {
                        "site": site,
                        "column": col,
                        "count": len(failed),
                        "sample": failed.tolist()[:5],
                    }
                )
                log.warn(
                    MODULE,
                    f"[{site}] {col}: {len(failed)} values failed duration parse",
                    str(failed.tolist()[:5]),
                )
            df[col] = parsed

    log.success(MODULE, f"Duration coercion done — {len(failures)} columns had parse failures")
    return failures


# ── 2e. Ratio extraction ──────────────────────────────────────────────────────


def coerce_ratios(
    phf: dict[str, pd.DataFrame],
    ptf_cols: list[str],
    ptf_mapping: dict[str, str],
) -> None:
    """A ratio written ``a:b`` keeps its right-hand side.

    These columns are recorded normalised — ``1:1.8`` — so the right side *is*
    the ratio and dividing would say something different. That is a property of
    how the sources write them, so it is documented here rather than changed.
    """
    log.step(MODULE, "Extracting ratio values (right side of ':', which is normalised to 1)")
    count = 0
    for col in ptf_cols:
        if ptf_mapping.get(col) != "ratio":
            continue
        for site, df in phf.items():
            if col not in df.columns:
                continue
            df[col] = df[col].apply(
                lambda x: x.split(":", 1)[1] if isinstance(x, str) and ":" in x else x
            )
            count += 1
            log.info(MODULE, f"[{site}] {col}: extracted right-side of ratio")
    log.success(MODULE, f"Ratio coercion done — {count} column-site pairs updated")


# ── 2f. Manual numeric scale corrections ─────────────────────────────────────


def apply_scale_corrections(phf: dict[str, pd.DataFrame]) -> None:
    """
    Site-specific numeric scale fixes identified during QC
    (e.g. Raritan stores certain cell counts without 1e6 divisor).
    """
    log.step(MODULE, "Applying numeric scale corrections")

    corrections = [
        ("Raritan", "Total Viable Cells/bag", 1e6),
        ("Raritan", "Total Viable Cells for Recovery (cells)", 1e6),
        ("Raritan", "Final Formulation CS5 Viable Cell Concentration A (cells/mL)", 1e6),
        ("Raritan", "Final Formulation CS5 Viable Cell Concentration B (cells/mL)", 1e6),
        ("Ghent", "Target Final Seeding Density per Area (VC/cm^2) G-Rex A", 1e6),
        ("Ghent", "Target Final Seeding Density per Area (VC/cm^2) G-Rex B", 1e6),
    ]
    for site, col, divisor in corrections:
        if col in phf[site].columns:
            phf[site][col] /= divisor
            log.info(MODULE, f"[{site}] {col}: divided by {divisor:.0e}")

    log.success(MODULE, f"{len(corrections)} scale corrections applied")


# ── 2g. Percentage range validation ──────────────────────────────────────────


def validate_percentages(
    phf: dict[str, pd.DataFrame],
    ptf_cols: list[str],
) -> list[dict]:
    log.step(MODULE, "Validating percentage columns (0–100 range)")
    issues: list[dict] = []

    for col in ptf_cols:
        if "(%" not in col:
            continue
        for site, df in phf.items():
            if col not in df.columns:
                continue
            s = df[col]
            mask = s.notna() & ((s < 0) | (s > 100))
            if mask.sum():
                issues.append(
                    {
                        "site": site,
                        "column": col,
                        "count": int(mask.sum()),
                        "values": s[mask].tolist()[:10],
                    }
                )
                log.warn(
                    MODULE,
                    f"[{site}] {col}: {mask.sum()} values outside [0,100]",
                    str(s[mask].tolist()[:10]),
                )

    if not issues:
        log.success(MODULE, "All percentage columns within [0, 100]")
    return issues


# ── 2h. Yes/No imputation ─────────────────────────────────────────────────────


def fill_yes_no(phf: dict[str, pd.DataFrame], defaults: dict | None = None) -> list[dict]:
    """Fill blanks in yes/no columns, but only where a config says what one means.

    A blank is unknown. Reading every blank as "No" invents observations — for a
    column recording whether something was seen, it turns "nobody looked" into
    "nothing there". So this fills only the columns ``cleaning.yes_no_defaults``
    names, in each site's own spelling, and records the count.
    """
    defaults = defaults or {}
    if not defaults:
        log.info(MODULE, "No yes/no defaults configured — blanks are left as unknown")
        return []

    log.step(MODULE, f"Filling {len(defaults)} configured yes/no columns")
    spellings = {"no": {"Yes": "No", "YES": "NO", "yes": "no", "Y": "N", "y": "n"}}
    imputations: list[dict] = []

    for site, df in phf.items():
        for col, meaning in defaults.items():
            if col not in df.columns:
                continue
            missing = int(df[col].isna().sum())
            if not missing:
                continue
            present = [str(v) for v in df[col].dropna().unique()]
            wanted = str(meaning)
            # Use the spelling this site already uses for that answer.
            same = next((v for v in present if v.strip().lower() == wanted.strip().lower()), None)
            if same is None and wanted.strip().lower() == "no":
                same = next(
                    (spellings["no"][v] for v in present if v in spellings["no"]),
                    wanted,
                )
            value = same or wanted
            df[col] = df[col].fillna(value)
            imputations.append({"site": site, "column": col, "imputed": missing, "value": value})
            log.info(MODULE, f"[{site}] {col}: {missing} blanks → '{value}' (from config)")

    log.success(MODULE, f"Yes/No filling done — {len(imputations)} column-site pairs")
    return imputations


# ── 2i. Categorical consistency check ────────────────────────────────────────


def check_categorical_consistency(
    phf: dict[str, pd.DataFrame],
    ptf_cols: list[str],
    ptf_mapping: dict[str, str],
) -> list[dict]:
    """Report categorical columns that have different levels across sites."""
    log.step(MODULE, "Checking categorical column consistency (Ghent vs Raritan)")
    diffs: list[dict] = []

    for col in ptf_cols:
        if ptf_mapping.get(col) != "categorical":
            continue
        if col not in phf.get("Ghent", pd.DataFrame()).columns:
            continue
        if col not in phf.get("Raritan", pd.DataFrame()).columns:
            continue

        s1 = set(phf["Ghent"][col].dropna().unique())
        s2 = set(phf["Raritan"][col].dropna().unique())
        if s1 != s2:
            diffs.append(
                {
                    "column": col,
                    "ghent_only": sorted(str(v) for v in (s1 - s2))[:10],
                    "raritan_only": sorted(str(v) for v in (s2 - s1))[:10],
                }
            )
            log.warn(
                MODULE,
                f"Categorical mismatch: '{col}'",
                f"Ghent-only: {sorted(s1 - s2)[:5]}  |  Raritan-only: {sorted(s2 - s1)[:5]}",
            )

    if not diffs:
        log.success(MODULE, "All shared categorical columns have consistent levels")
    else:
        log.warn(MODULE, f"{len(diffs)} categorical columns have different levels across sites")
    return diffs


# ── Convenience orchestrator ──────────────────────────────────────────────────


def run_all(
    phf: dict[str, pd.DataFrame],
    ptf_cols: list[str],
    ptf_mapping: dict[str, str],
    settings: dict | None = None,
) -> CleaningReport:
    """Run every cleaning pass in order and return a composite report."""
    settings = settings or {}
    report = strip_strings(phf, ptf_cols)
    report.settings = {
        "censored": settings.get("censored", "limit"),
        "duration_unit": settings.get("duration_unit", "minutes"),
        "yes_no_defaults": settings.get("yes_no_defaults") or {},
        "raritan_type_commercial": settings.get("raritan_type_commercial", True),
    }
    report.integrity_changes = apply_integrity_corrections(phf, settings)
    report.numeric_coercions = coerce_numeric(
        phf, ptf_cols, ptf_mapping, censored=report.settings["censored"], report=report
    )
    report.datetime_failures = coerce_datetimes(phf, ptf_cols, ptf_mapping)
    report.duration_failures = coerce_durations(
        phf, ptf_cols, ptf_mapping, numeric_unit=report.settings["duration_unit"]
    )
    coerce_ratios(phf, ptf_cols, ptf_mapping)
    apply_scale_corrections(phf)
    report.pct_out_of_range = validate_percentages(phf, ptf_cols)
    report.yesno_imputations = fill_yes_no(phf, report.settings["yes_no_defaults"])
    report.categorical_diffs = check_categorical_consistency(phf, ptf_cols, ptf_mapping)
    return report
