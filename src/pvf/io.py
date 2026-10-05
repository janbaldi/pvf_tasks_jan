"""Reading every source the PVF is built from.

The PTF specification, both sites' PHF files, the parameter-name mapping between
them, the re-manufacturing treatment-line supplement, the lentiviral vector
certificates of analysis, and the raw-material identifiers. Each loader takes a
:class:`Source`: a local file, or — when the workspace config says so — a
SharePoint location configured in ``pvf.yaml``. Nothing here knows an address.

Everything returned here is an *original* parameter. Nothing in this module
calculates anything from another column.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import features
from . import provenance as prov
from .logger import log

MODULE = "io"

#: The value types the PTF may declare, besides a bracketed ordinal category list.
VALUE_TYPES = frozenset({"numeric", "categorical", "datetime", "duration", "ratio", "boolean"})

#: The PTF column that says when in the process a parameter exists. Optional.
STAGE_COLUMN = "Available At"

#: The worksheet each source's data is on, in the files as the sites produce them.
#: A workspace can override any of these under ``sheets:`` in its config.
DEFAULT_SHEETS: dict[str, str | int] = {
    "ptf": 0,
    "pvf": 0,
    "phf_ghent": "BR Data",
    "phf_raritan": "BR Data",
    "param_mapping": "Overall",
    "remfg": "Treatment Line Data",
    "lv_coa": "CoA extracts",
    "raw_materials": 0,
    "investigations": "Master Query ALL COM",
    "site_maps": 0,
}
#: Rows of workbook furniture above the header, where a source has any.
DEFAULT_HEADERS = {"lv_coa": 2}


def read_table(path: str | Path, sheet: str | int = 0, header: int = 0) -> pd.DataFrame:
    """A local table, read the way its extension says: CSV, Parquet or Excel."""
    suffix = Path(path).suffix.lower()
    if suffix in (".csv", ".txt"):
        return pd.read_csv(path, header=header)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_excel(path, sheet_name=sheet, header=header)


@dataclass
class Source:
    """One input, and where it is read from.

    ``remote`` holds the keyword arguments the SharePoint reader takes; when it is
    set the source is read from there, otherwise from ``path``. Built by
    :func:`pvf.config.source` from the workspace config.
    """

    label: str
    path: str = ""
    sheet: str | int = 0
    header: int = 0
    remote: dict[str, Any] | None = None
    loader: Callable[..., pd.DataFrame] | None = None

    @property
    def is_remote(self) -> bool:
        return self.remote is not None

    @property
    def location(self) -> str:
        if self.remote is not None:
            return (
                f"SharePoint {self.remote['drive_id']}: {self.remote['sharepoint_path']} "
                f"[{self.remote['sheet_name']}]"
            )
        return self.path

    def read(self) -> pd.DataFrame:
        if self.remote is not None:
            kwargs = dict(self.remote)
            if self.header:
                kwargs["header"] = self.header
            return self.loader(**kwargs)
        if not self.path:
            raise FileNotFoundError(
                f"No path is configured for '{self.label}' (paths.{self.label})"
            )
        return read_table(self.path, self.sheet, self.header)

    def record(self, frame: pd.DataFrame | None) -> prov.Source:
        """What provenance says about this input, as it was actually read.

        A SharePoint read has no local bytes, so it is recorded by its location
        and a digest of the table that came back. Hashing the local path instead
        would record a copy the run never used.
        """
        if frame is None:
            why = "could not be read; the stage went on without it"
            return prov.missing(self.label, self.location, why)
        if self.is_remote:
            return prov.remote(self.label, self.location, frame)
        return prov.local(self.label, self.path)


def as_source(value: str | Path | Source, label: str) -> Source:
    """A plain path is a local source with the default sheet for its label."""
    if isinstance(value, Source):
        return value
    return Source(label, str(value), DEFAULT_SHEETS.get(label, 0), DEFAULT_HEADERS.get(label, 0))


def _read(source: Source, what: str) -> pd.DataFrame:
    log.step(MODULE, f"Loading {what}", source.location)
    return source.read()


# ── Public helpers ─────────────────────────────────────────────────────────────


def read_ptf(ptf: str | Path | Source) -> pd.DataFrame:
    """The Parameter Transfer File as it stands: one row per parameter.

    Every stage starts here. ``Parameter`` names what the PVF may hold and
    ``Value Type`` says what it holds — including, for an ordinal parameter, its
    categories in order, written as a bracketed list. An optional
    ``Available At`` column says from which process stage a parameter exists,
    which is what lets a predictive task check availability mechanically.
    """
    source = as_source(ptf, "ptf")
    frame = _read(source, "PTF")

    missing = [c for c in ("Parameter", "Value Type") if c not in frame.columns]
    if missing:
        raise ValueError(f"The PTF at {source.location} has no {' or '.join(missing)} column")

    blank = frame["Parameter"].isna() | (frame["Parameter"].astype(str).str.strip() == "")
    if blank.any():
        log.warn(MODULE, f"{int(blank.sum())} PTF rows have no parameter name and are ignored")
        frame = frame.loc[~blank]
    repeated = frame["Parameter"].astype(str).value_counts()
    repeated = repeated[repeated > 1]
    if len(repeated):
        log.warn(
            MODULE,
            f"{len(repeated)} PTF parameters are listed more than once; the first row of each wins",
            ", ".join(map(str, repeated.index[:10])),
        )
        frame = frame.drop_duplicates(subset="Parameter", keep="first")
    unknown = sorted(
        {
            str(value)
            for value in frame["Value Type"].dropna().unique()
            if str(value) not in VALUE_TYPES and not str(value).strip().startswith("[")
        }
    )
    if unknown:
        log.warn(
            MODULE,
            f"{len(unknown)} PTF value types are not ones this pipeline knows",
            ", ".join(unknown[:10]),
        )

    counts = frame["Value Type"].value_counts().to_dict()
    log.success(
        MODULE,
        f"PTF loaded — {len(frame)} parameters"
        + (f", with an '{STAGE_COLUMN}' column" if STAGE_COLUMN in frame.columns else ""),
        "  |  ".join(f"{k}: {v}" for k, v in counts.items()),
    )
    return frame


def load_ptf(ptf: str | Path | Source) -> tuple[list[str], dict[str, str]]:
    """The PTF as the build wants it: the parameter names, and their value types."""
    frame = read_ptf(ptf)
    return list(frame["Parameter"].values), dict(zip(frame["Parameter"], frame["Value Type"]))


def load_pvf(pvf: str | Path | Source) -> pd.DataFrame:
    """The merged PVF, as ``pvf build`` wrote it (Excel or its Parquet copy)."""
    source = as_source(pvf, "pvf")
    frame = _read(source, "PVF")
    if "Site Merged" not in frame.columns:
        raise ValueError(f"{source.location} has no 'Site Merged' column — is it a PVF?")
    log.success(MODULE, f"PVF loaded — {len(frame):,} batches × {frame.shape[1]:,} parameters")
    return frame


def load_investigations(investigations: str | Path | Source) -> pd.DataFrame | None:
    """The investigations team's Power Query workbook, if it is reachable.

    Only the PTF stage reads it, and only to see which parameters it uses. It
    is not part of the build, so a run without it carries on and says the source
    was not compared.
    """
    source = as_source(investigations, "investigations")
    try:
        return _read(source, "the investigations Power Query workbook")
    except Exception as exc:  # unreachable is a finding here, not a failure
        log.warn(MODULE, f"Investigations workbook not read: {exc}")
        return None


def load_phf_ghent(phf: str | Path | Source) -> pd.DataFrame:
    """Load the Ghent PHF."""
    frame = _read(as_source(phf, "phf_ghent"), "Ghent PHF")
    log.success(MODULE, f"Ghent PHF loaded — {len(frame):,} rows × {frame.shape[1]} columns")
    return frame


def load_phf_raritan(phf: str | Path | Source) -> pd.DataFrame:
    """Load the Raritan PHF."""
    frame = _read(as_source(phf, "phf_raritan"), "Raritan PHF")
    log.success(MODULE, f"Raritan PHF loaded — {len(frame):,} rows × {frame.shape[1]} columns")
    return frame


def load_site_mapping(
    mapping: str | Path | Source, exclude: list[str] | tuple[str, ...] = ()
) -> dict[str, str]:
    """
    Load the Ghent ↔ Raritan parameter-name mapping.

    Returns a cleaned dict  {raritan_col_name: ghent_col_name}. ``exclude`` names
    mapping rows known to be wrong (``mapping.exclude`` in the config).
    """
    df_mapping = _read(as_source(mapping, "param_mapping"), "site parameter name mapping")

    required = ["Parameter Name Raritan (CMD)", "Parameter Name Ghent (PHF)"]
    absent = [column for column in required if column not in df_mapping.columns]
    if absent:
        raise ValueError(f"The parameter mapping has no {' or '.join(absent)} column")

    name_mapping: dict[str, str] = {}
    conflicts: list[str] = []
    unusable: list[str] = []
    for source, target in zip(df_mapping[required[0]], df_mapping[required[1]]):
        if pd.isna(source) or not str(source).strip():
            continue
        source = str(source).strip()
        if pd.isna(target) or not str(target).strip():
            unusable.append(source)
            continue
        target = str(target).strip()
        if source in name_mapping and name_mapping[source] != target:
            conflicts.append(f"{source} → {name_mapping[source]} / {target}")
            continue
        name_mapping[source] = target

    excluded = [name for name in exclude if name_mapping.pop(name, None) is not None]
    if excluded:
        log.info(MODULE, f"{len(excluded)} mapping rows excluded by config", ", ".join(excluded))

    if conflicts:
        raise ValueError(
            "The parameter mapping sends one Raritan column to two different Ghent names: "
            + "; ".join(conflicts[:5])
        )
    if unusable:
        log.warn(
            MODULE,
            f"{len(unusable)} mapping rows have no Ghent name and were dropped",
            ", ".join(unusable[:10]),
        )
    log.success(MODULE, f"Site mapping loaded — {len(name_mapping)} usable mappings")
    return name_mapping


def apply_column_mapping(df_raritan: pd.DataFrame, name_mapping: dict) -> pd.DataFrame:
    """Rename Raritan columns to the Ghent names, suffixing only what collides.

    Two source columns can map to one name — the site records a country twice
    under different headings, say. The first of them, by position, keeps the
    plain name and each later one takes ``_1``, ``_2``… so which values ended up
    under which name is determined by the file, not by dictionary order. A
    suffix that is already a column name is skipped rather than reused.
    """
    log.step(MODULE, "Applying column mapping to Raritan")

    proposed = [str(name_mapping.get(column, column)) for column in df_raritan.columns]
    reserved = set(proposed)
    taken: set[str] = set()
    final: list[str] = []
    renamed: list[tuple[str, str]] = []

    for source, target in zip(df_raritan.columns, proposed):
        name = target
        if name in taken:
            index = 1
            while f"{target}_{index}" in taken or f"{target}_{index}" in reserved:
                index += 1
            name = f"{target}_{index}"
            renamed.append((str(source), name))
        taken.add(name)
        reserved.add(name)
        final.append(name)

    if len(set(final)) != len(final):
        raise ValueError("Column mapping produced duplicate names; the mapping file is wrong")

    df_raritan = df_raritan.copy()
    df_raritan.columns = final
    for old, new in renamed:
        log.warn(MODULE, f"Two columns claim one name: '{old}' is kept as '{new}'")

    log.success(
        MODULE,
        "Raritan column mapping applied",
        f"{len(renamed)} columns renamed to avoid duplicates",
    )
    return df_raritan


def join_key(df: pd.DataFrame, candidates: tuple[str, ...], what: str) -> str:
    """The column a join is keyed on, by name. Never "the first column"."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"{what} has no join key: none of {', '.join(candidates)} is a column of it "
        f"(it has {', '.join(map(str, df.columns[:8]))})"
    )


def normalise_key(series: pd.Series) -> pd.Series:
    """A join key with its whitespace flattened, as text."""
    return series.astype("string").str.strip()


def attach_supplement(
    df: pd.DataFrame, supplement: pd.DataFrame, key: str, columns: list[str]
) -> pd.DataFrame:
    """Join a per-batch lookup on, and refuse anything that would multiply rows.

    A lookup with the same key twice is either the same record written twice —
    dropped, with a count — or two different records, which is a data problem
    this cannot resolve by averaging. Rows with no key are dropped rather than
    left to match every other empty key.
    """
    lookup_key = join_key(supplement, (key,), "The supplement")
    wanted = [c for c in columns if c in supplement.columns]
    absent = [c for c in columns if c not in supplement.columns]
    if absent:
        log.warn(MODULE, f"The supplement does not carry {', '.join(absent)}")
    lookup = supplement[[lookup_key, *wanted]].copy()
    lookup[lookup_key] = normalise_key(lookup[lookup_key])

    empty = int(lookup[lookup_key].isna().sum())
    if empty:
        log.warn(MODULE, f"{empty} supplement rows have no {key} and were left out of the join")
        lookup = lookup.loc[lookup[lookup_key].notna()]

    before = len(lookup)
    lookup = lookup.drop_duplicates()
    if len(lookup) != before:
        log.info(MODULE, f"{before - len(lookup)} duplicate supplement rows removed")
    conflicting = lookup[lookup.duplicated(subset=lookup_key, keep=False)]
    if not conflicting.empty:
        raise ValueError(
            f"The supplement gives different values for the same {key}: "
            f"{sorted(set(conflicting[lookup_key].dropna()))[:5]}. "
            "One batch cannot have two answers"
        )

    rows = len(df)
    out = df.copy()
    out[key] = normalise_key(out[key])
    out = out.merge(lookup, on=key, how="left", validate="m:1")
    if len(out) != rows:
        raise ValueError(f"Joining the supplement changed the row count: {rows} → {len(out)}")
    log.info(MODULE, f"Supplement joined on {key} — {', '.join(wanted)}")
    return out


def load_remanufacturing_supplement(remfg: str | Path | Source) -> pd.DataFrame:
    """Load the treatment-line supplement for Raritan (Number of Prior Lines of Therapy)."""
    df = _read(as_source(remfg, "remfg"), "re-manufacturing treatment-line supplement")

    key = join_key(
        df,
        ("Atlas Batch Number", "Patient Lot/Batch #", "Batch Number"),
        "The re-manufacturing supplement",
    )
    df = df.rename(columns={key: "Patient Lot/Batch #"})
    empty = int(df["Patient Lot/Batch #"].isna().sum())
    if empty:
        log.warn(MODULE, f"{empty} supplement rows have no batch number and were dropped")
    df = df.dropna(subset=["Patient Lot/Batch #"])
    df = df.rename(
        columns={"Number of Prior Lines of Therapy (LGN)": "Number of Prior Lines of Therapy"}
    )
    log.success(MODULE, f"Re-MFG supplement loaded — {len(df):,} rows")
    return df


def load_lv_coa(
    lv_coa: str | Path | Source,
    ptf_cols: list[str],
    ph_nominal: float | None = None,
    osmo_nominal: float | None = None,
) -> pd.DataFrame:
    """
    Load LV CoA correlation data and engineer lot-level features for the
    "expected average patient-batch CAR%" model.

    Parameters
    ----------
    ph_nominal, osmo_nominal : float | None
        Release-spec MIDPOINT for the two-sided |deviation| features (Block 2).
        PRODUCT-SPECIFIC — pass from spec, never a data mean (a data-derived
        centre leaks). If None, that deviation feature is skipped with a warning.
    """
    df = _read(as_source(lv_coa, "lv_coa"), "vector certificates of analysis")
    log.success(MODULE, f"LVV CoA data loaded — {len(df):,} rows × {df.shape[1]} columns")

    # --- drop section-marker columns (headers "1".."5","42","42 ", empty) ---
    marker = [
        c
        for c in df.columns
        if isinstance(c, (int, float, np.integer)) or (isinstance(c, str) and c.strip().isdigit())
    ]
    if marker:
        df.drop(columns=marker, inplace=True)

    df["LV Batch"] = df["LV Batch"].astype("string").str.strip()
    df = df[df["CoA Available"] == "Yes"].copy()
    df = df.dropna(axis=1, how="all")  # removes unadjusted titer only if all-empty

    def norm(name: object) -> str:
        """A column name with its spacing and case flattened, for matching."""
        return re.sub(r"\s+", " ", str(name).strip().lower())

    lookup: dict[str, str] = {}
    for c in df.columns:
        lookup.setdefault(norm(c), c)

    def col(*cands: str) -> str | None:
        for cand in cands:
            hit = lookup.get(norm(cand))
            if hit is not None:
                return hit
        return None

    engineered: list[str] = []
    lineage: dict[str, tuple[str, ...]] = {}

    def add(name: str, series: pd.Series, *sources: str | None) -> None:
        """Create one derived certificate column, and record what it came from.

        The parsing belongs here, next to the workbook's own quirks, but the
        lineage belongs to the whole pipeline: a later stage has to know that a
        column is arithmetic over others before it can decide whether the target
        is inside it. ``features.coa_lineage()`` is the table this is checked
        against, so the two cannot silently drift apart.
        """
        df[name] = series
        engineered.append(name)
        lineage[name] = tuple(f"LV CoA {s}" for s in sources if s)

    # ------------------------------------------------------------------ #
    # Numeric coercion for EUROPEAN-LOCALE TEXT cells:
    #   decimal comma -> dot ("3,40E+07" -> 3.40e7, "7,5" -> 7.5)
    #   percent strings -> fractions ("20,0%" -> 0.20, "96%" -> 0.96)
    # Comma is the DECIMAL separator (no thousands grouping; big values use
    # scientific notation). Real numeric dtypes are returned unchanged.
    # ------------------------------------------------------------------ #
    def eu_num(s: pd.Series) -> pd.Series:
        if s.dtype.kind in "biufc":
            return s.astype(float)
        raw = s.astype("string").str.strip()
        is_pct = raw.str.contains("%", na=False)
        cleaned = raw.str.replace("%", "", regex=False).str.replace(",", ".", regex=False)
        out = pd.to_numeric(cleaned, errors="coerce")
        return out.where(~is_pct, out / 100.0)

    # ================================================================== #
    # BLOCK 4a — left-censored impurities ("<LOQ"), detected BEFORE coercion.
    # Below-LOQ means LOW, not UNKNOWN. Left as NaN it lands in XGBoost's
    # "missing" default branch and destroys the "clean = low" ordering, so we
    # (i) flag it and (ii) substitute LOQ/2 as a low sentinel. Concentration
    # columns only (that's where LOQ censoring lives).
    # ================================================================== #
    conc_impurity = [
        c
        for c in (
            col("HCP"),
            col("pDNA"),
            col("cDNA"),
            col("E1a cDNA"),
            col("E1b cDNA"),
        )
        if c is not None
    ]

    for c in conc_impurity:
        raw = df[c].astype("string").str.strip()
        loq = raw.str.extract(r"^<\s*([\d.,]+)", expand=False)  # "<0,5" -> "0,5"
        below = loq.notna()
        if below.any():
            add(f"{c} below LOQ", below.astype("int8"), c)
            loq_num = pd.to_numeric(loq.str.replace(",", ".", regex=False), errors="coerce") / 2.0
            df[c] = raw.where(~below, loq_num.astype("string"))  # LOQ/2 sentinel

    # Delivered-mass ("Calc *") columns — kept as features (see header note).
    calc_impurity = [
        c
        for c in (
            col("Calc HCP"),
            col("Calc pDNA"),
            col("Calc cDNA"),
            col("Calc E1a cDNA"),
            col("Calc E1b cDNA"),
        )
        if c is not None
    ]

    titer_adj = col("LV Titer")
    titer_unadj = col("LV Titer (unadjusted)")  # may survive but be PARTIAL
    p24 = col("Physical Titer p24 ELISA")
    vol = col("Volume")

    numeric_cols = [
        c
        for c in (
            titer_adj,
            titer_unadj,
            p24,
            col("PI Ratio"),
            col("CAR% - Donor 1"),
            col("CAR% - Donor 2"),
            col("CAR% - Average"),
            col("IFNg - Donor 1"),
            col("IFNg - Donor 2"),
            col("IFNg - Average"),
            col("pH"),
            col("Osmolality"),
            vol,
            col("DNA size"),
            *conc_impurity,
            *calc_impurity,
        )
        if c is not None
    ]
    for c in numeric_cols:
        df[c] = eu_num(df[c])

    # ================================================================== #
    # BLOCK 1 — constructed ratios / products.
    # XGBoost splits are axis-aligned, so a quotient/product must otherwise be
    # approximated by a deep staircase of splits — wasteful at n~tens of lots.
    # Precomputing hands the model the mechanism directly.
    # ================================================================== #

    # Delivered functional dose per G-Rex = titer x volume added.
    #   Proportional to effective MOI when cell seed is fixed; expected strongest
    #   chemistry feature. Titer basis chosen PER ROW: prefer UNADJUSTED (the
    #   physical amount actually delivered, and it dodges the adjusted-titer
    #   back-channel), fall back to ADJUSTED only where unadjusted is missing.
    #   If the titer adjustment is derived from a potency standard, the fallback
    #   rows partially back-channel the target — so we flag exactly those rows.
    if vol and (titer_unadj or titer_adj):
        if titer_unadj and titer_adj:
            dose_titer = df[titer_unadj].where(df[titer_unadj].notna(), df[titer_adj])
            n_fb = int(df[titer_unadj].isna().to_numpy().sum())
            add("Delivered Dose per GRex", dose_titer * df[vol], titer_unadj, titer_adj, vol)
            if n_fb:
                add(
                    "Delivered Dose used adj titer",
                    df[titer_unadj].isna().astype("int8"),
                    titer_unadj,
                )
                log.warn(
                    MODULE,
                    f"Delivered-dose: {n_fb}/{len(df)} lots fell back "
                    f"to ADJUSTED titer — potency back-channel possible "
                    f"on those rows",
                )
        else:
            basis = titer_unadj or titer_adj
            if basis == titer_adj:
                log.warn(
                    MODULE,
                    "Delivered-dose uses ADJUSTED titer (no unadjusted "
                    "column) — check for potency back-channel",
                )
            add("Delivered Dose per GRex", df[basis] * df[vol], basis, vol)

    # Titer adjustment factor = adjusted / unadjusted.
    #   How hard release titer was corrected; large corrections can flag assay
    #   drift. Defined only where BOTH exist; NaN elsewhere (Block 4b flags it).
    if titer_adj and titer_unadj:
        add(
            "Titer Adjustment Factor",
            df[titer_adj] / df[titer_unadj].replace(0, np.nan),
            titer_adj,
            titer_unadj,
        )

    # Specific infectivity = infectious titer / physical p24 titer.
    #   Functional particles per physical particle; empty/defective particles
    #   don't transduce, so this adds signal beyond raw titer. Built explicitly
    #   rather than reused from PI Ratio (different units/polarity here: PI Ratio
    #   is ~1e4 while titer/p24 is ~1 — NOT the same quantity).
    if titer_adj and p24:
        add("Specific Infectivity", df[titer_adj] / df[p24].replace(0, np.nan), titer_adj, p24)

    # E1b cDNA as a fraction of total residual DNA (concentration basis).
    #   *** LEAKAGE CAUTION: E1b cDNA is a known lot-FINGERPRINT risk — it can
    #   encode vector-lot identity rather than genuine quality. Validate under
    #   vector-lot GroupKFold; drop if it only helps within-lot.
    dna_cols = [
        c for c in (col("pDNA"), col("cDNA"), col("E1a cDNA"), col("E1b cDNA")) if c is not None
    ]
    e1b = col("E1b cDNA")
    if e1b and len(dna_cols) >= 2:
        total_dna = df[dna_cols].sum(axis=1, min_count=1)
        add("E1b Fraction of Total DNA", df[e1b] / total_dna.replace(0, np.nan), *dna_cols)

    # ================================================================== #
    # BLOCK 2 — two-sided spec deviations as monotone features.
    # pH / osmolality hurt on BOTH sides of spec (U-shaped). |x - nominal|
    # folds that into ONE monotone feature -> a single split instead of extra
    # depth to rediscover both tails. nominal = fixed spec midpoint, not a mean.
    # ================================================================== #
    for name, target, nominal in [
        ("pH", col("pH"), ph_nominal),
        ("Osmolality", col("Osmolality"), osmo_nominal),
    ]:
        if target is None:
            continue
        if nominal is None:
            log.warn(MODULE, f"{name} deviation skipped — pass {name.lower()}_nominal")
            continue
        add(f"{name} Abs Deviation", (df[target] - nominal).abs(), target)

    # ================================================================== #
    # BLOCK 3 — reference-donor summaries (features, not target).
    # ================================================================== #
    for metric in ["CAR%", "IFNg"]:
        d1, d2 = col(f"{metric} - Donor 1"), col(f"{metric} - Donor 2")
        if not (d1 and d2):
            continue
        pair = df[[d1, d2]]

        # MAX / MIN: best- and worst-donor readings. MIN matters because patient
        # apheresis material (pretreated, lymphopenic) often transduces WORSE
        # than healthy reference donors, so the weaker donor may track the
        # patient-population mean better than the average. Let the model choose.
        add(f"{metric} Donor MAX", pair.max(axis=1), d1, d2)
        add(f"{metric} Donor MIN", pair.min(axis=1), d1, d2)

        # DIFF (+ relative): divergent reference donors => a more
        # donor-context-sensitive lot, plausibly predicting a larger / more
        # variable gap between reference qualification and the patient mean.
        diff = pair.max(axis=1) - pair.min(axis=1)
        add(f"{metric} Donor DIFF", diff, d1, d2)
        add(f"{metric} Donor DIFF Rel", diff / pair.mean(axis=1).replace(0, np.nan), d1, d2)

        # Reference average (primary potency level). Prefer the CoA's own
        # "- Average" if present (it reconciles with the donor mean); else compute.
        if col(f"{metric} - Average") is None:
            add(f"{metric} Donor AVG", pair.mean(axis=1), d1, d2)

    # ================================================================== #
    # BLOCK 4b — assay-missingness indicators (distinct from below-LOQ).
    # "Not run" != "below LOQ". Which lots got the full panel can be informative,
    # so flag genuine NaN explicitly instead of relying only on the default branch.
    # Skip unadjusted titer (dropped below; its missingness is already encoded by
    # "Delivered Dose used adj titer").
    # ================================================================== #
    for c in numeric_cols:
        if c == titer_unadj:
            continue
        if df[c].isna().any():
            add(f"{c} missing", df[c].isna().astype("int8"), c)

    # ------------------------------------------------------------------ #
    # Drop identifiers / consumed raw columns, then namespace.
    # ------------------------------------------------------------------ #
    for c in ["CoA Available", "LV Batch #", "MFG Date", titer_unadj]:
        if c and c in df.columns:
            df.drop(columns=c, inplace=True)

    df.columns = ["Vector Lot"] + ["LV CoA " + str(c) for c in df.columns[1:]]
    engineered_prefixed = {"LV CoA " + n for n in engineered}

    missing_from_ptf = [c for c in df.columns if c not in ptf_cols and c not in engineered_prefixed]
    if missing_from_ptf:
        log.warn(
            MODULE,
            f"{len(missing_from_ptf)} CoA columns not found in PTF",
            ", ".join(missing_from_ptf),
        )

    # Derived columns are not PTF parameters, and that is a decision, not an
    # oversight — but each of them has to be visibly derived, with its inputs.
    declared = features.coa_lineage()
    undeclared = sorted(
        name for name in engineered_prefixed if f"LV CoA {name[7:]}" not in declared
    )
    if undeclared:
        log.warn(
            MODULE,
            f"{len(undeclared)} derived CoA columns have no entry in features.coa_lineage(); "
            "a task cannot check them for descent from its target",
            ", ".join(undeclared[:10]),
        )
    log.info(
        MODULE,
        f"Derived {len(engineered_prefixed)} LV CoA columns from other certificate columns "
        "(not PTF parameters; their lineage is in features.coa_lineage())",
        "; ".join(
            f"LV CoA {name} ← {', '.join(sources) or 'unrecorded'}"
            for name, sources in list(lineage.items())[:10]
        ),
    )

    log.success(MODULE, f"LV CoA loaded — {len(df):,} lots × {df.shape[1]} columns")
    return df


def load_raw_materials(raw_materials: str | Path | Source, ptf_cols: list[str]) -> pd.DataFrame:
    """Load the raw materials / consumables identifiers (a CSV export or a workbook)."""
    df = _read(as_source(raw_materials, "raw_materials"), "raw materials & consumables")

    df.columns = [
        str(c).replace("Query[", "").replace("]", "") if "Query[" in str(c) else str(c)
        for c in df.columns
    ]

    df = df.rename(columns={"targetbatchnumber": "Patient Lot/Batch #"})

    key = join_key(
        df,
        ("Patient Lot/Batch #", "Batch", "Batch Number", "Atlas Batch Number"),
        "The raw materials export",
    )
    df = df.rename(columns={key: "Patient Lot/Batch #"})
    df["Patient Lot/Batch #"] = normalise_key(df["Patient Lot/Batch #"])

    missing_from_ptf = [c for c in df.columns if c not in ptf_cols]
    if missing_from_ptf:
        log.warn(
            MODULE,
            f"{len(missing_from_ptf)} raw-material columns not in PTF",
            ", ".join(missing_from_ptf[:10]),
        )

    log.success(MODULE, f"Raw materials loaded — {df.shape[1] - 1} identifier columns")
    return df
