"""Parameters brought in from elsewhere, and parameters rewritten in place.

Everything here works on *original* parameters: measurements a source recorded.
Some arrive from a second system and are joined on; others are already in the
frame and get harmonised so that the two sites mean the same thing by them.
Nothing here calculates a new quantity — that is :mod:`ml.features`.

Each function returns the columns it added, so the report can name where every
column in the batch table came from.
"""

from __future__ import annotations

import os

import pandas as pd

from .io import Source, as_source, normalise_key
from .logger import log
from .names import col, site, value

MODULE = "enrich"

#: The variable that holds the Acridine Orange readings to fetch from Databricks:
#: ``name|parameter id|type`` entries separated by ``;``. It lives in ``.env``,
#: not in the code, because the parameter ids are internal.
AO_MAPPING_ENV = "DATABRICKS_AO_MAPPING"


def ao_mapping() -> str:
    """The Acridine Orange mapping from ``$DATABRICKS_AO_MAPPING``, as Databricks wants it."""
    raw = os.getenv(AO_MAPPING_ENV, "").strip()
    if not raw:
        raise RuntimeError(
            f"sources.databricks is on but ${AO_MAPPING_ENV} is not set — add it to the .env "
            "next to config.yaml (see .env.example)"
        )
    rows = []
    for entry in filter(None, (part.strip() for part in raw.split(";"))):
        fields = [field.strip() for field in entry.split("|")]
        if len(fields) != 3 or not all(fields):
            raise RuntimeError(f"${AO_MAPPING_ENV}: '{entry}' is not name|parameter id|type")
        rows.append("(" + ",".join(f'"{field}"' for field in fields) + ")")
    return "\n" + ",\n".join(rows) + "\n"


BATCH_KEY = "Patient Lot/Batch #"


def _lookup_merge(df: pd.DataFrame, lookup: pd.DataFrame, key: str, what: str) -> pd.DataFrame:
    """Join a lookup table on, and keep exactly the rows that went in.

    Two things go wrong silently in a left join: a lookup key that appears twice
    multiplies the batch rows, and an empty key on both sides matches everything
    to everything. Both are checked here rather than noticed later in a row count
    nobody reconciles.
    """
    if key not in df.columns:
        raise ValueError(f"{what}: the batch frame has no '{key}' column")
    if key not in lookup.columns:
        raise ValueError(f"{what}: the lookup has no '{key}' column")

    lookup = lookup.copy()
    lookup[key] = normalise_key(lookup[key])
    empty = int(lookup[key].isna().sum())
    if empty:
        log.warn(MODULE, f"{what}: {empty} lookup rows have no key and were left out")
        lookup = lookup.loc[lookup[key].notna()]

    before = len(lookup)
    lookup = lookup.drop_duplicates()
    if len(lookup) != before:
        log.info(MODULE, f"{what}: {before - len(lookup)} identical duplicate lookup rows removed")
    conflicting = sorted(set(lookup.loc[lookup.duplicated(subset=key, keep=False), key]))
    if conflicting:
        raise ValueError(
            f"{what}: the lookup holds different records under the same key "
            f"({conflicting[:5]}). Averaging them would invent a certificate"
        )

    rows = len(df)
    df = df.copy()
    df[key] = normalise_key(df[key])
    out = df.merge(lookup, on=key, how="left", validate="m:1")
    if len(out) != rows:
        raise ValueError(f"{what}: the join changed the batch count, {rows} → {len(out)}")
    return out


def add_viability_parameters(
    phf: dict[str, pd.DataFrame],
    catalog_cols: list[str],
    enabled: bool = False,
) -> list[str]:
    """Join the Acridine Orange viability and clump readings from Databricks.

    Whether this runs is a config decision (``sources.databricks``), not a
    consequence of a driver being importable: a run that reads the warehouse and
    one that does not produce different datasets, and which one happened has to
    be something the task file says.
    """
    if not enabled:
        log.info(
            MODULE,
            "sources.databricks is off — the Acridine Orange parameters are not in this run",
        )
        return []

    log.step(MODULE, "Adding Acridine Orange viability parameters")
    mapping = ao_mapping()
    try:
        from databricks_query_tool import DatabricksConnection, Mapping, QueryRunner
    except ImportError as exc:
        raise RuntimeError(
            "sources.databricks is on but databricks_query_tool is not installed"
        ) from exc

    with DatabricksConnection() as conn:
        df_add = QueryRunner(conn).run(Mapping.from_string(mapping))

    df_add = df_add.rename(columns={col("raw_materials_batch"): BATCH_KEY})
    added = [c for c in df_add.columns if c != BATCH_KEY]

    for name, df in phf.items():
        phf[name] = _lookup_merge(df, df_add, BATCH_KEY, f"[{name}] Acridine Orange")

    absent = [c for c in added if c not in catalog_cols]
    if absent:
        log.warn(
            MODULE,
            f"{len(absent)} Acridine Orange columns are not catalog parameters",
            ", ".join(absent),
        )

    log.success(MODULE, f"Acridine Orange parameters added — {len(added)} columns")
    return added


def merge_lv_coa(
    phf: dict[str, pd.DataFrame],
    df_lv_coa: pd.DataFrame,
) -> tuple[dict[str, dict], list[str]]:
    """Join the lentiviral vector certificate of analysis on the vector lot.

    Returns the per-site match rate and the columns added. A batch whose vector
    lot has no certificate keeps its row and gets empty CoA columns.
    """
    log.step(MODULE, "Merging the vector certificates")
    coverage: dict[str, dict] = {}
    lot = col("vector_lot")
    witness = col("coa_prefix") + col("coa_site_format")
    added = [c for c in df_lv_coa.columns if c != lot]

    for name, df in phf.items():
        phf[name] = _lookup_merge(df, df_lv_coa, lot, f"[{name}] vector certificates")

        total = len(phf[name])
        matched = int(phf[name][witness].notna().sum())
        coverage[name] = {
            "total": total,
            "matched": matched,
            "pct": round(100 * matched / total, 1) if total else 0,
        }
        log.info(
            MODULE,
            f"[{name}] certificate coverage: {matched}/{total} batches ({coverage[name]['pct']}%)",
        )

    log.success(MODULE, "Vector certificate merge complete")
    return coverage, added


def merge_raw_materials(
    phf: dict[str, pd.DataFrame],
    df_raw: pd.DataFrame,
) -> list[str]:
    """Join raw material and consumable lot identifiers on the batch number.

    A column the batch records already carry is dropped from the incoming frame rather
    than joined twice under a suffix.
    """
    log.step(MODULE, "Merging raw materials and consumables")
    added = [c for c in df_raw.columns if c != BATCH_KEY]

    for name, df in phf.items():
        duplicated = (set(df.columns) & set(df_raw.columns)) - {BATCH_KEY}
        phf[name] = _lookup_merge(
            df,
            df_raw.drop(columns=list(duplicated), errors="ignore"),
            BATCH_KEY,
            f"[{name}] raw materials",
        )

    log.success(MODULE, f"Raw materials merged — {len(added)} identifier columns added")
    return added


def enrich_clinical_sites(
    phf: dict[str, pd.DataFrame],
    site_maps: list[Source | str],
) -> tuple[list[str], list[str]]:
    """Give both sites a clinical site the other site would recognise.

    Site A encodes the clinical site in the patient identifier (a prefix, then
    the site); site B records an acronym that the mapping workbooks expand into
    an institution name. Which workbooks, in which SharePoint folder, is
    config.yaml's business (``sharepoint.site_maps``).

    Returns site B's acronyms no mapping file covers, and the columns added.
    """
    log.step(MODULE, "Enriching clinical site identifiers")
    site_id, clinical_site = col("clinical_site_id"), col("clinical_site")
    acronym, institution = col("site_map_acronym"), col("site_map_institution")
    prefix = value("patient_id_site_prefix")

    first = phf[site("a")]
    patient = first[col("patient_id")].fillna("Unknown")
    first[site_id] = patient.map(
        lambda x: x.split("-")[1] if isinstance(x, str) and x.startswith(prefix) else x
    ).replace("Unknown", pd.NA)
    log.info(MODULE, f"[{site('a')}] clinical site extracted from the patient identifier")

    second = phf[site("b")]
    second[site_id] = second[clinical_site].copy()

    frames = []
    for number, site_map in enumerate(site_maps, start=1):
        source = as_source(site_map, f"site_map_{number}")
        try:
            mapping = source.read()
            mapping[acronym] = mapping[acronym].astype(str)
            frames.append(mapping)
        except Exception as exc:  # one unreadable mapping file is reported, not fatal
            log.warn(MODULE, f"Could not load mapping file: {source.where} — {exc}")

    if not frames:
        log.warn(MODULE, f"No {site('b')} site mapping files loaded — acronyms unchanged")
        return [], [site_id]

    acronyms = (
        pd.concat(frames)
        .drop_duplicates([acronym], keep="last")
        .set_index(acronym)[institution]
        .to_dict()
    )

    unmapped: list[str] = []

    def expand(code):
        key = str(code).upper()
        if key in acronyms:
            return acronyms[key]
        unmapped.append(code)
        return code

    recorded = second[clinical_site].fillna("Unknown")
    second[clinical_site] = recorded.map(expand).replace("Unknown", pd.NA)

    unique = sorted({str(x) for x in unmapped})
    if unique:
        log.warn(
            MODULE,
            f"[{site('b')}] {len(unique)} clinical site acronyms have no mapping",
            ", ".join(unique[:20]),
        )
    else:
        log.success(MODULE, f"[{site('b')}] All clinical site acronyms resolved")

    return unique, [site_id]


def harmonise_countries(phf: dict[str, pd.DataFrame]) -> None:
    """Spell site B's countries the way site A spells them.

    A replacement rather than a lookup, and on the site's own frame rather than
    on the raw one it was loaded from: a lookup empties every country that is not
    in the table, and the raw frame no longer carries the corrections cleaning
    made.
    """
    log.step(MODULE, "Harmonising country names")
    second = phf[site("b")]
    country = col("country")
    second[country] = second[country].replace(
        {value("country_to_harmonise"): value("country_harmonised")}
    )
    log.success(MODULE, "Country harmonisation complete")
