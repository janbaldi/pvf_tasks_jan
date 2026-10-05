"""Parameters brought in from elsewhere, and parameters rewritten in place.

Everything here works on *original* parameters: measurements a source recorded.
Some arrive from a second system and are joined on; others are already in the
frame and get harmonised so that the two sites mean the same thing by them.
Nothing here calculates a new quantity — that is :mod:`pvf.features`.

Each function returns the columns it added, so the report can name where every
column in the PVF came from.
"""

from __future__ import annotations

import pandas as pd

from .io import normalise_key
from .logger import log

MODULE = "enrich"

#: Acridine Orange viability and clump readings, by their Databricks parameter id.
AO_MAPPING = """
("D10 Pre-Wash AO%1","0006/1CBF6/2CBF11/3CBF6/04MV","VL"),
("D10 Pre-Wash AO%2","0006/1CBF6/2CBF11/3CBF6/10MV","VL"),
("D10 Pre-Wash AO%3","0006/1CBF6/2CBF11/3CBF6/16MV","VL"),
("D0 Post-Thaw AO%","0001/1CBF4/2CBF10/3CBF6/05MV","VL"),
("D0 Post-Enrichment AO%1","0001/1CBF4/2CBF24/3CBF5/04MV","VL"),
("D0 Post-Enrichment AO%2","0001/1CBF4/2CBF24/3CBF5/10MV","VL"),
("D0 Post-Enrichment AO%3","0001/1CBF4/2CBF24/3CBF5/16MV","VL"),
("D3 Activated T-cells AO%1","0003/1CBF5/2CBF38/3CBF6/02MV","VL"),
("D3 Activated T-cells AO%2","0003/1CBF5/2CBF38/3CBF6/05MV","VL"),
("D3 Activated T-cells AO%3","0003/1CBF5/2CBF38/3CBF6/08MV","VL"),
("D0 SD viable cell concentration Enriched T-Cells","0001/1CBF4/2CBF24/3CBF6/05MV","VL"),
("D3 SD viable cell concentration Activated T-Cells","0003/1CBF5/2CBF38/3CBF6/15MV","VL"),
("D10 SD viable cell concentration pre-wash","0006/1CBF6/2CBF11/3CBF7/17MV","VL"),
("D1 Pre-massaging Clumps: Color of Clumps","0002/1CBF1/2CBF2/3CBF19/02LT","VL"),
("D1 Pre-massaging Clumps: Shape and Texture of Clumps","0002/1CBF1/2CBF2/3CBF19/03LT","VL"),
("D3 Clumps: Color of Clumps","0003/1CBF5/2CBF15/02LT","VL"),
("D3 Clumps: Shape and Texture of Clumps","0003/1CBF5/2CBF15/03LT","VL")
"""

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
    ptf_cols: list[str],
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
    try:
        from databricks_query_tool import DatabricksConnection, Mapping, QueryRunner
    except ImportError as exc:
        raise RuntimeError(
            "sources.databricks is on but databricks_query_tool is not installed"
        ) from exc

    with DatabricksConnection() as conn:
        df_add = QueryRunner(conn).run(Mapping.from_string(AO_MAPPING))

    df_add = df_add.rename(columns={"targetbatchnumber": BATCH_KEY})
    added = [c for c in df_add.columns if c != BATCH_KEY]

    for site, df in phf.items():
        phf[site] = _lookup_merge(df, df_add, BATCH_KEY, f"[{site}] Acridine Orange")

    absent = [c for c in added if c not in ptf_cols]
    if absent:
        log.warn(
            MODULE,
            f"{len(absent)} Acridine Orange columns are not PTF parameters",
            ", ".join(absent),
        )

    log.success(MODULE, f"Acridine Orange parameters added — {len(added)} columns")
    return added


def merge_lv_coa(
    phf: dict[str, pd.DataFrame],
    df_lv_coa: pd.DataFrame,
) -> tuple[dict[str, dict], list[str]]:
    """Join the lentiviral vector certificate of analysis on ``Vector Lot``.

    Returns the per-site match rate and the columns added. A batch whose vector
    lot has no certificate keeps its row and gets empty CoA columns.
    """
    log.step(MODULE, "Merging LV CoA data")
    coverage: dict[str, dict] = {}
    added = [c for c in df_lv_coa.columns if c != "Vector Lot"]

    for site, df in phf.items():
        phf[site] = _lookup_merge(df, df_lv_coa, "Vector Lot", f"[{site}] LV CoA")

        total = len(phf[site])
        matched = int(phf[site]["LV CoA Site/Format"].notna().sum())
        coverage[site] = {
            "total": total,
            "matched": matched,
            "pct": round(100 * matched / total, 1) if total else 0,
        }
        log.info(
            MODULE,
            f"[{site}] LV CoA coverage: {matched}/{total} batches ({coverage[site]['pct']}%)",
        )

    log.success(MODULE, "LV CoA merge complete")
    return coverage, added


def merge_raw_materials(
    phf: dict[str, pd.DataFrame],
    df_raw: pd.DataFrame,
) -> list[str]:
    """Join raw material and consumable lot identifiers on the batch number.

    A column the PHF already carries is dropped from the incoming frame rather
    than joined twice under a suffix.
    """
    log.step(MODULE, "Merging raw materials and consumables")
    added = [c for c in df_raw.columns if c != BATCH_KEY]

    for site, df in phf.items():
        duplicated = (set(df.columns) & set(df_raw.columns)) - {BATCH_KEY}
        phf[site] = _lookup_merge(
            df,
            df_raw.drop(columns=list(duplicated), errors="ignore"),
            BATCH_KEY,
            f"[{site}] raw materials",
        )

    log.success(MODULE, f"Raw materials merged — {len(added)} identifier columns added")
    return added


def enrich_clinical_sites(
    phf: dict[str, pd.DataFrame],
    raritan_mapping_paths: list[str],
    sharepoint_loader = None
) -> tuple[list[str], list[str]]:
    """Give both sites a clinical site the other site would recognise.

    Ghent encodes the site in the patient identifier (``EU-<site>-…``); Raritan
    records an acronym that two mapping files expand into an institution name.

    Returns the Raritan acronyms no mapping file covers, and the columns added.
    """
    log.step(MODULE, "Enriching clinical site identifiers")

    ghent = phf["Ghent"]
    patient = ghent["Patient ID"].fillna("Unknown")
    ghent["Clinical Site ID"] = patient.map(
        lambda x: x.split("-")[1] if isinstance(x, str) and x.startswith("EU-") else x
    ).replace("Unknown", pd.NA)
    log.info(MODULE, "[Ghent] Clinical Site ID extracted from Patient ID")

    raritan = phf["Raritan"]
    raritan["Clinical Site ID"] = raritan["Clinical Site"].copy()

    frames = []
    if sharepoint_loader is not None:
        for path in ["clinical_sites.xlsx","cross_data_meeting.xlsx","from_Hannelore.xlsx"]:
            try:
                mapping = sharepoint_loader(sharepoint_path="MS%26T%20MSAT%20Data%20Team//Reports/Adv%20Analystics%20%26%20AI/Data/mappings/"+path,
                                            sheet_name="Sheet1",
                                            drive_id="DRIVE_ID_GHENT",
                                            )
                mapping["Acronym"] = mapping["Acronym"].astype(str)
                frames.append(mapping)
            except Exception as exc:
                log.warn(MODULE, f"Could not load mapping file: {path} — {exc}")
    else:
        for path in raritan_mapping_paths:
            try:
                mapping = pd.read_excel(path, engine="openpyxl")
                mapping["Acronym"] = mapping["Acronym"].astype(str)
                frames.append(mapping)
            except Exception as exc:
                log.warn(MODULE, f"Could not load mapping file: {path} — {exc}")

    if not frames:
        log.warn(MODULE, "No Raritan site mapping files loaded — acronyms unchanged")
        return [], ["Clinical Site ID"]

    acronyms = (
        pd.concat(frames)
        .drop_duplicates(["Acronym"], keep="last")
        .set_index("Acronym")["Institution"]
        .to_dict()
    )

    unmapped: list[str] = []

    def expand(acronym):
        key = str(acronym).upper()
        if key in acronyms:
            return acronyms[key]
        unmapped.append(acronym)
        return acronym

    site = raritan["Clinical Site"].fillna("Unknown")
    raritan["Clinical Site"] = site.map(expand).replace("Unknown", pd.NA)

    unique = sorted({str(x) for x in unmapped})
    if unique:
        log.warn(
            MODULE,
            f"[Raritan] {len(unique)} clinical site acronyms have no mapping",
            ", ".join(unique[:20]),
        )
    else:
        log.success(MODULE, "[Raritan] All clinical site acronyms resolved")

    return unique, ["Clinical Site ID"]


def harmonise_countries(phf: dict[str, pd.DataFrame]) -> None:
    """Spell the Raritan countries the way Ghent spells them.

    A replacement rather than a lookup, and on the site's own frame rather than
    on the raw one it was loaded from: a lookup empties every country that is not
    in the table, and the raw frame no longer carries the corrections cleaning
    made — the Israeli clinical site among them.
    """
    log.step(MODULE, "Harmonising country names")
    raritan = phf["Raritan"]
    raritan["Country"] = raritan["Country"].replace({"United States": "United States of America"})
    log.success(MODULE, "Country harmonisation complete")
