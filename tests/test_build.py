"""The batch table the build produced, and the report that explains it.

Every question here is about stage two: both sites under one set of names, values
the cleaning had to correct, sources joined on, features computed or reported as
blocked, and the catalog gap in both directions.
"""

import pandas as pd

import dummy_data
from conftest import facts, figures, ghent, raritan, row_named, rows_naming, section, tables
from ml import stlite


# ---------------------------------------------------------------------------
# Loading and name mapping
# ---------------------------------------------------------------------------
def test_both_sites_reach_the_batch_table_under_one_set_of_names(built):
    batch_table = built["batch_table"]
    assert len(batch_table) == dummy_data.GHENT_BATCHES + dummy_data.RARITAN_BATCHES
    assert set(batch_table["Site Merged"]) == {"Ghent", "Raritan"}
    # Raritan records these under its own names; the mapping file renames them.
    for renamed in dummy_data.RARITAN_NAMES:
        assert raritan(batch_table)[renamed].notna().all(), renamed
    for raritan_name in dummy_data.RARITAN_NAMES.values():
        assert raritan_name not in batch_table.columns


def test_two_columns_claiming_one_name_are_both_kept_with_their_own_values(built):
    """The first column by position keeps the name; the second takes the suffix.

    Checking the values, not just the names: the bug this covers put the first
    column's values under the suffixed name and the second column's under the
    plain one, which no test of column presence can see.
    """
    batch_table = built["batch_table"]
    assert "Country_1" in batch_table.columns
    rows = raritan(batch_table)
    # 'Country' was Raritan's own column, harmonised to the Ghent spelling.
    assert set(rows["Country"].dropna()) <= {"United States of America", "Israel"}
    # 'Country Name' mapped onto the taken name and kept its own values.
    assert set(rows["Country_1"].dropna()) == {"USA"}


def test_the_supplement_reaches_the_site_it_belongs_to(built):
    batch_table = built["batch_table"]
    assert raritan(batch_table)["Number of Prior Lines of Therapy"].notna().all()
    assert ghent(batch_table)["Number of Prior Lines of Therapy"].isna().all()


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def test_cleaning_corrects_what_the_sources_got_wrong(built):
    batch_table = built["batch_table"]
    assert set(raritan(batch_table)["Type"]) == {"Commercial"}
    assert set(ghent(batch_table)["Mycoplasma"]) == {"not detected"}
    # The correction is Raritan's; Ghent's own spelling is left alone.
    assert "withdrawn" not in set(raritan(batch_table)["Non-Conformance Type"].dropna())
    assert "withdrawal" in set(raritan(batch_table)["Non-Conformance Type"].dropna())
    # A clump count that is really a mangled date.
    assert "45691" not in set(batch_table["Post-Mixing Clumps: # of Clumps"].astype(str))


def test_text_is_stripped_and_blanks_become_missing(built):
    sites = set(built["batch_table"]["Clinical Site"].dropna())
    assert "UZ Leuven" in sites and "UZ Leuven\xa0" not in sites
    assert "AZ Sint-Jan" in sites and " AZ Sint-Jan" not in sites
    assert "" not in sites


def test_values_are_coerced_to_the_type_the_catalog_declares(built):
    batch_table = built["batch_table"]
    # "<0.5 EU/mL" survives as a number, not as text.
    assert pd.api.types.is_numeric_dtype(batch_table["Endotoxin (EU/mL)"])
    assert batch_table["Endotoxin (EU/mL)"].notna().any()
    # A duration written as HH:MM becomes minutes.
    contact = batch_table["Day 3 Incubation Time (120min+/-30 min)"]
    assert pd.api.types.is_numeric_dtype(contact) and contact.max() < 24 * 60
    # A ratio keeps the side that carries the number.
    assert not batch_table["Post Thaw CD4:CD8 Ratio"].astype(str).str.contains(":").any()
    # Text in a numeric column is dropped rather than carried.
    assert batch_table["Recovery (%)"].isna().sum() == 1


def test_the_report_names_every_value_it_could_not_parse(built):
    payload = built["payloads"]["build"]
    assert rows_naming(payload, "not recorded"), "the unparseable timestamp"
    assert rows_naming(payload, "not started"), "the unparseable duration"
    assert rows_naming(payload, "not measured"), "the text in a numeric column"
    assert rows_naming(payload, "[105.0]"), "the percentage outside [0, 100]"


def test_yes_no_blanks_are_filled_in_each_site_own_spelling(built):
    batch_table = built["batch_table"]
    assert ghent(batch_table)["Pre-Activation Clumps: Y/N?"].notna().all()
    assert set(raritan(batch_table)["Pre-Activation Clumps: Y/N?"]) <= {"Y", "N"}


# ---------------------------------------------------------------------------
# Joined sources
# ---------------------------------------------------------------------------
def test_vector_certificates_join_where_there_is_one(built):
    batch_table = built["batch_table"]
    no_certificate = batch_table["Vector Lot"] == dummy_data.LOT_WITHOUT_COA
    assert no_certificate.any()
    assert batch_table.loc[no_certificate, "LV CoA Site/Format"].isna().all()
    assert batch_table.loc[~no_certificate, "LV CoA Site/Format"].notna().all()
    # European-locale text became numbers, and a below-LOQ impurity was flagged.
    assert pd.api.types.is_numeric_dtype(batch_table["LV CoA LV Titer"])
    assert "LV CoA HCP below LOQ" in batch_table.columns
    assert "LV CoA pH Abs Deviation" in batch_table.columns


def test_raw_materials_and_clinical_sites_are_resolved(built):
    batch_table = built["batch_table"]
    assert batch_table["Media Lot"].notna().any()
    assert "Memorial Sloan Kettering Cancer Center" in set(batch_table["Clinical Site"])
    assert "Rambam Health Care Campus" in set(batch_table["Clinical Site"])
    # An acronym no mapping file covers stays as it is, and is reported.
    assert dummy_data.UNMAPPED_ACRONYM in set(batch_table["Clinical Site"])
    assert rows_naming(built["payloads"]["build"], dummy_data.UNMAPPED_ACRONYM)
    assert ghent(batch_table)["Clinical Site ID"].notna().all()


def test_the_country_correction_survives_harmonisation(built):
    batch_table = built["batch_table"]
    assert "Israel" in set(raritan(batch_table)["Country"])
    assert set(raritan(batch_table)["Country"]) <= {"Israel", "United States of America"}
    assert set(ghent(batch_table)["Country"]) == {"Belgium", "Netherlands"}


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def test_features_are_computed_including_ones_built_on_other_features(built):
    batch_table = built["batch_table"]
    assert set(batch_table["growth profile"].dropna()) <= {"undergrowth", "normal", "overgrowth"}
    assert batch_table["Harvest Lactate Normalized"].notna().any()
    # MOI needs the titer from the certificate, so this proves the join fed it.
    assert batch_table["MOI_eff_pool"].notna().any()
    # The slope has two of its three days at both sites, and is computed anyway.
    assert batch_table["Glucose_perE6_slope_D6_D10"].notna().any()


def test_a_feature_missing_an_input_is_reported_and_not_invented(built):
    batch_table, payload = built["batch_table"], built["payloads"]["build"]
    assert "Log (PT CD4:CD8)" not in batch_table.columns
    blocked = rows_naming(payload, "Log (PT CD4:CD8)")
    assert any(dummy_data.BLOCKED_INPUT in str(row) for row in blocked)
    # Raritan alone lacks harvest glucose, so the gap is reported for one site.
    per_site = [
        row
        for row in rows_naming(payload, "D10 Glucose / E6 cells")
        if any(str(cell).startswith("missing:") for cell in row)
    ]
    assert per_site
    assert all(any("Raritan" in str(cell) for cell in row) for row in per_site)
    assert batch_table["D10 Glucose / E6 cells"].notna().any()  # Ghent still has it


def test_a_feature_the_catalog_does_not_list_is_reported_and_not_created(built):
    for name in dummy_data.UNLISTED_FEATURES:
        assert name not in built["batch_table"].columns
        assert rows_naming(built["payloads"]["build"], name), name


def test_no_feature_raised(built):
    """A calculation that throws is a defect, not a data condition."""
    errored = [
        row
        for block in tables(built["payloads"]["build"])
        for row in block["rows"]
        if any("Traceback" in str(cell) or "could not be broadcast" in str(cell) for cell in row)
    ]
    assert not errored


# ---------------------------------------------------------------------------
# The catalog gap, and the report itself
# ---------------------------------------------------------------------------
def test_the_report_lists_catalog_parameters_the_batch_table_does_not_have(built):
    coverage = section(built["payloads"]["build"], "catalog-coverage")
    missing = coverage["blocks"][0]
    names = {row[0] for row in missing["rows"]}
    for orphan in dummy_data.ORPHAN_PARAMETERS:
        assert orphan in names
        assert "no source supplies it" in str(
            dict(zip(missing["headers"], row_named(missing, orphan)))
        )
    # A blocked feature is a different kind of gap, and says so.
    assert "Log (PT CD4:CD8)" in names
    assert "blocked" in str(row_named(missing, "Log (PT CD4:CD8)"))


def test_the_inventory_says_where_every_parameter_came_from(built):
    inventory = section(built["payloads"]["build"], "inventory")["blocks"][0]
    origins = {row[0]: row[1] for row in inventory["rows"]}
    assert origins["growth profile"].startswith("Derived")
    assert origins["LV CoA LV Titer"] == "Vector certificate of analysis"
    assert origins["Media Lot"] == "Raw materials and consumables"
    assert origins["Site Merged"] == "Merge"
    assert origins["Local Comment"] == "Ghent PHF only"
    assert len(origins) == built["batch_table"].shape[1]


def test_columns_the_catalog_does_not_list_are_reported_too(built):
    names = [row[0] for row in rows_naming(built["payloads"]["build"], "Local Comment")]
    assert "Local Comment" in names
    assert rows_naming(built["payloads"]["build"], "CMD Comment")


def test_the_report_is_one_file_with_its_provenance_in_it(built):
    page = built["reports"]["build"].read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>")
    assert stlite.STLITE_VERSION in page
    panel = section(built["payloads"]["build"], "provenance")["blocks"][0]["blocks"]
    recorded = facts(panel[0])
    assert recorded["Run date"] and recorded["Seed"] == str(dummy_data.SEED)
    assert str(built["config_path"]) in recorded["Config"].replace("\\", "")
    # Every input was read from SharePoint, and every table it read is digested.
    sources = {row[0].replace("\\", ""): row for row in panel[1]["rows"]}
    assert {"catalog", "phf_ghent", "phf_raritan", "lv_coa", "site_map_1"} <= set(sources)
    assert all(row[1] == "SharePoint" for row in sources.values())
    assert all(len(row[3]) == 64 for name, row in sources.items() if "site_map" not in name)
    titles = [s["title"] for s in built["payloads"]["build"]["sections"]]
    assert titles[0] == "Overview"
    assert {"Cleaning", "Parameter inventory", "Catalog coverage", "Derived features"} <= set(
        titles
    )


def test_a_legend_has_a_row_of_its_own_under_the_title(built):
    """The bug this covers: legends sat in the title's margin and were drawn over it."""
    with_legend = [
        figure["spec"]["layout"]
        for figure in figures(built["payloads"]["build"])
        if figure["spec"]["layout"].get("showlegend")
    ]
    assert with_legend
    for layout in with_legend:
        assert layout["legend"]["yanchor"] == "bottom" and layout["legend"]["y"] >= 1
        assert layout["title"]["yref"] == "container" and layout["title"]["yanchor"] == "top"
        assert layout["margin"]["t"] >= 72


def test_the_build_writes_everything_into_the_batch_table_folder_on_sharepoint(built):
    import json

    from conftest import shared

    folder = shared(built["directory"], built["config"], "batch_table").parent
    assert {path.name for path in folder.iterdir()} == {
        "batch_data.xlsx",
        "batch_data.parquet",
        "build_report.html",
        "batch_table_manifest.json",
    }
    record = json.loads((folder / "batch_table_manifest.json").read_text(encoding="utf-8"))
    assert record["stage"] == "batch_table" and record["status"] == "complete"
    assert record["batch_table"]["columns"] == built["batch_table"].shape[1]
    kinds = {source["label"]: source["kind"] for source in record["provenance"]["sources"]}
    assert "investigations" not in kinds, "the build does not read it, so it does not record it"
    assert kinds["raw_materials"] == "SharePoint"
    assert record["written_to"].endswith("Data/batch_data/")
    assert {entry["path"] for entry in record["outputs"]} == {
        "batch_data.xlsx",
        "batch_data.parquet",
        "build_report.html",
    }
