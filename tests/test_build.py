"""The batch table the build produced, and the report that explains it.

Every question here is about stage two: both sites under one set of names, values
the cleaning had to correct, sources joined on, features computed or reported as
blocked, and the catalog gap in both directions.
"""

import pandas as pd

import dummy_data
from conftest import facts, figures, site_a, site_b, row_named, rows_naming, section, tables
from ml import stlite


# ---------------------------------------------------------------------------
# Loading and name mapping
# ---------------------------------------------------------------------------
def test_both_sites_reach_the_batch_table_under_one_set_of_names(built):
    batch_table = built["batch_table"]
    assert len(batch_table) == dummy_data.SITE_A_BATCHES + dummy_data.SITE_B_BATCHES
    assert set(batch_table["Site Merged"]) == {"SiteA", "SiteB"}
    # SiteB records these under its own names; the mapping file renames them.
    for renamed in dummy_data.SITE_B_NAMES:
        assert site_b(batch_table)[renamed].notna().all(), renamed
    for site_b_name in dummy_data.SITE_B_NAMES.values():
        assert site_b_name not in batch_table.columns


def test_two_columns_claiming_one_name_are_both_kept_with_their_own_values(built):
    """The first column by position keeps the name; the second takes the suffix.

    Checking the values, not just the names: the bug this covers put the first
    column's values under the suffixed name and the second column's under the
    plain one, which no test of column presence can see.
    """
    batch_table = built["batch_table"]
    assert "Country_1" in batch_table.columns
    rows = site_b(batch_table)
    # 'Country' was SiteB's own column, harmonised to the SiteA spelling.
    assert set(rows["Country"].dropna()) <= {"<value:country_harmonised>", "Israel"}
    # '<site_b_country>' mapped onto the taken name and kept its own values.
    assert set(rows["Country_1"].dropna()) == {"USA"}


def test_the_supplement_reaches_the_site_it_belongs_to(built):
    batch_table = built["batch_table"]
    assert site_b(batch_table)["<prior_lines>"].notna().all()
    assert site_a(batch_table)["<prior_lines>"].isna().all()


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def test_cleaning_corrects_what_the_sources_got_wrong(built):
    batch_table = built["batch_table"]
    assert set(site_b(batch_table)["Type"]) == {"Commercial"}
    assert set(site_a(batch_table)["Mycoplasma"]) == {"<value:mycoplasma_clean>"}
    # The correction is SiteB's; SiteA's own spelling is left alone.
    assert "withdrawn" not in set(site_b(batch_table)["<nonconformance>"].dropna())
    assert "withdrawal" in set(site_b(batch_table)["<nonconformance>"].dropna())
    # A clump count that is really a mangled date.
    assert "<value:mangled_count>" not in set(batch_table["<clumps_mix_count>"].astype(str))


def test_text_is_stripped_and_blanks_become_missing(built):
    sites = set(built["batch_table"]["<clinical_site>"].dropna())
    assert "UZ Leuven" in sites and "UZ Leuven\xa0" not in sites
    assert "AZ Sint-Jan" in sites and "<value:clinical_sites_a>" not in sites
    assert "" not in sites


def test_values_are_coerced_to_the_type_the_catalog_declares(built):
    batch_table = built["batch_table"]
    # "<0.5<value:endotoxin_unit>" survives as a number, not as text.
    assert pd.api.types.is_numeric_dtype(batch_table["<endotoxin>"])
    assert batch_table["<endotoxin>"].notna().any()
    # A duration written as HH:MM becomes minutes.
    contact = batch_table["<contact_time>"]
    assert pd.api.types.is_numeric_dtype(contact) and contact.max() < 24 * 60
    # A ratio keeps the side that carries the number.
    assert not batch_table["<pt_cd4_cd8_ratio>"].astype(str).str.contains(":").any()
    # Text in a numeric column is dropped rather than carried.
    assert batch_table["<recovery>"].isna().sum() == 1


def test_the_report_names_every_value_it_could_not_parse(built):
    payload = built["payloads"]["build"]
    assert rows_naming(payload, "not recorded"), "the unparseable timestamp"
    assert rows_naming(payload, "not started"), "the unparseable duration"
    assert rows_naming(payload, "not measured"), "the text in a numeric column"
    assert rows_naming(payload, "[105.0]"), "the percentage outside [0, 100]"


def test_yes_no_blanks_are_filled_in_each_site_own_spelling(built):
    batch_table = built["batch_table"]
    assert site_a(batch_table)["<pre_activation_clumps_yn>"].notna().all()
    assert set(site_b(batch_table)["<pre_activation_clumps_yn>"]) <= {"Y", "N"}


# ---------------------------------------------------------------------------
# Joined sources
# ---------------------------------------------------------------------------
def test_vector_certificates_join_where_there_is_one(built):
    batch_table = built["batch_table"]
    no_certificate = batch_table["<vector_lot>"] == dummy_data.LOT_WITHOUT_COA
    assert no_certificate.any()
    assert batch_table.loc[no_certificate, "<coa_prefix><coa_site_format>"].isna().all()
    assert batch_table.loc[~no_certificate, "<coa_prefix><coa_site_format>"].notna().all()
    # European-locale text became numbers, and a below-LOQ impurity was flagged.
    assert pd.api.types.is_numeric_dtype(batch_table["<coa_prefix><coa_titer>"])
    assert "<coa_prefix>HCP below LOQ" in batch_table.columns
    assert "<coa_prefix>pH Abs Deviation" in batch_table.columns


def test_raw_materials_and_clinical_sites_are_resolved(built):
    batch_table = built["batch_table"]
    assert batch_table["<media_lot>"].notna().any()
    assert "<value:institutions_second>" in set(batch_table["<clinical_site>"])
    assert "<value:institutions_second>" in set(batch_table["<clinical_site>"])
    # An acronym no mapping file covers stays as it is, and is reported.
    assert dummy_data.UNMAPPED_ACRONYM in set(batch_table["<clinical_site>"])
    assert rows_naming(built["payloads"]["build"], dummy_data.UNMAPPED_ACRONYM)
    assert site_a(batch_table)["<clinical_site_id>"].notna().all()


def test_the_country_correction_survives_harmonisation(built):
    batch_table = built["batch_table"]
    assert "Israel" in set(site_b(batch_table)["Country"])
    assert set(site_b(batch_table)["Country"]) <= {"Israel", "<value:country_harmonised>"}
    assert set(site_a(batch_table)["Country"]) == {"Belgium", "Netherlands"}


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def test_features_are_computed_including_ones_built_on_other_features(built):
    batch_table = built["batch_table"]
    assert set(batch_table["<f_growth_profile>"].dropna()) <= {"undergrowth", "normal", "overgrowth"}
    assert batch_table["<f_harvest_lactate_norm>"].notna().any()
    # MOI needs the titer from the certificate, so this proves the join fed it.
    assert batch_table["<f_moi_pool>"].notna().any()
    # The slope has two of its three days at both sites, and is computed anyway.
    assert batch_table["<f_glucose_slope>"].notna().any()


def test_a_feature_missing_an_input_is_reported_and_not_invented(built):
    batch_table, payload = built["batch_table"], built["payloads"]["build"]
    assert "<f_log_pt_cd4_cd8>" not in batch_table.columns
    blocked = rows_naming(payload, "<f_log_pt_cd4_cd8>")
    assert any(dummy_data.BLOCKED_INPUT in str(row) for row in blocked)
    # SiteB alone lacks harvest glucose, so the gap is reported for one site.
    per_site = [
        row
        for row in rows_naming(payload, "<f_glucose_per_cell_3>")
        if any(str(cell).startswith("missing:") for cell in row)
    ]
    assert per_site
    assert all(any("SiteB" in str(cell) for cell in row) for row in per_site)
    assert batch_table["<f_glucose_per_cell_3>"].notna().any()  # SiteA still has it


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
    assert "<f_log_pt_cd4_cd8>" in names
    assert "blocked" in str(row_named(missing, "<f_log_pt_cd4_cd8>"))


def test_the_inventory_says_where_every_parameter_came_from(built):
    inventory = section(built["payloads"]["build"], "inventory")["blocks"][0]
    origins = {row[0]: row[1] for row in inventory["rows"]}
    assert origins["<f_growth_profile>"].startswith("Derived")
    assert origins["<coa_prefix><coa_titer>"] == "Vector certificate of analysis"
    assert origins["<media_lot>"] == "Raw materials and consumables"
    assert origins["Site Merged"] == "Merge"
    assert origins["<site_a_comment>"] == "SiteA PHF only"
    assert len(origins) == built["batch_table"].shape[1]


def test_columns_the_catalog_does_not_list_are_reported_too(built):
    names = [row[0] for row in rows_naming(built["payloads"]["build"], "<site_a_comment>")]
    assert "<site_a_comment>" in names
    assert rows_naming(built["payloads"]["build"], "<site_b_comment>")


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
    assert {"catalog", "phf_site_a", "phf_site_b", "lv_coa", "site_map_1"} <= set(sources)
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
