"""The PVF the build produced, and the report that explains it.

Every question here is about stage two: both sites under one set of names, values
the cleaning had to correct, sources joined on, features computed or reported as
blocked, and the PTF gap in both directions.
"""

import pandas as pd

import dummy_data
from conftest import facts, figures, ghent, raritan, row_named, rows_naming, section, tables
from pvf import stlite


# ---------------------------------------------------------------------------
# Loading and name mapping
# ---------------------------------------------------------------------------
def test_both_sites_reach_the_pvf_under_one_set_of_names(built):
    pvf = built["pvf"]
    assert len(pvf) == dummy_data.GHENT_BATCHES + dummy_data.RARITAN_BATCHES
    assert set(pvf["Site Merged"]) == {"Ghent", "Raritan"}
    # Raritan records these under its own names; the mapping file renames them.
    for renamed in dummy_data.RARITAN_NAMES:
        assert raritan(pvf)[renamed].notna().all(), renamed
    for raritan_name in dummy_data.RARITAN_NAMES.values():
        assert raritan_name not in pvf.columns


def test_two_columns_claiming_one_name_are_both_kept_with_their_own_values(built):
    """The first column by position keeps the name; the second takes the suffix.

    Checking the values, not just the names: the bug this covers put the first
    column's values under the suffixed name and the second column's under the
    plain one, which no test of column presence can see.
    """
    pvf = built["pvf"]
    assert "Country_1" in pvf.columns
    rows = raritan(pvf)
    # 'Country' was Raritan's own column, harmonised to the Ghent spelling.
    assert set(rows["Country"].dropna()) <= {"United States of America", "Israel"}
    # 'Country Name' mapped onto the taken name and kept its own values.
    assert set(rows["Country_1"].dropna()) == {"USA"}


def test_the_supplement_reaches_the_site_it_belongs_to(built):
    pvf = built["pvf"]
    assert raritan(pvf)["Number of Prior Lines of Therapy"].notna().all()
    assert ghent(pvf)["Number of Prior Lines of Therapy"].isna().all()


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def test_cleaning_corrects_what_the_sources_got_wrong(built):
    pvf = built["pvf"]
    assert set(raritan(pvf)["Type"]) == {"Commercial"}
    assert set(ghent(pvf)["Mycoplasma"]) == {"not detected"}
    # The correction is Raritan's; Ghent's own spelling is left alone.
    assert "withdrawn" not in set(raritan(pvf)["Non-Conformance Type"].dropna())
    assert "withdrawal" in set(raritan(pvf)["Non-Conformance Type"].dropna())
    # A clump count that is really a mangled date.
    assert "45691" not in set(pvf["Post-Mixing Clumps: # of Clumps"].astype(str))


def test_text_is_stripped_and_blanks_become_missing(built):
    sites = set(built["pvf"]["Clinical Site"].dropna())
    assert "UZ Leuven" in sites and "UZ Leuven\xa0" not in sites
    assert "AZ Sint-Jan" in sites and " AZ Sint-Jan" not in sites
    assert "" not in sites


def test_values_are_coerced_to_the_type_the_ptf_declares(built):
    pvf = built["pvf"]
    # "<0.5 EU/mL" survives as a number, not as text.
    assert pd.api.types.is_numeric_dtype(pvf["Endotoxin (EU/mL)"])
    assert pvf["Endotoxin (EU/mL)"].notna().any()
    # A duration written as HH:MM becomes minutes.
    contact = pvf["Day 3 Incubation Time (120min+/-30 min)"]
    assert pd.api.types.is_numeric_dtype(contact) and contact.max() < 24 * 60
    # A ratio keeps the side that carries the number.
    assert not pvf["Post Thaw CD4:CD8 Ratio"].astype(str).str.contains(":").any()
    # Text in a numeric column is dropped rather than carried.
    assert pvf["Recovery (%)"].isna().sum() == 1


def test_the_report_names_every_value_it_could_not_parse(built):
    payload = built["payloads"]["build"]
    assert rows_naming(payload, "not recorded"), "the unparseable timestamp"
    assert rows_naming(payload, "not started"), "the unparseable duration"
    assert rows_naming(payload, "not measured"), "the text in a numeric column"
    assert rows_naming(payload, "[105.0]"), "the percentage outside [0, 100]"


def test_yes_no_blanks_are_filled_in_each_site_own_spelling(built):
    pvf = built["pvf"]
    assert ghent(pvf)["Pre-Activation Clumps: Y/N?"].notna().all()
    assert set(raritan(pvf)["Pre-Activation Clumps: Y/N?"]) <= {"Y", "N"}


# ---------------------------------------------------------------------------
# Joined sources
# ---------------------------------------------------------------------------
def test_vector_certificates_join_where_there_is_one(built):
    pvf = built["pvf"]
    no_certificate = pvf["Vector Lot"] == dummy_data.LOT_WITHOUT_COA
    assert no_certificate.any()
    assert pvf.loc[no_certificate, "LV CoA Site/Format"].isna().all()
    assert pvf.loc[~no_certificate, "LV CoA Site/Format"].notna().all()
    # European-locale text became numbers, and a below-LOQ impurity was flagged.
    assert pd.api.types.is_numeric_dtype(pvf["LV CoA LV Titer"])
    assert "LV CoA HCP below LOQ" in pvf.columns
    assert "LV CoA pH Abs Deviation" in pvf.columns


def test_raw_materials_and_clinical_sites_are_resolved(built):
    pvf = built["pvf"]
    assert pvf["Media Lot"].notna().any()
    assert "Memorial Sloan Kettering Cancer Center" in set(pvf["Clinical Site"])
    assert "Rambam Health Care Campus" in set(pvf["Clinical Site"])
    # An acronym no mapping file covers stays as it is, and is reported.
    assert dummy_data.UNMAPPED_ACRONYM in set(pvf["Clinical Site"])
    assert rows_naming(built["payloads"]["build"], dummy_data.UNMAPPED_ACRONYM)
    assert ghent(pvf)["Clinical Site ID"].notna().all()


def test_the_country_correction_survives_harmonisation(built):
    pvf = built["pvf"]
    assert "Israel" in set(raritan(pvf)["Country"])
    assert set(raritan(pvf)["Country"]) <= {"Israel", "United States of America"}
    assert set(ghent(pvf)["Country"]) == {"Belgium", "Netherlands"}


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def test_features_are_computed_including_ones_built_on_other_features(built):
    pvf = built["pvf"]
    assert set(pvf["growth profile"].dropna()) <= {"undergrowth", "normal", "overgrowth"}
    assert pvf["Harvest Lactate Normalized"].notna().any()
    # MOI needs the titer from the certificate, so this proves the join fed it.
    assert pvf["MOI_eff_pool"].notna().any()
    # The slope has two of its three days at both sites, and is computed anyway.
    assert pvf["Glucose_perE6_slope_D6_D10"].notna().any()


def test_a_feature_missing_an_input_is_reported_and_not_invented(built):
    pvf, payload = built["pvf"], built["payloads"]["build"]
    assert "Log (PT CD4:CD8)" not in pvf.columns
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
    assert pvf["D10 Glucose / E6 cells"].notna().any()  # Ghent still has it


def test_a_feature_the_ptf_does_not_list_is_reported_and_not_created(built):
    for name in dummy_data.UNLISTED_FEATURES:
        assert name not in built["pvf"].columns
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
# The PTF gap, and the report itself
# ---------------------------------------------------------------------------
def test_the_report_lists_ptf_parameters_the_pvf_does_not_have(built):
    coverage = section(built["payloads"]["build"], "ptf-coverage")
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
    assert len(origins) == built["pvf"].shape[1]


def test_columns_the_ptf_does_not_list_are_reported_too(built):
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
    # Every input the run read is hashed, and a file it could not read says so.
    sources = {row[0].replace("\\", ""): row for row in panel[1]["rows"]}
    assert {"ptf", "phf_ghent", "phf_raritan", "lv_coa", "site_map_1"} <= set(sources)
    assert all(row[1] == "local file" for row in sources.values())
    assert all(len(row[3]) == 64 for row in sources.values())
    titles = [s["title"] for s in built["payloads"]["build"]["sections"]]
    assert titles[0] == "Overview"
    assert {"Cleaning", "Parameter inventory", "PTF coverage", "Derived features"} <= set(titles)


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


def test_the_build_writes_its_manifest_beside_the_pvf(built):
    import json

    from pvf import cli

    record = json.loads(
        cli.manifest_path(built["config"]["paths"]["pvf"], "pvf").read_text(encoding="utf-8")
    )
    assert record["stage"] == "pvf" and record["status"] == "complete"
    assert record["pvf"]["columns"] == built["pvf"].shape[1]
    kinds = {source["label"]: source["kind"] for source in record["provenance"]["sources"]}
    assert "investigations" not in kinds, "the build does not read it, so it does not record it"
    assert kinds["raw_materials"] == "local file"
