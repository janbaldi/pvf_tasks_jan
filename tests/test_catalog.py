"""What the catalog stage found, and how it says it.

Stage one asks one question — which parameters do the sources record that the
schema has never been told about — so these tests check that it asks it of every
source, that it asks it through the SiteB name mapping rather than around it,
and that what it finds is usable by whoever maintains the catalog.
"""

import dummy_data
from conftest import facts, first_table, rows_naming, section


def test_every_source_is_compared(built):
    rows = {row[0]: row for row in first_table(built["payloads"]["catalog"], "sources")["rows"]}
    assert {"SiteA PHF", "SiteB batch data", "Investigations Power Query"} <= set(rows)
    assert all(row[3] == "read" for row in rows.values())


def test_the_columns_nobody_added_to_the_catalog_are_found(built):
    found = set(built["new_parameters"]["Parameter"])
    # The two site comment columns, and the investigations team's own working ones.
    assert {"<site_a_comment>", "<site_b_comment>"} <= found
    assert any("<investigation_id>" in name for name in found)
    assert "<days_to_close>" in found


def test_a_site_b_column_is_judged_by_the_name_it_maps_to(built):
    """SiteB's own names are not catalog parameters; what they map to is."""
    found = set(built["new_parameters"]["Parameter"])
    for site_b_name in dummy_data.SITE_B_NAMES.values():
        assert site_b_name not in found, site_b_name


def test_a_parameter_the_catalog_already_lists_is_not_reported(built):
    payload = built["payloads"]["catalog"]
    new = {row[0] for row in first_table(payload, "new")["rows"]}
    assert "<harvest_lactate>" not in new
    assert "Patient Lot/Batch #" not in new


def test_the_report_says_what_it_compared_and_what_produced_it(built):
    payload = built["payloads"]["catalog"]
    counts = facts(first_table(payload, "overview"))
    assert int(counts["Distinct names among them"]) == len(built["new_parameters"])
    assert int(counts["Parameters in the catalog"]) > 100
    assert rows_naming(payload, "new_parameters.csv"), "the file it wrote is named"
    assert section(payload, "provenance")


def test_the_console_names_every_column_the_catalog_does_not_list(built, caplog):
    from ml import cli

    cli.run_catalog(built["config"], config_path=built["config_path"])
    out = "\n".join(record.getMessage() for record in caplog.records)
    assert "SiteA PHF: columns the catalog does not list (1)" in out
    assert "  - <site_a_comment>" in out
    assert "\x1b[" not in out, "no colour codes when stdout is not a terminal"
    # What it read and wrote is listed, file by file, with the SharePoint folder.
    assert "Reading from SharePoint" in out and "Written to SharePoint" in out
    for name in ("catalog_report.html", "catalog_manifest.json", "new_parameters.csv"):
        assert name in out
    assert "DEMO_DRIVE › Data/catalog/" in out


def test_the_catalog_check_writes_into_the_catalog_folder_on_sharepoint(built):
    from conftest import shared

    folder = shared(built["directory"], built["config"], "catalog").parent
    assert {path.name for path in folder.iterdir()} == {
        "catalog.xlsx",
        "catalog_report.html",
        "catalog_manifest.json",
        "new_parameters.csv",
    }
