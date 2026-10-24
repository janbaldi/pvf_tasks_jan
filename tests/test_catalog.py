"""What the catalog stage found, and how it says it.

Stage one asks one question — which parameters do the sources record that the
schema has never been told about — so these tests check that it asks it of every
source, that it asks it through site B's name mapping rather than around it,
and that what it finds is usable by whoever maintains the catalog.
"""

import dummy_data
from conftest import facts, first_table, rows_naming, section
from dummy_data import c
from ml import names


def test_every_source_of_the_site_is_compared(built):
    rows = {row[0]: row for row in first_table(built["payloads"]["catalog"], "sources")["rows"]}
    assert {f"{names.site('a')} batch records", "Investigations Power Query"} <= set(rows)
    assert f"{names.site('b')} batch records" not in rows, "everything is about one site"
    assert all(row[3] == "read" for row in rows.values())
    rows_b = first_table(built["payloads_b"]["catalog"], "sources")["rows"]
    assert f"{names.site('b')} batch records" in {row[0] for row in rows_b}


def test_the_columns_nobody_added_to_the_catalog_are_found(built):
    found = set(built["new_parameters"]["Parameter"])
    # Each site's comment column, in its own check, and the investigations team's own ones.
    assert c("site_a_comment") in found and c("site_b_comment") not in found
    assert c("site_b_comment") in set(built["new_parameters_b"]["Parameter"])
    assert any(c("investigation_id") in name for name in found)
    assert c("days_to_close") in found


def test_a_site_b_column_is_judged_by_the_name_it_maps_to(built):
    """Site B's own names are not catalog parameters; what they map to is."""
    found = set(built["new_parameters_b"]["Parameter"])
    for own_name in dummy_data.site_b_names().values():
        assert own_name not in found, own_name


def test_a_parameter_the_catalog_already_lists_is_not_reported(built):
    payload = built["payloads"]["catalog"]
    new = {row[0] for row in first_table(payload, "new")["rows"]}
    assert c("harvest_lactate") not in new
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
    assert f"{names.site('a')} batch records: columns the catalog does not list (1)" in out
    assert f"  - {c('site_a_comment')}" in out
    assert "\x1b[" not in out, "no colour codes when stdout is not a terminal"
    # What it read and wrote is listed, file by file, with the SharePoint folder.
    assert "Reading from SharePoint" in out and "Written to SharePoint" in out
    for name in ("catalog_report.html", "catalog_manifest.json", "new_parameters.csv"):
        assert name in out
    assert "DEMO_DRIVE › Data/catalog/" in out


def test_the_catalog_check_writes_into_the_catalog_folder_on_sharepoint(built):
    from conftest import shared

    folder = shared(built["directory"], built["config"], "catalog").parent
    # The catalog and the register are shared; each site's check has a folder of its own.
    assert {path.name for path in folder.iterdir()} == {
        "catalog.xlsx",
        "parameter_screening.xlsx",
        names.site("a"),
        names.site("b"),
    }
    for site in (names.site("a"), names.site("b")):
        assert {path.name for path in (folder / site).iterdir()} == {
            "catalog_report.html",
            "catalog_manifest.json",
            "new_parameters.csv",
        }
