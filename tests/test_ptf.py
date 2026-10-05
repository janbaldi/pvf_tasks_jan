"""What the PTF stage found, and how it says it.

Stage one asks one question — which parameters do the sources record that the
schema has never been told about — so these tests check that it asks it of every
source, that it asks it through the Raritan name mapping rather than around it,
and that what it finds is usable by whoever maintains the PTF.
"""

import dummy_data
from conftest import facts, first_table, rows_naming, section


def test_every_source_is_compared(built):
    rows = {row[0]: row for row in first_table(built["payloads"]["ptf"], "sources")["rows"]}
    assert {"Ghent PHF", "Raritan batch data", "Investigations Power Query"} <= set(rows)
    assert all(row[3] == "read" for row in rows.values())


def test_the_columns_nobody_added_to_the_ptf_are_found(built):
    found = set(built["new_parameters"]["Parameter"])
    # The two site comment columns, and the investigations team's own working ones.
    assert {"Local Comment", "CMD Comment"} <= found
    assert any("Investigation ID" in name for name in found)
    assert "Days to Close" in found


def test_a_raritan_column_is_judged_by_the_name_it_maps_to(built):
    """Raritan's own names are not PTF parameters; what they map to is."""
    found = set(built["new_parameters"]["Parameter"])
    for raritan_name in dummy_data.RARITAN_NAMES.values():
        assert raritan_name not in found, raritan_name


def test_a_parameter_the_ptf_already_lists_is_not_reported(built):
    payload = built["payloads"]["ptf"]
    new = {row[0] for row in first_table(payload, "new")["rows"]}
    assert "Harvest Lactate (g/L)" not in new
    assert "Patient Lot/Batch #" not in new


def test_the_report_says_what_it_compared_and_what_produced_it(built):
    payload = built["payloads"]["ptf"]
    counts = facts(first_table(payload, "overview"))
    assert int(counts["Distinct names among them"]) == len(built["new_parameters"])
    assert int(counts["Parameters in the PTF"]) > 100
    assert rows_naming(payload, "new_parameters.csv"), "the file it wrote is named"
    assert section(payload, "provenance")


def test_the_console_names_every_column_the_ptf_does_not_list(built, caplog):
    from pvf import cli

    cli.run_ptf(built["config"], config_path=built["config_path"])
    out = "\n".join(record.getMessage() for record in caplog.records)
    assert "Ghent PHF: columns the PTF does not list (1)" in out
    assert "  - Local Comment" in out
    assert "\x1b[" not in out, "no colour codes when stdout is not a terminal"
    manifest = cli.manifest_path(built["config"]["paths"]["pvf"], "ptf")
    assert str(manifest) in out
