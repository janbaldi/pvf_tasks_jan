"""The correlation report: numerical parameters that rank the batches alike."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import dummy_data
from conftest import shared
from ml import correlation, names, report, screening, stlite


def test_parameters_that_rank_the_batches_alike_group_whatever_the_transform():
    rng = np.random.default_rng(0)
    x = rng.normal(size=120)
    frame = pd.DataFrame(
        {
            "id": range(120),
            "date": pd.date_range("2024-01-01", periods=120),
            "a": x,
            "b": np.exp(x),  # same order
            "c": -(x**3),  # reverse order: as redundant to a tree
            "d": rng.normal(size=120),
            "e": 2 * x,  # screened out for missingness: left out
            "f": x + 1,  # screened out as correlated: kept, to be reinstated
        }
    )
    frame.loc[:30, "b"] = np.nan
    types = dict.fromkeys("abcdef", "numeric")
    rows, report = correlation.for_build(
        frame, types, "id", "date", screened_out={"e": "retired", "f": correlation.REASON}
    )
    (group,) = report.groups
    assert sorted(group.members) == ["a", "b", "c", "f"]
    assert group.weakest > 0.99 and group.max_q < 1e-6 and group.min_batches == 89
    assert report.left_out == ["e"] and "e" not in report.values
    # The representative carries the others where they were measured: not b, with gaps.
    assert group.members[0] == group.representative != "b" and group.score > 0.9
    by_name = {row["Parameter"]: row for row in rows}
    assert by_name["d"]["Group"] == "" and by_name["b"]["Batches with a value"] == 89
    # The scatterplots: each member's value per batch, None where it has none.
    assert set(report.values) == {"a", "b", "c", "d", "f"}
    assert report.values["b"][:31] == [None] * 31 and report.values["a"][40] == float(
        f"{x[40]:.4g}"
    )


SITES = ("North", "South")
EVIDENCE = pd.DataFrame(
    {
        "Site": ["North"] * 4,
        "Parameter": ["X", "Y", "W", "Z"],
        "Missing %": [0, 0, 0, 0],
        "Pattern": ["Complete"] * 4,
        "Last measured": [""] * 4,
        "Group": [""] * 4,
        screening.CORRELATION_GROUP: ["Correlated 01"] * 3 + [""],
        screening.SUGGESTED: ["Y"] * 3 + [""],
    }
)


def _exclude(**row) -> pd.DataFrame:
    base = {
        "Action": "exclude",
        "Site": "North",
        "Reason": correlation.REASON,
        "Rationale": "ranks the batches as its representative does",
    }
    return pd.DataFrame([base | row])


def test_a_correlation_group_is_represented_by_the_suggested_member():
    added = screening.add([], _exclude(Group="Correlated 01"), EVIDENCE, "", "A. R.", SITES)
    assert [(d.parameter, d.represented_by) for d in added] == [("X", "Y"), ("W", "Y")]
    assert added[0].label == "S-0001: correlated with group (North), represented by 'Y'"
    # Named by hand, the representative stays out of the exclusion.
    chosen = _exclude(Group="Correlated 01", **{"Represented by": "W"})
    added = screening.add([], chosen, EVIDENCE, "", "A. R.", SITES)
    assert [(d.parameter, d.represented_by) for d in added] == [("X", "W"), ("Y", "W")]


def test_a_representative_must_stay_usable():
    standing = [
        screening.Decision(
            *("S-0001", "2026-01-01", "A", "exclude", "Z", "North", "", "retired", "x"),
            *("", "", "", "", ""),
        )
    ]
    with pytest.raises(screening.ScreeningError, match="'Z' is screened out at North"):
        screening.add(
            standing, _exclude(Parameter="X", **{"Represented by": "Z"}), EVIDENCE, "", "A", SITES
        )
    both = pd.concat(
        [
            _exclude(Parameter="X", **{"Represented by": "Y"}),
            _exclude(Parameter="Y", **{"Represented by": "W"}, Rationale="other"),
        ]
    )
    with pytest.raises(screening.ScreeningError, match="'Y', which the proposal excludes too"):
        screening.add([], both, EVIDENCE, "", "A", SITES)
    with pytest.raises(screening.ScreeningError, match="name the parameter that represents it"):
        screening.add([], _exclude(Parameter="Z"), EVIDENCE, "", "A", SITES)


def test_the_register_records_the_representative_and_still_reads_old_ones():
    (decision,) = screening.add(
        [], _exclude(Parameter="X", **{"Represented by": "Y"}), EVIDENCE, "", "A", SITES
    )
    frame = screening.to_frame([decision])
    assert screening.from_frame(frame, SITES)[0].represented_by == "Y"
    old = frame.drop(columns="Represented by").assign(**{"Represented by": ""})
    plain = screening.from_frame(frame.drop(columns="Represented by"), SITES)
    assert plain[0].represented_by == ""
    assert screening.digest(plain) == screening.digest(screening.from_frame(old, SITES))
    # Excluding the representative later is a reason to look again.
    later = [
        decision,
        *screening.add(
            [decision], _exclude(Parameter="Y", Reason="retired"), EVIDENCE, "", "A", SITES
        ),
    ]
    (finding,) = [f for f in screening.review(later, EVIDENCE, SITES) if f["Parameter"] == "X"]
    assert "its representative 'Y' is screened out too" in finding["Why look again"]


def test_the_task_catalog_lists_what_a_task_can_still_use():
    log = screening.add([], _exclude(Group="Correlated 01"), EVIDENCE, "", "A", SITES)
    log.append(replace(log[0], id="S-0002", parameter="Z", site="all", represented_by=""))
    catalog = pd.DataFrame({"Parameter": ["X", "Y", "W", "Z", "V"], "Value Type": ["numeric"] * 5})
    table = screening.task_catalog(
        catalog, ["X", "Y", "W", "Z"], screening.current(log, SITES), SITES
    )
    rows = table.set_index("Parameter")
    assert list(rows.index) == ["X", "Y", "W"], "Z excluded everywhere, V not in the batch table"
    assert rows.loc["X", "Usable at"] == "South" and rows.loc["Y", "Usable at"] == "North, South"
    assert rows.loc["Y", "Represents"] == "North: W, X"
    assert list(table.columns) == ["Parameter", "Usable at", "Represents", "Value Type"]


def test_the_build_writes_the_correlation_report_over_the_missingness_batches(built):
    page = shared(built["directory"], built["config"], "batch_table", "correlation_report.html")
    payload = stlite.read_payload(page.read_text(encoding="utf-8"))
    viewer = payload["sections"][0]["blocks"][0]
    assert viewer["kind"] == "rank_viewer"
    assert viewer["proposal"]["reasons"] == list(screening.REASONS)
    missing_page = shared(
        built["directory"], built["config"], "batch_table", "missingness_report.html"
    )
    missing = stlite.read_payload(missing_page.read_text(encoding="utf-8"))
    counted = {
        site["label"]: [b for b, bit in zip(site["batches"], site["in_statistics"]) if bit == "1"]
        for site in missing["sections"][0]["blocks"][0]["sites"]
    }
    for site in viewer["sites"]:
        # The same batches as the missingness statistics, and only numerical parameters.
        assert sorted(site["batches"]) == sorted(counted[site["label"]])
        members = {m for group in site["groups"].values() for m in group}
        assert members <= set(site["values"])
        assert not members & {row[0] for row in site["rows"]}, "grouped: in the group table"
        assert site["headers"][:2] == ["Parameter", "Register"]
        closest = site["headers"].index("Closest parameter")
        assert {row[closest] for row in site["rows"]} - {""} <= set(site["values"])
        assert all(len(values) == len(site["batches"]) for values in site["values"].values())
    # Screened out at site A for missingness, so not compared there.
    first = {site["label"]: site for site in viewer["sites"]}[names.site("a")]
    assert dummy_data.screened_parameter() not in first["values"]
    # Each group is drawn from its suggested representative.
    for site in viewer["sites"]:
        for row in site["group_rows"]:
            assert site["groups"][row[0]][0] == row[2]


def test_the_build_writes_the_task_catalog(built):
    path = shared(built["directory"], built["config"], "batch_table", "task_catalog.xlsx")
    table = pd.read_excel(path).set_index("Parameter")
    # Screened out at site A only: still usable over site B.
    assert table.loc[dummy_data.screened_parameter(), "Usable at"] == names.site("b")
    assert set(table.index) <= set(built["batch_table"].columns)


def test_a_ticked_pair_excludes_the_closest_represented_by_the_parameter():
    from test_screening import _proposal_rows

    # As the report's proposal builds it for a ticked parameter W whose closest is Z.
    rows, _ = _proposal_rows()(
        ["Z"], {}, {}, "North", "exclude", correlation.REASON, "same order", "2027-01-01", "A", "W"
    )
    (added,) = screening.add([], pd.DataFrame(rows), EVIDENCE, "", "", SITES)
    assert (added.parameter, added.represented_by, added.reason) == ("Z", "W", correlation.REASON)
    screened = {"Z": {"Decision": "S-0001", "Represented by": "W"}}
    assert report._register_note({"Parameter": "W", "Closest parameter": "Z"}, screened) == (
        "represents Z (its closest)"
    )
    assert report._register_note({"Parameter": "Z"}, screened) == (
        "excluded (S-0001), represented by W"
    )
