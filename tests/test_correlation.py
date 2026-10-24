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
            "e": 2 * x,
            "f": x + 1,
            "g": ["text"] * 120,  # not numerical in the catalog
            "h": [1.0] * 10 + [np.nan] * 110,  # too few values
            "i": [3.0] * 120,  # one value only
        }
    )
    frame.loc[:30, "b"] = np.nan
    types = dict.fromkeys("abcdefhi", "numeric") | {"g": "text"}
    rows, report = correlation.for_build(frame, types, "id", "date")
    (group,) = report.groups
    # Whatever the register says: the analysis never reads it.
    assert sorted(group.members) == ["a", "b", "c", "e", "f"]
    assert report.threshold == correlation.THRESHOLD == 0.9 and group.weakest >= 0.9
    assert group.weakest > 0.99 and group.max_q < 1e-6 and group.min_batches == 89
    assert report.not_analysed == {
        "g": "not numerical in the catalog",
        "h": f"fewer than {correlation.MIN_BATCHES} values",
        "i": "one value only",
    }
    # The representative carries the others where they were measured: not b, with gaps.
    assert group.members[0] == group.representative != "b" and group.score > 0.9
    assert set(group.scores) == set(group.members)
    assert group.scores[group.representative] == group.score == max(group.scores.values())
    by_name = {row["Parameter"]: row for row in rows}
    assert by_name["d"]["Group"] == "" and by_name["b"]["Batches with a value"] == 89
    # The scatterplots: each member's value per batch, None where it has none.
    assert set(report.values) == {"a", "b", "c", "d", "e", "f"}
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


def _retired(parameter: str) -> screening.Decision:
    return screening.Decision(
        *("S-0001", "2026-01-01", "A", "exclude", parameter, "North", "", "retired", "x"),
        *("", "", "", "", ""),
    )


def test_a_representative_must_stay_usable():
    with pytest.raises(screening.ScreeningError, match="'Z' at North represents X .* excluded"):
        screening.add(
            [_retired("Z")],
            _exclude(Parameter="X", **{"Represented by": "Z"}),
            EVIDENCE,
            "",
            "A",
            SITES,
        )
    both = pd.concat(
        [
            _exclude(Parameter="X", **{"Represented by": "Y"}),
            _exclude(Parameter="Y", **{"Represented by": "W"}, Rationale="other"),
        ]
    )
    with pytest.raises(screening.ScreeningError, match="'Y' at North represents X"):
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
    # Excluding the representative later, in the missingness report, is refused ...
    with pytest.raises(screening.ScreeningError, match="'Y' at North represents X"):
        screening.add(
            [decision], _exclude(Parameter="Y", Reason="retired"), EVIDENCE, "", "A", SITES
        )
    # ... but the build's own never-recorded exclusion can make it: a reason to look again.
    later = [decision, replace(_retired("Y"), id="S-0002", reason="never recorded")]
    (finding,) = [f for f in screening.review(later, EVIDENCE, SITES) if f["Parameter"] == "X"]
    assert "its representative 'Y' is screened out too" in finding["Why look again"]
    assert [c["Representative"] for c in screening.conflicts(later, SITES)] == ["Y"]


def test_a_group_moves_to_another_representative_in_one_proposal():
    log = screening.add([], _exclude(Group="Correlated 01"), EVIDENCE, "", "A", SITES)
    assert {d.parameter: d.represented_by for d in log} == {"X": "Y", "W": "Y"}
    # As the report writes it: W comes back, and represents the others.
    moved = pd.concat(
        [
            _exclude(Parameter="W", Action="reinstate"),
            _exclude(Parameter="X", **{"Represented by": "W"}),
            _exclude(Parameter="Y", **{"Represented by": "W"}),
        ]
    )
    log += screening.add(log, moved, EVIDENCE, "", "A", SITES)
    out = screening.excluded(log, SITES)
    assert {p: ds[0].represented_by for p, ds in out.items()} == {"X": "W", "Y": "W"}
    assert screening.conflicts(log, SITES) == []


def test_the_task_catalog_lists_what_one_task_can_use():
    log = screening.add([], _exclude(Group="Correlated 01"), EVIDENCE, "", "A", SITES)
    catalog = pd.DataFrame({"Parameter": ["X", "Y", "W", "Z", "V"], "Value Type": ["numeric"] * 5})
    # The task's own predictor choice kept Y and Z; the catalog keeps its order.
    table = screening.task_catalog(catalog, ["Z", "Y"], log, SITES)
    assert list(table["Parameter"]) == ["Y", "Z"]
    assert list(table.columns) == ["Parameter", "Represents", "Value Type"]
    assert table.set_index("Parameter").loc["Y", "Represents"] == "North: W, X"
    # A task over the other site: what Y stands in for at North is not its business.
    south = screening.task_catalog(catalog, ["Y"], log, SITES, ("South",))
    assert south.loc[0, "Represents"] == ""


def test_the_build_writes_the_correlation_report_over_the_missingness_batches(built):
    page = shared(built["directory"], built["config"], "batch_table", "correlation_report.html")
    payload = stlite.read_payload(page.read_text(encoding="utf-8"))
    viewer = payload["sections"][0]["blocks"][0]
    assert viewer["kind"] == "rank_viewer"
    assert viewer["proposal"]["reasons"] == [correlation.REASON]
    assert viewer["proposal"]["report"] == "correlation"
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
    # Screened out at site A in the missingness report: still compared, shown as such.
    first = {site["label"]: site for site in viewer["sites"]}[names.site("a")]
    assert dummy_data.screened_parameter() in first["values"]
    assert first["elsewhere"][dummy_data.screened_parameter()]["Decision"] == "S-0001"
    assert first["screened"] == {} and first["conflicts"] == []
    assert first["not_analysed"], "the catalog parameters not compared are listed, with why"
    # Each group is drawn from its suggested representative.
    for site in viewer["sites"]:
        assert site["group_headers"][-3:] == [
            "Best usable representative",
            "Excluded here",
            "Excluded elsewhere",
        ]
        for row in site["group_rows"]:
            assert site["groups"][row[0]][0] == row[2]


def test_a_new_task_gets_its_own_task_catalog(built, tmp_path):
    from ml import cli, dataset, io, taskconfig
    from ml import config as workspace

    folder = tmp_path / "expression_inputs"
    target = dummy_data.target()
    config = str(built["config_path"])
    assert cli.main(["--config", config, "task", "new", str(folder), "--target", target]) == 0
    table = pd.read_excel(folder / "task_catalog.xlsx").set_index("Parameter")
    # Not the target, and not what the register excludes at the task's sites.
    assert target not in table.index
    assert dummy_data.screened_parameter() not in table.index
    assert set(table.index) <= set(built["batch_table"].columns)
    # The build no longer writes one: it belongs to a task.
    assert not list((built["directory"] / "sharepoint").rglob("task_catalog.xlsx"))
    # Exactly what a run of the task picks as its predictors.
    spec = taskconfig.load(folder / "task.yaml")

    batch_table = io.load_batch_table(workspace.table_source(built["config"], "batch_table", None))
    catalog = io.read_catalog(workspace.table_source(built["config"], "catalog", None))
    register = cli._register(built["config"])
    assert spec.site == "a", "task new names the config's site"
    site = names.site(spec.site)
    screened = screening.excluded(screening.for_site(register.decisions, site), (site,))
    kept = dataset.candidate_predictors(
        batch_table,
        catalog,
        spec,
        dataset.TaskResult(spec),
        {p: screening.labels(ds) for p, ds in screened.items()},
    )
    assert set(table.index) == set(kept)
    # Written again from the task file as it is now.
    text = (folder / "task.yaml").read_text(encoding="utf-8")
    first = table.index[0]
    (folder / "task.yaml").write_text(
        text.replace(
            "  exclude:",
            f"  exclude:\n    - {{column: {first!r}, reason: not this time}}",
            1,
        ),
        encoding="utf-8",
    )
    assert cli.main(["--config", config, "task", "catalog", str(folder)]) == 0
    again = pd.read_excel(folder / "task_catalog.xlsx")
    assert first not in set(again["Parameter"]) and len(again) == len(table) - 1


def test_a_ticked_pair_excludes_the_closest_represented_by_the_parameter():
    from test_screening import _proposal_rows

    # As the report's proposal builds it for a ticked parameter W whose closest is Z.
    rows, _ = _proposal_rows()(
        ["Z"], {}, {}, "North", "exclude", correlation.REASON, "same order", "A", "W", "correlation"
    )
    (added,) = screening.add([], pd.DataFrame(rows), EVIDENCE, "", "", SITES)
    assert (added.parameter, added.represented_by, added.reason) == ("Z", "W", correlation.REASON)
    assert added.report == "correlation"


def test_the_best_usable_representative_skips_what_the_missingness_report_excludes():
    group = correlation.CorrelationGroup(
        name="Correlated 01",
        members=["A", "B", "C"],
        mean=0.95,
        weakest=0.92,
        min_batches=50,
        max_q=0.0,
        representative="A",
        score=0.9,
        scores={"A": 0.9, "B": 0.8, "C": 0.85},
    )
    assert report._usable_representative(group, {}) == "= suggested"
    assert report._usable_representative(group, {"A": {}}) == "C"
    assert report._usable_representative(group, dict.fromkeys("ABC", {})) == ""


def test_every_pair_in_a_group_reaches_the_threshold():
    # c is 0.96 with a but 0.86 with b: the mean reaches 0.9, the weakest pair does not.
    similarity = np.array([[1, 0.97, 0.96], [0.97, 1, 0.86], [0.96, 0.86, 1]])
    n, q = np.full((3, 3), 50), np.zeros((3, 3))
    names_ = ["a", "b", "c"]
    (average,) = correlation.grouped(similarity, n, q, names_, 0.9, "G", method="average")
    assert len(average.members) == 3 and average.weakest == 0.86
    (complete,) = correlation.grouped(similarity, n, q, names_, 0.9, "G", method="complete")
    assert len(complete.members) == 2 and complete.weakest >= 0.9
