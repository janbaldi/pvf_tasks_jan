"""The correlation report: numerical parameters that rank the batches alike."""

import numpy as np
import pandas as pd

import dummy_data
from conftest import shared
from ml import correlation, names, screening, stlite


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
    assert group.min_rho > 0.99 and group.max_q < 1e-6 and group.min_batches == 89
    assert report.left_out == ["e"] and "e" not in report.ranks
    by_name = {row["Parameter"]: row for row in rows}
    assert by_name["d"]["Group"] == "" and by_name["b"]["Batches with a value"] == 89
    # The strip: the lowest value is level 0, the highest the top level, no value ".".
    assert report.ranks["b"][:31] == "." * 31
    assert {report.ranks["a"][int(np.argmin(x))], report.ranks["a"][int(np.argmax(x))]} == {
        "0",
        np.base_repr(correlation.LEVELS - 1, 36).lower(),
    }


def test_a_correlation_group_can_be_screened_by_its_name():
    evidence = pd.DataFrame(
        {
            "Site": ["North"] * 3,
            "Parameter": ["X", "Y", "Z"],
            "Missing %": [0, 0, 0],
            "Pattern": ["Complete"] * 3,
            "Last measured": [""] * 3,
            "Group": ["", "", ""],
            screening.CORRELATION_GROUP: ["Correlated 01", "Correlated 01", ""],
        }
    )
    proposal = pd.DataFrame(
        [
            {
                "Action": "exclude",
                "Group": "Correlated 01",
                "Site": "North",
                "Reason": correlation.REASON,
                "Rationale": "rank the batches as Z does",
            }
        ]
    )
    added = screening.add([], proposal, evidence, "", "A. Reviewer", ("North", "South"))
    assert [d.parameter for d in added] == ["X", "Y"]


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
        assert set(site["patterns"]) == set(site["rho"]["parameters"])
        assert all(len(bits) == len(site["batches"]) for bits in site["patterns"].values())
        for group, members in site["groups"].items():
            assert len(members) >= 2 and set(members) <= set(site["patterns"])
    # Screened out at site A for missingness, so not compared there.
    first = {site["label"]: site for site in viewer["sites"]}[names.site("a")]
    assert dummy_data.screened_parameter() not in first["patterns"]
