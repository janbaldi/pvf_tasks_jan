"""The shifts report: parameters whose values fall into level groups over time, together."""

import numpy as np
import pandas as pd

from conftest import shared
from dummy_data import c
from ml import names, screening, shifts, stlite


def _lots(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    lot = np.arange(n) // 8
    level = rng.normal(10, 3, size=lot.max() + 1)
    frame = pd.DataFrame(
        {
            "id": range(n),
            "date": pd.date_range("2024-01-01", periods=n),
            "lot": [f"L{k}" for k in lot],
            # Measured per batch: very nearly the lot's value.
            "titre": level[lot] + rng.normal(0, 0.05, n),
            "expiry": pd.Timestamp("2025-01-01") + pd.to_timedelta(lot * 30, "D"),
            "other lot": np.arange(n) // 5,  # its own lots: level groups on other batches
            "trend": np.arange(n) * 0.1,
            "noisy trend": np.arange(n) * 0.1 + rng.normal(size=n),
            "random": rng.choice(list("abc"), n),
            # Follows the lot, but far from one value within it.
            "loosely per lot": level[lot] + rng.normal(0, 1.5, n),
            # A process date: advances with every batch, with a shutdown halfway.
            "process date": pd.date_range("2024-01-01", periods=n)
            + pd.to_timedelta((np.arange(n) >= n // 2) * 60, "D"),
            "screened": np.arange(n) // 8,
        }
    )
    frame.loc[10:30, "titre"] = np.nan
    return frame


def test_parameters_with_the_same_level_groups_group_and_trends_do_not():
    rows, report = shifts.for_build(_lots(), "id", "date", screened_out={"screened": "retired"})
    assert set(report.analysed) == {"lot", "titre", "expiry", "other lot"}
    assert report.left_out == ["screened"] and report.tested == 9
    (group,) = report.groups
    assert set(group.members) == {"lot", "titre", "expiry"}
    assert group.weakest > 0.95 and group.max_q < 1e-20
    # titre has gaps, so it carries less of the others: not the representative.
    assert group.representative == group.members[0] != "titre"
    by_name = {row["Parameter"]: row for row in rows}
    assert by_name["lot"]["Level groups"] == 25 and by_name["other lot"]["Group"] == ""
    assert by_name["expiry"]["Median jump"] == 30 and by_name["lot"]["Median jump"] is None
    assert (
        by_name["titre"]["Median jump"] > shifts.SEPARATION * by_name["titre"]["SD within groups"]
    )
    # The strip: one shade per level group, and each group's first batch and value.
    lot = report.segments["lot"]
    assert lot["starts"][:3] == [0, 8, 16] and lot["values"][:2] == ["L0", "L1"]
    assert len(set(report.levels["lot"][:8])) == 1 != len(set(report.levels["lot"][:9]))
    assert report.segments["expiry"]["values"][1] == "2025-01-31"


def test_a_parameter_excluded_as_shifting_stays_in_the_report():
    _, report = shifts.for_build(_lots(), "id", "date", screened_out={"titre": shifts.REASON})
    assert "titre" in report.analysed and report.left_out == []


def test_the_adjusted_rand_index():
    assert shifts.ari(np.array([0, 0, 1, 1]), np.array([5, 5, 7, 7])) == 1.0
    assert shifts.ari(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])) < 0


def test_a_shift_group_is_represented_by_its_suggested_member():
    evidence = pd.DataFrame(
        {
            "Site": ["North"] * 3,
            "Parameter": ["Lot", "Titre", "Expiry"],
            "Missing %": [0.0] * 3,
            "Pattern": ["Complete"] * 3,
            "Last measured": [""] * 3,
            "Group": [""] * 3,
            screening.SHIFT_GROUP: ["Shifting 01"] * 3,
            screening.SHIFT_SUGGESTED: ["Lot"] * 3,
        }
    )
    proposal = pd.DataFrame(
        [
            {
                "Action": "exclude",
                "Group": "Shifting 01",
                "Site": "North",
                "Reason": shifts.REASON,
                "Rationale": "the lot's certificate values",
            }
        ]
    )
    added = screening.add([], proposal, evidence, "", "A", ("North", "South"))
    assert [(d.parameter, d.represented_by) for d in added] == [("Titre", "Lot"), ("Expiry", "Lot")]


def test_the_build_writes_the_shifts_report_and_finds_the_demo_lots(built):
    page = shared(built["directory"], built["config"], "batch_table", "shifts_report.html")
    payload = stlite.read_payload(page.read_text(encoding="utf-8"))
    viewer = payload["sections"][0]["blocks"][0]
    assert viewer["kind"] == "shift_viewer" and viewer["proposal"]["reason"] == shifts.REASON
    # One per manufacturing site; a proposal names the data source its batches come from.
    assert viewer["site_label"] == "Manufacturing site"
    assert {(s["label"], s["register_site"]) for s in viewer["sites"]} == {
        (f"{names.site(k)} plant", names.site(k)) for k in "ab"
    }
    for site in viewer["sites"]:
        # The medium and cytokine lots change on the same batches at both sites.
        assert {c("media_lot"), c("cytokine_lot")} in [set(g) for g in site["groups"].values()]
        for parameter, levels in site["patterns"].items():
            starts = site["segments"][parameter]["starts"]
            assert len(levels) == len(site["batches"]) and levels[starts[0]] != "."
            assert len(site["segments"][parameter]["values"]) == len(starts)
    path = shared(built["directory"], built["config"], "batch_table", screening.EVIDENCE_FILE)
    evidence = pd.read_excel(path, sheet_name=screening.EVIDENCE_SHEET)
    at_a = evidence[evidence["Site"] == names.site("a")].set_index("Parameter")
    assert at_a.loc[c("media_lot"), screening.SHIFT_GROUP].startswith("Shifting")


def test_the_shifts_run_per_manufacturing_site():
    from types import SimpleNamespace

    from ml import cli

    plant = names.col("manufacturing_site")
    frame = _lots().rename(columns={"id": "Patient Lot/Batch #"})
    north, south = frame.iloc[:100].copy(), frame.iloc[100:].copy()
    north[plant] = ["P1"] * 60 + ["P2"] * 40
    south[plant] = "P2"
    cohorts = {"North": (north, None), "South": (south, None)}
    missing = {
        "North": (None, SimpleNamespace(screened={"lot": {"Reason": "retired"}})),
        "South": (None, SimpleNamespace(screened={})),
    }
    found = cli._shifts_by_plant(cohorts, missing, "date")
    assert set(found) == {"P1", "P2"}
    p1, p2 = found["P1"][1], found["P2"][1]
    # P2 makes batches for both sources: its decisions are for both.
    assert (p1.register_site, p2.register_site) == ("North", screening.ALL)
    assert p2.sources == ["North", "South"] and len(p2.batches) == 140
    # Screened out at North, so left out of every plant North's batches come from.
    assert "lot" in p1.left_out and "lot" in p2.left_out
    assert all(g.name.endswith(" @ P2") for g in p2.groups)
    # Without the column, each data source is one plant.
    found = cli._shifts_by_plant(
        {s: (b.drop(columns=plant), a) for s, (b, a) in cohorts.items()}, missing, "date"
    )
    assert set(found) == {"North", "South"}


def test_evidence_with_a_group_per_plant_still_names_the_members_and_the_suggestion():
    evidence = pd.DataFrame(
        {
            "Site": ["North"] * 3,
            "Parameter": ["Lot", "Titre", "Expiry"],
            "Missing %": [0.0] * 3,
            "Pattern": ["Complete"] * 3,
            "Last measured": [""] * 3,
            "Group": [None] * 3,
            screening.SHIFT_GROUP: ["Shifting 01 @ P1; Shifting 02 @ P2"] * 2
            + ["Shifting 02 @ P2"],
            screening.SHIFT_SUGGESTED: ["Lot; Expiry"] * 2 + ["Expiry"],
        }
    )
    row = {"Action": "exclude", "Site": "North", "Reason": shifts.REASON, "Rationale": "lots"}
    added = screening.add(
        [], pd.DataFrame([row | {"Group": "Shifting 02 @ P2"}]), evidence, "", "A", ("North",)
    )
    assert [(d.parameter, d.represented_by) for d in added] == [
        ("Lot", "Expiry"),
        ("Titre", "Expiry"),
    ]
    added = screening.add(
        [], pd.DataFrame([row | {"Group": "Shifting 01 @ P1"}]), evidence, "", "A", ("North",)
    )
    assert [(d.parameter, d.represented_by) for d in added] == [("Titre", "Lot")]
