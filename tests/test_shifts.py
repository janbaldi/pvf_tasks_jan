"""The shifts report: parameters that hold a value for runs of batches, shifting together."""

import numpy as np
import pandas as pd

from conftest import shared
from dummy_data import c
from ml import names, screening, shifts, stlite


def _lots(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    lot = np.arange(n) // 8
    frame = pd.DataFrame(
        {
            "id": range(n),
            "date": pd.date_range("2024-01-01", periods=n),
            "lot": [f"L{k}" for k in lot],
            "titre": (lot * 1.7 % 5).round(2),  # the lot's certificate value
            "expiry": pd.Timestamp("2025-01-01") + pd.to_timedelta(lot * 30, "D"),
            "other lot": np.arange(n) // 5,  # its own lots: shifts on other batches
            "trend": np.arange(n) * 0.1 + rng.normal(size=n),
            "random": rng.choice(list("abc"), n),
            "screened": np.arange(n) // 8,
        }
    )
    frame.loc[10:30, "titre"] = np.nan
    return frame


def test_parameters_that_shift_on_the_same_batches_group_and_trends_do_not():
    rows, report = shifts.for_build(_lots(), "id", "date", screened_out={"screened": "retired"})
    assert set(report.analysed) == {"lot", "titre", "expiry", "other lot"}
    assert report.left_out == ["screened"] and report.tested == 6
    (group,) = report.groups
    assert set(group.members) == {"lot", "titre", "expiry"}
    assert group.weakest == 1.0 and group.max_q < 1e-20
    # titre has gaps, so it carries less of the others: not the representative.
    assert group.representative == group.members[0] != "titre"
    by_name = {row["Parameter"]: row for row in rows}
    assert by_name["lot"]["Runs"] == 25 and by_name["other lot"]["Group"] == ""
    # The strip: runs flip colour, and each run's value is there for the hover.
    assert report.runs["lot"][:17] == "0" * 8 + "1" * 8 + "0"
    assert report.values["titre"][:3] == ["0", "1.7", "0.1"]
    assert report.values["expiry"][1] == "2025-01-31"


def test_a_parameter_excluded_as_shifting_stays_in_the_report():
    _, report = shifts.for_build(_lots(), "id", "date", screened_out={"titre": shifts.REASON})
    assert "titre" in report.analysed and report.left_out == []


def test_the_runs_test_is_wald_wolfowitz_for_two_values():
    codes = np.array([0, 0, 1, 1, 1, 0, 1, 0, 0, 1])
    runs, z = shifts.runs_test(codes)
    m, k, n = 5, 5, 10
    expected, variance = 1 + 2 * m * k / n, 2 * m * k * (2 * m * k - n) / (n**2 * (n - 1))
    assert runs == 6 and np.isclose(z, (runs - expected) / np.sqrt(variance))


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
    for site in viewer["sites"]:
        # The medium and cytokine lots change on the same batches at both sites.
        assert {c("media_lot"), c("cytokine_lot")} in [set(g) for g in site["groups"].values()]
        for parameter, bits in site["patterns"].items():
            flips = sum(a != b for a, b in zip(bits.replace(".", ""), bits.replace(".", "")[1:]))
            assert len(site["values"][parameter]) == flips + 1
    path = shared(built["directory"], built["config"], "batch_table", screening.EVIDENCE_FILE)
    evidence = pd.read_excel(path, sheet_name=screening.EVIDENCE_SHEET)
    at_a = evidence[evidence["Site"] == names.site("a")].set_index("Parameter")
    assert at_a.loc[c("media_lot"), screening.SHIFT_GROUP].startswith("Shifting")
