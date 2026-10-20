"""The screening register: decisions that leave parameters out of every task.

The register is a log: rows are only added, and the state today is the log
replayed in order. These tests hold the replay, the checks on a proposal, the
review of standing decisions against a new build, and the trail a task run
leaves of the decisions it applied.
"""

from __future__ import annotations

import copy
import json
from datetime import date

import pandas as pd
import pytest

import dummy_data
from conftest import EXPLORATORY, PREDICTIVE, csv, shared
from ml import cli, names, screening, share

SITES = ("North", "South")


def decision(number: int, action: str, parameter: str, site: str, **extra) -> screening.Decision:
    fields = {
        "id": f"S-{number:04d}",
        "date": "2026-01-01",
        "author": "A. Reviewer",
        "action": action,
        "parameter": parameter,
        "site": site,
        "group": "",
        "reason": "retired",
        "rationale": "stopped being recorded",
        "missing": "",
        "pattern": "",
        "last_measured": "",
        "batch_table": "",
        "review_by": "",
    }
    fields.update(extra)
    return screening.Decision(**fields)


def evidence(*rows: tuple) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "Site",
            "Parameter",
            "Missing %",
            "Pattern",
            "Last measured",
            "Group",
            screening.RECENT,
        ],
    )


# ---------------------------------------------------------------------------
# What the register says today
# ---------------------------------------------------------------------------
def test_the_last_decision_on_a_parameter_and_site_stands():
    log = [
        decision(1, "exclude", "X", "all"),
        decision(2, "reinstate", "X", "South"),
        decision(3, "exclude", "Y", "South"),
    ]
    state = screening.current(log, SITES)
    assert state[("X", "North")].action == "exclude"
    assert state[("X", "South")].action == "reinstate"
    assert set(screening.excluded(log, SITES, ("North",))) == {"X"}
    assert set(screening.excluded(log, SITES, ("South",))) == {"Y"}
    # A task over both sites leaves out what either site excludes.
    assert set(screening.excluded(log, SITES)) == {"X", "Y"}


def test_a_register_with_a_bad_row_is_refused_by_row():
    frame = screening.to_frame([decision(1, "drop", "X", "Elsewhere", reason="boring")])
    with pytest.raises(screening.ScreeningError) as caught:
        screening.from_frame(frame, SITES, "register.xlsx")
    for problem in ("Action 'drop'", "Site 'Elsewhere'", "Reason 'boring'"):
        assert problem in str(caught.value)


def test_a_register_edited_outside_the_tool_is_said_to_be(tmp_path, monkeypatch):
    monkeypatch.setenv(share.STANDIN_ENV, str(tmp_path))
    location = share.Location("D", "screening.xlsx")
    path = tmp_path / "D" / "screening.xlsx"
    path.parent.mkdir()
    screening.write([decision(1, "exclude", "X", "North")], path)
    assert screening.read(location, SITES).intact
    frame = pd.read_excel(path, sheet_name=screening.SHEET)
    frame.loc[0, "Rationale"] = "changed afterwards"
    check = pd.read_excel(path, sheet_name=screening.CHECK_SHEET)
    with pd.ExcelWriter(path) as workbook:
        frame.to_excel(workbook, sheet_name=screening.SHEET, index=False)
        check.to_excel(workbook, sheet_name=screening.CHECK_SHEET, index=False)
    assert not screening.read(location, SITES).intact


# ---------------------------------------------------------------------------
# Adding decisions
# ---------------------------------------------------------------------------
EVIDENCE = evidence(
    ("North", "X", 100.0, "Retired", "2025-06-01", "Missing 01", 0.0),
    ("North", "Y", 95.0, "Retired", "2025-06-01", "Missing 01", 0.0),
    ("North", "Z", 0.0, "Complete", "2026-09-01", "", 100.0),
    ("South", "X", 10.0, "Random gaps", "2026-09-01", "", 90.0),
)


def proposal(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame([{c: "" for c in screening.PROPOSAL_COLUMNS} | row for row in rows])


def test_a_group_becomes_one_decision_per_member_with_its_evidence():
    added = screening.add(
        [],
        proposal(
            {
                "Action": "exclude",
                "Group": "Missing 01",
                "Site": "North",
                "Reason": "retired",
                "Rationale": "the old assay",
            }
        ),
        EVIDENCE,
        "abc123",
        "A. Reviewer",
        SITES,
        today=date(2026, 10, 1),
    )
    assert [(d.id, d.parameter) for d in added] == [("S-0001", "X"), ("S-0001", "Y")]
    assert added[0].missing == "100" and added[0].pattern == "Retired"
    assert added[0].group == "Missing 01" and added[0].batch_table == "abc123"
    assert added[0].review_by == "2026-12-31", "a review date by default"


def test_an_all_sites_decision_records_the_evidence_of_each_site():
    (added,) = screening.add(
        [decision(4, "exclude", "Q", "North")],
        proposal(
            {
                "Action": "exclude",
                "Parameter": "X",
                "Site": "all",
                "Reason": "other",
                "Rationale": "why",
            }
        ),
        EVIDENCE,
        "",
        "A. Reviewer",
        SITES,
    )
    assert added.id == "S-0005", "numbers continue from the register"
    assert added.missing == "North 100; South 10"


def test_a_proposal_with_problems_adds_nothing_and_names_them():
    with pytest.raises(screening.ScreeningError) as caught:
        screening.add(
            [],
            proposal(
                {
                    "Action": "exclude",
                    "Parameter": "nope",
                    "Site": "North",
                    "Reason": "retired",
                    "Rationale": "x",
                },
                {
                    "Action": "exclude",
                    "Parameter": "X",
                    "Site": "North",
                    "Reason": "retired",
                    "Rationale": "",
                },
                {
                    "Action": "reinstate",
                    "Parameter": "Z",
                    "Site": "North",
                    "Reason": "other",
                    "Rationale": "x",
                },
                {
                    "Action": "exclude",
                    "Group": "Missing 01",
                    "Site": "all",
                    "Reason": "retired",
                    "Rationale": "x",
                },
            ),
            EVIDENCE,
            "",
            "A. Reviewer",
            SITES,
        )
    message = str(caught.value)
    assert "row 2: 'nope' is not a column" in message
    assert "row 3: Rationale is empty" in message
    assert "row 4: nothing to reinstate" in message
    assert "row 5: a Group is numbered per site" in message
    row = {
        "Action": "exclude",
        "Parameter": "X",
        "Site": "North",
        "Reason": "retired",
        "Rationale": "x",
    }
    with pytest.raises(screening.ScreeningError, match="--author"):
        screening.add([], proposal(row), EVIDENCE, "", " ", SITES)


# ---------------------------------------------------------------------------
# Looking again when new data comes in
# ---------------------------------------------------------------------------
def test_the_latest_build_says_which_exclusions_to_look_at_again():
    log = [
        decision(1, "exclude", "X", "South", missing="10.0", review_by="2030-01-01"),
        decision(2, "exclude", "Z", "North", missing="0.0", review_by="2026-01-31"),
        decision(3, "exclude", "Y", "North", missing="95.0", review_by="2030-01-01"),
    ]
    found = {
        f["Parameter"]: f["Why look again"]
        for f in screening.review(log, EVIDENCE, SITES, today=date(2026, 10, 1))
    }
    assert "measured again: 90%" in found["X"], "retired, and measured again"
    assert "review date 2026-01-31 has passed" in found["Z"]
    assert "Y" not in found, "nothing has changed for it"


# ---------------------------------------------------------------------------
# End to end: the build, the tasks, and `ml screening`
# ---------------------------------------------------------------------------
def test_a_task_leaves_out_what_the_register_excludes_at_its_sites(built):
    screened = dummy_data.screened_parameter()
    decisions = csv(built, PREDICTIVE, "parameters.csv")
    row = decisions[decisions["Parameter"] == screened].iloc[0]
    assert (row["Decision"], row["Stage"]) == ("excluded", "screening")
    assert "S-0001" in row["Reason"]
    # The release task is site B's, where the parameter is not excluded.
    other = csv(built, EXPLORATORY, "parameters.csv")
    assert not (other["Stage"] == "screening").any()


def test_a_run_records_and_keeps_the_register_it_applied(built):
    run = built["packages"][PREDICTIVE]
    record = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert record["screening"]["last_decision"] == "S-0001"
    assert record["screening"]["intact"] is True
    copy_ = pd.read_excel(run / "inputs" / "screening.xlsx", sheet_name=screening.SHEET)
    assert copy_["ID"].tolist() == ["S-0001"]


def test_the_build_writes_the_evidence_and_marks_screened_parameters(built):
    path = shared(built["directory"], built["config"], "batch_table", screening.EVIDENCE_FILE)
    table = pd.read_excel(path, sheet_name=screening.EVIDENCE_SHEET)
    marked = table[table["Screening"].fillna("") != ""]
    assert marked[["Site", "Parameter"]].values.tolist() == [
        [names.site("a"), dummy_data.screened_parameter()]
    ]
    assert marked["Screening"].iloc[0] == "excluded S-0001"
    about = pd.read_excel(path, sheet_name=screening.ABOUT_SHEET)
    assert "Batch table content digest" in set(about["Item"])


def test_adding_a_proposal_and_asking_for_the_status(built, tmp_path):
    config = copy.deepcopy(built["config"])
    config["sharepoint"]["screening"] = {
        "path": "Data/screening_test/parameter_screening.xlsx",
        "drive_id": dummy_data.DRIVE,
    }
    evidence_path = shared(built["directory"], config, "batch_table", screening.EVIDENCE_FILE)
    table = pd.read_excel(evidence_path, sheet_name=screening.EVIDENCE_SHEET)
    group = table[(table["Site"] == names.site("a")) & table["Group"].notna()].iloc[0]["Group"]
    path = tmp_path / "proposal.csv"
    proposal(
        {
            "Action": "exclude",
            "Group": group,
            "Site": names.site("a"),
            "Reason": "missing with group",
            "Rationale": "missing together",
        },
    ).to_csv(path, index=False)
    with pytest.raises(FileNotFoundError, match="--new"):
        cli.run_screening_add(config, path, "A. Reviewer")
    where = cli.run_screening_add(config, path, "A. Reviewer", new=True)
    register = screening.read(where, names.sites())
    assert register.intact and {d.id for d in register.decisions} == {"S-0001"}
    members = set(
        table[(table["Site"] == names.site("a")) & (table["Group"] == group)]["Parameter"]
    )
    assert {d.parameter for d in register.decisions} == members
    assert cli.run_screening_status(config) == []


# ---------------------------------------------------------------------------
# The proposal the missingness report writes
# ---------------------------------------------------------------------------
def _proposal_rows():
    """``proposal_rows`` from the page module, which imports streamlit and so cannot load here."""
    import ast

    from conftest import ROOT

    source = (ROOT / "src" / "ml" / "streamlit_app.py").read_text(encoding="utf-8")
    node = next(n for n in ast.parse(source).body if getattr(n, "name", "") == "proposal_rows")
    space: dict = {}
    exec(compile(ast.Module([node], []), "streamlit_app.py", "exec"), space)
    return space["proposal_rows"]


def test_the_report_writes_a_proposal_that_ml_screening_add_accepts():
    proposal_rows = _proposal_rows()
    group_of = {"X": "Missing 01", "Y": "Missing 01", "Z": ""}
    rows, skipped = proposal_rows(
        ["X", "Y", "Z"],
        group_of,
        {},
        "North",
        "exclude",
        "retired",
        " old assay ",
        "2027-01-01",
        "A. Reviewer",
    )
    assert skipped == []
    assert set(rows[0]) == set(screening.PROPOSAL_COLUMNS), "every column filled in"
    assert rows[0]["Rationale"] == "old assay" and rows[2]["Group"] == ""
    added = screening.add([], pd.DataFrame(rows), EVIDENCE, "", "", SITES)
    # The group's members are one decision; the parameter on its own is another.
    assert [(d.id, d.parameter) for d in added] == [
        ("S-0001", "X"),
        ("S-0001", "Y"),
        ("S-0002", "Z"),
    ]
    assert {d.author for d in added} == {"A. Reviewer"}
    assert added[0].review_by == "2027-01-01"


def test_the_report_only_reinstates_what_is_screened_out():
    proposal_rows = _proposal_rows()
    screened = {"X": {"Decision": "S-0001"}}
    rows, skipped = proposal_rows(
        ["X", "Z"],
        {},
        screened,
        "North",
        "reinstate",
        "other",
        "back in use",
        "2027-01-01",
        "A. Reviewer",
    )
    assert [r["Parameter"] for r in rows] == ["X"] and skipped == ["Z"]
    assert rows[0]["Review by"] == "", "a reinstatement has no review date"
    log = [decision(1, "exclude", "X", "North")]
    (added,) = screening.add(log, pd.DataFrame(rows), EVIDENCE, "", "", SITES)
    assert added.action == "reinstate"


def test_the_missingness_report_shows_what_is_screened_out(built):
    from conftest import shared
    from ml import stlite

    page = shared(built["directory"], built["config"], "batch_table", "missingness_report.html")
    payload = stlite.read_payload(page.read_text(encoding="utf-8"))
    viewer = next(b for b in payload["sections"][0]["blocks"] if b["kind"] == "missing_viewer")
    assert viewer["proposal"]["reasons"] == list(screening.REASONS)
    sites = {site["label"]: site for site in viewer["sites"]}
    first = sites[names.site("a")]
    assert first["screened"][dummy_data.screened_parameter()]["Decision"] == "S-0001"
    assert sites[names.site("b")]["screened"] == {}
    assert "Screened members" in first["group_headers"]


# ---------------------------------------------------------------------------
# A new task file lists what the register excludes
# ---------------------------------------------------------------------------
def test_a_new_task_file_lists_the_screened_parameters_as_exclusions(built, tmp_path):
    from ml import scaffold, taskconfig

    screened, as_of = cli._screened_for_new_task(built["config"])
    assert as_of == "S-0001"
    assert screened == [
        (dummy_data.screened_parameter(), f"screened out by S-0001: irregular ({names.site('a')})")
    ]
    path = scaffold.init(
        tmp_path / "new_task",
        dummy_data.target(),
        config=built["config_path"],
        screened=screened,
        screened_as_of=as_of,
    )
    text = path.read_text(encoding="utf-8")
    assert "Screened out by the screening register (last decision S-0001)" in text
    spec = taskconfig.load(path)
    assert dict(spec.predictors.exclude) == dict(screened)
    # Without a register there is nothing to list, and the template is as it was.
    plain = scaffold.init(tmp_path / "plain", dummy_data.target())
    assert "  exclude: []" in plain.read_text(encoding="utf-8")


def test_a_run_warns_when_a_listed_exclusion_was_reinstated(caplog):
    from ml import taskconfig

    spec = taskconfig.TaskSpec(
        predictors=taskconfig.Predictors(
            exclude=(("X", "screened out by S-0001: retired (North)"), ("Y", "my own reason"))
        )
    )
    cli._check_listed_screening(spec, {})
    assert "still excludes 'X'" in caplog.text
    assert "'Y'" not in caplog.text
