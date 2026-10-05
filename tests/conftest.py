"""One run of every stage, over made-up sources, shared by every test.

The real inputs are on a corporate share, so this is the only place the pipeline
runs end to end. It runs once per session: the stages are written to be run in
this order, and running them separately would not test that.

Two task files are run against the one PVF, because the claim worth testing is
that a different question needs a different YAML and nothing else.

Most questions are asked of the packages and the reports rather than of the
frames, because those are what anyone actually reads. The report's numbers come
out of the payload the run wrote, not out of the markup — the markup is
Streamlit's business and changes with its version.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dummy_data  # noqa: E402
from pvf import cli, stlite, taskconfig  # noqa: E402

#: The two example tasks, by the name each of them carries.
PREDICTIVE = "ghent_car_expression"
EXPLORATORY = "raritan_release"


@pytest.fixture(scope="session")
def built(tmp_path_factory) -> dict:
    """Everything the stages produced, for the tests to pick over."""
    directory = tmp_path_factory.mktemp("demo")
    config_path = dummy_data.write_config(directory)
    config = cli.load_config(config_path)

    reports = {
        "ptf": cli.run_ptf(config, config_path=config_path),
        "build": cli.run_build(config, mode="dev", config_path=config_path),
    }
    packages = {}
    for name in dummy_data.TASK_FILES:
        spec = taskconfig.load(directory / name)
        packages[spec.name] = cli.run_task(spec)

    return {
        "directory": directory,
        "config": config,
        "config_path": config_path,
        "pvf": pd.read_excel(config["paths"]["pvf"]),
        "new_parameters": pd.read_csv(config["paths"]["new_parameters"]),
        "reports": reports,
        "packages": packages,
        "payloads": {
            stage: stlite.read_payload(path.read_text(encoding="utf-8"))
            for stage, path in reports.items()
        },
    }


# ---------------------------------------------------------------------------
# Reading a package back
# ---------------------------------------------------------------------------
def artifact(built: dict, task: str, name: str) -> Path:
    """A file in a task folder, by its bare name; the layout puts it in its subfolder."""
    return built["packages"][task] / cli.TASK_LAYOUT.get(name, name)


def csv(built: dict, task: str, name: str) -> pd.DataFrame:
    return pd.read_csv(artifact(built, task, name))


def manifest(built: dict, task: str) -> dict:
    return json.loads(artifact(built, task, "manifest.json").read_text(encoding="utf-8"))


def payload(built: dict, task: str) -> dict:
    """The report's payload — the same object the page renders from."""
    return json.loads(artifact(built, task, "report_payload.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Reading a report back
# ---------------------------------------------------------------------------
def tables(payload: dict):
    """Every table in a report, whatever it is nested behind."""

    def walk(blocks):
        for block in blocks:
            if block["kind"] == "table":
                yield block
            elif block["kind"] in ("collapsed", "deferred"):
                yield from walk(block["blocks"])
            elif block["kind"] == "chooser":
                for option in block["options"]:
                    yield from walk(option["blocks"])

    for section_ in payload["sections"]:
        yield from walk(section_["blocks"])


def figures(payload: dict) -> list[dict]:
    """Every figure in a report, however deeply nested."""

    def walk(blocks):
        for block in blocks:
            if block["kind"] == "figure":
                yield block
            elif block["kind"] in ("collapsed", "deferred"):
                yield from walk(block["blocks"])
            elif block["kind"] == "chooser":
                for option in block["options"]:
                    yield from walk(option["blocks"])

    return [figure for section_ in payload["sections"] for figure in walk(section_["blocks"])]


def downloads(payload: dict) -> dict[str, dict]:
    """Every download button in a report, by the file name it offers."""

    found: dict[str, dict] = {}

    def walk(blocks):
        for block in blocks:
            if block["kind"] == "download":
                found[block["filename"]] = block
            elif block["kind"] in ("collapsed", "deferred"):
                walk(block["blocks"])
            elif block["kind"] == "chooser":
                for option in block["options"]:
                    walk(option["blocks"])

    for section_ in payload["sections"]:
        walk(section_["blocks"])
    return found


def rows_naming(payload: dict, value: str) -> list[list]:
    """Every report row that mentions this value in one of its cells."""
    return [
        row
        for block in tables(payload)
        for row in block["rows"]
        if any(value in str(cell) for cell in row)
    ]


def section(payload: dict, section_id: str) -> dict:
    return next(s for s in payload["sections"] if s["id"] == section_id)


def first_table(payload: dict, section_id: str) -> dict:
    return next(b for b in section(payload, section_id)["blocks"] if b["kind"] == "table")


def table_with(payload: dict, header: str) -> dict:
    """The first table that has this column."""
    return next(block for block in tables(payload) if header in block["headers"])


def facts(block: dict) -> dict[str, str]:
    """A two-column key/value table as a dict, with the bold markers taken off."""
    return {row[0].strip("*"): row[1] for row in block["rows"]}


def row_named(block: dict, first_cell: str) -> list:
    """The row of a table whose first cell is this."""
    return next(row for row in block["rows"] if row[0] == first_cell)


def ghent(pvf: pd.DataFrame) -> pd.DataFrame:
    return pvf[pvf["Site Merged"] == "Ghent"]


def raritan(pvf: pd.DataFrame) -> pd.DataFrame:
    return pvf[pvf["Site Merged"] == "Raritan"]
