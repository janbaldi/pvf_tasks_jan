"""Config, task workspaces, manifests, and the fixes the review asked for.

``config.yaml`` names the one catalog and the one batch table and holds every SharePoint
location; each task is a workspace whose runs carry the inputs they read; every
manifest has one format; a task can leave out whole catalog categories; a column constant
to measurement precision never reaches a model; and a recipe can be read back
and applied.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import dummy_data
from conftest import EXPLORATORY, PREDICTIVE, ROOT, artifact, csv, on_share, shared
from dummy_data import c, v
from ml import cli, corrections, dataset, encode, manifest, names, recipe, scaffold, taskconfig
from ml import config as workspace

CONFIG_TEXT = (ROOT / "config.yaml").read_text(encoding="utf-8")

#: Made-up SharePoint locations for the shipped config's ${NAME} placeholders.
SHAREPOINT_ENV = {
    "SHAREPOINT_DRIVE_A": "DRIVE_FOR_A",
    "SHAREPOINT_DRIVE_B": "DRIVE_FOR_B",
    "SHAREPOINT_VOCABULARY": "Team%20Site/Data/private/vocabulary.xlsx",
    "SHAREPOINT_CATALOG": "Team%20Site/Data/catalog/catalog.xlsx",
    "SHAREPOINT_BATCH_TABLE": "Team%20Site/Data/batch_data/batch_data.xlsx",
    "SHAREPOINT_PHF_SITE_A": "Batch data/site_a.xlsm",
    "SHAREPOINT_PHF_SITE_B": "General/Batch Data/site_b.xlsx",
    "SHAREPOINT_PARAM_MAPPING": "Team%20Site/mapping.xlsx",
    "SHAREPOINT_REMFG": "General/remfg.xlsx",
    "SHAREPOINT_LV_COA": "Team%20Site/Data/coa.xlsx",
    "SHAREPOINT_RAW_MATERIALS": "Team%20Site/Data/consumables.csv",
    "SHAREPOINT_INVESTIGATIONS": "Investigations/power_query.xlsx",
    "SHAREPOINT_SCREENING": "Team%20Site/Data/catalog/parameter_screening.xlsx",
    "SHAREPOINT_SITE_MAPS_FOLDER": "Team%20Site/Data/mappings/",
    "SHAREPOINT_SITE_MAPS_FILES": "first.xlsx, second.xlsx,third.xlsx",
}


@pytest.fixture(autouse=True)
def sharepoint_env(monkeypatch):
    """What a real .env holds: where each input is on SharePoint."""
    for name, value in SHAREPOINT_ENV.items():
        monkeypatch.setenv(name, value)


# ---------------------------------------------------------------------------
# Config: found, validated, and the only place a SharePoint address lives
# ---------------------------------------------------------------------------
def test_the_package_holds_no_sharepoint_address():
    source = Path(cli.__file__).parent
    code = "\n".join(path.read_text(encoding="utf-8") for path in source.glob("*.py"))
    assert "DRIVE_ID_" not in code and "%20" not in code
    # Every SharePoint location in config.yaml is a ${NAME} filled in from .env.
    sharepoint = yaml.safe_load(CONFIG_TEXT)["sharepoint"]
    values = [entry["path"] for label, entry in sharepoint.items() if label != "site_maps"]
    values += [entry["drive_id"] for entry in sharepoint.values()]
    values += [sharepoint["site_maps"]["folder"], sharepoint["site_maps"]["files"]]
    assert all(re.fullmatch(r"\$\{SHAREPOINT_[A-Z_]+\}", value) for value in values), values
    # .env.example names every variable and gives none of their values.
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    for name in SHAREPOINT_ENV:
        assert f"${{{name}}}" in CONFIG_TEXT and f'{name}=""' in example
    assert "DATABRICKS_AO_MAPPING=''" in example


def test_a_sharepoint_address_missing_from_the_environment_is_named(tmp_path, monkeypatch):
    monkeypatch.delenv("SHAREPOINT_CATALOG")
    monkeypatch.delenv("SHAREPOINT_REMFG")
    (tmp_path / "config.yaml").write_text(CONFIG_TEXT, encoding="utf-8")
    with pytest.raises(workspace.ConfigError) as raised:
        workspace.load(tmp_path / "config.yaml")
    assert "sharepoint.catalog.path: ${SHAREPOINT_CATALOG} is not set" in str(raised.value)
    assert "${SHAREPOINT_REMFG}" in str(raised.value) and ".env" in str(raised.value)


def test_sharepoint_addresses_are_read_from_the_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("SHAREPOINT_CATALOG")
    (tmp_path / "config.yaml").write_text(CONFIG_TEXT, encoding="utf-8")
    (tmp_path / ".env").write_text('SHAREPOINT_CATALOG="Elsewhere/catalog.xlsx"\n')
    config = workspace.load(tmp_path / "config.yaml")
    assert workspace.location(config, "catalog").path == "Elsewhere/catalog.xlsx"
    monkeypatch.delenv("SHAREPOINT_CATALOG")


def test_the_shipped_config_is_a_valid_config():
    config = workspace.load(ROOT / "config.yaml")
    assert workspace.location(config, "batch_table").name == "batch_data.xlsx"
    # Each site's outputs go into a folder named after it.
    site = names.site(config["site"])
    assert workspace.catalog_folder(config).path.endswith(f"/Data/catalog/{site}/")
    assert workspace.batch_table_folder(config).path.endswith(f"/Data/batch_data/{site}/")
    assert workspace.location(config, "catalog").path.endswith("/Data/catalog/catalog.xlsx")
    # It names columns by key only, and the keys exist.
    assert cli._unknown_keys(config) == []
    config["build"] = {"date_column": "no_such_key"}
    assert cli._unknown_keys(config) == [
        "build.date_column: 'no_such_key' is not a column key of the vocabulary (see ml.names)"
    ]
    config["build"] = {
        "missingness_cohort": {"column": "no_col", "value": "no_value", "all_batches_at": ["z"]}
    }
    problems = cli._unknown_keys(config)
    for setting in ("cohort.column", "cohort.value", "cohort.all_batches_at"):
        assert any(setting in p for p in problems), setting
    config["build"] = {
        "missingness_cohort": {"column": "no_col", "value": "no_value", "all_batches_at": ["z"]}
    }
    problems = cli._unknown_keys(config)
    for setting in ("cohort.column", "cohort.value", "cohort.all_batches_at"):
        assert any(setting in p for p in problems), setting


def test_a_key_the_vocabulary_workbook_adds_can_be_named_in_config():
    """The code's registry lists the keys it uses; a workbook may define more."""
    config = workspace.load(ROOT / "config.yaml")
    config["build"] = {"date_column": "release_date"}
    demo = names.demo()
    try:
        names.use(
            names.Vocabulary(
                {**demo.columns, "release_date": "Released on"},
                demo.values,
                demo.sites,
                demo.sheets,
                "a workbook",
            )
        )
        assert cli._unknown_keys(config) == []
        assert names.col("release_date") == "Released on"
    finally:
        names.use(demo)


def _fake_io_sharepoint(monkeypatch) -> list:
    """io_sharepoint as a module that records what it is asked, and the real share in use."""
    import sys
    import types

    from ml import share

    calls: list = []
    module = types.ModuleType("io_sharepoint")
    module.load_excel_from_sharepoint = lambda **kw: calls.append(("read", kw)) or pd.DataFrame()
    module.load_csv_from_sharepoint = lambda **kw: calls.append(("read csv", kw)) or pd.DataFrame()
    module.upload_file_to_sharepoint = lambda **kw: calls.append(("upload", kw))
    module.write_text_to_sharepoint = lambda **kw: calls.append(("write text", kw))
    module.download_file_from_sharepoint = lambda **kw: calls.append(("download", kw)) or b"{}"
    monkeypatch.setitem(sys.modules, "io_sharepoint", module)
    monkeypatch.delenv(share.STANDIN_ENV, raising=False)
    return calls


def test_sharepoint_locations_come_from_the_config(tmp_path, monkeypatch):
    calls = _fake_io_sharepoint(monkeypatch)
    config = workspace.load(ROOT / "config.yaml")
    workspace.source(config, "phf_site_b").read()
    assert calls == [
        (
            "read",
            {
                "sharepoint_path": "General/Batch Data/site_b.xlsx",
                "sheet_name": names.sheet("phf_site_b"),
                "drive_id": "DRIVE_FOR_B",
            },
        )
    ]
    maps = workspace.site_map_sources(config)
    assert [m.location.name for m in maps] == ["first.xlsx", "second.xlsx", "third.xlsx"]
    # The certificate workbook keeps its two rows of furniture above the header.
    assert workspace.source(config, "lv_coa").header == 2


def test_outputs_are_uploaded_into_the_folder_of_their_stage(tmp_path, monkeypatch):
    from ml import share

    calls = _fake_io_sharepoint(monkeypatch)
    config = workspace.load(ROOT / "config.yaml")
    report = tmp_path / "build_report.html"
    report.write_text("<html></html>")
    [(what, where)] = share.upload({"report": report}, workspace.batch_table_folder(config))
    assert what == "report" and where.name == "build_report.html"
    assert calls[0][1]["sharepoint_target_path"].endswith(f"/Data/batch_data/{names.site('a')}/")
    assert calls[0][1]["drive_id"] == "DRIVE_FOR_A"
    assert f"Team Site/Data/batch_data/{names.site('a')}/" in where.where, (
        "shown decoded, as a person reads it"
    )


def test_a_csv_source_is_read_as_csv(tmp_path, monkeypatch):
    calls = _fake_io_sharepoint(monkeypatch)
    config = workspace.load(ROOT / "config.yaml")
    workspace.source(config, "raw_materials").read()
    [(kind, kwargs)] = calls
    assert kind == "read csv", "the consumables export is a CSV file, not a workbook"
    assert kwargs["sharepoint_path"] == "Team%20Site/Data/consumables.csv"
    assert kwargs["drive_id"] == "DRIVE_FOR_A" and kwargs["header"] == 0
    assert kwargs["sep"] is None, "the delimiter is detected when config.yaml gives none"
    assert kwargs["encoding"] == "utf-8-sig"

    config["sharepoint"]["raw_materials"]["sep"] = ";"
    calls.clear()
    workspace.source(config, "raw_materials").read()
    assert calls[0][1]["sep"] == ";"


def test_a_csv_delimiter_must_be_text(tmp_path):
    text = CONFIG_TEXT.replace(
        'raw_materials:  {path: "${SHAREPOINT_RAW_MATERIALS}",',
        'raw_materials:  {sep: 3, path: "${SHAREPOINT_RAW_MATERIALS}",',
    )
    (tmp_path / "config.yaml").write_text(text)
    with pytest.raises(workspace.ConfigError, match=r"sharepoint\.raw_materials\.sep"):
        workspace.load(tmp_path / "config.yaml")


def test_a_manifest_is_written_and_read_back_through_io_sharepoint(tmp_path, monkeypatch):
    from ml import share

    calls = _fake_io_sharepoint(monkeypatch)
    config = workspace.load(ROOT / "config.yaml")
    target = workspace.batch_table_folder(config).file("batch_table_manifest.json")
    assert share.write_text("the manifest", "{}", target) == target
    assert calls[0] == (
        "write text",
        {
            "text": "{}",
            "sharepoint_target_folder_path": f"Team%20Site/Data/batch_data/{names.site('a')}/",
            "file_name": "batch_table_manifest.json",
            "drive_id": "DRIVE_FOR_A",
        },
    )
    assert share.read_text(target) == "{}"
    assert calls[1] == (
        "download",
        {
            "sharepoint_path": (
                f"Team%20Site/Data/batch_data/{names.site('a')}/batch_table_manifest.json"
            ),
            "drive_id": "DRIVE_FOR_A",
        },
    )


def test_without_io_sharepoint_a_run_stops_and_says_why(monkeypatch):
    import sys

    from ml import share

    monkeypatch.delenv(share.STANDIN_ENV, raising=False)
    monkeypatch.setitem(sys.modules, "io_sharepoint", None)
    with pytest.raises(share.ShareError, match="io_sharepoint is not installed"):
        share.backend()


def test_the_env_file_beside_the_config_is_loaded_with_it(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG_TEXT)
    (tmp_path / ".env").write_text("ML_TEST_SECRET=from-file\nMODE=DEV\n")
    monkeypatch.delenv("ML_TEST_SECRET", raising=False)
    monkeypatch.setenv("MODE", "PRD")  # the shell wins over the file
    config = workspace.load(config_path)
    import os

    assert os.environ["ML_TEST_SECRET"] == "from-file"
    assert os.environ["MODE"] == "PRD"
    assert config["__env__"].endswith(".env")
    assert "__env__" not in workspace.settings(config), "never part of the config digest"
    monkeypatch.delenv("ML_TEST_SECRET")


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"sources": {"location": "local"}}, "sources.location: no longer a setting"),
        ({"paths": {"catalog": "x.xlsx"}}, "paths: no longer a setting"),
        ({"upload": {"enabled": True}}, "upload: no longer a setting"),
        ({"sharepoint": {"catalog": {"sheet": "x"}}}, "sharepoint.catalog.path"),
        ({"cleaning": {"censord": "half"}}, "cleaning.censord"),
        ({"features": {"co2": 5}}, "features.co2"),
        ({"build": {"missingness_cohort": {"column": "batch_type"}}}, "missingness_cohort.value"),
        ({"build": {"missingness_cohort": {"value": "x", "column": "y", "on": 1}}}, "cohort.on"),
        ({"build": {"missingness_cohort": {"column": "batch_type"}}}, "missingness_cohort.value"),
        ({"build": {"missingness_cohort": {"value": "x", "column": "y", "on": 1}}}, "cohort.on"),
        ({"tasks": {"target": "x"}}, "unknown setting"),
        ({"build": {"correlation_threshold": 1.5}}, "build.correlation_threshold"),
    ],
)
def test_a_config_that_cannot_be_used_is_refused_by_field(tmp_path, change, field):
    config = yaml.safe_load(CONFIG_TEXT)
    config.update(change)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(workspace.ConfigError, match=field.replace(".", r"\.")):
        workspace.load(path)


def test_the_config_is_found_above_the_working_directory(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG_TEXT)
    deep = tmp_path / "tasks" / "drafts"
    deep.mkdir(parents=True)
    monkeypatch.delenv(workspace.CONFIG_ENV, raising=False)
    monkeypatch.chdir(deep)
    assert workspace.find() == config_path
    monkeypatch.setenv(workspace.CONFIG_ENV, "/somewhere/config.yaml")
    assert workspace.find() == Path("/somewhere/config.yaml")
    assert workspace.find("explicit.yaml") == Path("explicit.yaml")


# ---------------------------------------------------------------------------
# A task is a workspace of its own
# ---------------------------------------------------------------------------
def test_a_new_task_is_a_workspace_that_reads_the_configured_batch_table(built, tmp_path):
    folder = tmp_path / "analyses" / "my_question"
    path = scaffold.init(folder, c("fp_car_pct"), config=built["config_path"])
    assert path == folder.resolve() / "task.yaml"
    spec = taskconfig.load(folder)
    assert spec.name == "my_question"
    assert spec.batch_table is None, "read from SharePoint, where its config says"
    assert taskconfig.config_file(folder) == built["config_path"]
    assert spec.output_root == folder.resolve() / "runs"


def test_a_task_without_a_config_says_what_to_do(tmp_path):
    from ml import config as workspace

    with pytest.raises(workspace.ConfigError, match="no config.yaml says where"):
        workspace.table_source(None, "batch_table", None)


def test_the_command_line_checks_and_reports_errors_plainly(built, capsys):
    config = built["config"]["__path__"]
    task = str(built["directory"] / dummy_data.TASK_FILES[0])
    assert cli.main(["--config", config, "validate", "--task", task]) == 0
    assert cli.main(["--config", config, "task", "run", "--task", "no_such_task"]) == 2
    assert "does not exist" in capsys.readouterr().err


def test_init_links_the_task_to_the_config_it_was_made_with(built, tmp_path):
    folder = tmp_path / "q3"
    assert cli.main(["--config", str(built["config_path"]), "task", "new", str(folder)]) == 0
    text = (folder / "task.yaml").read_text(encoding="utf-8")
    assert "\nconfig: " in text and "# config:" not in text
    assert taskconfig.config_file(folder) == built["config_path"]


def test_inside_a_workspace_no_task_needs_naming(built, monkeypatch, caplog):
    monkeypatch.delenv(workspace.CONFIG_ENV, raising=False)
    monkeypatch.chdir(built["directory"] / dummy_data.TASK_FILES[0])
    assert cli.main(["validate"]) == 0
    assert "Task 'expression' is valid" in caplog.text


def test_a_run_carries_its_inputs_and_can_be_redone_from_its_folder_alone(built):
    run = built["packages"][PREDICTIVE]
    copied = pd.read_excel(run / "inputs" / "batch_table.xlsx")
    assert copied.shape == built["batch_table"].shape
    assert (run / "inputs" / "catalog.xlsx").is_file()
    settings = yaml.safe_load((run / "config" / "task.yaml").read_text(encoding="utf-8"))
    assert settings["inputs"] == {
        "batch_table": "../inputs/batch_table.xlsx",
        "catalog": "../inputs/catalog.xlsx",
        "screening": "../inputs/screening.xlsx",
    }

    again = cli.rerun(run)
    assert again.parent == run.parent, "the rerun is a new run of the same workspace"
    assert (again / "data" / "dataset.csv").read_bytes() == (
        run / "data" / "dataset.csv"
    ).read_bytes()
    record = json.loads((again / "manifest.json").read_text(encoding="utf-8"))
    stale = ("manifest", "catalog changed")
    assert not [w for w in record["warnings"] if any(word in w for word in stale)]


# ---------------------------------------------------------------------------
# Stale inputs are said to be stale
# ---------------------------------------------------------------------------
def test_a_build_says_when_the_catalog_check_found_new_parameters(built):
    record = manifest.read(
        shared(built["directory"], built["config"], "batch_table", "batch_table_manifest.json")
    )
    assert any("`ml check-catalog` found" in warning for warning in record["warnings"])


def test_a_task_says_when_the_catalog_changed_after_the_build(tmp_path):
    from ml import package, provenance

    batch_table = pd.DataFrame({"x": [1]})
    catalog = pd.DataFrame({"Parameter": ["x"], "Value Type": ["a newer catalog"]})
    built_with = {"label": "catalog", "digest": "the catalog the build read"}
    upstream = {
        "catalog": (None, "nowhere"),
        "batch_table": (json.dumps({"batch_table": {}, "inputs": [built_with]}), "somewhere"),
    }
    pkg = package.open_package(tmp_path / "runs", "t")
    sources = [
        provenance.remote("batch_table", "SharePoint D: b.xlsx", batch_table),
        provenance.remote("catalog", "SharePoint D: c.xlsx", catalog),
    ]
    cli._carry_upstream(pkg, upstream, sources, None)
    assert any("catalog changed after the batch table was built" in w for w in pkg.warnings)


# ---------------------------------------------------------------------------
# Columns left out by their catalog category
# ---------------------------------------------------------------------------
def test_a_catalog_category_can_be_left_out_by_name(built):
    catalog = pd.read_excel(shared(built["directory"], built["config"], "catalog"))
    late = set(catalog.loc[catalog["Available At"] == "Post-release", "Parameter"])
    decisions = csv(built, EXPLORATORY, "parameters.csv")
    excluded = decisions[decisions["Decision"] == "excluded"].set_index("Parameter")
    present = late & set(built["batch_table"].columns)
    assert present <= set(excluded.index), "every one is out, some already for another reason"
    left_out = decisions[decisions["Stage"] == "category exclusion"]
    assert len(left_out) and set(left_out["Parameter"]) <= present
    assert (left_out["Reason"] == "Catalog 'Available At' is 'Post-release'").all()
    columns = csv(built, EXPLORATORY, "dataset_columns.csv")
    assert not set(columns.loc[columns["Role"] == "predictor", "Source"]) & late


def test_include_by_catalog_keeps_only_those_categories():
    spec = taskconfig.parse(
        {
            "task": {"name": "t"},
            "target": {"column": "y"},
            "predictors": {"include_by_catalog": {"Category": ["Process"]}},
        },
        Path("."),
    )
    catalog = pd.DataFrame(
        {
            "Parameter": ["kept", "qc", "blank"],
            "Value Type": ["numeric"] * 3,
            "Category": ["Process", "QC", None],
        }
    )
    assert dataset._category_exclusions(catalog, spec) == {
        "qc": "Catalog 'Category' is 'QC', not an included category",
        "blank": "Catalog 'Category' is empty, not an included category",
    }


def test_a_category_column_the_catalog_does_not_have_is_refused():
    spec = taskconfig.parse(
        {
            "task": {"name": "t"},
            "target": {"column": "y"},
            "predictors": {"exclude_by_catalog": {"Categroy": ["QC"]}},
        },
        Path("."),
    )
    catalog = pd.DataFrame({"Parameter": ["x"], "Value Type": ["numeric"], "Category": ["QC"]})
    with pytest.raises(dataset.TaskRefused, match="no 'Categroy' column"):
        dataset._category_exclusions(catalog, spec)


# ---------------------------------------------------------------------------
# Corrections are data
# ---------------------------------------------------------------------------
def test_a_correction_rule_can_be_switched_off(tmp_path):
    frame = {
        names.site("b"): pd.DataFrame({c("batch_type"): [v("type_clinical"), None]}),
        names.site("a"): pd.DataFrame(),
    }
    rules = dummy_data.demo_corrections()
    changes = corrections.apply(frame, rules, corrections.INTEGRITY)
    assert set(frame[names.site("b")][c("batch_type")]) == {v("type_commercial")}
    assert next(ch for ch in changes if ch["column"] == c("batch_type"))["rows"] == 2

    frame = {
        names.site("b"): pd.DataFrame({c("batch_type"): [v("type_clinical")]}),
        names.site("a"): pd.DataFrame(),
    }
    corrections.apply(frame, rules, corrections.INTEGRITY, disabled={"site_b_batch_type"})
    assert frame[names.site("b")][c("batch_type")].tolist() == [v("type_clinical")]


def test_a_broken_corrections_table_names_the_rule():
    table = pd.DataFrame(
        [
            {
                "Name": "scale",
                "Site": names.site("a"),
                "Column": "x",
                "Action": "divide_by",
                "New": 0,
            },
            {"Name": "odd", "Site": names.site("a"), "Column": "y", "Action": "rename"},
        ]
    )
    with pytest.raises(corrections.CorrectionsError) as raised:
        corrections.from_frame(table, "the sheet")
    assert "the sheet row 2: divide_by needs a non-zero number" in str(raised.value)
    assert "row 3: Action 'rename' is not one of" in str(raised.value)


def test_corrections_survive_the_round_trip_through_their_sheet(tmp_path):
    rules = dummy_data.demo_corrections()
    path = tmp_path / "vocabulary.xlsx"
    corrections.to_frame(rules).to_excel(path, index=False)
    assert corrections.from_frame(pd.read_excel(path)) == rules


# ---------------------------------------------------------------------------
# One manifest format
# ---------------------------------------------------------------------------
def test_every_manifest_follows_the_one_format(built):
    paths = [
        on_share(
            built["directory"],
            workspace.catalog_folder(built["config"]).file("catalog_manifest.json"),
        ),
        shared(built["directory"], built["config"], "batch_table", "batch_table_manifest.json"),
        *(artifact(built, task, "manifest.json") for task in (PREDICTIVE, EXPLORATORY)),
    ]
    stages = []
    for path in paths:
        assert manifest.verify(path) == [], path
        record = manifest.read(path)
        assert list(record)[: len(manifest.ENVELOPE) - 1] == list(manifest.ENVELOPE[:-1])
        stages.append(record["stage"])
    assert stages == ["catalog", "batch_table", "task", "task"]


def test_a_manifest_names_the_files_it_describes_relative_to_itself(built):
    record = manifest.read(
        shared(built["directory"], built["config"], "batch_table", "batch_table_manifest.json")
    )
    written = {entry["what"]: entry["path"] for entry in record["outputs"]}
    assert (
        written["batch table"] == "batch_data.xlsx"
        and written["batch table (Parquet)"] == "batch_data.parquet"
    )
    assert record["batch_table"]["content_sha256"]
    labels = {entry["label"] for entry in record["inputs"]}
    assert labels >= {"catalog", "phf_site_a"} and not labels & {"phf_site_b", "site_map_1"}
    # Site B's build reads its own records, and what only it needs.
    record_b = manifest.read(
        shared(built["directory"], built["config_b"], "batch_table", "batch_table_manifest.json")
    )
    labels_b = {entry["label"] for entry in record_b["inputs"]}
    assert labels_b >= {"phf_site_b", "param_mapping", "remfg", "site_map_1"}
    assert "phf_site_a" not in labels_b


def test_a_changed_output_is_caught(built, tmp_path):
    source = artifact(built, PREDICTIVE, "manifest.json").parent
    copy = tmp_path / "run"

    shutil.copytree(source, copy)
    (copy / "data" / "dataset.csv").write_text("tampered\n", encoding="utf-8")
    assert any("changed since" in p for p in manifest.verify(copy / "manifest.json"))


# ---------------------------------------------------------------------------
# One row per batch
# ---------------------------------------------------------------------------
def test_a_batch_twice_in_a_cohort_is_refused():
    frame = pd.DataFrame({"Patient Lot/Batch #": [f"B{i}" for i in range(8)]})
    frame.loc[3, "Patient Lot/Batch #"] = "B0"
    spec = taskconfig.parse(
        {
            "task": {"name": "s"},
            "target": {"column": "outcome"},
            "roles": {"id": "Patient Lot/Batch #"},
        },
        Path("."),
    )
    with pytest.raises(dataset.TaskRefused, match="not unique"):
        dataset._check_unique_ids(frame, spec)


# ---------------------------------------------------------------------------
# Encoding hygiene
# ---------------------------------------------------------------------------
def test_floating_point_jitter_is_constant():
    jitter = pd.Series([0.015228, 0.0152285, 0.015229, 0.015228])
    assert encode.near_constant(jitter)
    assert not encode.near_constant(pd.Series([0.0, 0.015228, 0.015229]))


def test_no_constant_column_reaches_an_exported_table(built):
    for task in (EXPLORATORY, PREDICTIVE):
        table = csv(built, task, "dataset_encoded.csv")
        recipe_ = json.loads(artifact(built, task, "encoding.json").read_text(encoding="utf-8"))
        predictors = recipe_["output_columns"]
        constant = [c for c in predictors if encode.near_constant(table[c])]
        assert not constant, (task, constant[:5])


def test_a_missing_category_is_missing_in_every_one_hot_column():
    frame = pd.DataFrame({"site": ["A", "B", "C", None]})
    state, _ = encode.fit_one_hot(frame, ["site"], encode.TaskParams(), set())
    encoded, _ = encode.apply_one_hot(frame, state, encode.TaskParams())
    assert encoded.iloc[3].isna().all()
    assert encoded.iloc[:3].notna().all().all()


def test_a_small_cohort_is_not_target_encoded_or_hashed():
    frame = pd.DataFrame({"team": list("ABCDEF") * 3, "machine": [f"M{i}" for i in range(18)]})
    params = encode.TaskParams(onehot_max_categories=4, target_max_categories=8)
    routes = {r.column: r for r in encode.route_categoricals(frame, ["team"], params)}
    assert routes["team"].strategy == "hashing" or routes["team"].strategy == "one_hot"
    assert "rows per category" in routes["team"].reason


# ---------------------------------------------------------------------------
# The recipe is read back, and fits inside cross-validation
# ---------------------------------------------------------------------------
def test_a_recipe_applied_to_the_cohort_reproduces_the_package(built):
    run = artifact(built, PREDICTIVE, "manifest.json").parent
    encoded = csv(built, PREDICTIVE, "dataset_encoded.csv")
    state = json.loads(artifact(built, PREDICTIVE, "encoding.json").read_text(encoding="utf-8"))
    # The package's target encoding is cross-fitted; the recipe applies the frozen means.
    crossfitted = {f["name"] for f in state["target_encoding"].get("columns", {}).values()}
    batch_table = built["batch_table"].set_index("Patient Lot/Batch #")
    rows = batch_table.loc[encoded["Patient Lot/Batch #"]].copy().reset_index()
    applied = recipe.apply(run, rows)
    compared = [c for c in applied.columns if c not in crossfitted]
    assert compared
    pd.testing.assert_frame_equal(
        applied[compared].reset_index(drop=True).astype("float64"),
        encoded[compared].reset_index(drop=True).astype("float64"),
        check_exact=False,
        atol=1e-9,
    )


def test_every_encoded_column_belongs_to_exactly_one_explanation_group(built):
    for task in (PREDICTIVE, EXPLORATORY):
        groups = json.loads(artifact(built, task, "groups.json").read_text(encoding="utf-8"))
        encoded = csv(built, task, "dataset_encoded.csv")
        columns = [c for g in groups.values() for c in g["columns"]]
        assert sorted(columns) == sorted(
            json.loads(artifact(built, task, "encoding.json").read_text(encoding="utf-8"))[
                "output_columns"
            ]
        )
        assert set(columns) <= set(encoded.columns)
        clusters = csv(built, task, "clusters.csv").dropna(subset=["Cluster"])
        for name, group in groups.items():
            if len(group["parameters"]) > 1:  # a group of correlated parameters
                stability = clusters.loc[clusters["Cluster"] == name, "Stability"]
                assert (stability >= 0.8).all(), name
    # Every one-hot column of a category explains together with its category.
    groups = json.loads(artifact(built, EXPLORATORY, "groups.json").read_text(encoding="utf-8"))
    assert any(len(g["columns"]) > 1 and len(g["parameters"]) == 1 for g in groups.values())


def test_the_task_transformer_uses_the_runs_groups_in_every_fold(built):
    run = built["packages"][EXPLORATORY]
    groups = json.loads((run / "metadata" / "groups.json").read_text(encoding="utf-8"))
    fixed = [set(g["parameters"]) for g in groups.values() if len(g["parameters"]) > 1]
    X, y, _, spec = recipe.task_data(
        built["directory"] / dummy_data.TASK_FILES[1], built["config_b"]
    )
    half = X.iloc[: len(X) // 2 + 4]
    transformer = recipe.TaskTransformer(spec, run=run).fit(half, y.loc[half.index])
    used = [set(k["members"]) for k in transformer.preprocessor_.clusters["clusters"]]
    assert used and all(any(members <= group for group in fixed) for members in used)


def test_the_task_transformer_runs_inside_grouped_cross_validation(built):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, groups, spec = recipe.task_data(
        built["directory"] / dummy_data.TASK_FILES[0], built["config"]
    )
    model = make_pipeline(
        recipe.TaskTransformer(spec),
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-2, 4, 13)),
    )
    predictions = cross_val_predict(model, X, y, groups=groups, cv=GroupKFold(n_splits=3))
    assert len(predictions) == len(y) and np.isfinite(predictions).all()
    # Nothing scaled to unit variance became a wild extrapolation.
    assert np.abs(predictions - y.mean()).max() < 10 * y.std()


def test_the_missingness_cohort_keeps_every_batch_at_a_site_that_records_no_kind():
    config = {
        "build": {
            "missingness_cohort": {
                "column": "batch_type",
                "value": "type_commercial",
                "all_batches_at": ["b"],
            }
        }
    }
    kind = names.col("batch_type")
    wanted = names.value("type_commercial")
    first = pd.DataFrame({kind: [wanted, wanted.upper(), f" {wanted.lower()} ", "other", None]})
    kept, why = cli._missingness_cohort(first, names.site("a"), config)
    assert len(kept) == 3, "case and surrounding spaces do not matter"
    assert "3 of 5 batches" in why
    second = pd.DataFrame({kind: [None, None]})
    kept, why = cli._missingness_cohort(second, names.site("b"), config)
    assert len(kept) == 2, "a blank kind at an all-batches site still counts"
    assert cli._missingness_cohort(first, names.site("a"), {})[0] is first


# ---------------------------------------------------------------------------
# Everything is about one site
# ---------------------------------------------------------------------------
def test_a_config_names_the_one_site_it_is_about(tmp_path):
    for text, problem in (
        (CONFIG_TEXT.replace("\nsite: a\n", "\n"), "site: required"),
        (CONFIG_TEXT.replace("\nsite: a\n", "\nsite: c\n"), "site: required"),
    ):
        (tmp_path / "config.yaml").write_text(text)
        with pytest.raises(workspace.ConfigError, match=problem):
            workspace.load(tmp_path / "config.yaml")
    (tmp_path / "config.yaml").write_text(CONFIG_TEXT.replace("\nsite: a\n", "\nsite: b\n"))
    config = workspace.load(tmp_path / "config.yaml")
    assert workspace.site_key(config) == "b" and workspace.site_name(config) == names.site("b")
    assert workspace.batch_table_folder(config).path.endswith(f"/{names.site('b')}/")
    # Its build reads its own records, and what only site B needs.
    assert cli._build_inputs("b") == (
        "catalog",
        "phf_site_b",
        "param_mapping",
        "remfg",
        "lv_coa",
        "raw_materials",
    )
    assert cli._build_inputs("a") == ("catalog", "phf_site_a", "lv_coa", "raw_materials")


def test_a_task_names_its_site_and_refuses_another(tmp_path, caplog):
    base = {"task": {"name": "t"}, "target": {"column": "x"}}
    assert taskconfig.parse({**base, "site": "b"}, tmp_path).site == "b"
    with pytest.raises(taskconfig.TaskConfigError, match="cohort.sites: no longer a setting"):
        taskconfig.parse({**base, "cohort": {"sites": [names.site("a")]}}, tmp_path)
    with pytest.raises(taskconfig.TaskConfigError, match="site: 'c' is not a site key"):
        taskconfig.parse({**base, "site": "c"}, tmp_path)
    spec = taskconfig.parse({**base, "site": "b"}, tmp_path)
    with pytest.raises(dataset.TaskRefused, match="run it with a config whose site: is b"):
        dataset.task_site(spec, "a")
    assert dataset.task_site(spec, None) == "b", "a rerun goes by its own"
    # A task file from before takes the config's, and says so.
    older = taskconfig.parse(base, tmp_path)
    assert dataset.task_site(older, "a") == "a" and "add `site: a`" in caplog.text
    with pytest.raises(dataset.TaskRefused, match="names no site"):
        dataset.task_site(older, None)


def test_a_task_refuses_a_batch_table_of_another_site_or_of_both():
    table = pd.DataFrame({"Site Merged": [names.site("a"), names.site("a")]})
    dataset.check_batch_table_site(table, names.site("a"))
    for other in (names.site("b"), names.site("a")):
        both = pd.DataFrame({"Site Merged": [names.site("a"), names.site("b")]})
        with pytest.raises(dataset.TaskRefused, match="rebuild it with `ml build`"):
            dataset.check_batch_table_site(both, other)
    with pytest.raises(dataset.TaskRefused, match=f"is about {names.site('b')} only"):
        dataset.check_batch_table_site(table, names.site("b"))


def test_a_proposal_is_about_the_configured_site(built, tmp_path):
    from ml import screening

    path = tmp_path / "proposal.csv"
    pd.DataFrame(
        [{"Action": "exclude", "Parameter": "x", "Site": names.site("b"), "Reason": "retired"}]
    ).to_csv(path, index=False)
    with pytest.raises(screening.ScreeningError, match=f"config.yaml is about {names.site('a')}"):
        cli.run_screening_add(built["config"], path, "A. Reviewer")


def test_a_task_run_with_the_other_site_config_is_refused(built):
    spec = taskconfig.load(built["directory"] / dummy_data.TASK_FILES[0])
    assert spec.site == "a"
    with pytest.raises(dataset.TaskRefused, match="whose site: is a"):
        cli.run_task(built["config_b"], spec)
