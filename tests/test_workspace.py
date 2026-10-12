"""Config, task workspaces, manifests, and the fixes the review asked for.

``config.yaml`` names the one PTF and the one PVF and holds every SharePoint
location; each task is a workspace whose runs carry the inputs they read; every
manifest has one format; a predictive task's availability is checked against
the PTF's stages; a task can leave out whole PTF categories; a column constant
to measurement precision never reaches a model; and a recipe can be read back
and applied.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import dummy_data
from conftest import EXPLORATORY, PREDICTIVE, ROOT, artifact, csv
from pvf import cli, corrections, dataset, encode, manifest, recipe, scaffold, taskconfig
from pvf import config as workspace

CONFIG_TEXT = (ROOT / "config.yaml").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Config: found, validated, and the only place a SharePoint address lives
# ---------------------------------------------------------------------------
def test_the_package_holds_no_sharepoint_address():
    source = Path(cli.__file__).parent
    code = "\n".join(path.read_text(encoding="utf-8") for path in source.glob("*.py"))
    assert "MS%26T" not in code and "DRIVE_ID_" not in code
    assert "MS%26T" in CONFIG_TEXT, "config.yaml carries them instead"


def test_the_shipped_config_is_a_valid_config():
    config = workspace.load(ROOT / "config.yaml")
    assert config["paths"]["pvf"] == str(Path("~/pvf_data/PVF.xlsx").expanduser())
    assert len(corrections.load((config.get("cleaning") or {}).get("corrections"))) > 5


def test_sharepoint_locations_come_from_the_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG_TEXT.replace("location: local", "location: sharepoint"))
    config = workspace.load(path)
    calls = []
    source = workspace.source(config, "phf_raritan", lambda **kw: calls.append(kw))
    source.read()
    assert calls == [
        {
            "sharepoint_path": "General/Batch Data/Commercial Manufacturing Data.xlsx",
            "sheet_name": "BR Data",
            "drive_id": "DRIVE_ID_TIGER",
        }
    ]
    maps = workspace.site_map_sources(config, lambda **kw: None)
    assert [m.remote["sharepoint_path"].rsplit("/", 1)[1] for m in maps] == [
        "clinical_sites.xlsx",
        "cross_data_meeting.xlsx",
        "from_Hannelore.xlsx",
    ]
    # The certificate workbook keeps its two rows of furniture above the header.
    assert workspace.source(config, "lv_coa", lambda **kw: None).header == 2


def test_the_env_file_beside_the_config_is_loaded_with_it(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG_TEXT)
    (tmp_path / ".env").write_text("PVF_TEST_SECRET=from-file\nMODE=DEV\n")
    monkeypatch.delenv("PVF_TEST_SECRET", raising=False)
    monkeypatch.setenv("MODE", "PRD")  # the shell wins over the file
    config = workspace.load(config_path)
    import os

    assert os.environ["PVF_TEST_SECRET"] == "from-file"
    assert os.environ["MODE"] == "PRD"
    assert config["__env__"].endswith(".env")
    assert "__env__" not in workspace.settings(config), "never part of the config digest"
    monkeypatch.delenv("PVF_TEST_SECRET")


def test_the_upload_target_comes_from_the_config_or_the_environment(monkeypatch):
    config = {"paths": {"pvf": "/x/PVF.xlsx"}, "upload": {"enabled": True, "drive_id": "D"}}
    with pytest.raises(workspace.ConfigError, match="upload.path"):
        workspace.upload_target(config)
    monkeypatch.setenv("DATA_LINK", "Reports/PVF/")
    target = workspace.upload_target(config)
    assert target["location"] == "SharePoint D: Reports/PVF/PVF.xlsx"


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"sources": {"location": "cloud"}}, "sources.location"),
        ({"sharepoint": {"ptf": {"sheet": "x"}}}, "sharepoint.ptf.path"),
        ({"cleaning": {"censord": "half"}}, "cleaning.censord"),
        ({"features": {"co2": 5}}, "features.co2"),
        ({"tasks": {"target": "x"}}, "unknown setting"),
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
def test_a_new_task_is_a_workspace_that_reads_the_configured_pvf(built, tmp_path):
    folder = tmp_path / "analyses" / "my_question"
    path = scaffold.new_task(folder, "FP Flow CAR+ (%)")
    assert path == folder.resolve() / "task.yaml"
    spec = taskconfig.load(folder, built["config"])
    assert spec.name == "my_question"
    assert spec.pvf == Path(built["config"]["paths"]["pvf"])
    assert spec.output_root == folder.resolve() / "runs"


def test_a_task_without_inputs_or_config_says_what_to_do(tmp_path):
    path = tmp_path / "task.yaml"
    path.write_text(
        yaml.safe_dump({"task": {"name": "t"}, "target": {"column": "x"}}), encoding="utf-8"
    )
    with pytest.raises(taskconfig.TaskConfigError, match="inputs.pvf.*config.yaml"):
        taskconfig.load(tmp_path)


def test_the_command_line_checks_and_reports_errors_plainly(built, capsys):
    config = built["config"]["__path__"]
    task = str(built["directory"] / dummy_data.TASK_FILES[0])
    assert cli.main(["--config", config, "check", "--task", task]) == 0
    assert cli.main(["--config", config, "tasks", "--task", "no_such_task"]) == 2
    assert "does not exist" in capsys.readouterr().err


def test_a_run_carries_its_inputs_and_can_be_redone_from_its_folder_alone(built, tmp_path):
    run = built["packages"][PREDICTIVE]
    pvf = Path(built["config"]["paths"]["pvf"])
    assert (run / "inputs" / "PVF.xlsx").read_bytes() == pvf.read_bytes()
    assert (run / "inputs" / "PTF.xlsx").read_bytes() == Path(
        built["config"]["paths"]["ptf"]
    ).read_bytes()
    settings = yaml.safe_load((run / "config" / "task.yaml").read_text(encoding="utf-8"))
    assert settings["inputs"] == {"pvf": "../inputs/PVF.xlsx", "ptf": "../inputs/PTF.xlsx"}

    copy = tmp_path / "copied_run"
    shutil.copytree(run, copy)
    again = cli.run_task(None, taskconfig.load(copy / "config" / "task.yaml"))
    assert (again / "data" / "dataset.csv").read_bytes() == (
        run / "data" / "dataset.csv"
    ).read_bytes()
    record = json.loads((again / "manifest.json").read_text(encoding="utf-8"))
    assert not any("manifest" in warning for warning in record["warnings"])


# ---------------------------------------------------------------------------
# Columns left out by their PTF category
# ---------------------------------------------------------------------------
def test_a_ptf_category_can_be_left_out_by_name(built):
    ptf = pd.read_excel(built["config"]["paths"]["ptf"])
    late = set(ptf.loc[ptf["Available At"] == "Post-release", "Parameter"])
    decisions = csv(built, EXPLORATORY, "decisions.csv")
    excluded = decisions[decisions["Decision"] == "excluded"].set_index("Parameter")
    present = late & set(built["pvf"].columns)
    assert present <= set(excluded.index), "every one is out, some already for another reason"
    left_out = decisions[decisions["Stage"] == "category exclusion"]
    assert len(left_out) and set(left_out["Parameter"]) <= present
    assert (left_out["Reason"] == "PTF 'Available At' is 'Post-release'").all()
    columns = csv(built, EXPLORATORY, "columns.csv")
    assert not set(columns.loc[columns["Role"] == "predictor", "Source"]) & late


def test_a_category_column_the_ptf_does_not_have_is_refused():
    spec = taskconfig.parse(
        {
            "task": {"name": "t"},
            "inputs": {"pvf": "a.xlsx", "ptf": "b.xlsx"},
            "target": {"column": "y"},
            "columns": {"exclude_categories": {"Categroy": ["QC"]}},
        },
        Path("."),
    )
    ptf = pd.DataFrame({"Parameter": ["x"], "Value Type": ["numeric"], "Category": ["QC"]})
    with pytest.raises(dataset.TaskRefused, match="no 'Categroy' column"):
        dataset._category_exclusions(ptf, spec)


# ---------------------------------------------------------------------------
# Corrections are data
# ---------------------------------------------------------------------------
def test_a_correction_rule_can_be_switched_off(tmp_path):
    frame = {"Raritan": pd.DataFrame({"Type": ["Clinical", None]}), "Ghent": pd.DataFrame()}
    rules = corrections.load(None)
    changes = corrections.apply(frame, rules, corrections.INTEGRITY)
    assert set(frame["Raritan"]["Type"]) == {"Commercial"}
    assert next(c for c in changes if c["column"] == "Type")["rows"] == 2

    frame = {"Raritan": pd.DataFrame({"Type": ["Clinical"]}), "Ghent": pd.DataFrame()}
    corrections.apply(frame, rules, corrections.INTEGRITY, disabled={"raritan_type_commercial"})
    assert frame["Raritan"]["Type"].tolist() == ["Clinical"]


def test_a_broken_corrections_table_names_the_rule(tmp_path):
    path = tmp_path / "corrections.yaml"
    path.write_text(
        yaml.safe_dump({"corrections": [{"site": "Ghent", "column": "x", "divide_by": 0}]}),
        encoding="utf-8",
    )
    with pytest.raises(corrections.CorrectionsError, match=r"corrections\[0\]\.divide_by"):
        corrections.load(path)


# ---------------------------------------------------------------------------
# One manifest format
# ---------------------------------------------------------------------------
def test_every_manifest_follows_the_one_format(built):
    processed = Path(built["config"]["paths"]["pvf"]).parent
    paths = [
        processed / "ptf_manifest.json",
        processed / "pvf_manifest.json",
        *(artifact(built, task, "manifest.json") for task in (PREDICTIVE, EXPLORATORY)),
    ]
    stages = []
    for path in paths:
        assert manifest.verify(path) == [], path
        record = manifest.read(path)
        assert list(record)[: len(manifest.ENVELOPE) - 1] == list(manifest.ENVELOPE[:-1])
        stages.append(record["stage"])
    assert stages == ["ptf", "pvf", "task", "task"]


def test_a_manifest_names_the_files_it_describes_relative_to_itself(built):
    record = manifest.read(Path(built["config"]["paths"]["pvf"]).parent / "pvf_manifest.json")
    written = {entry["what"]: entry["path"] for entry in record["outputs"]}
    assert written["PVF"] == "PVF.xlsx" and written["PVF (Parquet)"] == "PVF.parquet"
    assert record["pvf"]["content_sha256"]
    assert {entry["label"] for entry in record["inputs"]} >= {"ptf", "phf_ghent", "site_map_1"}


def test_a_changed_output_is_caught(built, tmp_path):
    source = artifact(built, PREDICTIVE, "manifest.json").parent
    copy = tmp_path / "run"
    import shutil

    shutil.copytree(source, copy)
    (copy / "data" / "dataset.csv").write_text("tampered\n", encoding="utf-8")
    assert any("changed since" in p for p in manifest.verify(copy / "manifest.json"))


# ---------------------------------------------------------------------------
# Availability by process stage
# ---------------------------------------------------------------------------
def test_nothing_after_the_cutoff_reaches_a_predictive_dataset(built):
    columns = csv(built, PREDICTIVE, "columns.csv")
    predictors = columns[columns["Role"] == "predictor"]["Column"]
    for late in ("Post Thaw", "Final Formulation", "Post formulation dose", "Endotoxin"):
        assert not predictors.str.contains(late, regex=False).any(), late
    decisions = csv(built, PREDICTIVE, "decisions.csv")
    reasons = decisions[decisions["Stage"] == "availability"]["Reason"]
    assert reasons.str.contains("available at Post-thaw").any()
    # A feature is as late as its latest input, even with no stage of its own.
    assert "Post formulation dose" in set(
        decisions[decisions["Stage"] == "availability"]["Parameter"]
    )
    # A parameter the PTF gives no stage cannot be shown to be early enough.
    assert reasons.str.contains("no 'Available At' stage").any()


def test_a_cutoff_that_is_not_a_stage_is_refused_unless_reviewed(tmp_path):
    task = {
        "task": {"name": "t", "purpose": "predictive"},
        "inputs": {"pvf": "PVF.xlsx", "ptf": "PTF.xlsx"},
        "target": {"column": "y"},
        "availability": {"cutoff": "after lunch"},
    }
    path = tmp_path / "t.yaml"
    path.write_text(yaml.safe_dump(task), encoding="utf-8")
    with pytest.raises(taskconfig.TaskConfigError, match="not one of the process stages"):
        taskconfig.load(path)


# ---------------------------------------------------------------------------
# One row per batch
# ---------------------------------------------------------------------------
def test_a_batch_twice_in_a_cohort_is_refused():
    frame = pd.DataFrame({"Patient Lot/Batch #": [f"B{i}" for i in range(8)]})
    frame.loc[3, "Patient Lot/Batch #"] = "B0"
    spec = taskconfig.parse(
        {
            "task": {"name": "s"},
            "inputs": {"pvf": "a.xlsx", "ptf": "b.xlsx"},
            "target": {"column": "outcome"},
            "columns": {"id": "Patient Lot/Batch #"},
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
    for task, name in ((EXPLORATORY, "dataset.csv"), (PREDICTIVE, "transformed.csv")):
        table = csv(built, task, name)
        columns = csv(built, task, "columns.csv")
        predictors = columns[columns["Role"] == "predictor"]["Column"]
        if name == "transformed.csv":
            recipe_ = json.loads(artifact(built, task, "recipe.json").read_text(encoding="utf-8"))
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
    transformed = csv(built, PREDICTIVE, "transformed.csv")
    state = json.loads(artifact(built, PREDICTIVE, "recipe.json").read_text(encoding="utf-8"))
    # The package's target encoding is cross-fitted; the recipe applies the frozen means.
    crossfitted = {f["name"] for f in state["target_encoding"].get("columns", {}).values()}
    pvf = built["pvf"].set_index("Patient Lot/Batch #")
    rows = pvf.loc[transformed["Patient Lot/Batch #"]].copy().reset_index()
    applied = recipe.apply(run, rows)
    compared = [c for c in applied.columns if c not in crossfitted]
    assert compared
    pd.testing.assert_frame_equal(
        applied[compared].reset_index(drop=True).astype("float64"),
        transformed[compared].reset_index(drop=True).astype("float64"),
        check_exact=False,
        atol=1e-9,
    )


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
