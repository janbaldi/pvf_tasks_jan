"""Workspaces, config, manifests, and the fixes the review asked for.

The workspace keeps configuration and data apart from the code; the config
holds every SharePoint location; every manifest has one format; a predictive
task's availability is checked against the PTF's stages; a split is seeded and
keeps groups — and patients — whole; a column constant to measurement precision
never reaches a model; and a recipe can be read back and applied.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import dummy_data
from conftest import EXPLORATORY, PREDICTIVE, artifact, csv
from pvf import cli, corrections, dataset, encode, manifest, recipe, scaffold, taskconfig
from pvf import config as workspace


# ---------------------------------------------------------------------------
# Config: found, validated, and the only place a SharePoint address lives
# ---------------------------------------------------------------------------
def test_the_package_holds_no_sharepoint_address():
    source = Path(cli.__file__).parent
    code = "\n".join(path.read_text(encoding="utf-8") for path in source.glob("*.py"))
    assert "MS%26T" not in code and "DRIVE_ID_" not in code
    assert "MS%26T" in scaffold.template("pvf.yaml"), "the template carries them instead"


def test_the_config_template_is_a_valid_config(tmp_path):
    config_path = scaffold.init(tmp_path / "ws")
    config = workspace.load(config_path)
    assert config["paths"]["pvf"].endswith("data/processed/PVF.xlsx")
    assert (tmp_path / "ws" / "corrections.yaml").exists()
    assert {"data", "outputs", "tasks"} <= {p.name for p in (tmp_path / "ws").iterdir()}
    assert len(corrections.load(config["cleaning"]["corrections"])) > 5


def test_sharepoint_locations_come_from_the_config(tmp_path):
    text = scaffold.template("pvf.yaml").replace("location: local", "location: sharepoint")
    path = tmp_path / "pvf.yaml"
    path.write_text(text, encoding="utf-8")
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


def test_the_workspace_env_file_is_loaded_with_its_config(tmp_path, monkeypatch):
    config_path = scaffold.init(tmp_path / "ws")
    assert (tmp_path / "ws" / ".env.example").exists()
    (tmp_path / "ws" / ".env").write_text("PVF_TEST_SECRET=from-file\nMODE=DEV\n")
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
        ({"surprise": 1}, "unknown setting"),
    ],
)
def test_a_config_that_cannot_be_used_is_refused_by_field(tmp_path, change, field):
    config = yaml.safe_load(scaffold.template("pvf.yaml"))
    config.update(change)
    path = tmp_path / "pvf.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(workspace.ConfigError, match=field.replace(".", r"\.")):
        workspace.load(path)


def test_the_config_is_found_above_the_working_directory(tmp_path, monkeypatch):
    config_path = scaffold.init(tmp_path / "ws")
    deep = tmp_path / "ws" / "tasks" / "drafts"
    deep.mkdir(parents=True)
    monkeypatch.delenv(workspace.CONFIG_ENV, raising=False)
    monkeypatch.chdir(deep)
    assert workspace.find() == config_path
    monkeypatch.setenv(workspace.CONFIG_ENV, "/somewhere/pvf.yaml")
    assert workspace.find() == Path("/somewhere/pvf.yaml")
    assert workspace.find("explicit.yaml") == Path("explicit.yaml")


# ---------------------------------------------------------------------------
# A task can live anywhere
# ---------------------------------------------------------------------------
def test_a_new_task_outside_the_workspace_reads_the_workspace(built, tmp_path):
    path = scaffold.new_task(
        built["config"], str(tmp_path / "elsewhere" / "my_question.yaml"), "FP Flow CAR+ (%)"
    )
    spec = taskconfig.load(path)  # no config passed: the file says where its workspace is
    assert spec.pvf == Path(built["config"]["paths"]["pvf"])
    assert spec.output_root == Path(built["config"]["paths"]["tasks"])
    assert spec.workspace == built["config"]["__path__"]


def test_a_task_without_inputs_or_workspace_says_what_to_do(tmp_path):
    path = tmp_path / "task.yaml"
    path.write_text(
        yaml.safe_dump({"task": {"name": "t"}, "target": {"column": "x"}}), encoding="utf-8"
    )
    with pytest.raises(taskconfig.TaskConfigError, match="inputs.pvf.*workspace"):
        taskconfig.load(path)


def test_the_command_line_checks_and_reports_errors_plainly(built, capsys):
    config = built["config"]["__path__"]
    assert cli.main(["--config", config, "check"]) == 0
    assert cli.main(["--config", config, "tasks", "--task", "no_such_task.yaml"]) == 2
    assert "does not exist" in capsys.readouterr().err


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
# Splits: seeded, stratified, whole groups and whole patients
# ---------------------------------------------------------------------------
def split_frame(n: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Site Merged": "Ghent",
            "Patient Lot/Batch #": [f"B{i}" for i in range(n)],
            "Vector Lot": [f"LV-{i % 8}" for i in range(n)],
            # Patients 0..3 each have a batch in two different lots.
            "Patient ID": [f"P{i % 36}" for i in range(n)],
            "outcome": ["Pass", "Fail"] * (n // 2),
        }
    )


def split_spec(seed: int = 1, **split) -> taskconfig.TaskSpec:
    return taskconfig.parse(
        {
            "task": {"name": "s", "seed": seed},
            "inputs": {"pvf": "a.xlsx", "ptf": "b.xlsx"},
            "target": {"column": "outcome", "type": "binary", "positive_class": "Pass"},
            "columns": {
                "id": "Patient Lot/Batch #",
                "group": "Vector Lot",
                "patient": "Patient ID",
            },
            "split": {"strategy": "grouped", "validation_fraction": 0.25, **split},
        },
        Path("."),
    )


def test_a_grouped_split_keeps_patients_whole_and_moves_with_the_seed():
    frame = split_frame()
    target = (frame["outcome"] == "Pass").astype(int)
    held = []
    for seed in (1, 2, 3):
        spec = split_spec(seed)
        training, validation = dataset.assign_split(
            frame, frame.index, spec, dataset.TaskResult(spec=spec), target
        )
        for column in ("Vector Lot", "Patient ID"):
            assert not set(frame.loc[training, column]) & set(frame.loc[validation, column])
        held.append(frozenset(validation))
    again, _ = dataset.assign_split(
        frame, frame.index, split_spec(1), dataset.TaskResult(spec=split_spec(1)), target
    )
    assert frozenset(frame.index.difference(again)) == held[0], "same seed, same split"
    assert len(set(held)) > 1, "a different seed draws different groups"


def test_folds_keep_groups_whole_and_reach_splits_csv(built):
    splits = csv(built, PREDICTIVE, "splits.csv")
    dataset_ = csv(built, PREDICTIVE, "dataset.csv")
    folded = splits[splits["Split"] == "training"]
    assert folded["Fold"].notna().all()
    assert splits.loc[splits["Split"] == "validation", "Fold"].isna().all()
    lots = dataset_.set_index("Patient Lot/Batch #")["Vector Lot"]
    per_lot = folded.assign(lot=folded["Batch"].map(lots)).groupby("lot")["Fold"].nunique()
    assert (per_lot == 1).all()


def test_a_batch_twice_in_a_cohort_is_refused():
    frame = split_frame(8)
    frame.loc[3, "Patient Lot/Batch #"] = "B0"
    with pytest.raises(dataset.TaskRefused, match="not unique"):
        dataset._check_unique_ids(frame, split_spec())


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
        training = table[table["split"] == "training"] if "split" in table.columns else table
        constant = [c for c in predictors if encode.near_constant(training[c])]
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
def test_a_recipe_applied_to_the_validation_rows_reproduces_the_package(built):
    run = artifact(built, PREDICTIVE, "manifest.json").parent
    transformed = csv(built, PREDICTIVE, "transformed.csv")
    pvf = built["pvf"].set_index("Patient Lot/Batch #")
    validation = transformed[transformed["split"] == "validation"]
    rows = pvf.loc[validation["Patient Lot/Batch #"]].copy().reset_index()
    applied = recipe.apply(run, rows)
    expected = validation[list(applied.columns)].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        applied.reset_index(drop=True).astype("float64"),
        expected.astype("float64"),
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
