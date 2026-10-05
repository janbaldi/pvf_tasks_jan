"""The task packages, and the reports that explain them.

A task narrows the PVF to one cohort and one target and writes a folder someone
else has to be able to use without asking what is in it. What matters is that
the narrowing is visible, that the target and everything computed from it stay
out of the predictors, that every exported column can be read back to the
parameter it came from, and that the counts in the report are the counts in the
files.
"""

from __future__ import annotations

import json

import pandas as pd

import dummy_data
from conftest import (
    EXPLORATORY,
    PREDICTIVE,
    artifact,
    csv,
    downloads,
    facts,
    figures,
    first_table,
    manifest,
    payload,
    rows_naming,
    section,
    table_with,
)

REQUIRED = (
    "task.yaml",
    "dataset.csv",
    "columns.csv",
    "decisions.csv",
    "cohort.csv",
    "clusters.csv",
    "recipe.json",
    "report.html",
    "report_payload.json",
    "manifest.json",
    "run.log",
    "ptf_manifest.json",
    "pvf_manifest.json",
)


# ---------------------------------------------------------------------------
# The package
# ---------------------------------------------------------------------------
def test_a_finished_task_folder_holds_everything_it_promises(built):
    for task in (PREDICTIVE, EXPLORATORY):
        for name in REQUIRED:
            assert artifact(built, task, name).exists(), f"{task}/{name}"
        assert manifest(built, task)["status"] == "complete"
    # The transformed table has its untransformed inputs beside it; the raw one
    # is already that table and gets no second copy.
    assert artifact(built, EXPLORATORY, "raw_features.csv").exists()
    assert not artifact(built, PREDICTIVE, "raw_features.csv").exists()


def test_the_manifest_hashes_every_artefact_but_itself(built):
    recorded = manifest(built, PREDICTIVE)["artifacts"]
    names = {entry["name"] for entry in recorded}
    assert "manifest.json" not in names
    assert {"data/dataset.csv", "report/report.html", "metadata/columns.csv"} <= names
    assert all(len(entry["sha256"]) == 64 for entry in recorded)


def test_the_counts_agree_wherever_they_are_written(built):
    for task in (PREDICTIVE, EXPLORATORY):
        counts = manifest(built, task)["counts"]
        dataset = csv(built, task, "dataset.csv")
        columns = csv(built, task, "columns.csv")
        assert counts["dataset_rows"] == len(dataset)
        assert counts["dataset_columns"] == dataset.shape[1]
        assert list(columns["Column"]) == list(dataset.columns)

        overview = facts(first_table(payload(built, task), "overview"))
        assert int(overview["Batches in the dataset"].replace(",", "")) == len(dataset)

        funnel = first_table(payload(built, task), "cohort")["rows"]
        assert int(funnel[-1][-1].replace(",", "")) == len(dataset)


def test_two_runs_of_one_task_do_not_overwrite_each_other(built):
    from pvf import cli, taskconfig

    spec = taskconfig.load(built["directory"] / dummy_data.TASK_FILES[0])
    again = cli.run_task(spec)
    first = built["packages"][PREDICTIVE]
    assert again != first
    assert first.exists() and (first / cli.TASK_LAYOUT["dataset.csv"]).exists()
    assert again.parent == first.parent


def test_report_only_writes_no_dataset_and_says_so(built, tmp_path):
    from pvf import cli, taskconfig

    spec = taskconfig.load(built["directory"] / dummy_data.TASK_FILES[0])
    folder = cli.run_task(spec, report_only=True)
    assert (folder / cli.TASK_LAYOUT["report.html"]).exists()
    assert not (folder / cli.TASK_LAYOUT["dataset.csv"]).exists()
    assert not (folder / cli.TASK_LAYOUT["recipe.json"]).exists()
    record = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert record["mode"] == "report only"
    assert "not written" in record["dataset_role"]
    # And it left every earlier package alone.
    assert (built["packages"][PREDICTIVE] / cli.TASK_LAYOUT["dataset.csv"]).exists()


# ---------------------------------------------------------------------------
# Cohort and target
# ---------------------------------------------------------------------------
def test_the_cohort_is_narrower_than_the_pvf_and_says_how(built):
    dataset, pvf = csv(built, PREDICTIVE, "dataset.csv"), built["pvf"]
    assert len(dataset) < len(pvf)

    steps = first_table(payload(built, PREDICTIVE), "cohort")["rows"]
    left = [int(row[-1].replace(",", "")) for row in steps]
    assert left == sorted(left, reverse=True), "each filter can only remove batches"
    assert left[0] == len(pvf)
    assert left[-1] == len(dataset)


def test_withdrawals_are_excluded_however_the_source_spells_them(built):
    pvf = built["pvf"]
    kept = set(csv(built, PREDICTIVE, "dataset.csv")["Patient Lot/Batch #"])
    ghent = pvf[pvf["Site Merged"] == "Ghent"]
    spellings = ghent["Non-Conformance Type"].astype("string")
    assert {"withdrawn", "Withdrawal"} <= set(spellings.dropna()), "the fixture has both spellings"

    is_withdrawal = spellings.str.lower().isin({"withdrawn", "withdrawal"}).fillna(False)
    withdrawn = set(ghent.loc[is_withdrawal, "Patient Lot/Batch #"])
    assert withdrawn, "there are withdrawals to exclude"
    assert not (withdrawn & kept), "a withdrawal is a withdrawal however it is written"


def test_batches_without_a_target_are_counted_and_named_separately(built):
    report = payload(built, PREDICTIVE)
    reasons = table_with(report, "Why a batch is not in the dataset")
    assert any("target" in str(row[0]) for row in reasons["rows"])

    exclusions = csv(built, PREDICTIVE, "cohort.csv")
    assert len(exclusions) == len(built["pvf"])
    assert exclusions["Included"].sum() == len(csv(built, PREDICTIVE, "dataset.csv"))
    assert exclusions.loc[~exclusions["Included"], "Reason"].notna().all()


def test_a_balanced_binary_target_maps_both_classes(built):
    dataset = csv(built, EXPLORATORY, "dataset.csv")
    target = dataset["disposition_is_released"]
    assert set(target.dropna().unique()) == {0, 1}
    assert target.notna().all()
    # The declared positive class is 1, whether or not it is the rarer one.
    counts = target.value_counts()
    assert counts[1] > 0 and counts[0] > 0


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------
def test_the_target_and_everything_computed_from_it_stay_out(built):
    dataset = csv(built, PREDICTIVE, "dataset.csv")
    decisions = csv(built, PREDICTIVE, "decisions.csv")

    for leaked in ("VCN/cell", "Flow accuracy (effective)", dummy_data.TARGET):
        assert leaked not in dataset.columns
    lineage = decisions[decisions["Stage"] == "target lineage"]
    assert {"VCN/cell", "Flow accuracy (effective)"} <= set(lineage["Parameter"])
    assert all(dummy_data.TARGET in reason for reason in lineage["Reason"])


def test_roles_are_metadata_and_are_not_encoded(built):
    dataset = csv(built, PREDICTIVE, "dataset.csv")
    columns = csv(built, PREDICTIVE, "columns.csv")
    assert list(dataset.columns)[0] == "Patient Lot/Batch #"
    assert list(dataset.columns)[-1] == "fp_flow_car"
    assert list(columns["Column"]).count("Vector Lot") == 1
    assert columns.loc[columns["Column"] == "Vector Lot", "Role"].iloc[0] == "group"
    # No one-hot or target encoding built out of the grouping column.
    assert not [c for c in dataset.columns if c.startswith("Vector Lot_")]
    assert not [c for c in dataset.columns if c.startswith("patient_lot")]


def test_parameters_that_come_too_late_are_excluded_as_declared(built):
    decisions = csv(built, PREDICTIVE, "decisions.csv")
    late = decisions[decisions["Stage"] == "availability"]
    assert {"Disposition", "Endotoxin (EU/mL)"} <= set(late["Parameter"])
    assert all("prediction point" in reason for reason in late["Reason"])
    dataset = csv(built, PREDICTIVE, "dataset.csv")
    assert "Disposition" not in dataset.columns


def test_every_exported_column_has_exactly_one_dictionary_entry(built):
    for task in (PREDICTIVE, EXPLORATORY):
        dataset, columns = csv(built, task, "dataset.csv"), csv(built, task, "columns.csv")
        assert list(columns["Column"]) == list(dataset.columns)
        assert not columns["Column"].duplicated().any()
        assert columns["Role"].notna().all() and columns["Transformation"].notna().all()


def test_the_transformed_table_can_be_read_back_to_its_parameters(built):
    report = payload(built, EXPLORATORY)
    dictionary = {row[0]: row for row in table_with(report, "Meaning")["rows"]}
    dataset = csv(built, EXPLORATORY, "dataset.csv")
    metadata = {"Patient Lot/Batch #", "Vector Lot", "disposition_is_released"}
    for column in dataset.columns:
        if column not in metadata:
            assert column in dictionary, column
    strategies = {row[2] for row in dictionary.values()}
    assert {"one-hot", "binary", "ordinal"} <= strategies


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def test_each_categorical_parameter_went_to_the_encoder_its_size_calls_for(built):
    ghent = {row[0]: row for row in table_with(payload(built, PREDICTIVE), "Encoder")["rows"]}
    assert ghent[dummy_data.ONE_HOT_PARAMETER][3] == "one_hot"
    # A column with a value per batch identifies it rather than describing it.
    assert ghent["Patient ID"][3] == "skip"
    assert "identifier" in ghent["Patient ID"][4]

    # The Raritan cohort is large enough for the other two encoders to be chosen;
    # over fourteen Ghent training rows, ten incubators are near-identifying.
    raritan = {row[0]: row for row in table_with(payload(built, EXPLORATORY), "Encoder")["rows"]}
    assert raritan[dummy_data.TARGET_ENCODED_PARAMETER][3] == "target"
    assert raritan[dummy_data.HASHED_PARAMETER][3] == "hashing"


def test_target_encoding_is_cross_fitted_and_says_which_folds(built):
    prose = str(section(payload(built, EXPLORATORY), "parameters")["blocks"])
    assert "fold's training rows alone" in prose

    dataset = csv(built, EXPLORATORY, "dataset.csv")
    raw = csv(built, EXPLORATORY, "raw_features.csv")
    recipe = json.loads(artifact(built, EXPLORATORY, "recipe.json").read_text(encoding="utf-8"))
    encoded = [c for c in dataset.columns if c.endswith("_target_enc")]
    assert encoded

    for source, fitted in recipe["target_encoding"]["columns"].items():
        column = fitted["name"]
        if column not in dataset.columns:
            continue
        # The value a row got is not the value the whole-cohort fit would give
        # it: that is the difference between cross-fitted and leaked.
        whole_cohort = raw[source].astype(str).map(fitted["means"])
        assert not whole_cohort.equals(dataset[column]), column
        assert dataset[column].corr(dataset["disposition_is_released"]) < 0.99


def test_an_ordinal_parameter_becomes_a_rank_and_strays_are_left_missing(built):
    ranks = csv(built, EXPLORATORY, "dataset.csv")[dummy_data.ORDINAL_PARAMETER].dropna()
    assert set(ranks) <= set(range(1, len(dummy_data.ORDINAL_ORDER) + 1))


# ---------------------------------------------------------------------------
# Clusters
# ---------------------------------------------------------------------------
def test_report_only_clustering_leaves_the_predictors_alone(built):
    dataset = csv(built, PREDICTIVE, "dataset.csv")
    membership = csv(built, PREDICTIVE, "clusters.csv")
    assert manifest(built, PREDICTIVE)["counts"]["clusters"] > 0
    assert not [c for c in dataset.columns if c.endswith("_resid")]
    clustered = membership[membership["Action"] == "report_only"]["Parameter"]
    kept = set(dataset.columns)
    assert any(name in kept for name in clustered), "the parameters are still there, unchanged"


def test_linear_residuals_name_the_parameters_they_were_fitted_on(built):
    membership = csv(built, EXPLORATORY, "clusters.csv")
    residualised = membership[membership["Outcome"].astype(str).str.startswith("OLS on")]
    assert not residualised.empty
    assert residualised["Regressors"].notna().all()

    recipe = json.loads(artifact(built, EXPLORATORY, "recipe.json").read_text(encoding="utf-8"))
    for cluster in recipe["clusters"]["clusters"]:
        assert cluster["representative"] in set(cluster["members"])
        for fit in cluster["fits"]:
            if fit["fallback"]:
                continue
            assert fit["regressors"], fit
            assert len(fit["coefficients"]) == len(fit["regressors"])


def test_every_clustered_parameter_is_in_the_membership_export(built):
    membership = csv(built, EXPLORATORY, "clusters.csv")
    report = payload(built, EXPLORATORY)
    summary = table_with(report, "Representative")
    clusters = {row[0] for row in summary["rows"]}
    assert clusters <= set(membership["Cluster"].dropna())
    # Singletons and skipped parameters are in there too, not only members.
    assert (membership["Action"] == "skipped").any()
    assert (membership["Cluster"].isna() | (membership["Cluster"] == "")).any()


def test_a_dropped_duplicate_points_at_a_parameter_that_survived(built):
    decisions = csv(built, EXPLORATORY, "decisions.csv")
    dataset = csv(built, EXPLORATORY, "dataset.csv")
    duplicates = decisions[decisions["Stage"] == "near-duplicate"]
    assert not duplicates.empty
    for reason in duplicates["Reason"]:
        kept = reason.split("duplicates '")[1].split("'")[0]
        assert kept in dataset.columns or f"{kept}_resid" in dataset.columns, kept


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------
def test_the_report_offers_the_files_the_package_holds(built):
    offered = downloads(payload(built, PREDICTIVE))
    assert {"cohort.csv", "columns.csv", "decisions.csv", "clusters.csv"} <= set(offered)
    on_disk = artifact(built, PREDICTIVE, "decisions.csv").read_text(encoding="utf-8")
    assert offered["decisions.csv"]["text"].splitlines()[0] == on_disk.splitlines()[0]
    assert len(offered["decisions.csv"]["text"].splitlines()) == len(on_disk.splitlines())


def test_the_heavy_content_waits_to_be_asked_for(built):
    report = payload(built, EXPLORATORY)
    clusters = section(report, "clusters")
    picker = next(b for b in clusters["blocks"] if b["kind"] == "chooser")
    assert len(picker["options"]) == manifest(built, EXPLORATORY)["counts"]["clusters"]
    # Figures live inside the picker, so only the chosen cluster's are drawn.
    drawn = [b for b in clusters["blocks"] if b["kind"] == "figure"]
    assert not drawn
    assert figures(report), "they are in the payload, just not all on screen"


def test_the_report_does_not_claim_a_model_was_evaluated(built):
    for task in (PREDICTIVE, EXPLORATORY):
        prose = str(payload(built, task)["sections"]).lower()
        assert "no model was fitted" in prose
        assert "accuracy" not in prose.replace("flow accuracy", "")


def test_the_report_states_what_the_table_is_for(built):
    predictive = facts(first_table(payload(built, PREDICTIVE), "overview"))
    assert "raw modelling inputs" in predictive["What the table is"]
    exploratory = facts(first_table(payload(built, EXPLORATORY), "overview"))
    assert "exploratory" in exploratory["What the table is"]


def test_provenance_records_the_inputs_this_stage_actually_read(built):
    report = payload(built, PREDICTIVE)
    inputs = {row[0]: row for row in table_with(report, "Where it was read from")["rows"]}
    assert set(inputs) == {"pvf", "ptf"}
    assert all(len(row[3]) == 64 for row in inputs.values())
    log = artifact(built, PREDICTIVE, "run.log").read_text(encoding="utf-8")
    assert "Task 'ghent_car_expression'" in log
    assert "PVF build" not in log, "an earlier stage's events do not belong to this run"


def test_the_report_is_one_file_with_its_runtime_named(built):
    from pvf import stlite

    page = artifact(built, PREDICTIVE, "report.html").read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>")
    assert stlite.STLITE_VERSION in page
    assert "could not start" in page, "a boot failure has to say something useful"
    embedded = stlite.read_payload(page)
    assert embedded == payload(built, PREDICTIVE)


def test_the_figures_are_drawn_and_carry_their_data(built):
    drawn = figures(payload(built, EXPLORATORY))
    assert len(drawn) >= 3
    assert all("data" in figure["spec"] for figure in drawn)


def test_the_dataset_section_describes_the_table_on_disk(built):
    report = payload(built, EXPLORATORY)
    stated = facts(first_table(report, "dataset"))
    dataset = csv(built, EXPLORATORY, "dataset.csv")
    assert f"{len(dataset):,} batches" in stated["Shape"]
    assert str(dataset.shape[1]) in stated["Shape"].replace(",", "")
    assert pd.api.types.is_numeric_dtype(dataset["disposition_is_released"])


def test_a_task_names_itself_in_its_report(built):
    assert "ghent_car_expression" in payload(built, PREDICTIVE)["title"]
    assert rows_naming(payload(built, PREDICTIVE), "Which manufacturing parameters")


# ---------------------------------------------------------------------------
# The upstream manifests, and the new report sections
# ---------------------------------------------------------------------------
def test_the_task_carries_the_ptf_and_pvf_manifests_of_its_input(built):
    from pvf import provenance

    record = manifest(built, PREDICTIVE)
    assert record["upstream_manifests"] == {
        "ptf": "provenance/ptf_manifest.json",
        "pvf": "provenance/pvf_manifest.json",
    }
    pvf = json.loads(artifact(built, PREDICTIVE, "pvf_manifest.json").read_text(encoding="utf-8"))
    ptf = json.loads(artifact(built, PREDICTIVE, "ptf_manifest.json").read_text(encoding="utf-8"))
    # The PVF manifest is about the very file this task read.
    assert pvf["pvf"]["sha256"] == provenance.file_sha256(built["config"]["paths"]["pvf"])
    assert pvf["pvf"]["rows"] == len(built["pvf"])
    missing = set(pvf["ptf_parameters_not_in_pvf"])
    assert missing and not missing & set(built["pvf"].columns)
    assert "Local Comment" in ptf["not_in_ptf"]["Ghent PHF"]
    assert not any("manifest" in warning for warning in record["warnings"])


def test_the_task_folder_is_split_into_subfolders(built):
    folder = built["packages"][PREDICTIVE]
    top = {path.name for path in folder.iterdir()}
    assert top == {"manifest.json", "config", "data", "metadata", "report", "provenance", "logs"}
    names = {entry["name"] for entry in manifest(built, PREDICTIVE)["artifacts"]}
    assert "data/dataset.csv" in names and "metadata/clusters.csv" in names


def test_the_report_shows_every_cluster_not_only_the_first(built):
    report = payload(built, PREDICTIVE)
    clusters = section(report, "clusters")
    count = manifest(built, PREDICTIVE)["counts"]["clusters"]
    summary = table_with(report, "Max VIF before")
    assert len(summary["rows"]) == count
    every_member = table_with(report, "Output column")
    assert {row[0] for row in every_member["rows"]} == {row[0] for row in summary["rows"]}
    picker = next(b for b in clusters["blocks"] if b["kind"] == "chooser")
    assert len(picker["options"]) == count
    assert all(option["blocks"][0]["kind"] == "md" for option in picker["options"])
