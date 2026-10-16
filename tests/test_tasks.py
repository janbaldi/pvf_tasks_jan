"""The task packages, and the reports that explain them.

A task narrows the batch table to one cohort and one target and writes a folder someone
else has to be able to use without asking what is in it. What matters is that
the narrowing is visible, that the target and everything computed from it stay
out of the predictors, that every exported column can be read back to the
parameter it came from, and that the counts in the report are the counts in the
files.
"""

from __future__ import annotations

import json

import pandas as pd
import yaml

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
from dummy_data import c, v, vs
from ml import names
from ml.encode import sanitise

REQUIRED = (
    "task.yaml",
    "dataset.csv",
    "dataset_unencoded.csv",
    "dataset_columns.csv",
    "encoded_columns.csv",
    "parameters.csv",
    "batches.csv",
    "clusters.csv",
    "encoding.json",
    "groups.json",
    "report.html",
    "report_payload.json",
    "manifest.json",
    "run.log",
    "catalog_manifest.json",
    "batch_table_manifest.json",
)


# ---------------------------------------------------------------------------
# The package
# ---------------------------------------------------------------------------
def test_a_finished_task_folder_holds_everything_it_promises(built):
    for task in (PREDICTIVE, EXPLORATORY):
        for name in REQUIRED:
            assert artifact(built, task, name).exists(), f"{task}/{name}"
        assert manifest(built, task)["status"] == "complete"
        # dataset.csv is the encoded predictors for every task; the raw one is beside it.
        assert not artifact(built, task, "dataset_encoded.csv").exists()
        assert manifest(built, task)["dataset_role"].startswith("encoded modelling inputs")


#: The release task's binary target, as the dataset stage names it.
RELEASE_TARGET = f"{sanitise(c('disposition'))}_is_{sanitise(v('released'))}"


def test_the_manifest_hashes_every_artefact_but_itself(built):
    recorded = manifest(built, PREDICTIVE)["outputs"]
    names = {entry["path"] for entry in recorded}
    assert "manifest.json" not in names
    assert {"data/dataset.csv", "report/report.html", "metadata/dataset_columns.csv"} <= names
    assert all(len(entry["sha256"]) == 64 for entry in recorded)


def test_the_counts_agree_wherever_they_are_written(built):
    for task in (PREDICTIVE, EXPLORATORY):
        counts = manifest(built, task)["counts"]
        dataset = csv(built, task, "dataset.csv")
        unencoded = csv(built, task, "dataset_unencoded.csv")
        columns = csv(built, task, "dataset_columns.csv")
        assert counts["dataset_rows"] == len(dataset) == len(unencoded)
        assert counts["dataset_columns"] == dataset.shape[1]
        assert counts["unencoded_columns"] == unencoded.shape[1]
        assert list(columns["Column"]) == list(unencoded.columns)

        overview = facts(first_table(payload(built, task), "overview"))
        assert int(overview["Batches in the dataset"].replace(",", "")) == len(dataset)

        funnel = first_table(payload(built, task), "cohort")["rows"]
        assert int(funnel[-1][-1].replace(",", "")) == len(dataset)


def test_two_runs_of_one_task_do_not_overwrite_each_other(built):
    from ml import cli, taskconfig

    spec = taskconfig.load(built["directory"] / dummy_data.TASK_FILES[0])
    again = cli.run_task(built["config"], spec)
    first = built["packages"][PREDICTIVE]
    assert again != first
    assert first.exists() and (first / cli.TASK_LAYOUT["dataset.csv"]).exists()
    assert again.parent == first.parent


# ---------------------------------------------------------------------------
# Cohort and target
# ---------------------------------------------------------------------------
def test_the_cohort_is_narrower_than_the_batch_table_and_says_how(built):
    dataset, batch_table = csv(built, PREDICTIVE, "dataset.csv"), built["batch_table"]
    assert len(dataset) < len(batch_table)

    steps = first_table(payload(built, PREDICTIVE), "cohort")["rows"]
    left = [int(row[-1].replace(",", "")) for row in steps]
    assert left == sorted(left, reverse=True), "each filter can only remove batches"
    assert left[0] == len(batch_table)
    assert left[-1] == len(dataset)


def test_withdrawals_are_excluded_however_the_source_spells_them(built):
    batch_table = built["batch_table"]
    kept = set(csv(built, PREDICTIVE, "dataset.csv")["Patient Lot/Batch #"])
    first = batch_table[batch_table["Site Merged"] == names.site("a")]
    spellings = first[c("nonconformance")].astype("string")
    assert {vs("status_fix")[0], vs("nonconformance_raw")[5]} <= set(spellings.dropna()), (
        "the fixture has both spellings"
    )

    is_withdrawal = (
        spellings.str.lower().isin({vs("status_fix")[0], vs("status_fix")[1]}).fillna(False)
    )
    withdrawn = set(first.loc[is_withdrawal, "Patient Lot/Batch #"])
    assert withdrawn, "there are withdrawals to exclude"
    assert not (withdrawn & kept), "a withdrawal is a withdrawal however it is written"


def test_batches_without_a_target_are_counted_and_named_separately(built):
    report = payload(built, PREDICTIVE)
    reasons = table_with(report, "Why a batch is not in the dataset")
    assert any("target" in str(row[0]) for row in reasons["rows"])

    exclusions = csv(built, PREDICTIVE, "batches.csv")
    assert len(exclusions) == len(built["batch_table"])
    assert exclusions["Included"].sum() == len(csv(built, PREDICTIVE, "dataset.csv"))
    assert exclusions.loc[~exclusions["Included"], "Reason"].notna().all()


def test_a_balanced_binary_target_maps_both_classes(built):
    dataset = csv(built, EXPLORATORY, "dataset.csv")
    target = dataset[RELEASE_TARGET]
    assert set(target.dropna().unique()) == {0, 1}
    assert target.notna().all()
    # The declared positive class is 1, whether or not it is the rarer one.
    counts = target.value_counts()
    assert counts[1] > 0 and counts[0] > 0


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------
def test_the_target_and_everything_computed_from_it_stay_out(built):
    dataset = csv(built, PREDICTIVE, "dataset_unencoded.csv")
    decisions = csv(built, PREDICTIVE, "parameters.csv")

    for leaked in (c("f_vcn_per_cell"), c("f_flow_accuracy"), dummy_data.target()):
        assert leaked not in dataset.columns
    lineage = decisions[decisions["Stage"] == "target lineage"]
    assert {c("f_vcn_per_cell"), c("f_flow_accuracy")} <= set(lineage["Parameter"])
    assert all(dummy_data.target() in reason for reason in lineage["Reason"])


def test_roles_are_metadata_and_are_not_encoded(built):
    dataset = csv(built, PREDICTIVE, "dataset.csv")
    columns = csv(built, PREDICTIVE, "dataset_columns.csv")
    assert list(dataset.columns)[0] == "Patient Lot/Batch #"
    assert list(dataset.columns)[-1] == dummy_data.target_column()
    assert list(columns["Column"]).count(c("vector_lot")) == 1
    assert columns.loc[columns["Column"] == c("vector_lot"), "Role"].iloc[0] == "group"
    # No one-hot or target encoding built out of the grouping column.
    assert not [name for name in dataset.columns if name.startswith(c("vector_lot") + "_")]
    assert not [c for c in dataset.columns if c.startswith("patient_lot")]


def test_parameters_in_an_excluded_catalog_category_are_left_out(built):
    decisions = csv(built, PREDICTIVE, "parameters.csv")
    late = decisions[decisions["Stage"] == "category exclusion"]
    assert {c("disposition"), c("endotoxin")} <= set(late["Parameter"])
    assert all(reason.startswith("Catalog 'Available At' is") for reason in late["Reason"])
    dataset = csv(built, PREDICTIVE, "dataset_unencoded.csv")
    assert c("disposition") not in dataset.columns


def test_every_exported_column_has_exactly_one_dictionary_entry(built):
    for task in (PREDICTIVE, EXPLORATORY):
        dataset = csv(built, task, "dataset_unencoded.csv")
        columns = csv(built, task, "dataset_columns.csv")
        assert list(columns["Column"]) == list(dataset.columns)
        assert not columns["Column"].duplicated().any()
        assert columns["Role"].notna().all() and columns["Transformation"].notna().all()


def test_the_encoded_table_can_be_read_back_to_its_parameters(built):
    report = payload(built, EXPLORATORY)
    dictionary = {row[0]: row for row in table_with(report, "Meaning")["rows"]}
    dataset = csv(built, EXPLORATORY, "dataset.csv")
    metadata = {"Patient Lot/Batch #", c("vector_lot"), RELEASE_TARGET}
    for column in dataset.columns:
        if column not in metadata:
            assert column in dictionary, column
    strategies = {row[2] for row in dictionary.values()}
    assert {"one-hot", "binary", "ordinal"} <= strategies


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def test_each_categorical_parameter_went_to_the_encoder_its_size_calls_for(built):
    first = {row[0]: row for row in table_with(payload(built, PREDICTIVE), "Encoder")["rows"]}
    assert first[c("clinical_site")][3] == "one_hot"
    # A column with a value per batch identifies it rather than describing it.
    assert first[c("patient_id")][3] == "skip"
    assert "identifier" in first[c("patient_id")][4]

    # Site B's cohort is large enough for the other two encoders to be chosen;
    # over fourteen site A training rows, ten incubators are near-identifying.
    second = {row[0]: row for row in table_with(payload(built, EXPLORATORY), "Encoder")["rows"]}
    assert second[c("team")][3] == "target"
    assert second[c("incubator_id")][3] == "hashing"


def test_target_encoding_is_cross_fitted_and_says_which_folds(built):
    prose = str(section(payload(built, EXPLORATORY), "parameters")["blocks"])
    assert "fold's training rows alone" in prose

    dataset = csv(built, EXPLORATORY, "dataset.csv")
    raw = csv(built, EXPLORATORY, "dataset_unencoded.csv")
    recipe = json.loads(artifact(built, EXPLORATORY, "encoding.json").read_text(encoding="utf-8"))
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
        assert dataset[column].corr(dataset[RELEASE_TARGET]) < 0.99


def test_an_ordinal_parameter_becomes_a_rank_and_strays_are_left_missing(built):
    ranks = csv(built, EXPLORATORY, "dataset.csv")[c("ordinal_parameter")].dropna()
    assert set(ranks) <= set(range(1, len(dummy_data.ORDINAL_ORDER) + 1))


# ---------------------------------------------------------------------------
# Clusters
# ---------------------------------------------------------------------------
def test_report_only_clustering_leaves_the_predictors_alone(built):
    dataset = csv(built, PREDICTIVE, "dataset.csv")
    membership = csv(built, PREDICTIVE, "clusters.csv")
    assert manifest(built, PREDICTIVE)["counts"]["clusters"] > 0
    assert not [c for c in dataset.columns if c.endswith("_resid")]
    clustered = membership[membership["Action"] == "describe"]["Parameter"]
    kept = set(csv(built, PREDICTIVE, "dataset_unencoded.csv").columns)
    assert any(name in kept for name in clustered), "the parameters are still there, unchanged"


def test_a_summarised_group_becomes_one_score_fed_by_every_member(built):
    membership = csv(built, EXPLORATORY, "clusters.csv")
    summarised = membership[membership["Outcome"] == "summarised into the group score"]
    assert not summarised.empty
    assert (summarised["Output column"].str.endswith(" group")).all()
    assert (summarised["Stability"] >= 0.8).all(), "only stable groups are acted on"

    encoded = csv(built, EXPLORATORY, "dataset.csv")
    raw = csv(built, EXPLORATORY, "dataset_unencoded.csv")
    assert set(summarised["Output column"]) <= set(encoded.columns)
    # The score needs every member, so every member is a raw input of the dataset.
    assert set(summarised["Parameter"]) <= set(raw.columns)
    changed = csv(built, EXPLORATORY, "parameters.csv").query("Decision == 'changed'")
    assert set(summarised["Parameter"]) <= set(changed["Parameter"])


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
    decisions = csv(built, EXPLORATORY, "parameters.csv")
    dataset = csv(built, EXPLORATORY, "dataset_unencoded.csv")
    duplicates = decisions[decisions["Stage"] == "near-duplicate"]
    assert not duplicates.empty
    for reason in duplicates["Reason"]:
        kept = reason.split("duplicates '")[1].split("'")[0]
        assert kept in dataset.columns, kept


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------
def test_the_report_offers_the_files_the_package_holds(built):
    offered = downloads(payload(built, PREDICTIVE))
    assert {"batches.csv", "dataset_columns.csv", "parameters.csv", "clusters.csv"} <= set(offered)
    on_disk = artifact(built, PREDICTIVE, "parameters.csv").read_text(encoding="utf-8")
    assert offered["parameters.csv"]["text"].splitlines()[0] == on_disk.splitlines()[0]
    assert len(offered["parameters.csv"]["text"].splitlines()) == len(on_disk.splitlines())


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
    for task in (PREDICTIVE, EXPLORATORY):
        overview = facts(first_table(payload(built, task), "overview"))
        assert "encoded modelling inputs" in overview["What the table is"]
        assert "data/dataset\\_unencoded.csv" in overview["What the table is"]  # markdown


def test_provenance_records_the_inputs_this_stage_actually_read(built):
    report = payload(built, PREDICTIVE)
    rows = table_with(report, "Where it was read from")["rows"]
    inputs = {row[0].replace("\\_", "_"): row for row in rows}
    assert set(inputs) == {"batch_table", "catalog"}
    assert all(len(row[3]) == 64 for row in inputs.values())
    log = artifact(built, PREDICTIVE, "run.log").read_text(encoding="utf-8")
    assert "Task 'expression'" in log
    assert "Batch table build" not in log, "an earlier stage's events do not belong to this run"


def test_the_report_is_one_file_with_its_runtime_named(built):
    from ml import stlite

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
    assert pd.api.types.is_numeric_dtype(dataset[RELEASE_TARGET])


def test_a_task_names_itself_in_its_report(built):
    assert "expression" in payload(built, PREDICTIVE)["title"]
    assert rows_naming(payload(built, PREDICTIVE), "Which manufacturing parameters")


# ---------------------------------------------------------------------------
# The upstream manifests, and the new report sections
# ---------------------------------------------------------------------------
def test_the_task_carries_the_catalog_and_batch_table_manifests_of_its_input(built):

    record = manifest(built, PREDICTIVE)
    assert record["upstream_manifests"] == {
        "catalog": "inputs/catalog_manifest.json",
        "batch_table": "inputs/batch_table_manifest.json",
    }
    batch_table = json.loads(
        artifact(built, PREDICTIVE, "batch_table_manifest.json").read_text(encoding="utf-8")
    )
    catalog = json.loads(
        artifact(built, PREDICTIVE, "catalog_manifest.json").read_text(encoding="utf-8")
    )
    # The batch table manifest is about the very table this task read.
    read = {i["label"]: i["digest"] for i in record["inputs"]}
    assert batch_table["batch_table"]["content_sha256"] == read["batch_table"]
    assert batch_table["batch_table"]["rows"] == len(built["batch_table"])
    missing = set(batch_table["catalog_parameters_not_in_batch_table"])
    assert missing and not missing & set(built["batch_table"].columns)
    assert c("site_a_comment") in catalog["not_in_catalog"][f"{names.site('a')} batch records"]
    assert not any("manifest" in warning for warning in record["warnings"])


def test_the_task_folder_is_split_into_subfolders(built):
    folder = built["packages"][PREDICTIVE]
    top = {path.name for path in folder.iterdir()}
    assert top == {"manifest.json", "config", "data", "metadata", "report", "inputs", "logs"}
    assert folder.parent == built["directory"] / dummy_data.TASK_FILES[0] / "runs"
    names = {entry["path"] for entry in manifest(built, PREDICTIVE)["outputs"]}
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


# ---------------------------------------------------------------------------
# Two periods compared
# ---------------------------------------------------------------------------
def test_a_task_can_compare_two_periods_of_batches(built, tmp_path):
    """Batches from one date window against another: the second is 1, the first 0."""
    from ml import cli, taskconfig

    made = c("incubator_in_d0")  # the demo's batches start between 1 January and 20 February
    before, after = ("2026-01-01", "2026-01-10"), ("2026-02-10", "2026-02-20")

    task = yaml.safe_load((built["directory"] / dummy_data.TASK_FILES[0] / "task.yaml").read_text())
    task["config"] = str(built["config_path"])
    task["task"]["name"] = "periods"
    task["target"] = {
        "column": made,
        "type": "periods",
        "periods": [
            {"name": "before", "start": before[0], "stop": before[1]},
            {"name": "after", "start": after[0], "stop": after[1]},
        ],
    }
    (tmp_path / "task.yaml").write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")
    folder = cli.run_task(built["config"], taskconfig.load(tmp_path))

    table = pd.read_csv(folder / cli.TASK_LAYOUT["dataset_unencoded.csv"])
    assert set(table["period_is_after"]) == {0, 1}
    assert made not in table.columns, "the date that defines the periods is not a predictor"
    batches = pd.read_csv(folder / cli.TASK_LAYOUT.get("batches.csv", "batches.csv"))
    reasons = set(batches.loc[~batches["Included"], "Reason"])
    assert f"{made} is outside both periods" in reasons
