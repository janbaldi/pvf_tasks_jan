"""The failure modes worth a test of their own.

Each of these covers something the end-to-end run cannot show on its own: a
leak only visible by perturbing a target, a join that multiplies rows, a
clustering decision built on four shared batches, a package that must not look
finished when it is not. Small frames, no fixtures beyond what a case needs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dummy_data  # noqa: E402
from pvf import (  # noqa: E402
    cli,
    dataset,
    decorrelate,
    encode,
    enrich,
    io,
    package,
    provenance,
    taskconfig,
)
from pvf.encode import TaskParams  # noqa: E402


def params(**overrides) -> TaskParams:
    base = dict(target_encoding_smoothing=2.0, target_encoding_folds=4, seed=7)
    base.update(overrides)
    return TaskParams(**base)


# ---------------------------------------------------------------------------
# The task file is the contract
# ---------------------------------------------------------------------------
def minimal_task(**overrides) -> dict:
    task = {
        "task": {"name": "probe", "purpose": "exploratory", "seed": 1},
        "inputs": {"pvf": "PVF.xlsx", "ptf": "PTF.xlsx"},
        "output": {"root": "tasks"},
        "cohort": {"sites": ["Ghent"]},
        "target": {"column": "FP Flow CAR+ (%)"},
        "columns": {"id": "Patient Lot/Batch #", "group": "Vector Lot"},
    }
    for section, values in overrides.items():
        task.setdefault(section, {}).update(values)
    return task


def write_task(tmp_path: Path, task: dict, name: str = "task.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")
    return path


def test_a_column_name_with_a_hash_in_it_survives_the_yaml(tmp_path):
    """The bug this covers: an unquoted `Patient Lot/Batch #` is a comment."""
    path = write_task(tmp_path, minimal_task())
    assert taskconfig.load(path).roles.id == "Patient Lot/Batch #"
    assert "'Patient Lot/Batch #'" in path.read_text(encoding="utf-8")


def test_paths_resolve_against_the_task_file_wherever_it_is_run_from(tmp_path, monkeypatch):
    path = write_task(tmp_path, minimal_task())
    monkeypatch.chdir(tmp_path.parent)
    spec = taskconfig.load(path)
    assert spec.pvf == tmp_path / "PVF.xlsx"
    assert spec.output_root == tmp_path / "tasks"

    absolute = minimal_task()
    absolute["inputs"]["pvf"] = str(tmp_path / "elsewhere" / "PVF.xlsx")
    spec = taskconfig.load(write_task(tmp_path, absolute, "absolute.yaml"))
    assert spec.pvf == tmp_path / "elsewhere" / "PVF.xlsx"


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"task": {"purpose": "magic"}}, "task.purpose"),
        ({"target": {"type": "binary"}}, "target.positive_class"),
        ({"clustering": {"action": "pca"}}, "clustering.action"),
        ({"clustering": {"threshold": 2.5}}, "clustering.threshold"),
        ({"clustering": {"threshold": float("inf")}}, "clustering.threshold"),
        ({"quality": {"min_non_missing": 0}}, "quality.min_non_missing"),
        ({"encoding": {"target_encoding_folds": 1}}, "encoding.target_encoding_folds"),
        ({"cohort": {"filters": [{"column": "Type", "op": "matches", "value": "x"}]}}, "op"),
        ({"columns": {"include": ["Vector Lot"]}}, "columns.include"),
        ({"split": {"strategy": "chronological"}}, "split.order_column"),
        ({"task": {"unknown_setting": 1}}, "unknown setting"),
    ],
)
def test_a_task_that_cannot_be_run_is_refused_by_field(tmp_path, overrides, expected):
    path = write_task(tmp_path, minimal_task(**overrides))
    with pytest.raises(taskconfig.TaskConfigError) as raised:
        taskconfig.load(path)
    assert expected in str(raised.value)


def test_validation_happens_before_anything_is_written(tmp_path):
    bad = minimal_task(clustering={"action": "pca"})
    bad["output"]["root"] = str(tmp_path / "out")
    with pytest.raises(taskconfig.TaskConfigError):
        taskconfig.load(write_task(tmp_path, bad))
    assert not (tmp_path / "out").exists()


def test_a_predictive_task_has_to_have_been_reviewed(tmp_path):
    task = minimal_task(task={"purpose": "predictive"})
    with pytest.raises(taskconfig.TaskConfigError, match="availability.reviewed"):
        taskconfig.load(write_task(tmp_path, task))

    task["availability"] = {"reviewed": True, "cutoff": "harvest"}
    assert taskconfig.load(write_task(tmp_path, task, "reviewed.yaml")).predictive


def test_the_old_config_still_describes_a_task(tmp_path):
    config = {
        "seed": 3,
        "paths": {"pvf": "PVF.xlsx", "ptf": "PTF.xlsx", "tasks": "out"},
        "features": {},
        "tasks": {
            "site": "Ghent",
            "target": "FP Flow CAR+ (%)",
            "id_column": "Patient Lot/Batch #",
            "group_column": "Vector Lot",
            "decorrelation": "off",
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    spec = taskconfig.load(path)
    assert spec.roles.id == "Patient Lot/Batch #"
    assert spec.clustering.action == "off"
    # The cohort rules that used to be written into dataset.py are declared now.
    assert [f.column for f in spec.cohort.filters] == [
        "Type",
        "Manufacturing and Release Testing Completed? (Y/N)",
        "Non-Conformance Type",
    ]


# ---------------------------------------------------------------------------
# Cohort and target
# ---------------------------------------------------------------------------
def probe_spec(**overrides) -> taskconfig.TaskSpec:
    return taskconfig.parse(minimal_task(**overrides), Path("."), source="probe")


def probe_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Site Merged": ["Ghent"] * 6,
            "Patient Lot/Batch #": [f"B{i}" for i in range(6)],
            "Vector Lot": ["LV-1", "LV-2"] * 3,
            "Type": ["Commercial"] * 6,
            "Non-Conformance Type": [
                "withdrawn",
                "Withdrawal",
                None,
                "deviation",
                " WITHDRAWN ",
                None,
            ],
            "FP Flow CAR+ (%)": [30.0, 31.0, 32.0, 33.0, 34.0, 35.0],
        }
    )


def test_a_filter_whose_column_is_absent_stops_the_task(tmp_path):
    spec = probe_spec(
        cohort={"filters": [{"column": "Disposition", "op": "equals", "value": "Released"}]}
    )
    with pytest.raises(dataset.TaskRefused, match="Disposition"):
        dataset.select_cohort(probe_frame(), spec, dataset.TaskResult(spec=spec))


@pytest.mark.parametrize("spelling", ["withdrawn", "Withdrawal", " WITHDRAWN "])
def test_every_spelling_of_a_withdrawal_is_excluded_once_normalised(spelling):
    spec = probe_spec(
        cohort={
            "filters": [
                {
                    "column": "Non-Conformance Type",
                    "op": "not_in",
                    "value": ["withdrawal"],
                    "normalise": True,
                    "missing": "include",
                }
            ]
        }
    )
    result = dataset.TaskResult(spec=spec)
    cohort = dataset.select_cohort(probe_frame(), spec, result)
    assert spelling not in set(cohort["Non-Conformance Type"].dropna())
    assert len(cohort) == 3  # two blanks kept by the missing policy, one deviation


def test_a_missing_value_follows_the_declared_policy():
    for policy, expected in (("include", 3), ("exclude", 1)):
        spec = probe_spec(
            cohort={
                "filters": [
                    {
                        "column": "Non-Conformance Type",
                        "op": "not_in",
                        "value": ["withdrawal"],
                        "normalise": True,
                        "missing": policy,
                    }
                ]
            }
        )
        cohort = dataset.select_cohort(probe_frame(), spec, dataset.TaskResult(spec=spec))
        assert len(cohort) == expected, policy


@pytest.mark.parametrize(
    "values",
    [
        ["Pass", "Fail", "Pass", "Fail"],  # balanced: the old code lost one class
        ["Pass", "Pass", "Pass", "Fail"],
    ],
)
def test_a_binary_target_maps_both_classes_whatever_the_balance(values):
    spec = probe_spec(target={"type": "binary", "positive_class": "Pass", "column": "outcome"})
    frame = pd.DataFrame({"outcome": values})
    target, labelled = dataset.prepare_target(frame, spec, dataset.TaskResult(spec=spec))
    assert len(labelled) == len(values)
    assert set(target.dropna().unique()) == {0, 1}
    assert list(target) == [1 if v == "Pass" else 0 for v in values]


def test_a_target_nobody_measured_cannot_become_a_dataset():
    spec = probe_spec()
    frame = pd.DataFrame({"FP Flow CAR+ (%)": [None, None, "not measured"]})
    with pytest.raises(dataset.TaskRefused, match="no batch|No batch"):
        dataset.prepare_target(frame, spec, dataset.TaskResult(spec=spec))


def test_a_target_class_nobody_declared_is_refused():
    spec = probe_spec(
        target={"type": "binary", "positive_class": "Pass", "negative_class": "Fail", "column": "o"}
    )
    frame = pd.DataFrame({"o": ["Pass", "Fail", "Pending"]})
    with pytest.raises(dataset.TaskRefused, match="Pending"):
        dataset.prepare_target(frame, spec, dataset.TaskResult(spec=spec))


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------
def encoding_frame(n: int = 24) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"team": [("A", "B", "C", "D")[i % 4] for i in range(n)]})
    target = pd.Series(rng.normal(10, 2, n))
    return frame, target


def test_a_folds_encoding_does_not_move_when_its_own_targets_do():
    frame, target = encoding_frame()
    state, _ = encode.fit_target(frame, ["team"], target, params(), set())
    encoded, _ = encode.crossfit_target(frame, state, target, params())

    pairs, _, _ = encode.folds(target.index, params().target_encoding_folds, params().seed)
    _, held_out = pairs[0]
    moved = target.copy()
    moved.loc[held_out] += 100  # the values a leak would carry into the encoding

    state_again, _ = encode.fit_target(frame, ["team"], moved, params(), set())
    again, _ = encode.crossfit_target(frame, state_again, moved, params())
    column = state["columns"]["team"]["name"]
    pd.testing.assert_series_equal(
        encoded.loc[held_out, column], again.loc[held_out, column], check_names=False
    )


def test_what_the_training_rows_learned_does_not_move_with_the_holdout():
    frame, target = encoding_frame()
    training, holdout = frame.index[:16], frame.index[16:]
    state, _ = encode.fit_target(
        frame.loc[training], ["team"], target.loc[training], params(), set()
    )

    moved = target.copy()
    moved.loc[holdout] *= -5
    again, _ = encode.fit_target(
        frame.loc[training], ["team"], moved.loc[training], params(), set()
    )
    assert state["columns"]["team"]["means"] == again["columns"]["team"]["means"]
    assert state["prior"] == again["prior"]


def test_a_category_the_training_rows_never_saw_gets_the_training_prior():
    frame, target = encoding_frame()
    state, _ = encode.fit_target(frame, ["team"], target, params(), set())
    new_rows = pd.DataFrame({"team": ["A", "Z", None]})
    encoded, records = encode.apply_target(new_rows, state, params())
    column = state["columns"]["team"]["name"]
    assert encoded[column].iloc[0] == pytest.approx(state["columns"]["team"]["means"]["A"])
    assert encoded[column].iloc[1] == pytest.approx(state["prior"])
    assert pd.isna(encoded[column].iloc[2])
    assert "missing value stays missing" in records[0]["detail"]


def test_grouped_folds_never_split_a_group():
    index = pd.RangeIndex(18)
    groups = pd.Series([f"lot{i % 3}" for i in range(18)], index=index)
    pairs, strategy, _ = encode.folds(index, 5, 1, groups=groups)
    assert strategy == "grouped"
    for train, test in pairs:
        assert not set(groups.loc[train]) & set(groups.loc[test])


def test_too_few_groups_is_said_rather_than_worked_around():
    index = pd.RangeIndex(6)
    groups = pd.Series(["only"] * 6, index=index)
    pairs, strategy, note = encode.folds(index, 3, 1, groups=groups)
    assert pairs == [] and strategy == "grouped" and "at least two" in note


def test_a_chronological_fold_never_looks_forward():
    index = pd.RangeIndex(12)
    order = pd.Series(pd.date_range("2026-01-01", periods=12), index=index)
    pairs, strategy, note = encode.folds(index, 4, 1, order=order)
    assert strategy == "chronological" and "earliest block" in note
    for train, test in pairs:
        assert order.loc[train].max() < order.loc[test].min()


def test_a_grouped_split_keeps_whole_groups_on_one_side():
    spec = probe_spec(split={"strategy": "grouped", "validation_fraction": 0.34})
    frame = probe_frame()
    result = dataset.TaskResult(spec=spec)
    training, validation = dataset.assign_split(frame, frame.index, spec, result)
    lots = frame["Vector Lot"]
    assert not set(lots.loc[training]) & set(lots.loc[validation])
    assert result.split["strategy"] == "grouped"


def test_a_chronological_split_takes_the_last_batches():
    spec = probe_spec(
        split={"strategy": "chronological", "validation_fraction": 0.5, "order_column": "when"}
    )
    frame = probe_frame()
    frame["when"] = pd.date_range("2026-01-01", periods=len(frame))
    training, validation = dataset.assign_split(
        frame, frame.index, spec, dataset.TaskResult(spec=spec)
    )
    assert frame.loc[training, "when"].max() < frame.loc[validation, "when"].min()


def test_a_split_that_cannot_be_made_is_refused():
    spec = probe_spec(split={"strategy": "grouped", "validation_fraction": 0.3})
    frame = probe_frame()
    frame["Vector Lot"] = "one lot"
    with pytest.raises(dataset.TaskRefused, match="at least two distinct"):
        dataset.assign_split(frame, frame.index, spec, dataset.TaskResult(spec=spec))


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def test_the_first_column_of_a_pair_keeps_the_name_and_its_own_values():
    frame = pd.DataFrame({"a": [1], "b": [2]})
    mapped = io.apply_column_mapping(frame, {"a": "Country", "b": "Country"})
    assert mapped["Country"].tolist() == [1], "the first source column keeps the plain name"
    assert mapped["Country_1"].tolist() == [2]


def test_a_suffix_that_is_already_taken_is_not_reused():
    frame = pd.DataFrame({"a": [1], "b": [2], "Country_1": [3]})
    mapped = io.apply_column_mapping(frame, {"a": "Country", "b": "Country"})
    assert list(mapped.columns) == ["Country", "Country_2", "Country_1"]
    assert mapped["Country_1"].tolist() == [3]
    assert len(set(mapped.columns)) == mapped.shape[1]


def test_a_mapping_that_contradicts_itself_is_refused(tmp_path):
    path = tmp_path / "mapping.xlsx"
    pd.DataFrame(
        {
            "Parameter Name Raritan (CMD)": ["Weight", "Weight"],
            "Parameter Name Ghent (PHF)": ["Subject Weight (kg)", "Patient Weight"],
        }
    ).to_excel(path, sheet_name="Overall", index=False)
    with pytest.raises(ValueError, match="two different Ghent names"):
        io.load_site_mapping(str(path))


def test_a_lookup_with_two_answers_for_one_batch_is_refused():
    batches = pd.DataFrame({"Patient Lot/Batch #": ["B1", "B2"], "x": [1, 2]})
    lookup = pd.DataFrame({"Patient Lot/Batch #": ["B1", "B1"], "Media Lot": ["M1", "M2"]})
    with pytest.raises(ValueError, match="different records under the same key"):
        enrich._lookup_merge(batches, lookup, "Patient Lot/Batch #", "probe")


def test_an_identical_duplicate_lookup_row_does_not_multiply_batches():
    batches = pd.DataFrame({"Patient Lot/Batch #": ["B1", "B2"], "x": [1, 2]})
    lookup = pd.DataFrame({"Patient Lot/Batch #": ["B1", "B1"], "Media Lot": ["M1", "M1"]})
    joined = enrich._lookup_merge(batches, lookup, "Patient Lot/Batch #", "probe")
    assert len(joined) == 2
    assert joined.loc[joined["Patient Lot/Batch #"] == "B1", "Media Lot"].tolist() == ["M1"]


def test_an_empty_key_matches_nothing():
    batches = pd.DataFrame({"Patient Lot/Batch #": ["B1", None], "x": [1, 2]})
    lookup = pd.DataFrame({"Patient Lot/Batch #": [None, "B1"], "Media Lot": ["ghost", "M1"]})
    joined = enrich._lookup_merge(batches, lookup, "Patient Lot/Batch #", "probe")
    assert len(joined) == 2
    assert "ghost" not in set(joined["Media Lot"].dropna())


def test_a_join_key_is_found_by_name_not_by_position():
    frame = pd.DataFrame({"something else": [1], "LV Batch": ["LV-1"]})
    assert io.join_key(frame, ("LV Batch",), "probe") == "LV Batch"
    with pytest.raises(ValueError, match="no join key"):
        io.join_key(frame, ("Vector Lot",), "probe")


# ---------------------------------------------------------------------------
# Duplicates and clusters
# ---------------------------------------------------------------------------
def duplicate_frame() -> pd.DataFrame:
    base = np.arange(20, dtype=float)
    return pd.DataFrame(
        {
            "a": base,
            "b": base * 2 + 1,  # a duplicate of a
            "c": base + np.where(np.arange(20) % 2, 0.9, -0.9),  # close to a, not to b's tail
            "sparse": [1.0, 2.0] + [np.nan] * 18,
        }
    )


def test_a_dropped_duplicate_always_names_a_parameter_that_survived():
    frame = duplicate_frame()
    kept, decisions = encode.duplicate_selection(
        frame, list(frame.columns), params(duplicate_r2_threshold=0.99, duplicate_min_overlap=5)
    )
    assert decisions
    for entry in decisions:
        assert entry["Kept"] in kept
        assert entry["Dropped"] not in kept


def test_a_duplicate_pair_with_too_little_overlap_is_not_a_duplicate():
    frame = pd.DataFrame(
        {
            "full": np.arange(20, dtype=float),
            "short": [0.0, 1.0, 2.0, 3.0] + [np.nan] * 16,  # identical where it exists
        }
    )
    kept, decisions = encode.duplicate_selection(
        frame, list(frame.columns), params(duplicate_min_overlap=10)
    )
    assert set(kept) == {"full", "short"} and not decisions


def test_the_decisions_do_not_depend_on_the_order_of_the_columns():
    frame = duplicate_frame()
    forward, first = encode.duplicate_selection(frame, list(frame.columns), params())
    backward, second = encode.duplicate_selection(
        frame[list(reversed(frame.columns))], list(reversed(frame.columns)), params()
    )
    assert forward == backward
    assert [d["Dropped"] for d in first] == [d["Dropped"] for d in second]


def cluster_frame(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    base = rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "up": base,
            "down": -base * 3 + rng.normal(0, 0.05, n),  # strongly, negatively related
            "unrelated": rng.normal(0, 1, n),
            "constant": np.ones(n),
            "thin": [1.0, 5.0, 2.0, 9.0] + [np.nan] * (n - 4),
        }
    )


def test_a_negative_correlation_still_makes_a_group():
    report = decorrelate.fit(
        cluster_frame(),
        ["up", "down", "unrelated"],
        params(cluster_corr_threshold=0.7, cluster_min_overlap=10),
        action="report_only",
    )
    assert len(report.clusters) == 1
    assert set(report.clusters[0].members) == {"up", "down"}
    # The sign is kept for the reader, not folded away.
    matrix = report.clusters[0].corr_matrix_before
    assert min(min(row) for row in matrix) < 0


def test_a_pair_with_too_few_shared_batches_cannot_form_a_group():
    frame = cluster_frame()
    frame["thin_twin"] = frame["thin"] * 2
    report = decorrelate.fit(
        frame,
        ["thin", "thin_twin", "up"],
        params(cluster_corr_threshold=0.5, cluster_min_overlap=10),
        action="report_only",
    )
    assert not any({"thin", "thin_twin"} <= set(c.members) for c in report.clusters)
    reasons = {row["Parameter"]: row["Reason"] for row in report.skipped}
    assert "fewer than" in reasons["thin"]


def test_a_constant_parameter_is_skipped_with_a_reason():
    report = decorrelate.fit(
        cluster_frame(),
        ["up", "down", "constant"],
        params(cluster_min_overlap=5),
        action="report_only",
    )
    assert {"constant"} == {
        row["Parameter"] for row in report.skipped if row["Reason"] == "constant"
    }


def test_report_only_clustering_changes_no_value():
    frame = cluster_frame()
    report = decorrelate.fit(
        frame, ["up", "down"], params(cluster_min_overlap=5), action="report_only"
    )
    assert report.clusters and report.clusters[0].frame.empty
    assert decorrelate.transform(frame, report.state()).empty


def test_a_residual_can_be_applied_to_new_rows_without_refitting():
    frame = cluster_frame()
    report = decorrelate.fit(frame, ["up", "down"], params(cluster_min_overlap=5), action="linear")
    cluster = report.clusters[0]
    state = report.state()

    fitted = cluster.frame
    again = decorrelate.transform(frame, state)
    residual = f"{cluster.fits[0].column}{decorrelate.RESIDUAL_SUFFIX}"
    pd.testing.assert_series_equal(fitted[residual], again[residual], check_names=False)

    # And on rows the fit never saw.
    fresh = frame.iloc[:3] * 1.5
    assert decorrelate.transform(fresh, state)[residual].notna().all()


def test_too_little_data_keeps_the_parameter_rather_than_inventing_a_residual():
    """Three complete rows cannot carry a fit on two regressors."""
    frame = cluster_frame()
    frame["twin"] = frame["up"] * 2 + 0.01
    frame["sparse_twin"] = np.nan
    frame.loc[frame.index[:3], "sparse_twin"] = frame.loc[frame.index[:3], "up"] * 3

    report = decorrelate.fit(
        frame,
        ["up", "twin", "sparse_twin"],
        params(cluster_corr_threshold=0.6, cluster_min_overlap=3),
        action="linear",
    )
    assert report.clusters, "the three of them move together"
    kept = [fit for fit in report.clusters[0].fits if not fit.residualised]
    assert [fit.column for fit in kept] == ["sparse_twin"]
    assert "below the" in kept[0].fallback
    # It is still in the frame, under its own name, rather than as a "residual"
    # that is really a centred sparse series.
    assert "sparse_twin" in report.clusters[0].frame.columns
    assert f"sparse_twin{decorrelate.RESIDUAL_SUFFIX}" not in report.clusters[0].frame.columns


def test_an_undefined_vif_is_not_an_infinite_one():
    frame = cluster_frame()
    two_rows = frame[["up", "down"]].copy()
    two_rows.loc[two_rows.index[2:], "up"] = np.nan
    undefined, rows = decorrelate.max_vif(two_rows)
    assert np.isnan(undefined), "two complete rows cannot support a VIF"
    assert rows == 2

    collinear = pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0) * 2})
    worst, rows = decorrelate.max_vif(collinear)
    assert np.isinf(worst) and rows == 20


# ---------------------------------------------------------------------------
# Naming and encoders
# ---------------------------------------------------------------------------
def test_two_columns_that_want_one_output_name_both_get_one():
    taken: set[str] = set()
    assert encode.unique_name("harvest", taken) == "harvest"
    assert encode.unique_name("harvest", taken) == "harvest__2"
    assert encode.unique_name("harvest", taken) == "harvest__3"


def test_a_real_category_called_other_does_not_collide_with_the_tail():
    frame = pd.DataFrame({"site": ["Other"] * 5 + ["A"] * 4 + ["B"] * 3 + ["C", "D"]})
    state, _ = encode.fit_one_hot(frame, ["site"], params(one_hot_top_x=2), set())
    tail = state["site"]["other"]
    assert tail != "Other"
    encoded, _ = encode.apply_one_hot(frame, state, params())
    assert encoded[state["site"]["outputs"]["Other"]].sum() == 5
    assert encoded[state["site"]["outputs"][tail]].sum() == 5  # C, D and the three Bs


def test_a_missing_category_is_missing_unless_the_task_says_otherwise():
    frame = pd.DataFrame({"site": ["A"] * 5 + ["B"] * 4 + [None] * 3})
    state = encode.fit_binary(frame, ["site"], params(), set())
    encoded, _ = encode.apply_binary(frame, state, params())
    assert encoded.iloc[:9, 0].notna().all() and encoded.iloc[9:, 0].isna().all()

    # Asked for, absence becomes a category of its own and keeps one label
    # everywhere — which for a two-value column means three categories, so it
    # goes to one-hot instead and the missing rows get their own column.
    as_category = params(missing_category="category")
    routes = encode.route_categoricals(frame, ["site"], as_category)
    assert routes[0].categories == 3 and routes[0].strategy == "one_hot"
    state, _ = encode.fit_one_hot(frame, ["site"], as_category, set())
    encoded, _ = encode.apply_one_hot(frame, state, as_category)
    assert encode.MISSING_LABEL in state["site"]["categories"]
    missing_column = encoded[state["site"]["outputs"][encode.MISSING_LABEL]]
    assert missing_column.tolist() == [0] * 9 + [1] * 3


def test_hashing_puts_a_category_in_the_same_bucket_every_run():
    frame = pd.DataFrame({"machine": [f"INC-{i:02d}" for i in range(12)]})
    state, _ = encode.fit_hashed(frame, ["machine"], params(), set())
    first, _ = encode.apply_hashed(frame, state, params())
    second, _ = encode.apply_hashed(frame.iloc[::-1], state, params())
    pd.testing.assert_frame_equal(first, second.loc[first.index])


# ---------------------------------------------------------------------------
# The package
# ---------------------------------------------------------------------------
def test_a_package_is_only_complete_when_its_artefacts_are(tmp_path):
    pkg = package.open_package(tmp_path, "probe")
    pkg.write_text("dataset.csv", "a,b\n1,2\n")
    with pytest.raises(FileNotFoundError, match="report.html"):
        pkg.complete({}, required=("dataset.csv", "report.html"))
    assert pkg.staging.name.endswith(package.STAGING_SUFFIX)
    assert not pkg.destination.exists()


def test_a_failed_run_keeps_its_folder_and_says_it_failed(tmp_path):
    import json

    pkg = package.open_package(tmp_path, "probe")
    pkg.write_text("run.log", "something went wrong")
    folder = pkg.fail("ValueError: the PVF was not readable")
    assert folder.name.endswith(package.FAILED_SUFFIX)
    record = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed" and "not readable" in record["failure"]


def test_a_report_that_fails_cannot_leave_a_finished_package(tmp_path, monkeypatch):
    from pvf import report

    directory = tmp_path / "demo"
    config_path = dummy_data.write_config(directory)
    config = cli.load_config(config_path)
    cli.run_build(config, mode="dev")
    spec = taskconfig.load(directory / dummy_data.TASK_FILES[0])

    def explode(*args, **kwargs):
        raise RuntimeError("the report broke")

    monkeypatch.setattr(report, "build_task_payload", explode)
    with pytest.raises(RuntimeError, match="the report broke"):
        cli.run_task(config, spec)

    folders = list((spec.output_root / spec.name).iterdir())
    assert folders and all(f.name.endswith(package.FAILED_SUFFIX) for f in folders)
    assert not any(
        (f / "manifest.json").exists() and "complete" in (f / "manifest.json").read_text()
        for f in folders
    )


# ---------------------------------------------------------------------------
# Provenance records what was read, from where
# ---------------------------------------------------------------------------
def test_a_sharepoint_read_is_recorded_by_location_not_by_a_local_hash(tmp_path):
    """The bug this covers: with sources.location: sharepoint, provenance hashed
    the local copies named in config, which did not exist, and warned per file."""
    from pvf import config as workspace

    config = {
        "paths": {"phf_ghent": str(tmp_path / "absent" / "PHF.xlsm")},
        "sources": {"location": "sharepoint"},
        "sharepoint": {"phf_ghent": {"path": "Batch data/PHF.xlsm", "drive_id": "DRIVE_X"}},
    }
    frame = pd.DataFrame({"Patient Lot/Batch #": ["A", "B"], "Value": [1.0, 2.0]})
    reader = lambda **kwargs: frame  # noqa: E731 — stands in for io_sharepoint
    mark = cli.log.mark()

    source = workspace.source(config, "phf_ghent", reader)
    assert source.read() is frame
    record = source.record(frame)

    assert record.kind == "remote"
    assert record.location.startswith("SharePoint DRIVE_X: Batch data/PHF.xlsm [BR Data]")
    assert record.digest == provenance.frame_digest(frame)
    assert record.digest_of == "the table as it was consumed"
    assert not [e for e in cli.log.since(mark) if e.level == "WARN"]
    # A source with no SharePoint entry is still read, and hashed, where it is.
    local = tmp_path / "raw.csv"
    local.write_text("a\n1\n", encoding="utf-8")
    config["paths"]["raw_materials"] = str(local)
    assert workspace.source(config, "raw_materials", reader).record(frame).kind == "local file"
    assert workspace.source(config, "phf_ghent", reader).record(None).kind == "not read"


# ---------------------------------------------------------------------------
# The console
# ---------------------------------------------------------------------------
def test_the_scanner_bounces_end_to_end_and_stays_off_a_pipe():
    from pvf.logger import _Scanner, log

    scanner = _Scanner(log)
    cycle = [scanner.frame(tick) for tick in range(2 * (scanner.TRACK - 1))]
    assert all(len(frame) == scanner.TRACK and frame.count("#") == 1 for frame in cycle)
    assert all(frame.isascii() for frame in cycle), "every console can draw it"
    heads = [frame.index("#") for frame in cycle]
    assert min(heads) == 0 and max(heads) == scanner.TRACK - 1
    assert scanner.frame(len(cycle)) == cycle[0], "it loops"
    # The tests' stdout is not a terminal, so nothing animates into it.
    assert not log.animate


def test_a_cp1252_console_gets_ascii_and_never_an_encoding_error(monkeypatch):
    """The bug this covers: on a Windows console in cp1252 the progress bar's ♪
    raised "'charmap' codec can't encode character '\\u266a'" on every frame."""
    import io as stdio
    import time

    from pvf.logger import ASCII, PipelineLogger

    console = stdio.TextIOWrapper(stdio.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", console)
    logger = PipelineLogger()

    assert logger.glyph is ASCII
    assert logger._safe("PT → done") == "PT ? done"
    logger._scanner.DELAY = 0.0
    logger._scanner.start()
    time.sleep(0.3)
    logger._scanner.stop()
    console.flush()
    drawn = console.buffer.getvalue().decode("cp1252")
    assert "[" in drawn and "working" in drawn
