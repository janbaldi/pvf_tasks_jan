"""Using what a task fitted: on new batches, and inside cross-validation.

A task run writes its fitted preprocessing to ``metadata/recipe.json``. Two
things can be done with it here:

* :func:`apply` encodes new batches — a PVF built later, say — exactly as the
  training rows were encoded, with nothing refitted. ``pvf apply`` is this.
* :class:`TaskTransformer` is the same preprocessing as a scikit-learn step. A
  table preprocessed over every row is not a table to cross-validate on; put the
  transformer *inside* the pipeline instead, and every fold refits it on its own
  training rows::

      from sklearn.model_selection import GroupKFold, cross_val_score
      from sklearn.pipeline import make_pipeline
      from pvf.recipe import TaskTransformer, task_data

      X, y, groups, spec = task_data("tasks/ghent_car_expression.yaml")
      model = make_pipeline(TaskTransformer(spec), SimpleImputer(), StandardScaler(), Ridge())
      cross_val_score(model, X, y, groups=groups, cv=GroupKFold(5))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import dataset, io, taskconfig
from .dataset import Preprocessor, TaskResult, fit_preprocessing
from .logger import log

MODULE = "recipe"


def find(run: str | Path) -> Path:
    """The recipe of a task run, given the run folder or the recipe itself."""
    path = Path(run)
    if path.is_dir():
        path = path / "metadata" / "recipe.json"
    if not path.is_file():
        raise FileNotFoundError(f"No recipe at {path} — is this a task run folder?")
    return path


def apply(run: str | Path, rows: pd.DataFrame) -> pd.DataFrame:
    """Encode ``rows`` with a run's fitted recipe, and say what was missing."""
    pre = Preprocessor.load(find(run))
    absent = [column for column in pre.inputs if column not in rows.columns]
    if absent:
        log.warn(
            MODULE,
            f"{len(absent)} parameters the recipe uses are not in these rows; "
            "their columns come out empty",
            ", ".join(absent[:10]),
        )
    return pre.transform(rows)


class TaskTransformer(TransformerMixin, BaseEstimator):
    """A task's preprocessing as a scikit-learn transformer.

    ``fit`` learns category vocabularies, target means, near-duplicates,
    clusters and the sparsity and near-constant checks from the rows it is given;
    ``transform`` applies them. ``fit_transform`` cross-fits the target encoding
    on the training rows, as a task run does. ``X`` is the cohort (every PVF
    column, one row per batch); the grouping column, when present, keeps the
    target encoding's folds whole.
    """

    def __init__(
        self,
        spec: taskconfig.TaskSpec | None = None,
        ptf: pd.DataFrame | None = None,
        predictors: tuple[str, ...] = (),
    ):
        self.spec = spec
        self.ptf = ptf
        self.predictors = predictors

    def _fit(self, X: pd.DataFrame, y) -> pd.Series:
        y = pd.Series(np.asarray(y, dtype="float64"), index=X.index)
        ptf = self.ptf if self.ptf is not None else io.read_ptf(self.spec.ptf)
        result = TaskResult(spec=self.spec)
        result.target_column = "__target__"
        predictors = list(self.predictors) or dataset.candidate_predictors(
            X, ptf, self.spec, result
        )
        reserved = {
            name
            for name in (self.spec.roles.id, self.spec.roles.group, self.spec.roles.patient)
            if name
        } | {"__target__", "split", "fold"}
        self.preprocessor_ = fit_preprocessing(X, y, predictors, ptf, self.spec, result, reserved)
        self.feature_names_out_ = np.asarray(self.preprocessor_.output_columns, dtype=object)
        return y

    def fit(self, X: pd.DataFrame, y=None):
        self._fit(X, y)
        return self

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params: Any) -> pd.DataFrame:
        y = self._fit(X, y)
        groups = dataset.split_groups(X, X.index, self.spec)
        frame, _ = self.preprocessor_.crossfit_transform(X, y, groups=groups)
        return frame.astype("float64")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.preprocessor_.transform(X).astype("float64")

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


def task_data(
    task: str | Path, config: dict | None = None
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None, taskconfig.TaskSpec]:
    """A task's cohort, target and groups, ready for :class:`TaskTransformer`.

    The cohort filters and the target are applied exactly as ``pvf tasks`` does;
    nothing is encoded.
    """
    spec = taskconfig.load(task, config)
    pvf = io.load_pvf(spec.pvf)
    result = TaskResult(spec=spec)
    cohort = dataset.select_cohort(pvf, spec, result)
    target, labelled = dataset.prepare_target(cohort, spec, result)
    cohort = cohort.loc[labelled]
    groups = dataset.split_groups(cohort, labelled, spec)
    return cohort, target.loc[labelled].astype("float64"), groups, spec
