# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training loop implementation.

To describe argument shapes, we use the following variables:
      N - number of samples (rows)
      F - number of unique features
"""

import multiprocessing
import typing as t

from clearbox import features as F
from clearbox import formula
from clearbox.metrics import base as cbx_metrics
from clearbox.models import base as cbx_models
import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.model_selection
import tqdm
import tqdm.contrib.concurrent


_EPS = 1e-6


def _create_cv_split(
    df: pd.DataFrame,
    n_folds: int,
    seed: int,
    stratify_by_col: str,
    group_by_col: str,
) -> npt.NDArray[int]:
  """Create cross validation split of `n_folds`.

  Args:
    df: DataFrame to split into folds.
    n_folds: Number of folds to split `df` into.
    seed: Random seed integer to use for data shuffling before the split.
    stratify_by_col: Column to use for stratification, i.e. the we'll do our
      best to balance the distribuion of this column across the folds.
    group_by_col: Column to use for grouping, i.e. we'll make sure all rows from
      same group land either in train or in validation part, never in both.

  Returns:
    Numpy array where each component is a fold index of the corresponding row
    in `df`.
  """
  splitter = sklearn.model_selection.StratifiedGroupKFold(
      n_splits=n_folds, shuffle=True, random_state=seed
  )

  fold_idx = np.zeros((len(df),), dtype=np.int8)
  for i, (_, val_idx) in enumerate(
      splitter.split(df, y=df[stratify_by_col], groups=df[group_by_col])
  ):
    fold_idx[val_idx] = i
  return fold_idx


class _TrainModelTask(t.NamedTuple):
  """Input to the `_train_model` parallelization worker."""

  # Whole dataset as a data frame.
  df: pd.DataFrame
  # `pd.Series` of `bool` values where `True` value means that the row belongs
  # to the validation set.
  valid_mask: pd.Series
  # List of `df` columns to use as features.
  features: list[str]
  # Model builder instance, encapsulates all the parameters needed to build
  # a new model from scratch in the worker process.
  model_builder: cbx_models.Model
  # List of metrics to compute on train and validation sets.
  metrics: list[cbx_metrics.BaseMetric]
  # Column to use as an optimization target.
  target_col: str
  # Column to use as a query ID.
  query_col: str
  # Column representing the golden ranking. It almost always equals to
  # `target_col` in practice (and even defaults to it in the user-facing code).
  # At the same time allows defining optimization target different from the
  # golden ranking column.
  rank_col: str


class _TrainModelResult(t.NamedTuple):
  """Output of the `_train_model` parallelization worker."""

  # Array of tuned weights. Shape: (F,).
  weights: npt.NDArray[float]
  # Dictionary mapping metric names to the corresponding values.
  metric_row: dict[str, int | float]


def _train_model(task: _TrainModelTask) -> _TrainModelResult:
  """Worker function for training the model, basic parallelization unit.

  Args:
    task: Named tuple of the parameters describing the model to train. See the
      docstrings of `_TrainModelTask` class for more info about each field.

  Returns:
    `_TrainModelResult` named tuple encapsulating metrics and feature importance
    scores.
  """
  train_df, valid_df = (
      task.df[~task.valid_mask],
      task.df[task.valid_mask],
  )
  train_x, train_y = (
      train_df[task.features].values,
      train_df[task.target_col].values,
  )
  valid_x, _ = (
      valid_df[task.features].values,
      valid_df[task.target_col].values,
  )
  train_query_arr = train_df[task.query_col].values
  valid_query_arr = valid_df[task.query_col].values
  model = task.model_builder.new()
  model.fit(train_x, train_y, train_query_arr)

  train_score = model.predict(train_x)
  train_rank_y = train_df[task.rank_col].values
  valid_rank_y = valid_df[task.rank_col].values
  valid_score = model.predict(valid_x)
  metric_row = {}
  for metric in task.metrics:
    metric_row[f'train_{metric.name}'] = metric.compute(
        train_query_arr, train_score, train_rank_y
    )
    metric_row[f'valid_{metric.name}'] = metric.compute(
        valid_query_arr, valid_score, valid_rank_y
    )
  assert model.weights is not None
  return _TrainModelResult(
      weights=model.weights,
      metric_row=metric_row,
  )


class _IdentityModel(cbx_models.Model):
  """Builder for the ad-hoc identity model."""

  class _IdentityModelImpl(cbx_models.ModelImpl):
    """Implementation of the ad-hoc identity model.

    `fit` method does nothing and `predict` returns the value of the first
    feature as a score.
    """

    def __init__(self):
      self._n_features: int | None = None

    def fit(
        self,
        x: npt.NDArray[float],
        y: npt.NDArray[float],
        query: npt.NDArray[int],
    ):
      """Save the number of features to the attribute.

      Args:
        x: Array of input features. Shape: (N, F).
        y: Array of targets. Shape: (N,).
        query: Array of integer query IDs. Shape: (N,).
      """
      self._n_features = x.shape[1]

    @property
    def weights(self) -> npt.NDArray[float] | None:
      """Getter for the tuned weights of the model if they are available.

      Returns:
        Array of weights as floats if they are available, otherwise `None`.
        Shape: (F,).
      """
      if self._n_features is None:
        return None
      weights = np.zeros((self._n_features,))
      weights[0] = 1.0
      return weights

    def predict(self, x: npt.NDArray[float]) -> npt.NDArray[float]:
      """Return the first feature of `x` as predictions.

      Args:
        x: Array of input features. Shape: (N, F).

      Returns:
        Array of target predictions as floats. Shape: (N,).
      """
      return x[:, 0]

  def new(self) -> cbx_models.ModelImpl:
    """Create new identity model instance.

    Returns:
      `_IdentityModelImpl` instance.
    """
    return self._IdentityModelImpl()


class TrainingResults(t.NamedTuple):
  """`Trainer.train` return value, aggregates model training results."""

  # The result of tuning as a ranking formula object. It encapsulates the tuned
  # weights as well as the feature names in expected order.
  ranking_formula: formula.RankingFormula
  # DataFrame where columns are (seed, fold) index + metric names, rows are
  # metric values for a given training iteration.
  metrics: pd.DataFrame
  # Dictionary of metric values computed on test set. `None` if test set is not
  # provided.
  test_metrics: dict[str, float] | None


class Trainer:
  """Utility class that abstracts away train loop and cross validation."""

  def __init__(
      self,
      df: pd.DataFrame,
      seeds: list[int],
      n_folds: int,
      metrics: list[cbx_metrics.BaseMetric],
      target_col: str,
      query_col: str,
      rank_col: str | None = None,
      stratify_by_col: str | None = None,
      test_df: pd.DataFrame | None = None,
  ):
    """Trainer constructor.

    Args:
      df: Whole dataset as a data frame.
      seeds: List of random seed integers. For each of the seeds we generate a
        separate validation split of `n_folds` and train `n_folds` models.
      n_folds: Number of folds in one cross validation split.
      metrics: List of metrics to compute for each training iteration.
      target_col: Column to use as an optimization target.
      query_col: Column to use as a query ID.
      rank_col: Column representing the golden ranking, optional and defaults to
        `target_col`. It allows defining optimization target different from the
        golden ranking column.
      stratify_by_col: Column to use for stratification, i.e. the we'll do our
        best to balance the distribuion of this column across the folds.
      test_df: Optional dataset to run final model evaluation on.
    """
    self._df = df
    self._seeds = seeds
    self._n_folds = n_folds
    self._metrics = metrics
    self._target_col = target_col
    self._query_col = query_col
    self._rank_col = rank_col
    self._stratify_by_col = stratify_by_col
    self._test_df = test_df

  @property
  def metrics(self) -> list[cbx_metrics.BaseMetric]:
    """Getter for the metric list.

    Returns:
      A list of `cbx_metrics.BaseMetric` instances this trainer uses.
    """
    return self._metrics

  def train(
      self,
      model_builder: cbx_models.Model,
      features: list[F.Node],
      df: pd.DataFrame | None = None,
      metrics: list[cbx_metrics.BaseMetric] | None = None,
      target_col: str | None = None,
      query_col: str | None = None,
      rank_col: str | None = None,
      stratify_by_col: str | None = None,
      test_df: pd.DataFrame | None = None,
      print_progress: bool = False,
      num_parallel_workers: int | t.Literal['auto'] = 1,
  ) -> TrainingResults:
    """For each (seed, fold) pair train the model and compute metrics.

    Args:
      model_builder: Model builder instance, implements `cbx_models.Model`
        interface and encapsulates all the parameters needed to build a new
        model from scratch.
      features: List of feature nodes.
      df: Whole dataset as a data frame, overrides the corresponding argument of
        `__init__` if provided.
      metrics: List of metrics to compute for each training iteration, overrides
        the corresponding argument of `__init__` if provided.
      target_col: Column to use as an optimization target. Overrides the
        corresponding argument of `__init__` if provided.
      query_col: Column to use as a query ID. Overrides the corresponding
        argument of `__init__` if provided.
      rank_col: Column representing the golden ranking, optional and defaults to
        `target_col`. It allows defining optimization target different from the
        golden ranking column. Overrides the corresponding argument of
        `__init__` if provided.
      stratify_by_col: Column to use for stratification, i.e. the we'll do our
        best to balance the distribuion of this column across the folds.
        Overrides the corresponding argument of `__init__` if provided.
      test_df: Optional dataset to run final model evaluation on. Overrides the
        corresponding argument of `__init__` if provided.
      print_progress: If `True`, the progress is printed using `tqdm`.
      num_parallel_workers: Number of worker processes to spawn, each process
        handles the training for one seed & fold pair. If the value is 'auto',
        the number of workers is taken from `multiprocessing.cpu_count`. If the
        number is <2 no workers are spawned and all the processing happens in
        the main process.

    Returns:
      `TrainingResults` named tuple encapsulating metrics and feature importance
      scores if they are available.
    """
    df = df if df is not None else self._df
    metrics = metrics if metrics is not None else self._metrics
    target_col = target_col if target_col is not None else self._target_col
    query_col = query_col if query_col is not None else self._query_col
    rank_col = rank_col if rank_col is not None else self._rank_col
    if stratify_by_col is None:
      stratify_by_col = self._stratify_by_col
    if stratify_by_col is None:
      stratify_by_col = target_col
    test_df = test_df if test_df is not None else self._test_df
    n_workers: int
    if num_parallel_workers == 'auto':
      n_workers = multiprocessing.cpu_count()
    else:
      n_workers = num_parallel_workers

    if rank_col is None:
      rank_col = target_col

    feature_cols = [str(feature) for feature in features]
    # We use set literal to make sure we don't have duplicated columns as
    # same columns may be used for both `target_col` and `stratify_by_col` for
    # example.
    feature_df = df[
        list({query_col, rank_col, target_col, stratify_by_col})
    ].copy()
    for feature in features:
      feature_df[str(feature)] = feature.evaluate(df)

    seed_fold_pairs: list[tuple[int, int]] = []
    tasks: list[_TrainModelTask] = []
    for seed in self._seeds:
      folds = pd.Series(
          _create_cv_split(
              df,
              self._n_folds,
              group_by_col=query_col,
              stratify_by_col=stratify_by_col,
              seed=seed,
          )
      )
      for fold in range(self._n_folds):
        seed_fold_pairs.append((seed, fold))
        tasks.append(
            _TrainModelTask(
                df=feature_df,
                valid_mask=(folds == fold),
                features=feature_cols,
                model_builder=model_builder,
                metrics=metrics,
                target_col=target_col,
                query_col=query_col,
                rank_col=rank_col,
            )
        )
    results: list[_TrainModelResult]
    if n_workers > 1:
      if print_progress:
        results = tqdm.contrib.concurrent.process_map(
            _train_model, tasks, max_workers=n_workers
        )
      else:
        with multiprocessing.Pool(n_workers) as pool:
          results = pool.map(_train_model, tasks)
    else:
      task_it = tqdm.tqdm(tasks) if print_progress else tasks
      results = [_train_model(task) for task in task_it]

    metric_rows = []
    weight_arrs = []
    for (seed, fold), result in zip(seed_fold_pairs, results):
      metric_rows.append({
          'seed': seed,
          'fold': fold,
          **result.metric_row,
      })
      if result.weights is not None:
        weight_arrs.append(result.weights)

    weights = np.stack(weight_arrs, axis=0).mean(axis=0)
    # Normalize weights if possible to make them more readable.
    if abs(weights.max()) > _EPS:
      weights /= weights.max()
    ranking_formula = formula.RankingFormula(weights=weights, features=features)

    if test_df is not None:
      test_query_ids = test_df[query_col].values
      test_scores = ranking_formula.evaluate(test_df)
      test_targets = test_df[rank_col].values
      test_metrics: dict[str, float] = {}
      for metric in metrics:
        test_metrics[metric.name] = metric.compute(
            test_query_ids, test_scores, test_targets
        )
    else:
      test_metrics = None

    return TrainingResults(
        ranking_formula=ranking_formula,
        metrics=pd.DataFrame(metric_rows),
        test_metrics=test_metrics,
    )

  def get_feature_baseline(
      self,
      feature: F.Node,
      df: pd.DataFrame | None = None,
      metrics: list[cbx_metrics.BaseMetric] | None = None,
      target_col: str | None = None,
      query_col: str | None = None,
      rank_col: str | None = None,
      stratify_by_col: str | None = None,
      test_df: pd.DataFrame | None = None,
      print_progress: bool = False,
      num_parallel_workers: int = 1,
  ) -> pd.DataFrame:
    """Evaluate a baseline of using `feature_col` as predicted score.

    Args:
      feature: Feature node to use as prediction.
      df: Whole dataset as a data frame, overrides the corresponding argument of
        `__init__` if provided.
      metrics: List of metrics to compute for each training iteration, overrides
        the corresponding argument of `__init__` if provided.
      target_col: Column to use as an optimization target. Overrides the
        corresponding argument of `__init__` if provided.
      query_col: Column to use as a query ID. Overrides the corresponding
        argument of `__init__` if provided.
      rank_col: Column representing the golden ranking, optional and defaults to
        `target_col`. It allows defining optimization target different from the
        golden ranking column. Overrides the corresponding argument of
        `__init__` if provided.
      stratify_by_col: Column to use for stratification, i.e. the we'll do our
        best to balance the distribuion of this column across the folds.
        Overrides the corresponding argument of `__init__` if provided.
      test_df: Optional dataset to run final model evaluation on. Overrides the
        corresponding argument of `__init__` if provided.
      print_progress: If `True`, the progress is printed using `tqdm`.
      num_parallel_workers: Number of worker processes to spawn, each process
        handles the training for one seed & fold pair. If the value is 'auto',
        the number of workers is taken from `multiprocessing.cpu_count`. If the
        number is <2 no workers are spawned and all the processing happens in
        the main process.

    Returns:
      `pd.DataFrame` of metric values.
    """
    return self.train(
        model_builder=_IdentityModel(),
        features=[feature],
        df=df,
        metrics=metrics,
        query_col=query_col,
        target_col=target_col,
        rank_col=rank_col,
        stratify_by_col=stratify_by_col,
        test_df=test_df,
        print_progress=print_progress,
        num_parallel_workers=num_parallel_workers,
    ).metrics
