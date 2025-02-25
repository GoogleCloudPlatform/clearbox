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

"""The module provides a utility class for displaying the training results.
"""

from clearbox import training
from clearbox.metrics import base as cbx_metrics
from IPython.display import display
from IPython.display import HTML
import matplotlib.pyplot as plt
import pandas as pd


class Visualizer:
  """Utility class for visualizing training results."""

  def __init__(
      self,
      metrics: list[cbx_metrics.BaseMetric],
      boxplot_height: int = 7,
      boxplot_width: int = 7,
  ):
    """Visualizer constructor.

    Args:
      metrics: List of metric objects to use in visualizations.
      boxplot_height: Height of a single box plot in inches.
      boxplot_width: Width of a single box plot in inches.
    """
    self._metrics = metrics
    self._boxplot_height = boxplot_height
    self._boxplot_width = boxplot_width

  def _build_metric_summary_df(
      self, named_dfs: list[tuple[str, pd.DataFrame]], return_std: bool = False
  ) -> pd.DataFrame:
    """Build a data frame containing mean and std values for all the metrics.

    Args:
      named_dfs: List of `(name, dataframe)` tuples.
        - `name` is the string ID associated with the `dataframe` (typically
        model or feature name if it's a baseline)
        - `dataframe` contains `valid_{metric.name}` float column for each
        `metric` from `_metrics` attribute.
      return_std: If `True`, additional `{metric_name}_std` column will be
        added to the returned data frame containing standard deviation values
        for each metric across seed & fold combinations.
    Returns:
      Data frame with the following columns:
      - `model` column representing the ID associated with each input data frame
      - `{metric.name}_mean` column for each metric from `metrics` containing
        mean of each metric across all seed & fold combinations.
      - `{metric.name}_std` column for each metric from `metrics` containing
        standard deviation of each metric across all seed & fold combinations.
        Only present if `return_std` is `True`.
    """
    val_summary_rows = []
    for model_name, metric_df in named_dfs:
      val_summary_row = {
          "model": model_name,
      }
      for metric in self._metrics:
        val_summary_row[f"{metric.name}_mean"] = metric_df[
            f"valid_{metric.name}"
        ].mean()
        if return_std:
          val_summary_row[f"{metric.name}_std"] = metric_df[
              f"valid_{metric.name}"
          ].std()
      val_summary_rows.append(val_summary_row)
    return pd.DataFrame(val_summary_rows)

  def _draw_boxplots(self, named_dfs: list[tuple[str, pd.DataFrame]]):
    """Draw comparison boxplot for each metric from `_metrics` attribute.

    Args:
      named_dfs: List of `(name, dataframe)` tuples.
        - `name` is the string ID associated with the `dataframe` (typically
          feature name for a baseline, but it can be the name of other model as
          well).
        - `dataframe` contains `valid_{metric.name}` float column for each
          `metric` from `_metrics` attribute. The data frame of such format is
          typically the result of `trainer.get_feature_baseline` call.
    """
    f, axes = plt.subplots(
        1, len(self._metrics), sharey=True, squeeze=False
    )
    f.set_figheight(self._boxplot_height)
    f.set_figwidth(self._boxplot_width * len(self._metrics))
    for i, metric in enumerate(self._metrics):
      axes[0, i].set_xlabel(metric.name)
      axes[0, i].boxplot(
          [df[f"valid_{metric.name}"] for _, df in named_dfs],
          patch_artist=True,
          vert=False,
          medianprops={"color": "white", "linewidth": 1.5},
          boxprops={"color": "C0", "facecolor": "C0", "linewidth": 1.5},
          whiskerprops={"color": "C0", "linewidth": 1.5},
          capprops={"color": "C0", "linewidth": 1.5},
          labels=[x for x, _ in named_dfs],
      )

  def visualize_training_results(
      self,
      training_results: training.TrainingResults,
      baseline_named_dfs: list[tuple[str, pd.DataFrame]] | None = None,
      show_details: bool = False,
  ):
    """Visualize training results and how they perform against the baselines.

    Args:
      training_results: Results of the model training, typically it's the object
        returned by `train` method of `Trainer` class.
      baseline_named_dfs: List of `(name, dataframe)` tuples.
        - `name` is the string ID associated with the `dataframe` (typically
          feature name for a baseline, but it can be the name of other model as
          well).
        - `dataframe` contains `valid_{metric.name}` float column for each
          `metric` from `_metrics` attribute. The data frame of such format is
          typically the result of `trainer.get_feature_baseline` call.
      show_details: If `True`, the train and valid metrics for each seed & fold
        combination will be displayed.
    """
    baseline_named_dfs: list[tuple[str, pd.DataFrame]] = (
        baseline_named_dfs if baseline_named_dfs is not None else []
    )

    if show_details:
      display(HTML(training_results.metrics.to_html()))
    display(
        HTML(
            self._build_metric_summary_df([
                *baseline_named_dfs,
                ("model", training_results.metrics),
            ]).to_html()
        )
    )
    self._draw_boxplots(named_dfs=[
        *baseline_named_dfs,
        ("model", training_results.metrics),
    ])

  def visualize_model_comparison(
      self,
      named_dfs: list[tuple[str, pd.DataFrame]],
  ):
    """Visualize comparison of different models.

    Args:
      named_dfs: List of `(name, dataframe)` tuples.
        - `name` is the string ID associated with the `dataframe` (typically
          feature name for a baseline, but it can be the name of other model as
          well).
        - `dataframe` contains `valid_{metric.name}` float column for each
          `metric` from `_metrics` attribute. The data frame of such format is
          typically the result of either `trainer.get_feature_baseline(...)` or
          `trainer.train(...).metrics` calls.
    """
    display(HTML(self._build_metric_summary_df(named_dfs).to_html()))
    self._draw_boxplots(named_dfs=named_dfs)
