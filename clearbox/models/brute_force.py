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

"""Brute force models.

To describe argument shapes, we use the following variables:
      N - number of samples (rows)
      F - number of unique features
"""

import itertools

from clearbox.metrics import base as cbx_metrics
from clearbox.models import base
import numpy as np
import numpy.typing as npt
import tqdm


class _GridSearchLinearModelImpl(base.ModelImpl):
  """Linear model optimized using Grid Search over a configurable grid."""

  def __init__(
      self,
      metric: cbx_metrics.BaseMetric,
      grid_size: int = 10,
      print_progress: bool = False,
  ):
    """Grid Search Linear Model constructor.

    Args:
      metric:
        The metric to optimize for.
      grid_size:
        The number of steps to use for single feature in the grid.
      print_progress:
        Whether to print the progress of the grid search.
    """
    self._weights: npt.NDArray[float] | None = None
    self._metric = metric
    self._grid_size = grid_size
    self._print_progress = print_progress

  def fit(
      self,
      x: npt.NDArray[float],
      y: npt.NDArray[float],
      query: npt.NDArray[int],
  ):
    """Train the model using features `x`, targets `y` and query IDs `query`.

    Args:
      x: Array of input features. Shape: (N, F).
      y: Array of targets. Shape: (N,).
      query: Array of integer query IDs. Shape: (N,).
    """
    n_feat = x.shape[1]

    best_weights = np.zeros((n_feat,))
    best_metric_val = float('-inf')

    weights_it = itertools.product(*[
        np.array(np.linspace(0.0, 1.0, self._grid_size))
        for _ in range(n_feat)
    ])
    n_iter = int(self._grid_size**n_feat)
    if self._print_progress:
      weights_it = tqdm.tqdm(weights_it, total=n_iter)
    for weights in weights_it:
      weights = np.array(weights)
      scores = np.dot(x, weights)
      metric_val = self._metric.compute(query, scores, y)
      if metric_val > best_metric_val:
        best_weights = weights
        best_metric_val = metric_val
    self._weights = best_weights

  def predict(self, x: npt.NDArray[float]) -> npt.NDArray[float]:
    """Predicts the model using the best weights found during .fit().

    Args:
      x:
        The features to predict for. Shape: (N, F)
    Returns:
      The predicted scores. Shape: (N,)
    """
    assert (
        self._weights is not None
    ), 'Please call .fit() first to tune the model.'
    return np.dot(x, self._weights)

  @property
  def weights(self) -> npt.NDArray[float] | None:
    """Getter for the tuned weights of the model if they are available.

    Returns:
      Array of weights as floats if they are available, otherwise `None`.
      Shape: (F,).
    """
    return self._weights


class GridSearchLinearModel(base.Model):
  """Linear model optimized using Grid Search."""

  def __init__(
      self,
      metric: cbx_metrics.BaseMetric,
      grid_size: int = 10,
      print_progress: bool = False,
  ):
    """Grid Search Linear Model constructor.

    Args:
      metric: The metric to optimize for.
      grid_size: The number of steps to use for single feature in the grid.
      print_progress: Whether to print the progress of the grid search.
    """
    self._metric = metric
    self._grid_size = grid_size
    self._print_progress = print_progress

  def new(self) -> base.ModelImpl:
    """Creates and returns a new grid search linear model.

    Returns:
      `_GridSearchLinearModelImpl` instance which implements `base.ModelImpl`
      interface.
    """
    return _GridSearchLinearModelImpl(
        metric=self._metric,
        grid_size=self._grid_size,
        print_progress=self._print_progress,
    )
