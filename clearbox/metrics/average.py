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

"""The module provides the average aggregation metric implementation."""

import typing as t

from clearbox.metrics import base
import numpy as np
import numpy.typing as npt


class Average(base.BaseMetric):
  """Aggregator metric which takes an average of the metrics from `args`."""

  def __init__(
      self,
      metrics: t.Sequence[base.BaseMetric],
      name: str = 'average',
  ):
    """Average metric constructor.

    Args:
      metrics: Sequence of the metric objects to average. Although the metrics
        can be instances of different classes, it's up to the call site to
        ensure that those classes interpret `scores` and `targets` arguments of
        `compute` method in a compatible way as well as the numeric bounds of
        the scores to be averaged make sense.
      name: Optional name of the metric, string.
    """
    super().__init__()

    if not metrics:
      raise ValueError('`metrics` argument should be non-empty.')
    self._name = name
    self._metrics = metrics

  @property
  def name(self) -> str:
    return self._name

  def compute(
      self,
      query_ids: npt.NDArray[int],
      scores: npt.NDArray[float],
      targets: npt.NDArray[float],
  ) -> float:
    """Computes the average of the values produced by `metrics`.

    Args:
      query_ids: Numpy array where each component corresponds to the unique
        integer ID of the query of a given (query, doc) pair.
      scores: Numpy array where each component corresponds to the predicted
        score of a given (query, doc) pair. Larger score means better relevance
        of the `doc` for the `query`.
      targets: Numpy array where each component represent a golden score for
        this particular (query, doc) pair. Exact interpretation of the score is
        to be defined and described by the classes of `metrics`.

    Returns:
      Metric value as a float, bounds and meaning of it is to be defined by the
        classes of `metrics`.
    """
    self._validate_compute_args(query_ids, scores, targets)

    return np.mean([
        m.compute(query_ids=query_ids, scores=scores, targets=targets)
        for m in self._metrics
    ])
