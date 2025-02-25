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

"""The module provides the base metric interface."""

import abc

import numpy.typing as npt


class BaseMetric(abc.ABC):
  """Base metric interface, every metric should implement it."""

  @property
  @abc.abstractmethod
  def name(self) -> str:
    """The method should return the name of the metric."""
    raise NotImplementedError()

  @abc.abstractmethod
  def compute(
      self,
      query_ids: npt.NDArray[int],
      scores: npt.NDArray[float],
      targets: npt.NDArray[float],
  ) -> float:
    """The method for computing single metric value.

    Args:
      query_ids: Numpy array where each component corresponds to the unique
        integer ID of the query of a given (query, doc) pair.
      scores: Numpy array where each component corresponds to the predicted
        score of a given (query, doc) pair. Larger score means better relevance
        of the `doc` for the `query`.
      targets: Numpy array where each component represent a golden score for
        this particular (query, doc) pair. Exact interpretation of the score is
        to be defined and described by the subclass.

    Returns:
      Metric value as a float, bounds and meaning of it is to be defined by the
        subclass.
    """
    raise NotImplementedError()

  def _validate_compute_args(
      self,
      query_ids: npt.NDArray[int],
      scores: npt.NDArray[float],
      targets: npt.NDArray[float],
  ):
    if query_ids.shape != scores.shape or scores.shape != targets.shape:
      raise ValueError(
          '`query_ids`, `scores` and `targets` should be of the same shape.'
      )
    if not query_ids.shape or query_ids.shape[0] == 0:
      raise ValueError('Input arguments should be non-empty.')
