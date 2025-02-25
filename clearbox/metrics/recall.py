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

"""The module provides the Recall@K implementation."""

import typing as t

from clearbox.metrics import base
import numpy as np
import numpy.typing as npt
import pandas as pd


class RecallAtK(base.BaseMetric):
  """Recall@K implementation."""

  def __init__(
      self,
      k: int,
      max_queries_for_loop_impl: int = 500,
      bin_threshold: float = 0.5,
  ):
    """RecallAtK constructor.

    Args:
      k: K parameter of the recall, should be >= 1.
      max_queries_for_loop_impl: Maximum number of unique queries at which we
        compute the metric using custom implementation based on the for loop
        over the queries. The default implementation is prefferable for a small
        number of queries, otherwise Pandas-based implementation is faster.
      bin_threshold: Threshold for binarizing float targets.
    """
    super().__init__()

    self._k = k
    self._max_queries_for_loop_impl = max_queries_for_loop_impl
    self._bin_threshold = bin_threshold

  @property
  def name(self) -> str:
    return f'recall@{self._k}'

  def _binarize_target(
      self, target: npt.NDArray[float]
  ) -> npt.NDArray[np.uint8]:
    """Convert float target to 0 or 1 based on `_bin_threshold` attribute.

    If all the values in the array are either 0 or 1, we perform a simple cast
    to `np.uint8`. Otherwise we do the convertsion using the threshold.

    Args:
      target: Numpy array of float targets to be converted to {0, 1}.

    Returns:
      Numpy array of target values in {0, 1}.
    """
    if np.logical_or(target == 0, target == 1).all():
      return target.astype(np.uint8)
    return (target >= self._bin_threshold).astype(np.uint8)

  def compute(
      self,
      query_ids: npt.NDArray[int],
      scores: npt.NDArray[float],
      targets: npt.NDArray[float],
  ) -> float:
    """Computes Recall@K.

    Args:
      query_ids: Numpy array where each component corresponds to the unique
        integer ID of the query of a given (query, doc) pair.
      scores: Numpy array where each component corresponds to the predicted
        score of a given (query, doc) pair. Larger score means better relevance
        of the `doc` for the `query`.
      targets: Numpy array where each component represent a golden score for
        this particular (query, doc) pair. Float scores will be converted to {0,
        1} using `_convert_target_to_01` method. 1 means the `doc` is relevant
        for the `query`.

    Returns:
      Recall@K value, float between 0 and 1 inclusively.
    """
    self._validate_compute_args(query_ids, scores, targets)

    targets = self._binarize_target(targets)
    if targets.sum() == 0:
      raise ValueError('Expected at least 1 positive sample in `targets`.')

    unique_query_ids = np.unique(query_ids)
    # Choosing the preferable implementation based on the number of unique
    # queries. See description of `max_queries_for_loop_impl` constructor
    # argument for more details.
    if len(unique_query_ids) <= self._max_queries_for_loop_impl:
      n_correct, n_applicable = 0, 0
      for qi in unique_query_ids:
        query_mask = query_ids == qi
        if targets[query_mask].sum() == 0:
          continue
        score_perm = np.argsort(scores[query_mask])
        dn_correct = targets[query_mask][score_perm][-self._k :].sum()
        dn_applicable = min(targets[query_mask].sum(), self._k)
        assert dn_correct <= dn_applicable
        n_correct += dn_correct
        n_applicable += dn_applicable
      assert n_applicable > 0
    else:
      df = pd.DataFrame({
          'uid': range(len(scores)),
          'query': query_ids,
          'score': scores,
          'target': targets,
      })
      n_correct = (
          # Using `t.cast` to fix weird type annotation of
          # `sort_values(...) -> None` which actually returns `pd.DataFrame`.
          t.cast(pd.DataFrame, df.sort_values(
              ['query', 'score'], ascending=[True, False]
          ))
          .groupby('query')
          .head(self._k)
          .groupby('query')['target']
          .sum()
          .sum()
      )
      n_applicable = (
          df.groupby('query')['target'].sum().clip(upper=self._k).sum()
      )
    return n_correct / n_applicable
