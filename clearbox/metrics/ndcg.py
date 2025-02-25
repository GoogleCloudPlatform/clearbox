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

"""The module provides the NDCG implementation."""

import collections

from clearbox.metrics import base
import numpy as np
import numpy.typing as npt
import sklearn.metrics


class NDCG(base.BaseMetric):
  """NDCG metric implementation."""

  def __init__(self, k: int | None = None, pw_decay_factor: float = 1.0):
    """NDCG constructor.

    Args:
      k: Number of top docs to take into account when computing the metric for
        each query. We take all the docs into account when it's `None`.
      pw_decay_factor: Position weighted decay factor to apply to relevance
        target, float in (0, 1] range.
    """
    super().__init__()

    if k is not None and k < 1:
      raise ValueError(
          f'If provided, `k` should be integer >= 1, received {k}.'
      )
    if pw_decay_factor <= 0.0 or pw_decay_factor > 1.0:
      raise ValueError('`pw_decay_factor` should be in (0, 1] range.')

    self._k = k
    self._pw_decay_factor = pw_decay_factor

  @property
  def name(self) -> str:
    return f'ndcg@{self._k}' if self._k is not None else 'ndcg'

  def compute(
      self,
      query_ids: npt.NDArray[int],
      scores: npt.NDArray[float],
      targets: npt.NDArray[float],
  ) -> float:
    """Computes NDCG.

    Args:
      query_ids: Numpy array where each component corresponds to the unique
        integer ID of the query of a given (query, doc) pair.
      scores: Numpy array where each component corresponds to the predicted
        score of a given (query, doc) pair. Larger score means better relevance
        of the `doc` for the `query`.
      targets: Numpy array where each component represent a relevance score of a
        given (query, doc) pair.

    Returns:
      NDCG value, float between 0 and 1 inclusively.
    """
    self._validate_compute_args(query_ids, scores, targets)

    qid_counter = collections.Counter(query_ids)
    pred_matrix = np.zeros(
        (len(qid_counter), max(qid_counter.values()))
    )
    true_matrix = np.zeros_like(pred_matrix)
    for i, (qi, n_docs) in enumerate(qid_counter.items()):
      pred_matrix[i, :n_docs] = scores[query_ids == qi]
      true_matrix[i, : n_docs] = targets[query_ids == qi]
    if self._pw_decay_factor < 1.:
      decay_matrix = self._pw_decay_factor ** np.argsort(-true_matrix, axis=1)
      true_matrix *= decay_matrix
    return sklearn.metrics.ndcg_score(true_matrix, pred_matrix, k=self._k)
