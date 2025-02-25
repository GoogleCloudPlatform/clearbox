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

"""The module provides the AUROC implementation."""

from clearbox.metrics import base
import numpy.typing as npt
import sklearn.metrics


class AUROC(base.BaseMetric):
  """Area Under RO Curve implementation."""

  @property
  def name(self) -> str:
    return 'auroc'

  def compute(
      self,
      query_ids: npt.NDArray[int],
      scores: npt.NDArray[float],
      targets: npt.NDArray[float],
  ) -> float:
    """Computes AUROC.

    Args:
      query_ids: Numpy array where each component corresponds to the unique
        integer ID of the query of a given (query, doc) pair. Ignored by AUROC.
      scores: Numpy array where each component corresponds to the predicted
        score of a given (query, doc) pair. Each score is a float from 0 to 1.
      targets: Numpy array where each component represent a binary label
        indicator of a given (query, doc) pair.

    Returns:
      AUROC value, float between 0 and 1 inclusively.
    """
    self._validate_compute_args(query_ids, scores, targets)

    return sklearn.metrics.roc_auc_score(targets, scores)
