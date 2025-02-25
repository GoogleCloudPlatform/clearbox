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

"""Ranking formula abstraction.

To describe argument shapes, we use the following variables:
      N - number of samples (rows)
      F - number of unique features
"""

from clearbox import features as F
import numpy.typing as npt
import pandas as pd


class RankingFormula:
  """Result of the tuning as a weighted average of the input features."""

  def __init__(self, weights: npt.NDArray[float], features: list[F.Node]):
    """Ranking formula constructor.

    Args:
      weights: Array of floating point weights of the features. Shape: (F,).
      features: List of feature nodes, should be of the same size as `weights`.
    """
    if len(weights) != len(features):
      raise ValueError(
          "`weights` and `features` should be of the same length,"
          f" currently `len(weights)` = {len(weights)}, `len(features)` ="
          f" {len(features)}."
      )

    self._weights = weights
    self._features = features
    self._node = F.Add([f * w for f, w in zip(features, weights)])

  @property
  def feature_names(self) -> list[str]:
    """Getter for the list of feature names.

    Returns:
      The list of feature names as strings.
    """
    return [f.serialize_to_ranking_expression() for f in self._features]

  @property
  def weights(self) -> npt.NDArray[float]:
    """Getter for the array of feature weights.

    Returns:
      Array of feature weights as floats. Shape: (F,).
    """
    return self._weights

  def evaluate(self, x: pd.DataFrame) -> npt.NDArray[float]:
    """Evaluate ranking formula for feature values `x`.

    Args:
      x: Data Frame of input featues to evaluate the formula on. Shape: (N, F).

    Returns:
      Array of scores, same length. Shape: (N,).
    """
    return self._node.evaluate(x)

  def __str__(self) -> str:
    return f"{self._node}"

  def serialize_to_ranking_expression(self) -> str:
    """Compile the ranking formula to a ranking expression format."""
    return self._node.serialize_to_ranking_expression()
