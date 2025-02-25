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

"""Tests for the AUROC metric."""

from absl.testing import absltest
from absl.testing import parameterized
from clearbox.metrics import auroc
import numpy as np
import numpy.typing as npt


class AUROCTest(parameterized.TestCase):

  def test_name(self):
    metric = auroc.AUROC()
    self.assertEqual(metric.name, "auroc")

  @parameterized.named_parameters(
      (
          "perfect_prediction",
          np.array([0, 0, 1, 1]),
          np.array([0.9, 0.1, 0.8, 0.2]),
          np.array([1.0, 0.0, 1.0, 0.0]),
          1.0,
      ),
      (
          "random_prediction",
          np.array([0, 0, 1, 1]),
          np.array([0.5, 0.5, 0.5, 0.5]),
          np.array([1.0, 0.0, 1.0, 0.0]),
          0.5,
      ),
  )
  def test_auroc_returns_correct_value(
      self,
      query_ids: npt.NDArray[int],
      scores: npt.NDArray[float],
      targets: npt.NDArray[float],
      expected_value: float,
  ):
    metric = auroc.AUROC()
    actual_value = metric.compute(
        query_ids=query_ids, scores=scores, targets=targets
    )
    self.assertAlmostEqual(actual_value, expected_value, places=4)

  @parameterized.named_parameters(
      (
          "all_positive",
          np.array([0, 0, 1, 1]),
          np.array([0.9, 0.8, 0.7, 0.6]),
          np.array([1.0, 1.0, 1.0, 1.0]),
      ),
      (
          "all_negative",
          np.array([0, 0, 0, 0]),
          np.array([0.1, 0.2, 0.3, 0.4]),
          np.array([0.0, 0.0, 0.0, 0.0]),
      ),
      (
          "empty input",
          np.array([]),
          np.array([]),
          np.array([]),
      ),
      (
          "input shapes mismatch",
          np.array([0, 1]),
          np.array([0.1, 0.2, 0.3]),
          np.array([0.3, 0.2, 0.1]),
      ),
  )
  def test_auroc_raises_value_error(
      self,
      query_ids: npt.NDArray[int],
      scores: npt.NDArray[float],
      targets: npt.NDArray[float],
  ):
    metric = auroc.AUROC()
    with self.assertRaises(ValueError):
      metric.compute(
          query_ids=query_ids, scores=scores, targets=targets
      )


if __name__ == "__main__":
  absltest.main()
