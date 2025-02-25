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

"""Tests for Recall@K implementation."""

from absl.testing import absltest
from absl.testing import parameterized
from clearbox.metrics import recall
import numpy as np
import numpy.typing as npt


class RecallTest(parameterized.TestCase):

  def test_pandas_implementation(self):
    metric = recall.RecallAtK(k=2, max_queries_for_loop_impl=1)
    self.assertAlmostEqual(
        metric.compute(
            query_ids=np.array([0, 0, 0, 1, 1, 1]),
            scores=np.array([0.0, 0.5, 1.0, 0.3, 0.4, 0.1]),
            targets=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        ),
        0.5,
    )

  def test_zero_target(self):
    metric = recall.RecallAtK(k=3)
    with self.assertRaises(ValueError):
      metric.compute(
          query_ids=np.array([0, 0, 0, 1, 1, 1]),
          scores=np.array([0.0, 0.5, 1.0, 0.3, 0.4, 0.1]),
          targets=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
      )

  @parameterized.named_parameters(
      (
          "k1",
          np.array([0, 0, 0, 1, 1, 1]),
          np.array([0.0, 0.5, 1.0, 0.3, 0.4, 0.1]),
          np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
          1,
          0.0,
      ),
      (
          "k2",
          np.array([0, 0, 0, 1, 1, 1]),
          np.array([0.0, 0.5, 1.0, 0.3, 0.4, 0.1]),
          np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
          2,
          0.5,
      ),
      (
          "k2_with_negative_only_query",
          np.array([0, 0, 0, 1, 1, 1, 2]),
          np.array([0.0, 0.5, 1.0, 0.3, 0.4, 0.1, 0.1]),
          np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
          1,
          0.0,
      ),
      (
          "k3",
          np.array([0, 0, 0, 1, 1, 1]),
          np.array([0.0, 0.5, 1.0, 0.3, 0.4, 0.1]),
          np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
          3,
          1.0,
      ),
  )
  def test_recall_returns_correct_value(
      self,
      query_ids: npt.NDArray[int],
      scores: npt.NDArray[float],
      targets: npt.NDArray[float],
      k: int,
      expected_value: float,
  ):
    metric = recall.RecallAtK(k=k)
    actual_value = metric.compute(query_ids, scores, targets)
    self.assertAlmostEqual(actual_value, expected_value, places=4)

  @parameterized.named_parameters(
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
  def test_recall_raises_value_error(
      self,
      query_ids: npt.NDArray[int],
      scores: npt.NDArray[float],
      targets: npt.NDArray[float],
  ):
    metric = recall.RecallAtK(k=1)
    with self.assertRaises(ValueError):
      metric.compute(query_ids, scores, targets)

  @parameterized.named_parameters(
      ("k_1", 1, "recall@1"),
      (
          "k_2",
          2,
          "recall@2",
      ),
  )
  def test_name_returns_correct_value(self, k: int, expected_value: str):
    metric = recall.RecallAtK(k=k)
    self.assertEqual(metric.name, expected_value)


if __name__ == "__main__":
  absltest.main()
