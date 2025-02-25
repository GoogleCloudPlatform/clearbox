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

"""Tests for the NDCG implementation."""

from absl.testing import absltest
from absl.testing import parameterized
from clearbox.metrics import ndcg
import numpy as np
import numpy.typing as npt


class NDCGTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          "all_correct",
          np.array([0, 0, 0, 1, 1, 1]),
          np.array([0.1, 0.2, 0.3, 0.3, 0.2, 0.1]),
          np.array([1.0, 2.0, 3.0, 3.0, 2.0, 1.0]),
          None,
          1.0,
      ),
      (
          "order_mismatch",
          np.array([0, 0, 0, 1, 1, 1]),
          np.array([0.1, 0.2, 0.3, 0.3, 0.2, 0.1]),
          np.array([3.0, 2.0, 1.0, 3.0, 2.0, 1.0]),
          None,
          0.89499,
      ),
      (
          "order_mismatch_at_2",
          np.array([0, 0, 0, 1, 1, 1]),
          np.array([0.1, 0.2, 0.3, 0.3, 0.2, 0.1]),
          np.array([3.0, 2.0, 1.0, 3.0, 2.0, 1.0]),
          2,
          0.76536,
      ),
      (
          "single_query",
          np.array([0, 0, 0]),
          np.array([0.1, 0.3, 0.2]),
          np.array([1.0, 2.0, 3.0]),
          None,
          0.92249,
      ),
      (
          "non_consecutive_query_ids",
          np.array([0, 0, 0, 2, 2, 2]),
          np.array([0.1, 0.2, 0.3, 0.3, 0.2, 0.1]),
          np.array([1.0, 2.0, 3.0, 3.0, 2.0, 1.0]),
          None,
          1.0,
      ),
  )
  def test_ndcg_returns_correct_value(
      self,
      query_ids: npt.NDArray[int],
      scores: npt.NDArray[float],
      targets: npt.NDArray[float],
      k: int | None,
      expected_value: float,
  ):
    metric = ndcg.NDCG(k=k)
    actual_value = metric.compute(
        query_ids=query_ids, scores=scores, targets=targets
    )
    self.assertAlmostEqual(actual_value, expected_value, places=4)

  def test_pw_decay_factor(self):
    query_ids = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.3, 0.2, 0.1])
    targets = np.array([3.0, 2.0, 1.0, 3.0, 2.0, 1.0])
    pw_targets = np.array(
        [3.0 * 1.0, 2.0 * 0.5, 1.0 * 0.25, 3.0 * 1.0, 2.0 * 0.5, 1.0 * 0.25]
    )

    self.assertAlmostEqual(
        ndcg.NDCG(pw_decay_factor=0.5).compute(query_ids, scores, targets),
        ndcg.NDCG().compute(query_ids, scores, pw_targets),
    )

  def test_init_validation(self):
    with self.assertRaises(ValueError):
      ndcg.NDCG(k=-1)
    with self.assertRaises(ValueError):
      ndcg.NDCG(pw_decay_factor=2.0)
    with self.assertRaises(ValueError):
      ndcg.NDCG(pw_decay_factor=-1.0)

  @parameterized.named_parameters(
      ("k_3", 3, "ndcg@3"), ("k_none", None, "ndcg")
  )
  def test_name_returns_correct_value(self, k: int, expected_name: str):
    metric = ndcg.NDCG(k=k)
    self.assertEqual(metric.name, expected_name)

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
  def test_ndcg_raises_value_error(
      self,
      query_ids: npt.NDArray[int],
      scores: npt.NDArray[float],
      targets: npt.NDArray[float],
  ):
    metric = ndcg.NDCG()
    with self.assertRaises(ValueError):
      metric.compute(query_ids, scores, targets)


if __name__ == "__main__":
  absltest.main()
