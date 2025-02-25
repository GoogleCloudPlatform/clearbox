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

"""Tests for the Average implementation."""

from absl.testing import absltest
from clearbox.metrics import average
from clearbox.metrics import base
import numpy as np
import numpy.typing as npt


class _MinScoreMetric(base.BaseMetric):

  @property
  def name(self) -> str:
    return "min"

  def compute(
      self,
      query_ids: npt.NDArray[float],
      scores: npt.NDArray[float],
      targets: npt.NDArray[float],
  ) -> float:
    return scores.min()


class _MaxScoreMetric(base.BaseMetric):

  @property
  def name(self) -> str:
    return "max"

  def compute(
      self,
      query_ids: npt.NDArray[float],
      scores: npt.NDArray[float],
      targets: npt.NDArray[float],
  ) -> float:
    return scores.max()


class AverageTest(absltest.TestCase):

  def test_returns_correct_result(self):
    metric = average.Average(
        name="min_max_avg",
        metrics=[
            _MinScoreMetric(),
            _MaxScoreMetric(),
        ],
    )
    self.assertAlmostEqual(
        metric.compute(
            query_ids=np.array([0, 0, 0]),
            scores=np.array([1.0, 2.0, 3.0]),
            targets=np.array([0.1, 0.2, 0.3]),
        ),
        2.0,
    )
    self.assertAlmostEqual(
        metric.compute(
            query_ids=np.array([0, 0, 0]),
            scores=np.array([2.0, 4.0, 8.0]),
            targets=np.array([0.1, 0.2, 0.3]),
        ),
        5.0,
    )

  def test_empty_metrics(self):
    with self.assertRaises(ValueError):
      average.Average([])

  def test_different_shape_of_args(self):
    metric = average.Average(
        name="min_max_avg",
        metrics=[
            _MinScoreMetric(),
            _MaxScoreMetric(),
        ],
    )
    with self.assertRaises(ValueError):
      metric.compute(
          query_ids=np.array([0, 0, 0, 1]),
          scores=np.array([2.0, 4.0, 8.0]),
          targets=np.array([0.1, 0.2, 0.3]),
      )

  def test_empty_args(self):
    metric = average.Average(
        name="min_max_avg",
        metrics=[
            _MinScoreMetric(),
            _MaxScoreMetric(),
        ],
    )
    with self.assertRaises(ValueError):
      metric.compute(
          query_ids=np.array([]),
          scores=np.array([]),
          targets=np.array([]),
      )

  def test_default_name(self):
    metric = average.Average(
        metrics=[
            _MinScoreMetric(),
            _MaxScoreMetric(),
        ],
    )
    self.assertEqual(metric.name, "average")

  def test_custom_name(self):
    metric = average.Average(
        name="avg",
        metrics=[
            _MinScoreMetric(),
            _MaxScoreMetric(),
        ],
    )
    self.assertEqual(metric.name, "avg")


if __name__ == "__main__":
  absltest.main()
