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

"""Tests for brute force models."""

import random

from absl.testing import absltest
from absl.testing import parameterized
from clearbox.metrics import recall
from clearbox.models import brute_force
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection


_RANDOM_SEED = 42


class GridSearchLinearModelTest(parameterized.TestCase):

  @parameterized.named_parameters(*[
      {
          'testcase_name': 'max_at_0.5_0.5',
          'gmax_w1': 0.5,
          'gmax_w2': 0.5,
          'grid_size': 21,
      },
      {
          'testcase_name': 'max_at_0.0_1.0',
          'gmax_w1': 0.0,
          'gmax_w2': 1.0,
          'grid_size': 11,
      },
      {
          'testcase_name': 'max_at_0.8_0.2',
          'gmax_w1': 0.8,
          'gmax_w2': 0.2,
          'grid_size': 41,
      },
  ])
  def test_fit_predict_works(
      self,
      gmax_w1: float,
      gmax_w2: float,
      grid_size: int,
  ):
    random.seed(_RANDOM_SEED)
    np.random.seed(_RANDOM_SEED)

    metric = recall.RecallAtK(3, bin_threshold=0.7)

    x = np.random.rand(512, 2)
    # We use predefined `gmax_w1`, `gmax_w2` to guarantee that the global
    # maximum of the metric exists in the grid.
    y = x[:, 0] * gmax_w1 + x[:, 1] * gmax_w2
    train_x, valid_x, train_y, valid_y = (
        sklearn.model_selection.train_test_split(x, y, test_size=0.3)
    )
    train_query = np.random.randint(0, 36, size=train_y.shape)
    valid_query = np.random.randint(0, 18, size=valid_y.shape) + 36

    model = brute_force.GridSearchLinearModel(
        metric=metric,
        grid_size=grid_size,
    ).new()
    model.fit(train_x, train_y, train_query)
    valid_y_hat = model.predict(valid_x)
    self.assertGreater(
        metric.compute(valid_query, valid_y_hat, valid_y), 0.95)


if __name__ == '__main__':
  absltest.main()
