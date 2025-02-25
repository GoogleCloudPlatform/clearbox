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

"""Tests for regression models."""

import itertools
import random

from absl.testing import absltest
from absl.testing import parameterized
from clearbox.models import regression
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

_RANDOM_SEED = 42


class LinRegModelTest(parameterized.TestCase):

  @parameterized.named_parameters(
      *[
          {
              'testcase_name': (
                  f'positive_{positive}'
                  f'_fit_intercept_{fit_intercept}_apply_scaler_{apply_scaler}'
              ),
              'positive': positive,
              'fit_intercept': fit_intercept,
              'apply_scaler': apply_scaler,
          }
          for positive, fit_intercept, apply_scaler in itertools.product(
              [True, False], [True, False], [True, False]
          )
      ],
  )
  def test_fully_functional_model_is_created(
      self, positive: bool, fit_intercept: bool, apply_scaler: bool
  ):
    random.seed(_RANDOM_SEED)
    np.random.seed(_RANDOM_SEED)

    model = regression.LinRegModel(
        positive=positive,
        fit_intercept=fit_intercept,
        apply_scaler=apply_scaler,
    ).new()

    # Make sure the model can fit linear regression data.
    x, y = sklearn.datasets.make_regression(n_samples=100, n_features=4)
    train_x, valid_x, train_y, valid_y = (
        sklearn.model_selection.train_test_split(x, y, test_size=0.2)
    )
    model.fit(train_x, train_y, query=np.zeros_like(train_y))
    self.assertGreater(
        sklearn.metrics.r2_score(valid_y, model.predict(valid_x)), 0.95
    )

    # Weights should not be `None`.
    self.assertIsNotNone(model.weights)
    assert model.weights is not None

    # If `positive` is `True`, all the coefficients should be >= 0.
    if positive:
      self.assertTrue((model.weights >= 0).all())

    if fit_intercept or apply_scaler:
      # Dot product between input and weights equals return value of `predict`
      # up to a constant term.
      const_terms = model.predict(valid_x) - np.dot(valid_x, model.weights)
      self.assertTrue(np.allclose(const_terms, const_terms[0]))
    else:
      # Dot product between input and weights equals return value of `predict`.
      self.assertTrue(
          np.allclose(model.predict(valid_x), np.dot(valid_x, model.weights))
      )


if __name__ == '__main__':
  absltest.main()
