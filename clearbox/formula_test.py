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

"""Tests for the ranking formula abstraction."""

from absl.testing import absltest
from clearbox import features as F
from clearbox import formula
import numpy as np
import pandas as pd


class FormulaTest(absltest.TestCase):

  def test_init_validation(self):
    with self.assertRaises(ValueError):
      formula.RankingFormula(
          weights=np.array([1.0, 2.0]), features=[F.Signal("f1")]
      )

  def test_evaluate_works_as_expected(self):
    x = pd.DataFrame({
        "f1": [3.0, 1.0],
        "f2": [2.0, 2.0],
        "f3": [1.0, 3.0],
    })
    expected = np.array([1.4, 1.0])
    rf = formula.RankingFormula(
        weights=np.array([0.3, 0.2, 0.1]),
        features=[F.Signal("f1"), F.Signal("f2"), F.Signal("f3")],
    )
    self.assertTrue(np.allclose(rf.evaluate(x), expected))

  def test_evaluate_raises_when_input_shape_is_wrong(self):
    rf = formula.RankingFormula(
        weights=np.array([0.3, 0.2, 0.1]),
        features=[F.Signal("f1"), F.Signal("f2"), F.Signal("f3")],
    )
    with self.assertRaises(ValueError):
      rf.evaluate(pd.DataFrame({"f1": [1.0], "f2": [2.0]}))
    with self.assertRaises(ValueError):
      rf.evaluate(pd.DataFrame({"f1": [1.0, 2.0]}))

  def test_weights_returns_expected_value(self):
    rf = formula.RankingFormula(
        weights=np.array([0.3, 0.2, 0.1]),
        features=[F.Signal("f1"), F.Signal("f2"), F.Signal("f3")],
    )
    self.assertTrue(np.allclose(rf.weights, np.array([0.3, 0.2, 0.1])))

  def test_feature_names_returns_expected_value(self):
    rf = formula.RankingFormula(
        weights=np.array([0.3, 0.2, 0.1]),
        features=[F.Signal("f1"), F.Signal("f2"), F.Signal("f3")],
    )
    self.assertEqual(rf.feature_names, ["f1", "f2", "f3"])

  def test_str_returns_expected_value(self):
    rf = formula.RankingFormula(
        weights=np.array([0.3, 0.2, 0.1]),
        features=[F.Signal("f1"), F.Signal("f2"), F.Signal("f3")],
    )
    self.assertEqual(
        str(rf),
        (
            'Signal("f1") * Constant(0.3)'
            ' + Signal("f2") * Constant(0.2)'
            ' + Signal("f3") * Constant(0.1)'
        ),
    )

  def test_serialize_to_ranking_expression_returns_expected_value(self):
    rf = formula.RankingFormula(
        weights=np.array([0.3, 0.2, 0.1]),
        features=[F.Signal("f1"), F.Signal("f2"), F.Signal("f3")],
    )
    self.assertEqual(
        rf.serialize_to_ranking_expression(), "f1 * 0.3 + f2 * 0.2 + f3 * 0.1"
    )


if __name__ == "__main__":
  absltest.main()
