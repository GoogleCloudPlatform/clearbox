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

"""Tests for the feature generation utilities."""

import math

from absl.testing import absltest
from clearbox import features as F
# pylint: disable=g-importing-member
from clearbox.features import signals as S
import numpy as np
import pandas as pd


class SignalTest(absltest.TestCase):

  def test_evaluate(self):
    signal_data = pd.DataFrame({
        "signal_1": [1.0, 2.0, 3.0],
        "signal_2": [4.0, 5.0, 6.0],
    })
    self.assertAlmostEqual(
        (
            F.Signal("signal_1").evaluate(signal_data)
            - np.array([1.0, 2.0, 3.0])
        ).sum(),
        0.0,
    )
    self.assertAlmostEqual(
        (
            F.Signal("signal_2").evaluate(signal_data)
            - np.array([4.0, 5.0, 6.0])
        ).sum(),
        0.0,
    )
    with self.assertRaises(ValueError):
      F.Signal("signal_3").evaluate(signal_data)

  def test_str(self):
    self.assertEqual(str(F.Signal("signal_1")), 'Signal("signal_1")')
    self.assertEqual(
        str(F.Signal("signal_2", encode_values=True)),
        'Signal("signal_2", encode_values=True)',
    )

  def test_serialize_to_ranking_expression(self):
    self.assertEqual(
        F.Signal("signal_1").serialize_to_ranking_expression(),
        "signal_1",
    )
    with self.assertRaises(ValueError):
      F.Signal("signal_1", encode_values=True).serialize_to_ranking_expression()


class ConstantTest(absltest.TestCase):

  def test_evaluate(self):
    signal_data = pd.DataFrame({
        "signal_1": [1.0, 2.0, 3.0],
        "signal_2": [4.0, 5.0, 6.0],
    })
    self.assertAlmostEqual(
        (
            F.Constant(1.0).evaluate(signal_data) - np.array([1.0, 1.0, 1.0])
        ).sum(),
        0.0,
    )
    with self.assertRaises(ValueError):
      F.Constant(1.0).evaluate(pd.DataFrame())

  def test_str(self):
    self.assertEqual(str(F.Constant(1.0)), "Constant(1.0)")

  def test_serialize_to_ranking_expression(self):
    self.assertEqual(
        F.Constant(1.0).serialize_to_ranking_expression(),
        "1.0",
    )


class LogTest(absltest.TestCase):

  def test_evaluate(self):
    signal_data = pd.DataFrame({
        "signal_1": [math.exp(1.0), math.exp(2.0), math.exp(3.0)],
        "signal_2": [math.exp(4.0), math.exp(5.0), math.exp(6.0)],
    })
    self.assertAlmostEqual(
        (
            F.Log(F.Signal("signal_1")).evaluate(signal_data)
            - np.array([1.0, 2.0, 3.0])
        ).sum(),
        0.0,
    )
    self.assertAlmostEqual(
        (
            F.Log(F.Signal("signal_2")).evaluate(signal_data)
            - np.array([4.0, 5.0, 6.0])
        ).sum(),
        0.0,
    )

  def test_str(self):
    self.assertEqual(
        str(F.Log(F.Signal("signal_1"))), 'Log(Signal("signal_1"))'
    )

  def test_serialize_to_ranking_expression(self):
    self.assertEqual(
        F.Log(F.Signal("signal_1")).serialize_to_ranking_expression(),
        "log(signal_1)",
    )


class ExpTest(absltest.TestCase):

  def test_evaluate(self):
    signal_data = pd.DataFrame({
        "signal_1": [1.0, 2.0, 3.0],
        "signal_2": [4.0, 5.0, 6.0],
    })
    self.assertAlmostEqual(
        (
            F.Exp(F.Signal("signal_1")).evaluate(signal_data)
            - np.array([math.exp(1.0), math.exp(2.0), math.exp(3.0)])
        ).sum(),
        0.0,
    )
    self.assertAlmostEqual(
        (
            F.Exp(F.Signal("signal_2")).evaluate(signal_data)
            - np.array([math.exp(4.0), math.exp(5.0), math.exp(6.0)])
        ).sum(),
        0.0,
    )

  def test_str(self):
    self.assertEqual(
        str(F.Exp(F.Signal("signal_1"))), 'Exp(Signal("signal_1"))'
    )

  def test_serialize_to_ranking_expression(self):
    self.assertEqual(
        F.Exp(F.Signal("signal_1")).serialize_to_ranking_expression(),
        "exp(signal_1)",
    )


class RRTest(absltest.TestCase):

  def test_evaluate(self):
    signal_data = pd.DataFrame({
        "signal_1": [1.0, 2.0, 3.0],
        "signal_2": [6.0, 5.0, 4.0],
        "query": ["q1", "q1", "q2"],
    })
    self.assertAlmostEqual(
        (
            F.RR(F.Signal("signal_1"), 1.0).evaluate(signal_data)
            - np.array([1 / (1 + 3.0), 1 / (1 + 2.0), 1 / (1 + 1.0)])
        ).sum(),
        0.0,
    )
    self.assertAlmostEqual(
        (
            F.RR(
                F.Signal("signal_2"),
                1.0,
                group_by=F.Signal("query", encode_values=True),
            ).evaluate(signal_data)
            - np.array([1 / (1 + 1.0), 1 / (1 + 2.0), 1 / (1 + 1.0)])
        ).sum(),
        0.0,
    )
    with self.assertRaises(ValueError):
      F.RR(F.Signal("signal_1"), -1.0)

  def test_str(self):
    self.assertEqual(
        str(F.RR(F.Signal("signal_1"), 1.0)), 'RR(Signal("signal_1"), k=1.0)'
    )

  def test_serialize_to_ranking_expression(self):
    self.assertEqual(
        F.RR(
            F.Signal("signal_1"),
            k=1.0,
            group_by=F.Signal("signal_2", encode_values=True),
        ).serialize_to_ranking_expression(),
        "rr(signal_1, 1.0)",
    )


class AddTest(absltest.TestCase):

  def test_evaluate(self):
    signal_data = pd.DataFrame({
        "signal_1": [1.0, 2.0, 3.0],
        "signal_2": [4.0, 5.0, 6.0],
    })
    self.assertAlmostEqual(
        (
            F.Add([F.Signal("signal_1"), F.Signal("signal_2")]).evaluate(
                signal_data
            )
            - np.array([5.0, 7.0, 9.0])
        ).sum(),
        0.0,
    )
    with self.assertRaises(ValueError):
      F.Add([])

  def test_str(self):
    self.assertEqual(
        str(F.Add([F.Signal("signal_1"), F.Signal("signal_2")])),
        'Signal("signal_1") + Signal("signal_2")',
    )

  def test_serialize_to_ranking_expression(self):
    self.assertEqual(
        F.Add(
            [F.Signal("signal_1"), F.Signal("signal_2")]
        ).serialize_to_ranking_expression(),
        "signal_1 + signal_2",
    )


class MulTest(absltest.TestCase):

  def test_evaluate(self):
    signal_data = pd.DataFrame({
        "signal_1": [1.0, 2.0, 3.0],
        "signal_2": [4.0, 5.0, 6.0],
    })
    self.assertAlmostEqual(
        (
            F.Mul([F.Signal("signal_1"), F.Signal("signal_2")]).evaluate(
                signal_data
            )
            - np.array([4.0, 10.0, 18.0])
        ).sum(),
        0.0,
    )
    with self.assertRaises(ValueError):
      F.Mul([])

  def test_str(self):
    self.assertEqual(
        str(F.Mul([F.Signal("signal_1"), F.Signal("signal_2")])),
        'Signal("signal_1") * Signal("signal_2")',
    )

  def test_serialize_to_ranking_expression(self):
    self.assertEqual(
        F.Mul(
            [F.Signal("signal_1"), F.Signal("signal_2")]
        ).serialize_to_ranking_expression(),
        "signal_1 * signal_2",
    )


class IsNaNTest(absltest.TestCase):

  def test_evaluate(self):
    signal_data = pd.DataFrame({
        "signal_1": [1.0, 2.0, np.nan],
        "signal_2": [4.0, np.nan, 6.0],
    })
    self.assertAlmostEqual(
        (
            F.IsNaN(F.Signal("signal_1")).evaluate(signal_data)
            - np.array([0.0, 0.0, 1.0])
        ).sum(),
        0.0,
    )
    self.assertAlmostEqual(
        (
            F.IsNaN(F.Signal("signal_2")).evaluate(signal_data)
            - np.array([0.0, 1.0, 0.0])
        ).sum(),
        0.0,
    )

  def test_str(self):
    self.assertEqual(
        str(F.IsNaN(F.Signal("signal_1"))), 'IsNaN(Signal("signal_1"))'
    )

  def test_serialize_to_ranking_expression(self):
    self.assertEqual(
        F.IsNaN(F.Signal("signal_1")).serialize_to_ranking_expression(),
        "is_nan(signal_1)",
    )


class FillNaNTest(absltest.TestCase):

  def test_evaluate(self):
    signal_data = pd.DataFrame({
        "signal_1": [1.0, 2.0, np.nan],
        "signal_2": [4.0, np.nan, 6.0],
    })
    self.assertAlmostEqual(
        (
            F.FillNaN(F.Signal("signal_1"), F.Signal("signal_2")).evaluate(
                signal_data
            )
            - np.array([1.0, 2.0, 6.0])
        ).sum(),
        0.0,
    )

  def test_str(self):
    self.assertEqual(
        str(F.FillNaN(F.Signal("signal_1"), F.Signal("signal_2"))),
        'FillNaN(Signal("signal_1"), Signal("signal_2"))',
    )

  def test_serialize_to_ranking_expression(self):
    self.assertEqual(
        F.FillNaN(
            F.Signal("signal_1"), F.Signal("signal_2")
        ).serialize_to_ranking_expression(),
        "fill_nan(signal_1, signal_2)",
    )


class ModuleTest(absltest.TestCase):

  def test_parens_override_operator_precedence(self):
    self.assertEqual(
        (
            (F.Signal("signal_1") + F.Signal("signal_2")) * F.Signal("signal_3")
        ).serialize_to_ranking_expression(),
        "(signal_1 + signal_2) * signal_3",
    )

  def test_end_to_end(self):
    signal_data = pd.DataFrame({
        "signal_1": [math.log(1.0), math.log(2.0), math.log(3.0)],
        "signal_2": [1.0, 1.0, 0.0],
        "signal_3": [math.exp(4.0), math.exp(5.0), math.exp(6.0)],
    })
    y = (
        -F.Exp(F.Signal("signal_1")) * F.Signal("signal_2")
        + F.Log(F.Signal("signal_3")) * 0.5
    )
    self.assertAlmostEqual(
        (y.evaluate(signal_data) - np.array([1.0, 0.5, 3.0])).sum(), 0.0
    )
    self.assertEqual(
        str(y),
        (
            'Exp(Signal("signal_1")) * Constant(-1.0) * Signal("signal_2")'
            ' + Log(Signal("signal_3")) * Constant(0.5)'
        ),
    )

  def test_signal_provider(self):
    signal = S.signal_1
    self.assertIsInstance(signal, F.Signal)
    self.assertEqual(str(signal), 'Signal("signal_1")')


if __name__ == "__main__":
  absltest.main()
