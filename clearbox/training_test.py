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

"""Tests for the training utility."""

import itertools
import random

from absl.testing import absltest
from absl.testing import parameterized
from clearbox import features as F
from clearbox import formula
from clearbox import training
from clearbox.metrics import recall
from clearbox.models import brute_force
import numpy as np
import pandas as pd

_RANDOM_SEED = 42


class TrainerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "in_main_process",
          "gmax_w1": 0.5,
          "gmax_w2": 0.5,
          "num_parallel_workers": 1,
      },
      {
          "testcase_name": "with_parallel_workers",
          "gmax_w1": 0.5,
          "gmax_w2": 0.5,
          "num_parallel_workers": 4,
      },
      {
          "testcase_name": "eval_on_test",
          "gmax_w1": 0.5,
          "gmax_w2": 0.5,
          "num_parallel_workers": 1,
          "eval_on_test": True,
      },
  )
  def test_train_works_as_expected(
      self,
      gmax_w1: float,
      gmax_w2: float,
      num_parallel_workers: int,
      eval_on_test: bool = False,
  ):
    random.seed(_RANDOM_SEED)
    np.random.seed(_RANDOM_SEED)

    x = np.random.rand(512, 2)
    # We use predefined `gmax_w1`, `gmax_w2` to guarantee that the global
    # maximum of the metric exists in the grid.
    y = x[:, 0] * gmax_w1 + x[:, 1] * gmax_w2
    query = np.random.randint(0, 32, size=y.shape)

    if eval_on_test:
      test_mask = query > 24
      df = pd.DataFrame({
          "f1": x[~test_mask, 0],
          "f2": x[~test_mask, 1],
          "target": y[~test_mask],
          "target_bin": np.round(y[~test_mask] * 10),
          "query": query[~test_mask],
      })
      test_df = pd.DataFrame({
          "f1": x[test_mask, 0],
          "f2": x[test_mask, 1],
          "target": y[test_mask],
          "target_bin": np.round(y[test_mask] * 10),
          "query": query[test_mask],
      })
    else:
      df = pd.DataFrame({
          "f1": x[:, 0],
          "f2": x[:, 1],
          "target": y,
          "target_bin": np.round(y * 10),
          "query": query,
      })
      test_df = None

    trainer = training.Trainer(
        df=df,
        seeds=[
            15,
            21,
        ],
        n_folds=3,
        metrics=[
            recall.RecallAtK(1, bin_threshold=0.7),
            recall.RecallAtK(3, bin_threshold=0.7),
            recall.RecallAtK(5, bin_threshold=0.7),
        ],
        target_col="target",
        query_col="query",
        stratify_by_col="target_bin",
        test_df=test_df,
    )
    result = trainer.train(
        brute_force.GridSearchLinearModel(
            metric=recall.RecallAtK(k=3, bin_threshold=0.75)
        ),
        features=[F.Signal("f1"), F.Signal("f2")],
        num_parallel_workers=num_parallel_workers,
    )

    self.assertLen(result.metrics, 6)
    recall_at_k_thresholds = {
        1: 0.9,
        3: 0.95,
        5: 1.0,
    }
    for dataset, k in itertools.product(["train", "valid"], [1, 3, 5]):
      col_name = f"{dataset}_recall@{k}"
      self.assertIn(col_name, result.metrics.columns)
      self.assertTrue(
          np.all(result.metrics[col_name] >= recall_at_k_thresholds[k])
      )

    self.assertIsInstance(result.ranking_formula, formula.RankingFormula)
    self.assertEqual(result.ranking_formula.feature_names, ["f1", "f2"])
    self.assertAlmostEqual(
        result.ranking_formula.weights[0] / gmax_w1,
        result.ranking_formula.weights[1] / gmax_w2,
    )

    if test_df is not None:
      self.assertIsNotNone(result.test_metrics)
      for k in [1, 3, 5]:
        key = f"recall@{k}"
        self.assertIn(key, result.test_metrics)
        self.assertGreaterEqual(
            result.test_metrics[key], recall_at_k_thresholds[k]
        )
    else:
      self.assertIsNone(result.test_metrics)

  @parameterized.named_parameters(
      {
          "testcase_name": "in_main_process",
          "num_parallel_workers": 1,
      },
      {
          "testcase_name": "with_parallel_workers",
          "num_parallel_workers": 4,
      },
  )
  def test_get_feature_baseline_works_as_expected(
      self, num_parallel_workers: int
  ):
    random.seed(_RANDOM_SEED)
    np.random.seed(_RANDOM_SEED)

    x = np.random.rand(512, 2)
    # We use predefined `gmax_w1`, `gmax_w2` to guarantee that the global
    # maximum of the metric exists in the grid.
    y = x[:, 0] * 2
    query = np.random.randint(0, 32, size=y.shape)

    df = pd.DataFrame({
        "f1": x[:, 0],
        "f2": x[:, 1],
        "target": y,
        "target_bin": np.round(y * 10),
        "query": query,
    })
    trainer = training.Trainer(
        df=df,
        seeds=[15, 21,],
        n_folds=3,
        metrics=[
            recall.RecallAtK(1, bin_threshold=0.7),
            recall.RecallAtK(3, bin_threshold=0.7),
            recall.RecallAtK(5, bin_threshold=0.7),
        ],
        target_col="target",
        query_col="query",
        stratify_by_col="target_bin",
    )
    result_metrics = trainer.get_feature_baseline(
        feature=F.Signal("f1"),
        num_parallel_workers=num_parallel_workers,
    )

    self.assertLen(result_metrics, 6)
    recall_at_k_thresholds = {
        1: 0.9,
        3: 0.95,
        5: 1.0,
    }
    for dataset, k in itertools.product(["train", "valid"], [1, 3, 5]):
      col_name = f"{dataset}_recall@{k}"
      self.assertIn(col_name, result_metrics.columns)
      self.assertTrue(
          np.all(result_metrics[col_name] >= recall_at_k_thresholds[k])
      )

  def test_metrics_property(self):
    metrics = [
        recall.RecallAtK(1, bin_threshold=0.7),
        recall.RecallAtK(3, bin_threshold=0.7),
        recall.RecallAtK(5, bin_threshold=0.7),
    ]
    trainer = training.Trainer(
        df=pd.DataFrame(),
        seeds=[15, 21,],
        n_folds=3,
        metrics=metrics,
        target_col="target",
        query_col="query",
        stratify_by_col="target_bin",
    )
    self.assertEqual(len(trainer.metrics), len(metrics))
    for act_m, exp_m in zip(trainer.metrics, metrics):
      self.assertIsInstance(act_m, recall.RecallAtK)
      self.assertEqual(act_m.name, exp_m.name)


if __name__ == "__main__":
  absltest.main()
