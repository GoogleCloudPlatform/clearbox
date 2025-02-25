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

"""Probabilistic models.

To describe argument shapes, we use the following variables:
      N - number of samples (rows)
      F - number of unique features
"""

import itertools
import statistics
import typing as t

from clearbox.metrics import base as cbx_metrics
from clearbox.models import base
import numpy as np
import numpy.typing as npt
import sklearn.gaussian_process
import sklearn.linear_model
import tqdm


_SKLearnProbabilisticModel = t.Union[
    sklearn.gaussian_process.GaussianProcessRegressor,
    sklearn.linear_model.BayesianRidge,
]


class _SurrogateModelImpl:
  """Base interface for surrogate models."""

  def fit(self, x: npt.NDArray[float], y: npt.NDArray[float]):
    """Train the model using features `x` and targets `y`.

    Args:
      x: Array of input features. Shape: (N, F).
      y: Array of targets. Shape: (N,).
    """
    raise NotImplementedError()

  def predict(
      self, x: npt.NDArray[float]
  ) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    """Predict mean and standard deviation of the target for features `x`.

    Args:
      x: Array of input features. Shape: (N, F).

    Returns:
      Mean and standard deviation of target predictions as arrays.
      Shape: (N,), (N,).
    """
    raise NotImplementedError()


class _SKLearnSurrogateModelImpl(_SurrogateModelImpl):
  """Surrogate model wrapper for the probabilistic models from SKLearn."""

  def __init__(self, model: _SKLearnProbabilisticModel):
    """Constructor.

    Args:
      model: SKLearn model to wrap. See `_SKLearnProbabilisticModel` for the
        list of supported models.
    """
    self._model = model

  def fit(self, x: npt.NDArray[float], y: npt.NDArray[float]):
    """Train the model using features `x` and targets `y`.

    Args:
      x: Array of input features. Shape: (N, F).
      y: Array of targets. Shape: (N,).
    """
    self._model.fit(x, y)

  def predict(
      self, x: npt.NDArray[float]
  ) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    """Predict mean and standard deviation of the target for features `x`.

    Args:
      x: Array of input features. Shape: (N, F).

    Returns:
      Mean and standard deviation of target predictions as arrays.
      Shape: (N,), (N,).
    """
    return self._model.predict(x, return_std=True)


class SurrogateModel:
  """Surrogate model builder class.

  Use this class to create a new instance of the certain `_SurrogateModelImpl`
  subclass from scratch.

  The primary use case of this class is to provide the train loop implementation
  with a simple way of building a new instance of the model. The API is
  identical to `glassbox.models.base.Model`.
  """

  def new(self) -> _SurrogateModelImpl:
    """Creates a new instance of the surrogate model implementation.

    Returns:
      An instance of the `_SurrogateModelImpl` subclass.
    """
    raise NotImplementedError()


class GaussianProcessModel(SurrogateModel):
  """Gaussian Process surrogate model implementation.

  Wrapper over `sklearn.gaussian_process.GaussianProcessRegressor`.
  """

  def __init__(
      self, kernel: sklearn.gaussian_process.kernels.Kernel | None = None
  ):
    """GP model constructor.

    Args:
      kernel: The kernel specifying the covariance function of the GP. If None
        is passed, the following kernel is used: `ConstantKernel(1.0,
        constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")`.
    """
    self._kernel = kernel

  def new(self) -> _SurrogateModelImpl:
    """Creates and returns a new Gaussian Process model.

    Returns:
      `_SKLearnSurrogateModelImpl` wrapper over
      `sklearn.gaussian_process.GaussianProcessRegressor`.
    """
    return _SKLearnSurrogateModelImpl(
        sklearn.gaussian_process.GaussianProcessRegressor(kernel=self._kernel)
    )


class BayesianRidgeModel(SurrogateModel):
  """Implementation of Bayesian Ridge surrogate model.

  Wrapper over `sklearn.linear_model.BayesianRidge`.
  """

  def new(self) -> _SurrogateModelImpl:
    """Creates and returns a new Bayesian Ridge model.

    Returns:
      `_SKLearnSurrogateModelImpl` wrapper over
      `sklearn.linear_model.BayesianRidge`.
    """
    return _SKLearnSurrogateModelImpl(sklearn.linear_model.BayesianRidge())


class AcquisitionFunction:
  """Base interface for acquisition function."""

  def __call__(
      self,
      mean: npt.NDArray[float],
      std: npt.NDArray[float],
      prev_best_mean: float = 0.0,
  ) -> npt.NDArray[float]:
    """Score each candidate point based on predicted `mean` and `std` values.

    Args:
      mean: Array of mean predictions for the candidate points.
      std: Array of standard deviation predictions for the candidate points.
      prev_best_mean: Best score from the previous iteration, useful when
        computing the probability of improvement or expected improvement.

    Returns:
      Array of scores of the candidate points. Larger score is the
      recommendation to prefer this particular candidate over the ones with
      the lower scores.
    """
    raise NotImplementedError()


class ExpectedImprovement(AcquisitionFunction):
  """Expectation of Improvment function of normally distributed random variable."""

  def __init__(
      self,
      xi: float = 1e-3,
      exploit_coef: float = 1.0,
      explore_coef: float = 1.0,
  ):
    """Expected Improvement acquisition function constructor.

    Args:
      xi: Minimal improvement over the previous best score to be considered.
      exploit_coef: The scaling factor for the exploitation component (prefer
        the candidates with large expectation/mean value).
      explore_coef: The scaling factor for the exploration component (prefer the
        candidates with high uncertainty/std of the predicted value).
    """
    self._xi = xi
    self._exploit_coef = exploit_coef
    self._explore_coef = explore_coef

    self._normal = statistics.NormalDist()

  def _norm_cdf(self, xs: npt.NDArray[float]) -> npt.NDArray[float]:
    return np.array([self._normal.cdf(x) for x in xs])

  def _norm_pdf(self, xs: npt.NDArray[float]) -> npt.NDArray[float]:
    return np.array([self._normal.pdf(x) for x in xs])

  def __call__(
      self,
      mean: npt.NDArray[float],
      std: npt.NDArray[float],
      prev_best_mean: float = 0.0,
  ) -> npt.NDArray[float]:
    """Score each candidate point based on predicted `mean` and `std` values.

    Assuming the normality of the underlying distribution, we compute the
    expectation of the improvement for each candidate.

    Args:
      mean: Array of mean predictions for the candidate points.
      std: Array of standard deviation predictions for the candidate points.
      prev_best_mean: Best score from the previous iteration, useful when
        computing the probability of improvement or expected improvement.

    Returns:
      Array of scores of the candidate points. Larger score is the
      recommendation to prefer this particular candidate over the ones with
      the lower scores.
    """
    exp_impr = np.zeros_like(mean)
    mask = std > 0
    improve = prev_best_mean - self._xi - mean[mask]
    improve_norm = improve / std[mask]
    exploit_score = improve * self._norm_cdf(improve_norm) * self._exploit_coef
    explore_score = (
        std[mask] * self._norm_pdf(improve_norm) * self._explore_coef
    )
    exp_impr[mask] = exploit_score + explore_score
    return exp_impr


class MaxMean(AcquisitionFunction):
  """Acquisition function that returns predicted mean value as a score.

  Dumbest but computationally efficient acquisition function. In a nutshell, it
  completely ignores the exploration part of the exploration-vs-exploitation
  dilemma by pursuing the candidates with the highest predicted mean value.
  As a result, the probabilistic nature of the surrogate model is ignored.
  """

  def __call__(
      self,
      mean: npt.NDArray[float],
      std: npt.NDArray[float],
      prev_best_mean: float = 0.0,
  ) -> npt.NDArray[float]:
    """Score each candidate point based on predicted `mean` values.

    Args:
      mean: Array of mean predictions for the candidate points.
      std: Array of standard deviation predictions for the candidate points,
        ignored.
      prev_best_mean: Best score from the previous iteration, useful when
        computing the probability of improvement or expected improvement,
        ignored.

    Returns:
      Array of scores of the candidate points. Larger score is the
      recommendation to prefer this particular candidate over the ones with
      the lower scores.
    """
    return mean


class _BayesOptLinearModelImpl(base.ModelImpl):
  """Implementation of the Bayesian Optimization for linear model."""

  def __init__(
      self,
      surrogate_model: SurrogateModel,
      acquisition_function: AcquisitionFunction,
      metric: cbx_metrics.BaseMetric,
      n_opt_steps: int,
      batch_size: int,
      seed_batch_size: int,
      grid_size: int,
      print_progress: bool = False,
  ):
    """Bayesian Optimization over linear model constructor.

    Args:
      surrogate_model: The surrogate model builder object to use for mean and
        standard deviation predictions.
      acquisition_function: Acquisition function object which is used to pick
        the next candidate points to compute the optimized function for.
      metric: The metric to optimize, the values of the metric are used as
        targets in the optimization process.
      n_opt_steps: Number of the optimization steps to perform.
      batch_size: Number of candidates pick on each optimization step.
      seed_batch_size: Number of the candidates to compute the `metric` value
        for before the first optimization step. Those values are then used as
        the initial training set for the `surrogate_model`.
      grid_size: The number of steps to use for single feature in the grid. In
        the canonical implementation of BO the candidates are sampled from the
        multivariate normal distribution at each optimization step. But, as our
        implementation is basically a logical evolution of Grid Search
        algorithm, we retain the same uniform grid and use it as a source of
        candidates.
      print_progress: Whether to print the progress of the optimization.
    """
    self._surrogate_model = surrogate_model
    self._acquisition_function = acquisition_function
    self._metric = metric
    self._n_opt_steps = n_opt_steps
    self._batch_size = batch_size
    self._seed_batch_size = seed_batch_size
    self._grid_size = grid_size
    self._print_progress = print_progress

    self._weights: npt.NDArray[float] | None = None

  def fit(
      self,
      x: npt.NDArray[float],
      y: npt.NDArray[float],
      query: npt.NDArray[int],
  ):
    """Train the model using features `x`, targets `y` and query IDs `query`.

    1. Generate a grid of all possible weight combinations using based on the
    `grid_size` value.
    2. Randomly sample `seed_batch_size` weight arrays from the grid, compute
    `metric` values for each of those and train the initial version of the
    `surrogate_model` on the resulting weight & metric value pairs.
    3. Repeat `n_opt_steps` times:
      I. Use the `surrogate_model` to compute the approximation of the `metric`
      values for the whole grid (excluding the samples that have been used to
      train the `surrogate_model`).
      II. Use the `acquisition_function` to score the predictions from previous
      step.
      III. Take first `batch_size` candidates with the largest score from the
      previous step and compute the `metric` value for them.
      IV. Add weights and the `metric` values from the previous step to the
      train set of the `surrogate_model` and retrain it on the larger dataset.
    4. From the set of computed `metric` values, pick the index of the largest
    component and save the corresponding array of weights to `self._weights`.

    Args:
      x: Array of input features. Shape: (N, F).
      y: Array of targets. Shape: (N,).
      query: Array of integer query IDs. Shape: (N,).
    """
    # x: N x F
    n_feat = x.shape[1]
    # Generate weight matrix aka grid.
    # weight_matrix: M x F, where M = (self._grid_size) ** F
    weight_matrix = np.array(
        list(
            itertools.product(*[
                list(np.linspace(0.0, 1.0, self._grid_size))
                for _ in range(n_feat)
            ])
        )
    )
    # Matrix of shape (M, 2)
    index = np.stack(
        [
            np.array(list(range(weight_matrix.shape[0]))),
            np.zeros((weight_matrix.shape[0],), dtype='int'),
        ],
        axis=1,
    )
    seed_index = np.random.choice(
        index[:, 0], size=self._seed_batch_size, replace=False
    )
    index[seed_index, 1] = 1
    wx = weight_matrix[index[:, 1] == 1, :]
    wy_parts = []
    seed_scores_it: t.Iterable[npt.NDArray[float]]
    if self._print_progress:
      seed_scores_it = tqdm.tqdm(np.dot(wx, x.T), desc='Seed batch')
    else:
      seed_scores_it = np.dot(wx, x.T)
    for scores in seed_scores_it:
      wy_parts.append(self._metric.compute(query, scores, y))
    wy = np.array(wy_parts)
    if self._print_progress:
      it = tqdm.tqdm(range(self._n_opt_steps), total=self._n_opt_steps)
    else:
      it = range(self._n_opt_steps)
    for _ in it:
      sur_model = self._surrogate_model.new()
      sur_model.fit(wx, wy)
      pidx = index[index[:, 1] == 0][:, 0]
      pwx = weight_matrix[index[:, 1] == 0]
      pwy_mean, pwy_std = sur_model.predict(pwx)
      acq_score = self._acquisition_function(pwy_mean, pwy_std, wy.max())
      idxs_to_explore = np.argsort(acq_score)[-self._batch_size :]
      wx = np.concat([wx, pwx[idxs_to_explore]], axis=0)
      wy_deltas = []
      for scores in np.dot(pwx[idxs_to_explore], x.T):
        wy_deltas.append(self._metric.compute(query, scores, y))
      wy_delta = np.array(wy_deltas)
      wy = np.concatenate([wy, wy_delta], axis=0)
      index[pidx[idxs_to_explore], 1] = 1
    assert wx is not None and wy is not None
    self._weights = wx[np.argmax(wy)]

  def predict(self, x: npt.NDArray[float]) -> npt.NDArray[float]:
    """Predicts the model using the best weights found during .fit().

    Args:
      x: The features to predict for. Shape: (N, F)

    Returns:
      The predicted scores. Shape: (N,)
    """
    assert (
        self._weights is not None
    ), 'Please call .fit() first to tune the model.'
    return np.dot(x, self._weights)

  @property
  def weights(self) -> npt.NDArray[float] | None:
    """Getter for the tuned weights of the model if they are available.

    Returns:
      Array of weights as floats if they are available, otherwise `None`.
      Shape: (F,).
    """
    return self._weights


class BayesOptLinearModel(base.Model):
  """Linear model with weights tuned using Bayesian Optimization algorithm."""

  def __init__(
      self,
      surrogate_model: SurrogateModel,
      acquisition_function: AcquisitionFunction,
      metric: cbx_metrics.BaseMetric,
      n_opt_steps: int,
      batch_size: int,
      seed_batch_size: int,
      grid_size: int,
      print_progress: bool = False,
  ):
    """Constructor of the Bayesian Optimization model builder.

    Args:
      surrogate_model:
        The surrogate model builder object to use for mean and standard
        deviation predictions.
      acquisition_function:
        Acquisition function object which is used to pick the next candidate
        points to compute the optimized function for.
      metric:
        The metric to optimize, the values of the metric are used as targets in
        the optimization process.
      n_opt_steps: Number of the optimization steps to perform.
      batch_size: Number of candidates pick on each optimization step.
      seed_batch_size:
        Number of the candidates to compute the `metric` value for before the
        first optimization step. Those values are then used as the initial
        training set for the `surrogate_model`.
      grid_size:
        The number of steps to use for single feature in the grid. In the
        canonical implementation of BO the candidates are sampled from the
        multivariate normal distribution at each optimization step. But, as our
        implementation is basically a logical evolution of Grid Search
        algorithm, we retrain the same uniform grid and use it as a source of
        candidates.
      print_progress: Whether to print the progress of the optimization.
    """
    self._surrogate_model = surrogate_model
    self._acquisition_function = acquisition_function
    self._metric = metric
    self._n_opt_steps = n_opt_steps
    self._batch_size = batch_size
    self._seed_batch_size = seed_batch_size
    self._grid_size = grid_size
    self._print_progress = print_progress

  def new(self) -> base.ModelImpl:
    """Creates and returns a new Bayesian Optimization model.

    Returns:
      `_BayesOptLinearModelImpl` instance which implements `base.ModelImpl`
      interface.
    """
    return _BayesOptLinearModelImpl(
        surrogate_model=self._surrogate_model,
        acquisition_function=self._acquisition_function,
        metric=self._metric,
        n_opt_steps=self._n_opt_steps,
        batch_size=self._batch_size,
        seed_batch_size=self._seed_batch_size,
        grid_size=self._grid_size,
        print_progress=self._print_progress,
    )
