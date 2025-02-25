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

"""The module contains the implementation of regression models."""

from clearbox.models import base
import numpy.typing as npt
import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing


class _LinRegModelImpl(base.ModelImpl):
  """Linear regression model implementation."""

  def __init__(
      self,
      positive: bool,
      fit_intercept: bool = False,
      apply_scaler: bool = True,
  ):
    """Linear regression model implementation constructor.

    Args:
      positive: When set to `True`, forces the coefficients to be positive.
        Unlike in SKLearn, the default is `True` as in most of the cases the
        features we work with are relevance scores of various sorts positively
        correlated with target.
      fit_intercept: Whether to calculate the intercept for this model, disabled
        by default.
      apply_scaler: Whether to do the normalization of the features using
        `sklearn.preprocessing.StandardScaler`, enabled by default.
    """
    super().__init__()
    self._scaler = (
        sklearn.preprocessing.StandardScaler() if apply_scaler else None
    )
    self._linreg = sklearn.linear_model.LinearRegression(
        positive=positive,
        fit_intercept=fit_intercept,
    )

    pipeline_steps = []
    if self._scaler is not None:
      pipeline_steps.append(("scaler", self._scaler))
    pipeline_steps.append(("model", self._linreg))
    self._pipeline = sklearn.pipeline.Pipeline(pipeline_steps)

  def fit(
      self,
      x: npt.NDArray[float],
      y: npt.NDArray[float],
      query: npt.NDArray[float],
  ):
    """Train the model using features `x`, targets `y` and query IDs `query`.

    Args:
      x: Array of input features. Shape: (N, F).
      y: Array of targets. Shape: (N,).
      query: Array of integer query IDs. Shape: (N,).
    """
    self._pipeline.fit(x, y)

  def predict(self, x: npt.NDArray[float]) -> npt.NDArray[float]:
    """Predict the target for features `x`.

    Args:
      x: Array of input features. Shape: (N, F).

    Returns:
      Array of target predictions as floats. Shape: (N,).
    """
    return self._pipeline.predict(x)

  @property
  def weights(self) -> npt.NDArray[float] | None:
    """Getter for the tuned weights of the model if they are available.

    In `weights` we ignore bias/intercept term as well as the mean part of
    standard scaling. While those components may be useful when optimizing
    towards some contiguous target, they are redundant for inference as they do
    not depend on the feature values.

    Returns:
      Array of weights as floats if they are available, otherwise `None`.
      Shape: (F,).
    """
    scale_by: float | npt.NDArray[float] = 1.0
    if self._scaler is not None and self._scaler.scale_ is not None:
      scale_by = self._scaler.scale_
    return self._linreg.coef_ / scale_by


class LinRegModel(base.Model):
  """Linear regression model."""

  def __init__(
      self,
      positive: bool = True,
      fit_intercept: bool = False,
      apply_scaler: bool = True,
  ):
    """Linear regression model constructor.

    Args:
      positive: When set to `True`, forces the coefficients to be positive.
        Unlike in SKLearn, the default is `True` as in most of the cases the
        features we work with are relevance scores of various sorts positively
        correlated with target.
      fit_intercept: Whether to calculate the intercept for this model, disabled
        by default.
      apply_scaler: Whether to do the normalization of the features using
        `sklearn.preprocessing.StandardScaler`, enabled by default.
    """
    self._positive = positive
    self._fit_intercept = fit_intercept
    self._apply_scaler = apply_scaler

  def new(self) -> base.ModelImpl:
    """Creates and returns a new linear regression model.

    Returns:
      `_LinRegModelImpl` instance wrapping as SKLearn implementation of
      linear regression.
    """
    return _LinRegModelImpl(
        positive=self._positive,
        fit_intercept=self._fit_intercept,
        apply_scaler=self._apply_scaler,
    )
