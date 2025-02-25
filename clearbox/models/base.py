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

"""The module provides base classes for model implementations.

To describe argument shapes, we use the following variables:
      N - number of samples (rows)
      F - number of unique features
"""

import abc

import numpy.typing as npt


class ModelImpl(abc.ABC):
  """Base class for a model predicting a single score."""

  @abc.abstractmethod
  def fit(
      self,
      x: npt.NDArray[float],
      y: npt.NDArray[float],
      query: npt.NDArray[int],
  ):
    """Train the model using features `x`, targets `y` and query IDs `query`.

    Args:
      x: Array of input features. Shape: (N, F).
      y: Array of targets. Shape: (N,).
      query: Array of integer query IDs. Shape: (N,).
    """
    raise NotImplementedError()

  @property
  def weights(self) -> npt.NDArray[float] | None:
    """Getter for the tuned weights of the model if they are available.

    Returns:
      Array of weights as floats if they are available, otherwise `None`.
      Shape: (F,).
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def predict(self, x: npt.NDArray[float]) -> npt.NDArray[float]:
    """Predict the target for features `x`.

    Args:
      x: Array of input features. Shape: (N, F).
    Returns:
      Array of target predictions as floats. Shape: (N,).
    """
    raise NotImplementedError()


class Model(abc.ABC):
  """Model builder class.

  Use this class to create a new instance of the certain `ModelImpl` subclass
  from scratch.

  The primary use case of this class is to provide the train loop implementation
  with a simple way of building a new instance of the model without any prior
  knowledge of its constructor arguments. This is useful for the K-fold training
  setup where we need to ensure we create a new model using same set of
  arguments from scratch for each fold.
  """

  @abc.abstractmethod
  def new(self) -> ModelImpl:
    """Creates a new instance of the model implementation.

    Returns:
      An instance of the `ModelImpl` subclass.
    """
    raise NotImplementedError()
