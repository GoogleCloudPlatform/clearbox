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

"""Feature generation utilities."""

from __future__ import annotations

import abc
import typing as t

import numpy as np
import numpy.typing as npt
import pandas as pd


def encode(srs: pd.Series) -> pd.Series:
  """Encode the `pd.Series` of hashable objects into a `pd.Series` of int IDs.

  This may be useful if the feature you wanna group on in the cross validation
  is represented by string or any other non-int hashable object.

  Args:
    srs: `pd.Series` of hashable objects.

  Returns:
    `pd.Series` where each component is a unique `int` ID of the corresponding
    value from `srs`.
  """
  value_to_id = {value: id_ for id_, value in enumerate(sorted(srs.unique()))}
  # Manual type cast is needed as `apply` may return `pd.DataFrame` in some
  # cases which are not relevant for this function (e.g. when lambda returns
  # collection).
  return t.cast(pd.Series, srs.apply(lambda x: value_to_id[x]))


class Node(abc.ABC):
  """Base class for a node of the Formula expression tree."""

  @abc.abstractmethod
  def evaluate(self, signal_data: pd.DataFrame) -> npt.NDArray[float]:
    """Evaluate the node on `signal_data`.

    Args:
      signal_data: `pd.DataFrame` of signal values.

    Returns:
      Array of results as floats.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def _serialize_to_str(self) -> str:
    """Serialize the node to a string."""
    raise NotImplementedError()

  @abc.abstractmethod
  def serialize_to_ranking_expression(self) -> str:
    """Serialize the node to the ranking expression format."""
    raise NotImplementedError()

  def __str__(self) -> str:
    """Return a string representation of the node."""
    return self._serialize_to_str()

  def __add__(self, other: Node | float) -> Node:
    """Add two nodes together.

    Syntactic sugar for `Add([self, other])`. This method allows you to write:
    ```
    F.Signal("signal_1") + 1.0
    F.Signal("signal_1") + F.Signal("signal_2")
    ```

    Args:
      other: Another node or a float value to add to `self`.

    Returns:
      A new `Add` node.
    """
    if isinstance(other, float):
      other = Constant(other)
    return Add([self, other])

  def __mul__(self, other: Node | float) -> Node:
    """Multiply two nodes together.

    Syntactic sugar for `Mul([self, other])`. This method allows you to write:
    ```
    F.Signal("signal_1") * 2.0
    F.Signal("signal_1") * F.Signal("signal_2")
    ```

    Args:
      other: Another node or a float value to multiply to `self`.

    Returns:
      A new `Mul` node.
    """
    if isinstance(other, float) or isinstance(other, int):
      other = Constant(other)
    return Mul([self, other])

  def __neg__(self) -> Node:
    """Negate the node."""
    return self * -1.0

  def __sub__(self, other: Node | float) -> Node:
    """Subtract `other` from `self`.

    Syntactic sugar for `self + other * -1.0`. This method allows you to write:
    ```
    F.Signal("signal_1") - 1.0
    F.Signal("signal_1") - F.Signal("signal_2")
    ```

    Args:
      other: Another node or a float value to subtract from `self`.

    Returns:
      A new `Node` that implements subtraction.
    """
    if isinstance(other, float):
      other = Constant(other)
    return self + (-other)


class Signal(Node):
  """Node that evaluates to an array of signal values."""

  def __init__(self, name: str, encode_values: bool = False):
    """Initialize the `Signal` node.

    Args:
      name: Name of the signal.
      encode_values: If `True`, the signal values are encoded into unique `int`
        IDs. This may be useful if the signal values are non-int hashable
        objects (e.g. strings). Encoded signals cannot be serialized to proto
        but they are still useful for train-only transformations (e.g. grouping
        by a query for reciprocal rank).
    """
    super().__init__()

    self._name = name
    self._encode_values = encode_values

  def evaluate(self, signal_data: pd.DataFrame) -> npt.NDArray[float]:
    """Evaluate the node on `signal_data`.

    Args:
      signal_data: `pd.DataFrame` of signal values.

    Returns:
      Array of results as floats.
    """
    if self._name not in signal_data.columns:
      raise ValueError(f"Signal '{self._name}' not found in `signal_data`.")
    data = signal_data[self._name]
    if self._encode_values:
      data = encode(data)
    return data.values

  def _serialize_to_str(self) -> str:
    if self._encode_values:
      return f'Signal("{self._name}", encode_values=True)'
    else:
      return f'Signal("{self._name}")'

  def serialize_to_ranking_expression(self) -> str:
    """Serialize the node to the ranking expression format."""
    if self._encode_values:
      raise ValueError(
          "Signal with encoded values cannot be serialized to the ranking"
          " expression format."
      )
    return self._name


class Constant(Node):
  """Node that evaluates to an array of constant values."""

  def __init__(self, value: float):
    """Initialize the `Constant` node.

    Args:
      value: Constant value the node evaluates to.
    """
    self._value = value

  def evaluate(self, signal_data: pd.DataFrame) -> npt.NDArray[float]:
    """Evaluate the node on `signal_data`.

    Args:
      signal_data: `pd.DataFrame` of signal values.

    Returns:
      Array of results as floats.
    """
    if signal_data.empty:
      raise ValueError("`signal_data` should not be empty.")
    return np.full(signal_data.shape[0], self._value)

  def _serialize_to_str(self) -> str:
    return f"Constant({self._value})"

  def serialize_to_ranking_expression(self) -> str:
    """Serialize the node to the ranking expression format."""
    # If the value has more than 4 digits after the decimal point, we round it
    # to 4 digits, otherwise we just print the value as is.
    return min(f"{self._value}", f"{self._value:.4f}")


class Log(Node):
  """Node that evaluates to a log transformaion of the `arg` node."""

  def __init__(self, arg: Node):
    """Initialize the `Log` node.

    Args:
      arg: Node to evaluate and apply log transformation to.
    """
    self._arg = arg

  def evaluate(self, signal_data: pd.DataFrame) -> npt.NDArray[float]:
    """Evaluate the node on `signal_data`.

    Args:
      signal_data: `pd.DataFrame` of signal values.

    Returns:
      Array of results as floats.
    """
    return np.log(self._arg.evaluate(signal_data))

  def _serialize_to_str(self) -> str:
    return f"Log({self._arg})"

  def serialize_to_ranking_expression(self) -> str:
    """Serialize the node to the ranking expression format."""
    return f"log({self._arg.serialize_to_ranking_expression()})"


class Exp(Node):
  """Node that evaluates to an exponential transformaion of the `arg` node."""

  def __init__(self, arg: Node):
    """Initialize the `Exp` node.

    Args:
      arg: Node to evaluate and apply exponential transformation to.
    """
    self._arg = arg

  def evaluate(self, signal_data: pd.DataFrame) -> npt.NDArray[float]:
    """Evaluate the node on `signal_data`.

    Args:
      signal_data: `pd.DataFrame` of signal values.

    Returns:
      Array of results as floats.
    """
    return np.exp(self._arg.evaluate(signal_data))

  def _serialize_to_str(self) -> str:
    return f"Exp({self._arg})"

  def serialize_to_ranking_expression(self) -> str:
    """Serialize the node to the ranking expression format."""
    return f"exp({self._arg.serialize_to_ranking_expression()})"


class RR(Node):
  """Node that evaluates to a reciprocal rank of the `arg` node."""

  def __init__(self, arg: Node, k: float, group_by: Node | None = None):
    """Initialize the recipical rank node.

    Args:
      arg: Node to evaluate and apply reciprocal rank transformation to.
      k: Constant to add to the ranks in the denominator of the reciprocal rank
        formula.
      group_by: If provided, the node is evaluated and the value is used to
        group the `arg` node evaluation results into a non-overlapping clusters.
        The reciprocal rank is then computed separately for each cluster.
        `group_by` is not supported in the Proto format so it's not serialized.
    """
    if k < 0:
      raise ValueError("`k` should be non-negative.")
    self._arg = arg
    self._k = k
    self._group_by = group_by

  def evaluate(self, signal_data: pd.DataFrame) -> npt.NDArray[float]:
    """Evaluate the node on `signal_data`.

    Args:
      signal_data: `pd.DataFrame` of signal values.

    Returns:
      Array of results as floats.
    """
    arg_data = self._arg.evaluate(signal_data)
    if self._group_by is not None:
      group_by_data = self._group_by.evaluate(signal_data)
      ranks = (
          pd.DataFrame({"groupby": group_by_data, "arg": -arg_data})
          .groupby("groupby")["arg"]
          .rank()
          .values
      )
    else:
      ranks = pd.Series(-arg_data).rank().values
    return 1.0 / (ranks + self._k)

  def _serialize_to_str(self) -> str:
    if self._group_by is not None:
      return f"RR({self._arg}, k={self._k}, group_by={self._group_by})"
    else:
      return f"RR({self._arg}, k={self._k})"

  def serialize_to_ranking_expression(self) -> str:
    """Serialize the node to the ranking expression format.

    `group_by` argument is not supported in the Ranking Expression format so
    it's not serialized.

    Returns:
      Ranking Expression representation of the node.
    """
    return f"rr({self._arg.serialize_to_ranking_expression()}, {self._k})"


class Add(Node):
  """Node that evaluates to an addition of multiple `args` nodes."""

  def __init__(self, args: list[Node]):
    """Initialize the `Add` node.

    Args:
      args: List of nodes to evaluate and sum up the results. Cannot be empty.
    """
    if not args:
      raise ValueError("`args` should not be empty.")

    self._args = args

  def __add__(self, other: Node | float) -> Node:
    """Shortcut implementation to keep 1 `Add` node for more than 2 args."""
    return Add([*self._args, other])

  def evaluate(self, signal_data: pd.DataFrame) -> npt.NDArray[float]:
    """Evaluate the node on `signal_data`.

    Args:
      signal_data: `pd.DataFrame` of signal values.

    Returns:
      Array of results as floats.
    """
    return np.sum(
        np.stack([arg.evaluate(signal_data) for arg in self._args], axis=0),
        axis=0,
    )

  def _serialize_to_str(self) -> str:
    return " + ".join([str(arg) for arg in self._args])

  def serialize_to_ranking_expression(self) -> str:
    """Serialize the node to the ranking expression format."""
    return " + ".join(
        [arg.serialize_to_ranking_expression() for arg in self._args]
    )


class Mul(Node):
  """Node that evaluates to a multiplication of multiple `args` nodes."""

  def __init__(self, args: list[Node]):
    """Initialize the `Mul` node.

    Args:
      args: List of nodes to evaluate and multiply the results. Cannot be empty.
    """
    if not args:
      raise ValueError("`args` should not be empty.")
    self._args = args

  def __mul__(self, other: Node | float) -> Node:
    """Shortcut implementation to keep 1 `Mul` node for more than 2 args."""
    return Mul([*self._args, other])

  def evaluate(self, signal_data: pd.DataFrame) -> npt.NDArray[float]:
    """Evaluate the node on `signal_data`.

    Args:
      signal_data: `pd.DataFrame` of signal values.

    Returns:
      Array of results as floats.
    """
    return np.prod(
        np.stack([arg.evaluate(signal_data) for arg in self._args], axis=0),
        axis=0,
    )

  def _serialize_arg_to_str(self, arg: Node) -> str:
    if isinstance(arg, Add):
      # Add parens around the Add node to preserve the intended operator
      # precedence.
      return f"({arg})"
    else:
      return str(arg)

  def _serialize_arg_to_ranking_expression(self, arg: Node) -> str:
    if isinstance(arg, Add):
      # Add parens around the Add node to preserve the intended operator
      # precedence.
      return f"({arg.serialize_to_ranking_expression()})"
    else:
      return arg.serialize_to_ranking_expression()

  def _serialize_to_str(self) -> str:
    return " * ".join([self._serialize_arg_to_str(arg) for arg in self._args])

  def serialize_to_ranking_expression(self) -> str:
    """Serialize the node to the ranking expression format."""
    return " * ".join(
        [self._serialize_arg_to_ranking_expression(arg) for arg in self._args]
    )


class IsNaN(Node):
  """Node that evaluates to a binary mask of NaN values in the `arg` node."""

  def __init__(self, arg: Node):
    """Initialize the `IsNaN` node.

    Args:
      arg: Node to evaluate and generate the is-NaN mask for. Mask values are
        floats.
    """
    self._arg = arg

  def evaluate(self, signal_data: pd.DataFrame) -> npt.NDArray[float]:
    """Evaluate the node on `signal_data`.

    Args:
      signal_data: `pd.DataFrame` of signal values.

    Returns:
      Array of results as floats.
    """
    return np.isnan(self._arg.evaluate(signal_data)).astype(float)

  def _serialize_to_str(self) -> str:
    return f"IsNaN({self._arg})"

  def serialize_to_ranking_expression(self) -> str:
    """Serialize the node to the ranking expression format."""
    return f"is_nan({self._arg.serialize_to_ranking_expression()})"


class FillNaN(Node):
  """Node that fills NaNs in the `arg` node with values from `fill_with`."""

  def __init__(self, arg: Node, fill_with: Node):
    """Initialize the `FillNaN` node.

    Args:
      arg: Node to evaluate and fill NaN values in.
      fill_with: Node to evaluate and use the values to fill NaN values in
        `arg`.
    """
    self._arg = arg
    self._fill_with = fill_with

  def evaluate(self, signal_data: pd.DataFrame) -> npt.NDArray[float]:
    """Evaluate the node on `signal_data`.

    Args:
      signal_data: `pd.DataFrame` of signal values.

    Returns:
      Array of results as floats.
    """
    arg_data = self._arg.evaluate(signal_data)
    fill_with_data = self._fill_with.evaluate(signal_data)
    return np.where(np.isnan(arg_data), fill_with_data, arg_data)

  def _serialize_to_str(self) -> str:
    return f"FillNaN({self._arg}, {self._fill_with})"

  def serialize_to_ranking_expression(self) -> str:
    """Serialize the node to the ranking expression format."""
    return (
        f"fill_nan({self._arg.serialize_to_ranking_expression()}, "
        f"{self._fill_with.serialize_to_ranking_expression()})"
    )


class _SignalProvider:
  """Syntactic sugar for creating `Signal` nodes.

  The class provides a way to access the `Signal` nodes by their names without
  explicitly creating them. We override the `__getattr__` method to return a
  `Signal` node with the given name.

  So, instead of:
  > from clearbox import features as F
  > F.Signal("signal_1")

  you can do:
  > from clearbox.features import signals as S
  > S.signal_1
  """

  def __getattr__(self, name: str, /) -> Node:
    return Signal(name)


signals = _SignalProvider()
