"""Shape-preserving activation functions implemented with NumPy."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

__concepts__ = {"Sigmoid": ["sigmoid"], "Softmax": ["softmax"]}


def _as_finite_real_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """Convert real numeric input of any shape to finite float64, or raise."""
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numbers, excluding booleans")
    with np.errstate(over="ignore", invalid="ignore"):
        array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite float64 values")
    return array


def sigmoid(x: ArrayLike) -> NDArray[np.float64]:
    """Return sigmoid(x) as float64 with x's shape (scalar input has shape ()).

    For x >= 0 use 1 / (1 + exp(-x)); otherwise use exp(x) / (1 + exp(x)).
    The two branches avoid overflow. Non-real or non-finite inputs raise
    TypeError or ValueError respectively. Empty arrays retain their shape.
    """
    values = _as_finite_real_array(x, "x")
    result = np.empty_like(values)
    nonnegative = values >= 0
    with np.errstate(under="ignore"):
        result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
        exponential = np.exp(values[~nonnegative])
        result[~nonnegative] = exponential / (1.0 + exponential)
    return result


def softmax(x: ArrayLike, axis: int = -1) -> NDArray[np.float64]:
    """Normalize exp(x) along one axis, returning float64 with the same shape.

    Subtract the maximum along the chosen axis before exponentiation.
    Input must be a nonempty, finite real array with at least one dimension.
    Invalid numeric input, an invalid axis, or an empty array raises
    TypeError or ValueError. A scalar has no normalization axis and is rejected.
    """
    values = _as_finite_real_array(x, "x")
    if values.ndim == 0 or values.size == 0:
        raise ValueError("softmax requires a nonempty array with an axis")
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise TypeError("axis must be an integer")
    if not -values.ndim <= axis < values.ndim:
        raise ValueError("axis is outside the input dimensions")
    # Opposite finite float64 extremes can subtract to -inf; exp(-inf)=0
    # is the appropriate limiting contribution to the normalized probability.
    with np.errstate(over="ignore", under="ignore"):
        shifted = values - np.max(values, axis=axis, keepdims=True)
        weights = np.exp(shifted)
    return weights / np.sum(weights, axis=axis, keepdims=True)
