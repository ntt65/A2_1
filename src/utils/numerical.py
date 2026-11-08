"""Scalar numerical differentiation with explicit finite-input contracts."""

from collections.abc import Callable

import numpy as np

from src.utils.math import _as_finite_real_array

__concepts__ = {"Central_Difference": ["central_difference"]}


def _finite_scalar(value: object, name: str) -> float:
    """Return a finite real scalar as float; reject inputs with shape other than ()."""
    array = _as_finite_real_array(value, name)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar with shape ()")
    return float(array)


def central_difference(
    function: Callable[[float], float], x: float, h: float = 1e-5
) -> float:
    """Approximate f'(x) = (f(x+h)-f(x-h))/(2h), returning a scalar float.

    x, h, and both function outputs must be finite real scalars (shape ()).
    h must be positive and distinguish x+h and x-h from x in float64.
    Non-real values raise TypeError. Invalid shapes, steps, sampling points,
    or unrepresentable differences/results raise ValueError. Exceptions raised
    by the supplied function propagate without being hidden.
    """
    if not callable(function):
        raise TypeError("function must be callable")
    point = _finite_scalar(x, "x")
    step = _finite_scalar(h, "h")
    if step <= 0:
        raise ValueError("h must be positive")
    upper, lower = point + step, point - step
    denominator = 2.0 * step
    if not all(np.isfinite(value) for value in (upper, lower, denominator)):
        raise ValueError("sampling points and 2*h must be finite")
    if upper == point or lower == point:
        raise ValueError("h is too small to distinguish x+h and x-h from x")
    value_upper = _finite_scalar(function(upper), "function(x+h)")
    value_lower = _finite_scalar(function(lower), "function(x-h)")
    derivative = (value_upper - value_lower) / denominator
    if not np.isfinite(derivative):
        raise ValueError("the central difference is not representable as a finite float64")
    return derivative
