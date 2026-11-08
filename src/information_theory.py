"""Discrete Shannon information measures in nats, implemented with NumPy.

Probabilities are validated, never clipped or renormalized. The class axis
must be nonempty. The conventions 0 log(0)=0 and p>0,q=0 => +inf preserve
support mismatches rather than replacing them with an arbitrary epsilon.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.utils.math import _as_finite_real_array

__concepts__ = {
    "Entropy": ["entropy"],
    "Cross_Entropy": ["cross_entropy"],
    "KL_Divergence": ["kl_divergence"],
    "Probability_Simplex_Validation": ["_distribution"],
}


def _distribution(value: ArrayLike, axis: int, name: str) -> tuple[NDArray[np.float64], int]:
    """Validate a real distribution array and return float64 values plus a positive axis.

    Every class vector has entries in [0,1] and sum within absolute 1e-12 of
    one (rtol=0). Scalar input or an empty class axis raises ValueError; empty
    batch dimensions are allowed. Invalid numeric/axis types raise TypeError.
    Arrays are never modified, clipped, normalized or randomly initialized.
    """
    values = _as_finite_real_array(value, name)
    if values.ndim == 0:
        raise ValueError(f"{name} must have a class axis, not be scalar")
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise TypeError("axis must be an integer, excluding booleans")
    if not -values.ndim <= axis < values.ndim:
        raise ValueError("axis is outside the input dimensions")
    axis = int(axis) % values.ndim
    if values.shape[axis] == 0:
        raise ValueError("the class axis must be nonempty")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError(f"{name} probabilities must lie in [0,1]")
    if not np.all(np.abs(np.sum(values, axis=axis) - 1.0) <= 1e-12):
        raise ValueError(f"{name} must sum to 1 on the class axis (atol=1e-12, rtol=0)")
    return values, axis


def _pair(p: ArrayLike, q: ArrayLike, axis: int) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Validate equally shaped target/model distributions, rejecting all broadcasting."""
    target, normalized_axis = _distribution(p, axis, "p")
    model, _ = _distribution(q, axis, "q")
    if target.shape != model.shape:
        raise ValueError("p and q must have exactly the same shape; broadcasting is not allowed")
    return target, model, normalized_axis


def _sum_classes(terms: NDArray[np.float64], axis: int) -> float | NDArray[np.float64]:
    """Sum information terms along the validated class axis, returning float for shape ()."""
    result = np.sum(terms, axis=axis)
    return float(result) if np.ndim(result) == 0 else result


def entropy(p: ArrayLike, axis: int = -1) -> float | NDArray[np.float64]:
    """Return H(p)=-sum(p*ln(p)) in nats, reducing the indicated class axis.

    p has at least one dimension, finite real probabilities in [0,1], and
    class sums 1 within atol=1e-12, rtol=0. The result has p.shape with axis
    removed (a Python float for a single vector); empty batch axes are allowed.
    The convention 0*ln(0)=0 is implemented with masks, without clipping.
    Invalid types raise TypeError; invalid values, shape or axis raise ValueError.
    The input and global RNG are unchanged. Computation uses float64.
    """
    values, axis = _distribution(p, axis, "p")
    logarithms = np.zeros_like(values)
    np.log(values, out=logarithms, where=values > 0)
    with np.errstate(under="ignore"):
        terms = -values * logarithms
    return _sum_classes(terms, axis)


def cross_entropy(p: ArrayLike, q: ArrayLike, axis: int = -1) -> float | NDArray[np.float64]:
    """Return CE(p,q)=-sum(p*ln(q)) in nats with exact support-boundary semantics.

    p and q must have identical shape, real finite entries in [0,1], a nonempty
    class axis and class sums 1 within atol=1e-12, rtol=0. There is no implicit
    broadcasting, clipping or normalization. Terms where p=0 are zero even
    when q=0; any p>0,q=0 contributes +inf. Output removes axis, returning a
    Python float for a single vector or float64 ndarray for batches. Invalid
    types raise TypeError; invalid values/shapes/axis raise ValueError. Caller
    arrays and global RNG are unchanged. Natural logs give nats, not bits.
    """
    target, model, axis = _pair(p, q, axis)
    positive_support = (target > 0) & (model > 0)
    logarithms = np.zeros_like(model)
    np.log(model, out=logarithms, where=positive_support)
    with np.errstate(under="ignore"):
        terms = -target * logarithms
    terms[(target > 0) & (model == 0)] = np.inf
    return _sum_classes(terms, axis)


def kl_divergence(p: ArrayLike, q: ArrayLike, axis: int = -1) -> float | NDArray[np.float64]:
    """Return KL(p||q)=sum(p*(ln(p)-ln(q))) in nats, reducing the class axis.

    The difference of logs avoids overflow in p/q for tiny positive q. Inputs
    obey cross_entropy's exact-shape, [0,1], finite, normalization and axis
    contracts. A zero p term contributes zero; p>0,q=0 contributes +inf. No
    epsilon or clipping is used, including for small negative roundoff in the
    final sum. Output is a Python float for a vector or float64 batch ndarray
    with the class axis removed. TypeError/ValueError identify invalid input.
    Inputs and RNG are unchanged; finite-support CE=H+KL holds up to roundoff.
    """
    target, model, axis = _pair(p, q, axis)
    positive_support = (target > 0) & (model > 0)
    target_log, model_log = np.zeros_like(target), np.zeros_like(model)
    np.log(target, out=target_log, where=positive_support)
    np.log(model, out=model_log, where=positive_support)
    with np.errstate(under="ignore"):
        terms = target * (target_log - model_log)
    terms[(target > 0) & (model == 0)] = np.inf
    return _sum_classes(terms, axis)
