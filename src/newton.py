"""Newton updates for finite real vectors with positive-definite Hessians.

The method solves a linear system instead of forming an inverse. It supports
fixed damping, not a line search or a global convergence guarantee.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.optimizer import VanillaGD, _finite_result, _output_array, _step_inputs, _vector

__concepts__ = {
    "Newton_Method": ["NewtonMethod"],
    "Quadratic_Hessian": ["circle_hessian", "elliptic_hessian"],
}


class NewtonMethod(VanillaGD):
    """Solve H(theta) direction = gradient and return theta - lr*direction.

    theta and gradient are finite real vectors (n,); hessian receives an
    isolated point copy and returns a finite real matrix (n,n). Matrices must
    be symmetric within atol=rtol=1e-12. Exactly symmetric matrices are not
    averaged; accepted asymmetric pairs use an overflow-safe midpoint,
    mirrored to both triangles before Cholesky and np.linalg.solve. This
    removes accepted asymmetry, not negative or singular curvature. Midpoints
    remain subject to float64 rounding when the exact value is unrepresentable.
    For wholly subnormal H, H and gradient are divided by max(abs(H)) before
    factorization/solve to avoid subnormal elimination; an unrepresentable
    scaled gradient is rejected rather than accepted as a finite direction.
    lr is a scalar in (0,1]; lr=1 is the full Newton step. The class is stateless
    and works with run_optimization, but fixed damping provides no general
    convergence guarantee for arbitrary objectives or starting points.

    Invalid input/callback types raise TypeError. Invalid shapes, damping,
    asymmetry, singular or non-positive-definite Hessians raise ValueError.
    Nonfinite Hessian output, factorization, direction or update raises the
    shared FloatingPointError subtype handled by the trajectory runner.
    Other callback exceptions propagate unchanged. Inputs are never modified.
    """

    def __init__(self, hessian: Callable[[NDArray[np.float64]], ArrayLike], lr: float = 1.0):
        """Store callable H:(n,)->(n,n) and finite damping 0<lr<=1; reject other inputs."""
        if not callable(hessian):
            raise TypeError("hessian must be callable")
        super().__init__(lr)
        if self.lr > 1:
            raise ValueError("Newton lr must satisfy 0 < lr <= 1")
        self.hessian = hessian

    def step(self, theta: ArrayLike, gradient: ArrayLike) -> NDArray[np.float64]:
        """Return a new float64 (n,) Newton update after validating SPD H(theta).

        The callback receives a copy of theta. Type/shape/curvature errors
        raise TypeError/ValueError; nonfinite computed values raise
        FloatingPointError. No Hessian inverse or eigensolver is called.
        """
        point, derivative = _step_inputs(theta, gradient)
        dimension = point.size
        matrix = _output_array(self.hessian(point.copy()), (dimension, dimension), "hessian output")
        with np.errstate(over="ignore", invalid="ignore"):
            symmetric = np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-12)
        if not symmetric:
            raise ValueError("hessian must be symmetric within atol=rtol=1e-12")
        if not np.array_equal(matrix, matrix.T):
            # Preserve exact symmetry, especially subnormal diagonal entries.
            # Same-sign subtraction cannot overflow; opposite-sign addition
            # cannot overflow. Mirror one midpoint so both triangles agree.
            row, column = np.triu_indices(dimension, k=1)
            upper, lower = matrix[row, column], matrix[column, row]
            same_sign = np.signbit(upper) == np.signbit(lower)
            midpoint = np.empty_like(upper)
            midpoint[same_sign] = upper[same_sign] + 0.5 * (lower[same_sign] - upper[same_sign])
            midpoint[~same_sign] = 0.5 * (upper[~same_sign] + lower[~same_sign])
            matrix[row, column] = midpoint
            matrix[column, row] = midpoint
        _finite_result(matrix, "Newton symmetric Hessian")
        system, right_hand_side = matrix, derivative
        scale = float(np.max(np.abs(matrix)))
        if 0 < scale < np.finfo(np.float64).tiny:
            # Solve the equivalent normalized system before LAPACK elimination.
            # An all-zero Hessian instead reaches the explicit SPD rejection.
            with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
                system = matrix / scale
                right_hand_side = derivative / scale
            _finite_result(right_hand_side, "Newton scaled gradient")
        try:
            with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
                factor = np.linalg.cholesky(system)
        except np.linalg.LinAlgError as error:
            raise ValueError("hessian must be positive definite and nonsingular") from error
        _finite_result(factor, "Newton Cholesky factor")
        try:
            with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
                direction = np.linalg.solve(system, right_hand_side)
        except np.linalg.LinAlgError as error:
            raise ValueError("hessian linear system is numerically singular") from error
        _finite_result(direction, "Newton direction")
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            candidate = point - self.lr * direction
        return _finite_result(candidate, "Newton update")


def circle_hessian(point: ArrayLike) -> NDArray[np.float64]:
    """Return 2*I with shape (2,2) for f(x,y)=x**2+y**2 at finite point (2,).

    The input is validated and preserved; every call returns an owned float64
    matrix. Non-real input raises TypeError; nonfinite or wrong shape raises
    ValueError, matching the shared vector contract.
    """
    value = _vector(point, "point")
    if value.shape != (2,):
        raise ValueError("circle Hessian point must have shape (2,)")
    return np.diag([2.0, 2.0])


def elliptic_hessian(point: ArrayLike) -> NDArray[np.float64]:
    """Return diag(2,20) with shape (2,2) for f(x,y)=x**2+10*y**2.

    point must be a finite real vector (2,) and is preserved. The returned
    float64 matrix is owned by the caller. Non-real input raises TypeError;
    nonfinite or wrong shape raises ValueError.
    """
    value = _vector(point, "point")
    if value.shape != (2,):
        raise ValueError("elliptic Hessian point must have shape (2,)")
    return np.diag([2.0, 20.0])
