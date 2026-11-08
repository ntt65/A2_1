"""Numerical derivatives and the geometry of f(x, y) = x**2 + y**2."""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.utils.math import _as_finite_real_array
from src.utils.numerical import central_difference

__concepts__ = {
    "Numerical_Derivative": ["numerical_derivative"],
    "Numerical_Gradient": ["numerical_gradient"],
    "Quadratic_Surface": ["quadratic"],
    "Quadratic_Gradient": ["quadratic_gradient"],
    "Gradient_Visualization": ["gradient_figure"],
    "Derivative_Error_Study": ["derivative_error_study"],
    "Derivative_Error_Visualization": ["derivative_error_figure"],
}


def numerical_derivative(
    function: Callable[[float], float], x: float, h: float = 1e-5
) -> float:
    """Return the scalar central difference of a scalar function at x.

    Delegate (f(x+h)-f(x-h))/(2h) and input checks to the shared utility.
    x, h, and function outputs have shape (); h must be positive, finite,
    and distinguish both samples from x in float64. Invalid types raise
    TypeError; invalid shapes, steps, or nonfinite results raise ValueError.
    User function exceptions propagate. The function must be deterministic
    and have three continuous derivatives near x for the usual O(h**2)
    truncation estimate. A finite result does not establish differentiability.
    """
    return central_difference(function, x, h)


def numerical_gradient(
    function: Callable[[NDArray[np.float64]], float],
    point: ArrayLike,
    h: float = 1e-5,
) -> NDArray[np.float64]:
    """Approximate a scalar function's gradient at a nonempty point (n,).

    For coordinate i, use the shared central difference of
    t -> function(point with coordinate i replaced by t). The returned
    float64 array has shape (n,). Each evaluation receives an independent
    copy, so mutation of a sample cannot change the input or other samples.
    The objective should be deterministic and sufficiently smooth near point;
    a finite result does not establish differentiability. Its output must be a finite real
    scalar. A positive finite scalar h must resolve both samples for every
    coordinate. Invalid numeric types raise TypeError; invalid shapes,
    sampling points, steps, or outputs raise ValueError. User exceptions
    propagate, and no random state is consumed or reset.
    """
    if not callable(function):
        raise TypeError("function must be callable")
    values = _as_finite_real_array(point, "point")
    if values.ndim != 1 or values.size == 0:
        raise ValueError("point must be a nonempty one-dimensional array")
    values = values.copy()
    gradient = np.empty_like(values)
    for index in range(values.size):

        def along_coordinate(coordinate: float) -> float:
            """Evaluate the scalar objective on an isolated sample of shape (n,)."""
            sample = values.copy()
            sample[index] = coordinate
            return function(sample)

        gradient[index] = central_difference(along_coordinate, float(values[index]), h)
    return gradient


def quadratic(x: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
    """Return x**2 + y**2 with the broadcast shape of real finite x and y.

    Scalars produce a float64 array of shape (); arrays use NumPy's standard
    broadcasting. Inputs are not modified. Non-real inputs raise TypeError;
    incompatible shapes, nonfinite inputs, or overflowing results raise
    ValueError. As with ordinary float64 arithmetic, tiny squares can round
    to zero; this function does not provide arbitrary precision.
    """
    x_values = _as_finite_real_array(x, "x")
    y_values = _as_finite_real_array(y, "y")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = np.asarray(x_values**2 + y_values**2, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("quadratic result must be representable as finite float64")
    return result


def quadratic_gradient(points: ArrayLike) -> NDArray[np.float64]:
    """Return the analytical gradient (2x, 2y) for points with shape (..., 2).

    Accept one point (2,) or a nonempty batch with a final coordinate axis
    of length two; return a new float64 array with the same shape. The origin
    maps to the zero vector, which has no direction. Non-real input raises
    TypeError; empty/incorrect shapes, nonfinite inputs, and overflowing
    results raise ValueError. The input is never modified.
    """
    values = _as_finite_real_array(points, "points")
    if values.ndim == 0 or values.shape[-1] != 2 or values.size == 0:
        raise ValueError("points must be nonempty with shape (..., 2)")
    with np.errstate(over="ignore", invalid="ignore"):
        result = 2.0 * values
    if not np.all(np.isfinite(result)):
        raise ValueError("quadratic gradient must be representable as finite float64")
    return result




def _gradient_samples() -> np.ndarray:
    """Return eight radius-two circle samples and the origin, shape (9,2).

    The eight angles are 0, pi/4, ..., 7*pi/4. A fresh float64 array is
    returned, so callers may modify it without changing later figures.
    """
    angles = np.arange(8, dtype=np.float64) * (np.pi / 4.0)
    circle = 2.0 * np.column_stack((np.cos(angles), np.sin(angles)))
    return np.vstack((circle, np.zeros((1, 2))))


def gradient_figure():
    """Return a Figure and Axes showing gradients normal to circular contours.

    The fixed objective is x**2+y**2 on [-3.5,3.5] in both coordinates.
    Eight samples on f=4 carry outward gradient and counterclockwise tangent
    arrows. Both are normalized for display (lengths 0.7 and 0.55 coordinate
    units), not a claim that their mathematical magnitudes are equal.
    The origin has zero gradient and is marked without a direction arrow.
    This function saves nothing; the caller owns Figure cleanup.
    """
    from src.utils.plotting import contour_base
    import matplotlib.pyplot as plt

    figure, axes = contour_base(
        quadratic, (-3.5, 3.5), (-3.5, 3.5), points=141, levels=12,
        title="Gradient is normal to a level curve: f(x,y) = x² + y²",
    )
    try:
        figure.set_size_inches(8.4, 7.2)
        figure.set_layout_engine("constrained")
        samples = _gradient_samples()
        points = samples[:-1]
        gradients = quadratic_gradient(points)
        gradient_directions = gradients / np.linalg.norm(gradients, axis=1, keepdims=True)
        tangents = np.column_stack((-points[:, 1], points[:, 0]))
        tangent_directions = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)
        angles = np.linspace(0.0, 2.0 * np.pi, 361)
        axes.plot(2.0 * np.cos(angles), 2.0 * np.sin(angles), color="#424242",
                  linewidth=1.5, linestyle="--", label="Sample level curve: f = 4")
        axes.scatter(points[:, 0], points[:, 1], color="#303030", s=18, zorder=4)
        axes.quiver(
            points[:, 0], points[:, 1], gradient_directions[:, 0], gradient_directions[:, 1],
            angles="xy", scale_units="xy", scale=1.0 / 0.7, color="#cf4b16",
            width=0.006, zorder=5, label="Gradient direction (2x, 2y)",
        )
        axes.quiver(
            points[:, 0], points[:, 1], tangent_directions[:, 0], tangent_directions[:, 1],
            angles="xy", scale_units="xy", scale=1.0 / 0.55, color="#2466b0",
            width=0.005, zorder=5, label="Tangent direction (-y, x)",
        )
        axes.scatter([0.0], [0.0], marker="x", color="#202020", s=55, zorder=6)
        axes.annotate("Origin: zero gradient\n(no direction)", xy=(0.0, 0.0),
                      xytext=(0.15, -0.55), fontsize=9,
                      bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"})
        axes.set(xlim=(-3.5, 3.5), ylim=(-3.5, 3.5))
        axes.legend(loc="upper right", fontsize=8, framealpha=0.95)
        figure.supxlabel("Arrow lengths are normalized for display; axes use equal scales.", fontsize=9)
        return figure, axes
    except Exception:
        plt.close(figure)
        raise


def derivative_error_study() -> dict[str, np.ndarray]:
    """Return central-difference estimates and errors at x=3 for 61 step sizes.

    Each dict value is float64 shape (61,): h spans 1e-15 through 1, and
    quadratic_derivative/quadratic_error compare x**2 with exact derivative 6;
    sine_derivative/sine_error compare sin(x) with cos(3). All calls reuse
    numerical_derivative. Errors are raw absolute values, including exact
    floating-point zeros. Smaller h need not improve accuracy monotonically.
    """
    steps = np.logspace(-15.0, 0.0, 61)
    quadratic_estimates = np.array([
        numerical_derivative(lambda x: x * x, 3.0, float(h)) for h in steps
    ])
    sine_estimates = np.array([
        numerical_derivative(np.sin, 3.0, float(h)) for h in steps
    ])
    return {
        "h": steps,
        "quadratic_derivative": quadratic_estimates,
        "quadratic_error": np.abs(quadratic_estimates - 6.0),
        "sine_derivative": sine_estimates,
        "sine_error": np.abs(sine_estimates - np.cos(3.0)),
    }


def derivative_error_figure():
    """Return a Figure and Axes comparing step-size errors at x=3.

    The x-axis is logarithmic. The absolute-error y-axis uses symlog with a
    linear band [0,1e-15] so exact zero errors remain zero and are visible;
    no artificial positive floor replaces the numerical measurements.
    x**2 has no central-difference truncation error in exact arithmetic.
    sin(x) illustrates the leading abs(cos(3))*h**2/6 truncation term.
    No file is saved; the caller owns Figure cleanup.
    """
    import matplotlib.pyplot as plt

    study = derivative_error_study()
    figure, axes = plt.subplots(figsize=(9.0, 5.5), layout="constrained")
    try:
        steps = study["h"]
        axes.plot(steps, study["quadratic_error"], marker="o", markersize=3,
                  color="#cf4b16", label="x²: measured error (exact derivative 6)")
        axes.plot(steps, study["sine_error"], marker="s", markersize=3,
                  color="#2466b0", label="sin(x): measured error (exact derivative cos(3))")
        axes.plot(steps, abs(np.cos(3.0)) * steps**2 / 6.0,
                  linestyle="--", color="#555555", label="sin(x): leading O(h²) truncation term")
        axes.set_xscale("log")
        axes.set_yscale("symlog", linthresh=1e-15)
        maximum = max(float(np.max(study["quadratic_error"])), float(np.max(study["sine_error"])))
        axes.set(ylim=(0.0, maximum * 2.0), xlabel="Central-difference step h",
                 ylabel="Absolute derivative error",
                 title="Step-size error at x = 3\nQuadratic central difference has no truncation error in exact arithmetic")
        axes.grid(True, which="major", alpha=0.25)
        axes.legend(fontsize=8, loc="upper center")
        figure.supxlabel("Y scale: logarithmic above 1e-15, linear near zero; zero errors are retained.", fontsize=9)
        return figure, axes
    except Exception:
        plt.close(figure)
        raise
