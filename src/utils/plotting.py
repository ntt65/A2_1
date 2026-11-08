"""Reusable contour grids and figures, independent of any mission objective."""

from collections.abc import Callable, Sequence

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.utils.math import _as_finite_real_array

__concepts__ = {"Contour_Grid": ["contour_grid"], "Contour_Plot": ["contour_base"]}


def _interval(bounds: Sequence[float], name: str) -> tuple[float, float]:
    """Validate a finite increasing interval of shape (2,), or raise an error."""
    interval = _as_finite_real_array(bounds, name)
    if interval.shape != (2,) or interval[0] >= interval[1]:
        raise ValueError(f"{name} must contain two strictly increasing bounds")
    return float(interval[0]), float(interval[1])


def contour_grid(
    function: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    x_range: Sequence[float] = (-5.0, 5.0),
    y_range: Sequence[float] = (-5.0, 5.0),
    points: int = 101,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return X, Y, f(X,Y), each float64 with shape (points, points).

    The x/y ranges are finite, increasing pairs. points is an integer >= 2.
    function must accept the two mesh arrays and return finite real values of
    the exact same shape; scalar broadcasting is deliberately not inferred.
    Invalid argument types raise TypeError; bad ranges, sizes, output shapes,
    and non-finite grids or function outputs raise ValueError.
    """
    if not callable(function):
        raise TypeError("function must be callable")
    x_limits, y_limits = _interval(x_range, "x_range"), _interval(y_range, "y_range")
    if isinstance(points, (bool, np.bool_)) or not isinstance(points, (int, np.integer)):
        raise TypeError("points must be an integer")
    if points < 2:
        raise ValueError("points must be at least 2")
    with np.errstate(over="ignore", invalid="ignore"):
        x_values = np.linspace(*x_limits, points)
        y_values = np.linspace(*y_limits, points)
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
        raise ValueError("range is too wide to construct a finite grid")
    x_grid, y_grid = np.meshgrid(x_values, y_values, indexing="xy")
    values = _as_finite_real_array(function(x_grid, y_grid), "function output")
    if values.shape != x_grid.shape:
        raise ValueError("function output must have shape (points, points)")
    return x_grid, y_grid, values


def contour_base(
    function: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    x_range: Sequence[float] = (-5.0, 5.0),
    y_range: Sequence[float] = (-5.0, 5.0),
    points: int = 101,
    levels: int = 20,
    title: str = "Function contours",
) -> tuple[Figure, Axes]:
    """Draw a labeled contour base and return its Figure and Axes for reuse.

    Grid arguments follow contour_grid; levels is a positive integer.
    No file is saved or window shown, and the caller owns figure cleanup.
    x/y labels and equal axis scaling make geometric overlays comparable.
    Invalid grid arguments or levels raise TypeError or ValueError.
    """
    if isinstance(levels, (bool, np.bool_)) or not isinstance(levels, (int, np.integer)):
        raise TypeError("levels must be an integer")
    if levels < 1:
        raise ValueError("levels must be positive")
    x_grid, y_grid, values = contour_grid(function, x_range, y_range, points)
    figure, axes = plt.subplots()
    try:
        contours = axes.contour(x_grid, y_grid, values, levels=levels)
        axes.clabel(contours, inline=True, fontsize=8)
        axes.set(xlabel="x", ylabel="y", title=title)
        axes.set_aspect("equal", adjustable="box")
    except Exception:
        plt.close(figure)
        raise
    return figure, axes
