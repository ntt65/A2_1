"""NumPy gradient descent and momentum with finite, aligned trajectory records."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.utils.math import _as_finite_real_array
from src.utils.numerical import _finite_scalar

__concepts__ = {
    "Vanilla_GD": ["VanillaGD"],
    "Momentum_Update": ["Momentum"],
    "Optimization_Trajectory": ["OptimizationResult", "run_optimization"],
    "Elliptic_Objective": ["elliptic", "elliptic_gradient"],
    "Optimizer_Comparison_Visualization": ["optimizer_comparison_figure"],
    "Learning_Rate_Stability_Visualization": ["learning_rate_figure"],
}


class _NonfiniteResult(FloatingPointError):
    """Identify this module's nonrepresentable arithmetic or callback outputs."""


def _vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """Validate a finite, nonempty real vector (n,); invalid type/shape raises."""
    array = _as_finite_real_array(value, name)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    return array


def _step_inputs(theta: ArrayLike, gradient: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return finite theta and gradient vectors with matching (n,) shapes."""
    point = _vector(theta, "theta")
    derivative = _vector(gradient, "gradient")
    if point.shape != derivative.shape:
        raise ValueError("theta and gradient must have identical shapes")
    return point, derivative


def _finite_result(value: np.ndarray, name: str) -> np.ndarray:
    """Return finite computed data, or raise an internal FloatingPointError subtype."""
    if not np.all(np.isfinite(value)):
        raise _NonfiniteResult(f"{name} is not representable as finite float64")
    return value


class VanillaGD:
    """Stateless theta_next = theta - lr*gradient for nonempty vectors (n,).

    lr must be a positive finite scalar. Inputs are preserved and each step
    returns a new float64 array. Invalid inputs raise TypeError/ValueError;
    unrepresentable arithmetic raises FloatingPointError.
    """

    def __init__(self, lr: float):
        """Store a positive finite scalar learning rate; reject other values."""
        self.lr = _finite_scalar(lr, "lr")
        if self.lr <= 0:
            raise ValueError("lr must be positive")

    def reset(self) -> None:
        """Reset this stateless method; accept no state and return None."""

    def step(self, theta: ArrayLike, gradient: ArrayLike) -> NDArray[np.float64]:
        """Return theta-lr*gradient with shape (n,), without mutating either input."""
        point, derivative = _step_inputs(theta, gradient)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            candidate = point - self.lr * derivative
        return _finite_result(candidate, "GD update")


class Momentum(VanillaGD):
    """Momentum v_next=beta*v+gradient, theta_next=theta-lr*v_next, v0=0.

    lr is positive finite and beta is a finite scalar in [0,1). Input shapes
    are (n,) and must agree with established velocity until reset(). Failed
    updates do not commit velocity. velocity returns a copy, never internal
    mutable state. There is no (1-beta) factor in this convention.
    """

    def __init__(self, lr: float, beta: float = 0.9):
        """Validate scalar hyperparameters and initialize velocity to None."""
        super().__init__(lr)
        self.beta = _finite_scalar(beta, "beta")
        if not 0 <= self.beta < 1:
            raise ValueError("beta must satisfy 0 <= beta < 1")
        self._velocity = None

    @property
    def velocity(self) -> NDArray[np.float64] | None:
        """Return a copy of velocity (n,), or None before first update/reset."""
        return None if self._velocity is None else self._velocity.copy()

    def reset(self) -> None:
        """Clear velocity so the next step starts with a matching zero vector."""
        self._velocity = None

    def step(self, theta: ArrayLike, gradient: ArrayLike) -> NDArray[np.float64]:
        """Return a finite momentum update and atomically commit finite velocity."""
        point, derivative = _step_inputs(theta, gradient)
        if self._velocity is not None and self._velocity.shape != point.shape:
            raise ValueError("reset momentum before changing the parameter shape")
        previous = np.zeros_like(point) if self._velocity is None else self._velocity
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            velocity = self.beta * previous + derivative
            candidate = point - self.lr * velocity
        _finite_result(velocity, "momentum velocity")
        _finite_result(candidate, "momentum update")
        self._velocity = velocity.copy()
        return candidate


@dataclass(frozen=True)
class OptimizationResult:
    """Accepted trajectory: path(k+1,n), losses/gradient_norms(k+1), k updates.

    status is budget_exhausted, gradient_tolerance, or nonfinite. Arrays are
    owned by this result and contain finite values only; frozen fields do
    not make the underlying NumPy arrays immutable. A rejected candidate is
    absent from all histories, so iterations equals len(path)-1.
    """

    path: NDArray[np.float64]
    losses: NDArray[np.float64]
    gradient_norms: NDArray[np.float64]
    iterations: int
    status: str


def _output_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Validate callback type/shape, distinguishing nonfinite output from bad contracts."""
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numbers, excluding booleans")
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        array = array.astype(np.float64, copy=True)
    return _finite_result(array, name)


def _stable_norm(vector: np.ndarray) -> float:
    """Compute a scaled Euclidean norm for a finite vector, rejecting norm overflow."""
    scale = float(np.max(np.abs(vector)))
    if scale == 0:
        return 0.0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = scale * float(np.sqrt(np.sum((vector / scale)**2)))
    if not np.isfinite(result):
        raise _NonfiniteResult("gradient norm exceeds float64 range")
    return result


def _evaluate(objective: Callable, gradient: Callable, point: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Evaluate isolated point copies and return scalar loss, gradient(n,), norm."""
    loss = _output_array(objective(point.copy()), (), "objective output")
    derivative = _output_array(gradient(point.copy()), point.shape, "gradient output")
    return float(loss), derivative, _stable_norm(derivative)


def run_optimization(
    objective: Callable[[NDArray[np.float64]], float],
    gradient: Callable[[NDArray[np.float64]], ArrayLike],
    initial: ArrayLike,
    optimizer: VanillaGD,
    steps: int = 100,
    gradient_tolerance: float | None = None,
) -> OptimizationResult:
    """Run fixed-budget optimization on a scalar objective and point (n,).

    Each run resets optimizer state. Default steps=100 means up to 100
    accepted updates, with initial metrics retained. Optional nonnegative
    gradient_tolerance enables norm-based early stopping, including at the
    initial point. objective returns a scalar and gradient returns (n,).
    Each callback receives an isolated copy. Initial nonfinite metrics raise
    ValueError; later nonfinite outputs or this module's arithmetic stop with
    status nonfinite and restore the last accepted momentum state. Wrong
    types/shapes and exceptions raised by user callbacks propagate unchanged.
    Optimizer must be VanillaGD or Momentum; input initial is preserved.
    """
    if not callable(objective) or not callable(gradient):
        raise TypeError("objective and gradient must be callable")
    if not isinstance(optimizer, VanillaGD):
        raise TypeError("optimizer must be VanillaGD or Momentum")
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)):
        raise TypeError("steps must be an integer")
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    tolerance = None if gradient_tolerance is None else _finite_scalar(gradient_tolerance, "gradient_tolerance")
    if tolerance is not None and tolerance < 0:
        raise ValueError("gradient_tolerance must be nonnegative")
    point = _vector(initial, "initial").copy()
    optimizer.reset()
    try:
        loss, derivative, norm = _evaluate(objective, gradient, point)
    except _NonfiniteResult as error:
        raise ValueError("initial objective, gradient, and norm must be finite") from error
    path, losses, norms = [point.copy()], [loss], [norm]
    status = "budget_exhausted"
    for _ in range(int(steps)):
        if tolerance is not None and norm <= tolerance:
            status = "gradient_tolerance"
            break
        previous_velocity = optimizer.velocity if isinstance(optimizer, Momentum) else None
        try:
            candidate = optimizer.step(point, derivative)
            next_loss, next_derivative, next_norm = _evaluate(objective, gradient, candidate)
        except _NonfiniteResult:
            if isinstance(optimizer, Momentum):
                optimizer._velocity = previous_velocity
            status = "nonfinite"
            break
        point, loss, derivative, norm = candidate, next_loss, next_derivative, next_norm
        path.append(point.copy())
        losses.append(loss)
        norms.append(norm)
    if status == "budget_exhausted" and tolerance is not None and norm <= tolerance:
        status = "gradient_tolerance"
    return OptimizationResult(np.array(path), np.array(losses), np.array(norms), len(path)-1, status)


def elliptic(theta: ArrayLike) -> float:
    """Return x**2+10*y**2 for a finite real point (2,); overflow raises FloatingPointError."""
    point = _vector(theta, "theta")
    if point.shape != (2,):
        raise ValueError("elliptic theta must have shape (2,)")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = np.asarray(point[0]**2 + 10.0*point[1]**2)
    return float(_finite_result(result, "elliptic objective"))


def elliptic_gradient(theta: ArrayLike) -> NDArray[np.float64]:
    """Return (2*x,20*y) for a finite real point (2,); overflow raises FloatingPointError."""
    point = _vector(theta, "theta")
    if point.shape != (2,):
        raise ValueError("elliptic theta must have shape (2,)")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = point * np.array([2.0, 20.0])
    return _finite_result(result, "elliptic gradient")



def optimizer_comparison_figure(runs: dict, objective, title: str):
    """Return Figure/Axes(3,) for two-dimensional paths, losses, and gradient norms.

    runs maps display labels to records with finite path(k+1,2), losses(k+1,)
    and gradient_norms(k+1,). objective accepts two mesh arrays and returns
    the same shape. The experiment caller supplies validated recorded runs.
    Path axes use equal scales. Symlog preserves exact zero loss/norm values.
    No file is saved; the caller closes the returned Figure. Bad records raise.
    """
    import matplotlib.pyplot as plt
    from src.utils.plotting import contour_grid

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    try:
        extent = max(1.0, max(float(np.max(np.abs(run["path"]))) for run in runs.values())) * 1.15
        x, y, z = contour_grid(objective, (-extent, extent), (-extent, extent), points=101)
        contours = axes[0].contour(x, y, z, levels=12, cmap="Greys", alpha=0.55)
        axes[0].clabel(contours, inline=True, fontsize=7)
        for index, (label, run) in enumerate(runs.items()):
            color = ("#2466b0", "#cf4b16", "#29864a")[index % 3]
            path = np.asarray(run["path"])
            updates = np.arange(len(path))
            axes[0].plot(path[:, 0], path[:, 1], "o--", markersize=2.5,
                         linewidth=1.2, color=color, label=label)
            axes[0].scatter(*path[0], marker="s", color=color, s=45, zorder=4)
            axes[0].scatter(*path[-1], marker="x", color=color, s=50, zorder=5)
            axes[1].plot(updates, run["losses"], color=color, label=label)
            axes[2].plot(updates, run["gradient_norms"], color=color, label=label)
        axes[0].set(xlabel="x", ylabel="y", title="Paths (square: start, x: end)",
                    xlim=(-extent, extent), ylim=(-extent, extent))
        axes[0].set_aspect("equal", adjustable="box")
        for axis, ylabel, threshold in zip(axes[1:], ("Objective value", "Gradient norm"), (1e-18, 1e-12)):
            axis.set(xlabel="Completed updates", ylabel=ylabel, title=ylabel)
            axis.set_yscale("symlog", linthresh=threshold)
            axis.set_ylim(bottom=0)
            axis.grid(True, alpha=0.25)
        for axis in axes:
            axis.legend(fontsize=8)
        figure.suptitle(title, fontsize=12)
        figure.supxlabel("Same initial point and update budget; symlog axes retain zero values.", fontsize=9)
        return figure, axes
    except Exception:
        plt.close(figure)
        raise


def learning_rate_figure(runs: dict):
    """Return Figure/Axes(3,) for circle GD learning rates 0.5, 1.0, and 1.1.

    runs maps these numeric rates to validated trajectory records. Each panel
    gets its own equal x/y scales so divergence cannot hide the other cases.
    Initial and final coordinates are included; no values are clipped.
    No file is saved; the caller owns Figure cleanup. Invalid records raise.
    """
    import matplotlib.pyplot as plt
    from src.calculus import quadratic
    from src.utils.plotting import contour_grid

    figure, axes = plt.subplots(1, 3, figsize=(14.4, 4.8), layout="constrained")
    try:
        descriptions = {0.5: "One-step convergence", 1.0: "Undamped sign oscillation", 1.1: "Growing oscillation"}
        for axis, rate in zip(axes, (0.5, 1.0, 1.1)):
            run = runs[rate]
            path = np.asarray(run["path"])
            extent = max(1.0, float(np.max(np.abs(path)))) * 1.15
            x, y, z = contour_grid(quadratic, (-extent, extent), (-extent, extent), points=81)
            axis.contour(x, y, z, levels=10, cmap="Greys", alpha=0.5)
            axis.plot(path[:, 0], path[:, 1], "o--", color="#2466b0", markersize=4, linewidth=1,
                      label="GD path")
            axis.scatter(*path[0], marker="s", color="#cf4b16", s=55, zorder=5, label="Start")
            axis.scatter(*path[-1], marker="x", color="#202020", s=70, zorder=6, label="End")
            axis.set(xlabel="x", ylabel="y", xlim=(-extent, extent), ylim=(-extent, extent),
                     title=f"lr={rate:g}: {descriptions[rate]}\nFinal radius={run['final_norm']:.4g}")
            axis.set_aspect("equal", adjustable="box")
            axis.legend(fontsize=8, loc="upper left")
        figure.suptitle("Circle GD: theta[t+1] = (1 - 2 lr) theta[t]; start=(5,5), 20 updates", fontsize=12)
        figure.supxlabel("Each panel has its own coordinate range; lr=0.5 does not diverge.", fontsize=9)
        return figure, axes
    except Exception:
        plt.close(figure)
        raise
