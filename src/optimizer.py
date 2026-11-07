"""Optimization module implementing Vanilla GD and Momentum."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)


class VanillaGD:
    """Vanilla gradient descent optimizer."""

    def __init__(self, lr: float):
        self.lr = lr

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Apply one gradient descent step."""
        return params - self.lr * grads


class Momentum:
    """Momentum optimizer."""

    def __init__(self, lr: float, beta: float = 0.9):
        self.lr = lr
        self.beta = beta
        self.velocity: np.ndarray | None = None

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Apply one momentum step."""
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        self.velocity = self.beta * self.velocity + grads
        return params - self.lr * self.velocity


def f_quadratic(p: np.ndarray) -> float:
    """f(x,y)=x^2+y^2."""
    return float(p[0] ** 2 + p[1] ** 2)


def grad_quadratic(p: np.ndarray) -> np.ndarray:
    """Gradient of f(x,y)=x^2+y^2."""
    return np.array([2 * p[0], 2 * p[1]], dtype=float)


def f_elliptic(p: np.ndarray) -> float:
    """f(x,y)=x^2+10y^2."""
    return float(p[0] ** 2 + 10 * p[1] ** 2)


def grad_elliptic(p: np.ndarray) -> np.ndarray:
    """Gradient of f(x,y)=x^2+10y^2."""
    return np.array([2 * p[0], 20 * p[1]], dtype=float)


def optimize_path(optimizer, grad_fn, start: np.ndarray, steps: int = 100) -> np.ndarray:
    """Return optimization trajectory including start point."""
    params = start.astype(float).copy()
    path = [params.copy()]
    for _ in range(steps):
        grads = grad_fn(params)
        params = optimizer.step(params, grads)
        path.append(params.copy())
    return np.array(path)


def reached_origin(path: np.ndarray, radius: float = 0.1) -> bool:
    """Check if final point is within radius from origin."""
    return bool(np.linalg.norm(path[-1]) <= radius)


def visualize_path(function_name: str, path_dict: dict, output_path: str | Path) -> Path:
    """Visualize optimizer paths on contour plot."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.linspace(-6, 6, 300)
    y = np.linspace(-6, 6, 300)
    xx, yy = np.meshgrid(x, y)

    if function_name == "quadratic":
        zz = xx**2 + yy**2
    elif function_name == "elliptic":
        zz = xx**2 + 10 * yy**2
    else:
        raise ValueError("function_name must be quadratic or elliptic")

    plt.figure(figsize=(7, 6))
    plt.contour(xx, yy, zz, levels=30, cmap="viridis")

    for label, path in path_dict.items():
        plt.plot(path[:, 0], path[:, 1], "o--", linewidth=1.5, markersize=3, label=label)

    plt.title(f"Convergence Path ({function_name})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def run_optimizer_experiments(output_dir: str | Path = "outputs") -> dict:
    """Run required optimizer experiments and save figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = np.array([5.0, 5.0])

    vanilla = optimize_path(VanillaGD(lr=0.1), grad_quadratic, start, steps=100)
    momentum = optimize_path(Momentum(lr=0.1, beta=0.9), grad_quadratic, start, steps=100)
    diverging = optimize_path(VanillaGD(lr=1.1), grad_quadratic, start, steps=30)

    visualize_path(
        "quadratic",
        {
            "Vanilla GD (lr=0.1)": vanilla,
            "Momentum (lr=0.1, beta=0.9)": momentum,
        },
        output_dir / "optimizer_quadratic_compare.png",
    )

    visualize_path(
        "quadratic",
        {
            "Vanilla GD (lr=1.1)": diverging,
        },
        output_dir / "optimizer_divergence_lr_1_1.png",
    )

    vanilla_elliptic = optimize_path(VanillaGD(lr=0.05), grad_elliptic, start, steps=100)
    momentum_elliptic = optimize_path(Momentum(lr=0.05, beta=0.9), grad_elliptic, start, steps=100)

    visualize_path(
        "elliptic",
        {
            "Vanilla GD (lr=0.05)": vanilla_elliptic,
            "Momentum (lr=0.05, beta=0.9)": momentum_elliptic,
        },
        output_dir / "optimizer_elliptic_compare.png",
    )

    return {
        "quadratic_vanilla_final": vanilla[-1],
        "quadratic_momentum_final": momentum[-1],
        "quadratic_vanilla_within_radius_0_1": reached_origin(vanilla, radius=0.1),
        "quadratic_divergence_max_norm": float(np.max(np.linalg.norm(diverging, axis=1))),
        "elliptic_vanilla_final": vanilla_elliptic[-1],
        "elliptic_momentum_final": momentum_elliptic[-1],
    }
