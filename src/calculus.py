"""Calculus utilities for gradients and numerical differentiation."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)


def numerical_derivative(func: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """Compute derivative by central difference: f'(x) ≈ (f(x+h)-f(x-h))/(2h)."""
    return float((func(x + h) - func(x - h)) / (2 * h))


def verify_quadratic_derivative() -> dict:
    """Verify f(x)=x^2 derivative at x=3 against analytical value 6."""
    f = lambda v: v**2
    estimate = numerical_derivative(f, 3.0)
    analytical = 6.0
    error = abs(estimate - analytical)
    return {
        "estimate": estimate,
        "analytical": analytical,
        "absolute_error": error,
        "within_1e-4": error <= 1e-4,
    }


def gradient_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return gradient of f(x,y)=x^2+y^2, i.e., (2x, 2y)."""
    return 2 * x, 2 * y


def visualize_gradient_contours(output_path: str | Path = "outputs/gradient_contour.png") -> Path:
    """Plot contours of f(x,y)=x^2+y^2 and gradient arrows."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    xx, yy = np.meshgrid(x, y)
    zz = xx**2 + yy**2

    sample = np.array(
        [
            [-2.0, -1.0],
            [-1.5, 1.5],
            [1.0, -2.0],
            [2.0, 1.0],
            [0.5, 2.5],
        ]
    )
    gx, gy = gradient_xy(sample[:, 0], sample[:, 1])

    plt.figure(figsize=(7, 6))
    cs = plt.contour(xx, yy, zz, levels=15, cmap="viridis")
    plt.clabel(cs, inline=1, fontsize=8)
    plt.quiver(sample[:, 0], sample[:, 1], gx, gy, color="red", angles="xy", scale_units="xy", scale=8)
    plt.title("Gradient vectors on contours of f(x,y)=x²+y²")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path
