"""Reproduce Step 3 differentiation errors and gradient geometry with NumPy."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.calculus import (
    derivative_error_figure, derivative_error_study, gradient_figure,
    numerical_derivative, numerical_gradient, quadratic, quadratic_gradient,
)
from src.utils.random import set_seed

__concepts__ = {"Calculus_Experiment": ["run_experiment", "main"]}


def run_experiment(output_dir: Path) -> dict:
    """Write two PNGs and JSON with asserted scalar and vector errors.

    The experiment uses float64, seed42, x=3/h=1e-5, eight radius-two
    points and the origin. Numerical gradients have shape (9,2). Geometry
    is checked only at nonzero points; a zero vector has no direction.
    File creation errors and failed numerical assertions propagate.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    derivative = numerical_derivative(lambda x: x*x, 3.0, h=1e-5)
    derivative_error = abs(derivative - 6.0)
    assert derivative_error <= 1e-4

    angles = np.arange(8) * np.pi / 4
    points = np.vstack((2*np.column_stack((np.cos(angles), np.sin(angles))), [0, 0]))
    analytic = quadratic_gradient(points)
    numeric = np.array([
        numerical_gradient(lambda p: quadratic(p[0], p[1]), point, h=1e-5)
        for point in points
    ])
    np.testing.assert_allclose(analytic, 2*points, atol=0, rtol=0)
    gradient_error = float(np.max(np.abs(numeric - analytic)))
    assert gradient_error <= 1e-8
    tangent = np.column_stack((-points[:-1, 1], points[:-1, 0]))
    normal_unit = analytic[:-1] / np.linalg.norm(analytic[:-1], axis=1, keepdims=True)
    tangent_unit = tangent / np.linalg.norm(tangent, axis=1, keepdims=True)
    normalized_dot = np.sum(normal_unit * tangent_unit, axis=1)
    orthogonality_error = float(np.max(np.abs(normalized_dot)))
    assert orthogonality_error <= 1e-12
    np.testing.assert_array_equal(analytic[-1], [0, 0])
    np.testing.assert_array_equal(numeric[-1], [0, 0])

    sweep = derivative_error_study()
    report = {
        "seed": 42, "numpy_version": np.__version__, "dtype": "float64",
        "derivative": {
            "function": "x**2", "x": 3.0, "h": 1e-5,
            "numerical": derivative, "analytic": 6.0,
            "absolute_error": derivative_error, "tolerance": 1e-4,
        },
        "gradient": {
            "function": "x**2+y**2", "points": points.tolist(),
            "analytic": analytic.tolist(), "numerical": numeric.tolist(),
            "maximum_absolute_error": gradient_error, "tolerance": 1e-8,
            "nonzero_tangents": tangent.tolist(),
            "normalized_dot_products": normalized_dot.tolist(),
            "maximum_normalized_dot": orthogonality_error,
            "orthogonality_tolerance": 1e-12,
            "origin_gradient": analytic[-1].tolist(), "origin_direction": None,
        },
        "step_size_study": {key: np.asarray(value).tolist() for key, value in sweep.items()},
        "notes": [
            "Quadratic central differences have zero truncation error in exact arithmetic.",
            "Sine central differences have O(h**2) truncation error; small h suffers roundoff.",
            "No monotonic error decrease or universal best h is asserted.",
            "The origin gradient is zero and has no direction.",
            "Arrow lengths are normalized for display; metric gradients are unnormalized.",
        ],
    }
    for filename, make_figure in (
        ("gradient_geometry.png", gradient_figure),
        ("derivative_error.png", derivative_error_figure),
    ):
        figure, _ = make_figure()
        try:
            figure.savefig(output_dir / filename, dpi=140)
        finally:
            plt.close(figure)
    (output_dir / "calculus_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
        encoding="utf-8", newline="\n",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the Step 3 experiment from CLI; --output-dir selects a writable folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    arguments = parser.parse_args(argv)
    report = run_experiment(arguments.output_dir)
    print(f"Derivative error: {report['derivative']['absolute_error']:.3e}; "
          f"gradient error: {report['gradient']['maximum_absolute_error']:.3e}; "
          f"orthogonality: {report['gradient']['maximum_normalized_dot']:.3e}")
    print(f"Figures and metrics saved in {arguments.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
