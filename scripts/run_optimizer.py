"""Reproduce Step 4 optimizer paths, objective histories, and stability cases."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.calculus import quadratic, quadratic_gradient
from src.optimizer import (
    Momentum, VanillaGD, elliptic, elliptic_gradient, learning_rate_figure,
    optimizer_comparison_figure, run_optimization,
)
from src.utils.cache_manager import CacheManager, make_cache_key
from src.utils.random import set_seed

__concepts__ = {
    "Optimizer_Trajectory_Cache": ["cached_trajectory"],
    "Optimizer_Experiment": ["run_experiment", "main"],
}


def cached_trajectory(
    objective_name: str, method: str, lr: float, steps: int = 100,
    beta: float = 0.9, cache: CacheManager | None = None,
) -> dict:
    """Compute/cache one fixed two-dimensional experiment from initial [5,5].

    objective_name is 'circle' or 'ellipse'; method is 'GD' or 'Momentum'.
    The record stores path(k+1,2), losses/norms(k+1), metadata and termination.
    Key inputs include the entire initial point, objective/algorithm version,
    method, lr, beta, budget, tolerance policy and NumPy version. No RNG is
    consumed. Invalid names raise ValueError; optimizer input errors propagate.
    """
    if objective_name not in ("circle", "ellipse"):
        raise ValueError("objective_name must be circle or ellipse")
    if method not in ("GD", "Momentum"):
        raise ValueError("method must be GD or Momentum")
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)):
        raise TypeError("steps must be an integer")
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    steps = int(steps)
    initial = np.array([5.0, 5.0])
    optimizer = VanillaGD(lr) if method == "GD" else Momentum(lr, beta)
    objective = (lambda p: quadratic(p[0], p[1])) if objective_name == "circle" else elliptic
    gradient = quadratic_gradient if objective_name == "circle" else elliptic_gradient
    parameters = {
        "objective": objective_name, "method": method, "lr": float(lr),
        "beta": float(beta) if method == "Momentum" else None,
        "steps": steps, "gradient_tolerance": None, "numpy": np.__version__,
    }

    def compute() -> dict:
        """Run with a fresh optimizer and return supported cache values and raw histories."""
        result = run_optimization(objective, gradient, initial, optimizer, steps=steps)
        radii = np.linalg.norm(result.path, axis=1)
        hits = np.flatnonzero(radii <= 0.1)
        outside = np.flatnonzero(radii > 0.1)
        remaining_start = int(outside[-1] + 1) if outside.size else 0
        return {
            **parameters, "initial": initial.copy(), "path": result.path,
            "losses": result.losses, "gradient_norms": result.gradient_norms,
            "iterations": result.iterations, "status": result.status,
            "final_norm": float(radii[-1]),
            "first_radius_0_1": int(hits[0]) if hits.size else None,
            "remaining_inside_radius_0_1_through_budget": remaining_start if remaining_start < len(radii) else None,
        }

    if cache is None:
        return compute()
    key = make_cache_key("optimizer_trajectory", "fixed-quadratics-v1", initial, parameters)
    return cache.get_or_compute(key, compute)


def _json_record(record: dict) -> dict:
    """Convert a flat trajectory record's ndarrays to JSON lists without changing values."""
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in record.items()}


def run_experiment(output_dir: Path, use_cache: bool = True) -> dict:
    """Write three comparison PNGs and a JSON report after independent assertions.

    Uses seed42 and fixed initial [5,5]. Circle/ellipse comparisons both have
    100 updates; stability panels have 20 finite updates. Cached calculations
    and uncached calculations have the same numerical contract. File errors
    and failed assertions propagate; all figures are closed after saving.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    cache = CacheManager(".cache/optimizer", enabled=use_cache)
    comparisons = {}
    for name, rate, surface in (
        ("circle", 0.1, quadratic), ("ellipse", 0.05, lambda x, y: x*x + 10*y*y),
    ):
        runs = {method: cached_trajectory(name, method, rate, cache=cache) for method in ("GD", "Momentum")}
        assert all(run["path"].shape == (101, 2) and run["iterations"] == 100 for run in runs.values())
        assert all(run["status"] == "budget_exhausted" for run in runs.values())
        if name == "circle":
            expected = np.full(2, 5.0 * 0.8**100)
            np.testing.assert_allclose(runs["GD"]["path"][-1], expected, atol=1e-22, rtol=1e-12)
            assert runs["GD"]["final_norm"] <= 0.1
        figure, _ = optimizer_comparison_figure(
            runs, surface, f"{name.title()}: lr={rate:g}, Momentum beta=0.9, start=(5,5), 100 updates",
        )
        try:
            figure.savefig(output_dir / f"optimizer_{name}.png", dpi=140)
        finally:
            plt.close(figure)
        comparisons[name] = {method: _json_record(run) for method, run in runs.items()}

    learning_rates = {rate: cached_trajectory("circle", "GD", rate, steps=20, cache=cache)
                      for rate in (0.5, 1.0, 1.1)}
    np.testing.assert_array_equal(learning_rates[0.5]["path"][1:], np.zeros((20, 2)))
    np.testing.assert_array_equal(learning_rates[1.0]["path"],
                                  np.tile([5.0, 5.0], (21, 1)) * (-1.0)**np.arange(21)[:, None])
    assert np.all(np.diff(learning_rates[1.1]["losses"]) > 0)
    assert np.all(np.isfinite(learning_rates[1.1]["path"]))
    figure, _ = learning_rate_figure(learning_rates)
    try:
        figure.savefig(output_dir / "optimizer_learning_rates.png", dpi=140)
    finally:
        plt.close(figure)
    report = {
        "seed": 42, "numpy_version": np.__version__, "dtype": "float64",
        "comparisons": comparisons,
        "learning_rates": {str(rate): _json_record(run) for rate, run in learning_rates.items()},
        "notes": [
            "Momentum uses v_next=beta*v+gradient, theta_next=theta-lr*v_next, with v0=0.",
            "Fixed budgets include the initial point; optional early stopping is not used here.",
            "Circle GD convergence is 0<lr<1; lr=0.5 reaches zero in one update.",
            "Ellipse GD convergence is 0<lr<0.1; lr=0.05 eliminates y in one update.",
            "Momentum is not asserted to be faster at every step or for every setting.",
            "first_radius_0_1 is a first crossing, not a guarantee of remaining inside the radius.",
            "remaining_inside_radius_0_1_through_budget describes only the recorded finite horizon.",
        ],
    }
    (output_dir / "optimizer_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
        encoding="utf-8", newline="\n",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the reproducible optimizer CLI with optional output folder/cache bypass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--no-cache", action="store_true")
    arguments = parser.parse_args(argv)
    report = run_experiment(arguments.output_dir, not arguments.no_cache)
    for name, runs in report["comparisons"].items():
        print(f"{name}: " + "; ".join(f"{method} radius={run['final_norm']:.6e}, loss={run['losses'][-1]:.6e}"
                                      for method, run in runs.items()))
    print(f"Figures and metrics saved in {arguments.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
