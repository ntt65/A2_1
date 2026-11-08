"""Reproduce the mission's Adam and Newton comparisons with declared budgets."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.run_optimizer_sensitivity import radius_summary
from src.adam import Adam
from src.calculus import quadratic, quadratic_gradient
from src.newton import NewtonMethod, circle_hessian, elliptic_hessian
from src.optimizer import (
    Momentum, VanillaGD, elliptic, elliptic_gradient,
    optimizer_comparison_figure, run_optimization,
)
from src.utils.cache_manager import CacheManager, make_cache_key
from src.utils.math import _as_finite_real_array
from src.utils.numerical import _finite_scalar
from src.utils.random import set_seed

__concepts__ = {
    "Bonus_Optimizer_Comparison": ["trajectory", "run_experiment", "main"],
    "Bonus_Optimizer_Cache": ["trajectory"],
}


def trajectory(objective_name: str, method: str, lr: float, steps: int = 200,
               initial=(5.0, 5.0), gradient_tolerance=None,
               cache: CacheManager | None = None) -> dict:
    """Compute/cache a declared quadratic path(k+1,2) and loss/norm histories.

    Names are circle/ellipse and GD/Momentum/Adam/Newton. All inputs become
    float64; initial is exactly (2,), steps a nonnegative integer, and optional
    tolerance a finite nonnegative scalar. Method hyperparameters are fixed
    and recorded. Cache identity includes all numerical inputs, dtype/shape,
    versions, seed and stopping policy. Invalid inputs propagate exceptions.
    No RNG is consumed and cached returns own their arrays.
    """
    if objective_name not in ("circle", "ellipse"):
        raise ValueError("objective_name must be circle or ellipse")
    if method not in ("GD", "Momentum", "Adam", "Newton"):
        raise ValueError("method must be GD, Momentum, Adam or Newton")
    point = _as_finite_real_array(initial, "initial").copy()
    if point.shape != (2,):
        raise ValueError("initial must have shape (2,)")
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)):
        raise TypeError("steps must be an integer")
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    tolerance = None if gradient_tolerance is None else _finite_scalar(gradient_tolerance, "gradient_tolerance")
    if tolerance is not None and tolerance < 0:
        raise ValueError("gradient_tolerance must be nonnegative")
    hessian = circle_hessian if objective_name == "circle" else elliptic_hessian
    if method == "GD":
        optimizer, settings = VanillaGD(lr), {}
    elif method == "Momentum":
        optimizer, settings = Momentum(lr, beta=0.9), {"beta": 0.9}
    elif method == "Adam":
        settings = {"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8}
        optimizer = Adam(lr=lr, **settings)
    else:
        optimizer, settings = NewtonMethod(hessian, lr=lr), {"hessian_policy": "symmetric_positive_definite"}
    objective = (lambda p: quadratic(p[0], p[1])) if objective_name == "circle" else elliptic
    gradient = quadratic_gradient if objective_name == "circle" else elliptic_gradient
    parameters = {"objective": objective_name, "method": method, "lr": optimizer.lr,
                  "settings": settings, "steps_budget": int(steps),
                  "gradient_tolerance": tolerance, "seed": 42, "numpy_version": np.__version__}

    def compute() -> dict:
        """Run a fresh optimizer and return finite histories plus radius statistics."""
        result = run_optimization(objective, gradient, point, optimizer, steps=int(steps),
                                  gradient_tolerance=tolerance)
        return {**parameters, "initial": point.copy(), "path": result.path,
                "losses": result.losses, "gradient_norms": result.gradient_norms,
                "iterations": result.iterations, "status": result.status,
                **radius_summary(np.linalg.norm(result.path, axis=1))}

    if cache is None:
        return compute()
    key = make_cache_key("bonus_optimizer", "adam-corrected-newton-spd-v1", point, parameters)
    return cache.get_or_compute(key, compute)


def json_record(record: dict) -> dict:
    """Convert the flat record's NumPy arrays to lists without altering scalar metadata."""
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in record.items()}


def report_markdown(report: dict) -> str:
    """Format measured speed/accuracy and distinguish update counts from total work."""
    lines = ["# Adam and Newton: executed comparison", "",
             "Reproduce: `python -m scripts.run_bonus_optimizers --no-cache`.", "",
             "float64; seed 42; start=(5,5). Adam: beta1=.9, beta2=.999, epsilon=1e-8; "
             "Momentum: beta=.9. Same rate within each three-method comparison; 200 updates.", "",
             "| Objective | Method | lr | Updates | Final radius | Final loss | First <=.1 | Stay through budget |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, methods in report["adam_comparisons"].items():
        for method, record in methods.items():
            lines.append(f"| {name} | {method} | {record['lr']} | {record['iterations']} | "
                         f"{record['final_radius']:.6e} | {record['losses'][-1]:.6e} | "
                         f"{record['first_radius_0_1']} | {record['remaining_inside_radius_0_1_through_budget']} |")
    lines += ["", "None means the criterion was not achieved; a first crossing can be followed by an exit. "
              "A finite-horizon stay is not an asymptotic guarantee. These fixed settings do not establish "
              "that one method is universally faster; equal rates have different meanings for normalized Adam updates.", "",
              "## Newton versus GD", "",
              "Ellipse x^2+10y^2, same initial point, maximum 200 updates, "
              "same stopping criterion ||gradient|| <= 1e-8. GD lr=.05; pure Newton lr=1.", "",
              "| Method | Accepted updates | Final gradient norm | Final loss | Stop |",
              "|---|---:|---:|---:|---|"]
    for method, record in report["newton_comparison"].items():
        lines.append(f"| {method} | {record['iterations']} | {record['gradient_norms'][-1]:.6e} | "
                     f"{record['losses'][-1]:.6e} | {record['status']} |")
    lines += ["", "Newton solves H*d=g with H=diag(2,20), hence it reaches this quadratic's minimizer "
              "in one update. It also requires Hessian construction and a dense linear solve per update "
              "(generally O(n^2) storage and O(n^3) solve work). Update counts are not runtime comparisons. "
              "The implementation requires a finite symmetric positive-definite Hessian and rejects singular/indefinite inputs. "
              "Fixed damping does not guarantee global convergence on arbitrary nonlinear objectives.", ""]
    for figure in report["figures"]:
        lines += [f"![{figure['filename']}]({figure['filename']})", ""]
    return "\n".join(lines)


def run_experiment(output_dir: Path, use_cache: bool = True) -> dict:
    """Generate three PNGs plus JSON/Markdown after independent quadratic checks.

    The execution boundary seeds once. Existing artifacts are not rewritten.
    Paths are (201,2) in fixed-budget comparisons; early-stop Newton/GD paths
    contain accepted updates plus their initial point. Failed assertions or
    filesystem errors propagate and each Figure is closed after saving.
    """
    set_seed(42)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = CacheManager(".cache/bonus_optimizers", enabled=use_cache)
    comparisons, figures = {}, []
    for name, rate in (("circle", 0.1), ("ellipse", 0.05)):
        runs = {method: trajectory(name, method, rate, cache=cache) for method in ("GD", "Momentum", "Adam")}
        for record in runs.values():
            assert record["path"].shape == (201, 2) and record["iterations"] == 200
            assert record["status"] == "budget_exhausted"
            assert np.all(np.isfinite(record["path"]))
        factors = np.array([1-2*rate, 1-2*rate if name == "circle" else 1-20*rate])
        expected = np.full((201, 2), 5.0) * factors[None, :]**np.arange(201)[:, None]
        np.testing.assert_allclose(runs["GD"]["path"], expected, atol=1e-14, rtol=1e-12)
        surface = quadratic if name == "circle" else (lambda x, y: x*x + 10*y*y)
        figure, _ = optimizer_comparison_figure(
            runs, surface, f"{name.title()}: GD / Momentum / Adam, lr={rate}, start=(5,5), 200 updates")
        filename = f"bonus_adam_{name}.png"
        try:
            figure.savefig(output_dir / filename, dpi=140)
        finally:
            plt.close(figure)
        figures.append({"filename": filename, "dpi": 140})
        comparisons[name] = {method: json_record(run) for method, run in runs.items()}
    newton = {method: trajectory("ellipse", method, rate, gradient_tolerance=1e-8, cache=cache)
              for method, rate in (("GD", 0.05), ("Newton", 1.0))}
    np.testing.assert_allclose(newton["Newton"]["path"][-1], [0, 0], atol=1e-12, rtol=0)
    assert newton["Newton"]["iterations"] == 1
    assert all(record["status"] == "gradient_tolerance" for record in newton.values())
    figure, _ = optimizer_comparison_figure(
        newton, lambda x, y: x*x+10*y*y,
        "Ellipse: GD lr=.05 / Newton lr=1; start=(5,5); max 200; gradient tolerance 1e-8")
    try:
        figure.savefig(output_dir / "bonus_newton_ellipse.png", dpi=140)
    finally:
        plt.close(figure)
    figures.append({"filename": "bonus_newton_ellipse.png", "dpi": 140})
    report = {"seed": 42, "dtype": "float64", "numpy_version": np.__version__,
              "adam_comparisons": comparisons,
              "newton_comparison": {method: json_record(run) for method, run in newton.items()},
              "figures": figures, "all_assertions_passed": True}
    (output_dir / "bonus_optimizer_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    (output_dir / "bonus_optimizer_report.md").write_text(report_markdown(report), encoding="utf-8", newline="\n")
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the declared optimizer experiments; successful checks return zero."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)
    report = run_experiment(args.output_dir, not args.no_cache)
    for name, methods in report["adam_comparisons"].items():
        print(name+": "+"; ".join(f"{method} radius={r['final_radius']:.6e}, stay={r['remaining_inside_radius_0_1_through_budget']}"
                                  for method, r in methods.items()))
    print("Newton/GD accepted updates: "+str({method: r["iterations"] for method, r in report["newton_comparison"].items()}))
    print(f"Bonus optimizer PNG, JSON and Markdown saved in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
