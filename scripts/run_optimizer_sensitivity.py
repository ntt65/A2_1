"""Reproduce deterministic GD/Momentum sensitivity over five declared starts."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.calculus import quadratic, quadratic_gradient
from src.optimizer import Momentum, VanillaGD, elliptic, elliptic_gradient, run_optimization
from src.utils.cache_manager import CacheManager, make_cache_key
from src.utils.math import _as_finite_real_array
from src.utils.numerical import _finite_scalar
from src.utils.plotting import contour_grid
from src.utils.random import set_seed

__concepts__ = {
    "Optimizer_Sensitivity": ["trajectory", "radius_summary", "aggregate", "run_experiment"],
    "Optimizer_Stability_Evidence": ["stability_evidence"],
}

INITIAL_POINTS = ((5.0, 5.0), (-5.0, 5.0), (1.0, -3.0), (-2.0, -1.0), (0.0, 0.0))


def radius_summary(radii, radius: float = 0.1) -> dict:
    """Summarize finite radii (k+1,) including t=0; invalid data raise.

    first_radius_0_1 is the first recorded crossing. remaining_inside starts
    immediately after the last radius greater than the threshold, if any
    accepted point remains. Neither quantity guarantees an infinite-horizon
    property. Missing crossings/stays are None, never the budget value.
    """
    values = _as_finite_real_array(radii, "radii")
    threshold = _finite_scalar(radius, "radius")
    if values.ndim != 1 or not values.size or np.any(values < 0):
        raise ValueError("radii must be a nonempty nonnegative vector")
    if threshold <= 0:
        raise ValueError("radius must be positive")
    inside = np.flatnonzero(values <= threshold)
    outside = np.flatnonzero(values > threshold)
    stay = int(outside[-1] + 1) if outside.size else 0
    return {
        "radius_threshold": threshold,
        "first_radius_0_1": int(inside[0]) if inside.size else None,
        "remaining_inside_radius_0_1_through_budget": stay if stay < len(values) else None,
        "final_radius": float(values[-1]),
    }


def trajectory(objective_name: str, method: str, initial, lr: float,
               steps: int = 100, beta: float = 0.9, cache: CacheManager | None = None) -> dict:
    """Return a cached float64 path(k+1,2) with aligned loss/norm arrays.

    Reuse the existing GD/Momentum implementation for circle or ellipse.
    Initial shape is exactly (2,); steps is a nonnegative integer; the core
    validates lr and beta. Cache identity includes all input coordinates,
    shape/dtype, algorithm/objective version, rates, budget, seed and NumPy.
    Invalid names/types/shapes raise. The function does not consume RNG.
    """
    if objective_name not in ("circle", "ellipse") or method not in ("GD", "Momentum"):
        raise ValueError("use circle/ellipse and GD/Momentum")
    point = _as_finite_real_array(initial, "initial").copy()
    if point.shape != (2,):
        raise ValueError("initial must have shape (2,)")
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)):
        raise TypeError("steps must be an integer")
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    optimizer = VanillaGD(lr) if method == "GD" else Momentum(lr, beta)
    objective = (lambda p: quadratic(p[0], p[1])) if objective_name == "circle" else elliptic
    gradient = quadratic_gradient if objective_name == "circle" else elliptic_gradient
    parameters = {
        "objective": objective_name, "method": method, "lr": optimizer.lr,
        "beta": optimizer.beta if method == "Momentum" else None,
        "steps": int(steps), "gradient_tolerance": None, "seed": 42,
        "numpy_version": np.__version__, "radius_threshold": 0.1,
    }

    def compute() -> dict:
        """Compute a fresh finite trajectory and its finite-horizon radius summary."""
        result = run_optimization(objective, gradient, point, optimizer, steps=int(steps))
        radii = np.linalg.norm(result.path, axis=1)
        return {**parameters, "initial": point.copy(), "path": result.path,
                "losses": result.losses, "gradient_norms": result.gradient_norms,
                "iterations": result.iterations, "status": result.status,
                **radius_summary(radii)}

    if cache is None:
        return compute()
    key = make_cache_key("optimizer_sensitivity", "quadratic-sensitivity-v1", point, parameters)
    return cache.get_or_compute(key, compute)


def aggregate(records: list[dict]) -> dict:
    """Summarize a nonempty declared collection without substituting missing hits.

    Means/variances of crossing/staying updates use only successful records;
    counts and denominators remain explicit. Final-radius statistics include
    every record. Population variances use ddof=0; this deterministic set is
    not an estimate of a random population. Malformed records raise.
    """
    if not records:
        raise ValueError("records must be nonempty")
    summary = {"sample_count": len(records), "variance_ddof": 0}
    for field in ("first_radius_0_1", "remaining_inside_radius_0_1_through_budget"):
        values = [record[field] for record in records if record[field] is not None]
        summary[field] = {
            "success_count": len(values), "failure_count": len(records) - len(values),
            "mean_successful_updates": float(np.mean(values)) if values else None,
            "variance_successful_updates": float(np.var(values, ddof=0)) if values else None,
        }
    final = np.asarray([record["final_radius"] for record in records], dtype=np.float64)
    summary["final_radius_all_samples"] = {
        "mean": float(np.mean(final)), "variance": float(np.var(final, ddof=0)),
        "minimum": float(np.min(final)), "maximum": float(np.max(final)),
    }
    return summary


def stability_evidence() -> list[dict]:
    """Verify circle GD at lr=.5/1/1.1 by the closed recurrence, returning scalars.

    For theta0=(5,5), theta[t]=(1-2lr)**t theta0 and loss multiplier is
    (1-2lr)**2. Twenty finite updates establish the observed behavior; the
    theoretical classification follows this specific quadratic recurrence.
    Failed independent checks raise AssertionError.
    """
    evidence = []
    for rate, classification in ((0.5, "one_step_convergence"),
                                 (1.0, "constant_radius_sign_oscillation"),
                                 (1.1, "divergence_for_nonzero_initial")):
        record = trajectory("circle", "GD", [5.0, 5.0], rate, steps=20)
        multiplier = 1.0 - 2.0 * rate
        expected = np.full((21, 2), 5.0) * multiplier ** np.arange(21)[:, None]
        np.testing.assert_allclose(record["path"], expected, atol=1e-12, rtol=1e-13)
        assert np.all(np.isfinite(record["losses"]))
        nonzero = record["losses"][:-1] != 0
        observed = record["losses"][1:][nonzero] / record["losses"][:-1][nonzero]
        np.testing.assert_allclose(observed, multiplier**2, atol=1e-13, rtol=1e-13)
        evidence.append({
            "lr": rate, "steps": 20, "initial": [5.0, 5.0],
            "coordinate_multiplier": multiplier, "theoretical_loss_multiplier": multiplier**2,
            "observed_loss_ratio_min": float(np.min(observed)),
            "observed_loss_ratio_max": float(np.max(observed)),
            "observed_ratio_count": int(len(observed)), "classification": classification,
            "final_radius": record["final_radius"], "all_values_finite": True,
            "observed_strict_loss_increase": bool(np.all(np.diff(record["losses"]) > 0)),
        })
    return evidence


def sensitivity_figure(experiments: dict):
    """Return Figure/Axes(2,2) with each method's five paths on equal spatial axes.

    Inputs are the validated experiment records. Coordinates are unitless;
    separate panels prevent method/start legends from obscuring trajectories.
    The caller saves/closes the Figure. Invalid records propagate errors.
    """
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), layout="constrained")
    colors = ("#245c9e", "#d06a19", "#14866a", "#a2428c", "#272727")
    try:
        for row, (objective, rates) in enumerate(experiments.items()):
            surface = quadratic if objective == "circle" else (lambda x, y: x*x + 10*y*y)
            x, y, z = contour_grid(surface, (-6, 6), (-6, 6), points=101)
            for column, (method, group) in enumerate(rates.items()):
                axis = axes[row, column]
                axis.contour(x, y, z, levels=12, cmap="Greys", alpha=0.4)
                for index, record in enumerate(group["runs"]):
                    path = np.asarray(record["path"])
                    initial = record["initial"]
                    axis.plot(path[:, 0], path[:, 1], "--", color=colors[index], linewidth=1.2,
                              label=f"s{index+1}: ({initial[0]:g}, {initial[1]:g})")
                    axis.scatter(*path[0], color=colors[index], s=35, marker="s", zorder=4)
                    axis.scatter(*path[-1], color=colors[index], s=40, marker="x", zorder=5)
                axis.set(xlabel="x (unitless)", ylabel="y (unitless)", xlim=(-6, 6), ylim=(-6, 6),
                         title=f"{objective.title()} / {method}, lr={group['lr']:g}")
                axis.set_aspect("equal", adjustable="box")
                axis.legend(fontsize=8, loc="lower right")
        figure.suptitle("Sensitivity to five fixed initial points: GD and Momentum", fontsize=14)
        figure.supxlabel("100 updates; Momentum beta=0.9; square=start, x=end; zero start included", fontsize=10)
        return figure, axes
    except Exception:
        plt.close(figure)
        raise


def _json_record(record: dict) -> dict:
    """Convert one flat record's arrays to lists; scalar measurements remain unchanged."""
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in record.items()}


def report_markdown(report: dict) -> str:
    """Format tested sensitivity/statistics and recurrence evidence as reusable Markdown."""
    lines = ["# Optimizer sensitivity and stability evidence", "",
             "Reproduce: `python -m scripts.run_optimizer_sensitivity --no-cache`.", "",
             "Five declared starts: (5,5), (-5,5), (1,-3), (-2,-1), (0,0). "
             "float64; seed 42; 100 updates; Momentum beta=0.9; radius threshold=0.1.", "",
             "This is a deterministic sensitivity set, not repeated random trials. "
             "The optimum start is deliberately included. Statistics use population variance (ddof=0). "
             "Missing hits remain null; first crossing does not imply staying inside. "
             "A stay is measured only through the recorded budget.", "",
             "| Objective | Method | Start | lr | Updates | Final radius | First <=0.1 | Stay through budget |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for name, methods in report["experiments"].items():
        for method, group in methods.items():
            for run in group["runs"]:
                lines.append(f"| {name} | {method} | {run['initial']} | {run['lr']:g} | {run['iterations']} "
                             f"| {run['final_radius']:.6e} | {run['first_radius_0_1']} "
                             f"| {run['remaining_inside_radius_0_1_through_budget']} |")
    lines += ["", "## Conditional success statistics", "",
              "Means/variances below include only starts meeting the corresponding criterion. "
              "The JSON additionally includes all-start final-radius statistics and every raw trajectory.", "",
              "| Objective | Method | Metric | Success / all | Mean updates | Variance updates² |",
              "|---|---|---|---:|---:|---:|"]
    for name, methods in report["experiments"].items():
        for method, group in methods.items():
            for field, label in (("first_radius_0_1", "first crossing"),
                                 ("remaining_inside_radius_0_1_through_budget", "stay through budget")):
                item = group["summary"][field]
                lines.append(f"| {name} | {method} | {label} | {item['success_count']} / 5 "
                             f"| {item['mean_successful_updates']} | {item['variance_successful_updates']} |")
    lines += ["", "## Circle GD stability", "",
              "For a nonzero start, theta[t]=(1-2lr)^t theta[0]; loss multiplies by (1-2lr)^2. "
              "Classification follows this exact quadratic recurrence, not an arbitrary divergence threshold.", "",
              "| lr | Coordinate factor | Loss factor | Observed ratio range | Classification |",
              "|---:|---:|---:|---|---|"]
    for item in report["stability"]:
        lines.append(f"| {item['lr']} | {item['coordinate_multiplier']:.6g} "
                     f"| {item['theoretical_loss_multiplier']:.6g} "
                     f"| [{item['observed_loss_ratio_min']:.6g}, {item['observed_loss_ratio_max']:.6g}] "
                     f"| {item['classification']} |")
    lines += ["", "Ratios with zero denominators are excluded; lr=0.5 has only the first ratio (0). "
              "All 20 observed updates are finite. Newton/Adam and claims of universal superiority are outside this report.",
              "", "![Fixed-start trajectories](optimizer_sensitivity.png)", ""]
    return "\n".join(lines)


def run_experiment(output_dir: Path, use_cache: bool = True) -> dict:
    """Write PNG/JSON/Markdown for the fixed experiment; return JSON-safe evidence.

    Set seed42 exactly once at this execution boundary. Cache-off bypasses
    reads/writes. Filesystem errors and independent assertion failures
    propagate; each run must have (101,2) path and finite aligned histories.
    """
    set_seed(42)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = CacheManager(".cache/optimizer_sensitivity", enabled=use_cache)
    experiments = {}
    for name, rate in (("circle", 0.1), ("ellipse", 0.05)):
        methods = {}
        for method in ("GD", "Momentum"):
            records = [trajectory(name, method, start, rate, cache=cache) for start in INITIAL_POINTS]
            for record in records:
                assert record["path"].shape == (101, 2) and record["iterations"] == 100
                assert record["status"] == "budget_exhausted"
                if method == "GD":
                    factors = np.array([1-2*rate, 1-2*rate if name == "circle" else 1-20*rate])
                    expected = record["initial"] * factors[None, :] ** np.arange(101)[:, None]
                    np.testing.assert_allclose(record["path"], expected, atol=1e-14, rtol=1e-12)
            methods[method] = {"lr": rate, "summary": aggregate(records),
                               "runs": [_json_record(record) for record in records]}
        experiments[name] = methods
    report = {"seed": 42, "dtype": "float64", "numpy_version": np.__version__,
              "initial_points": [list(p) for p in INITIAL_POINTS], "steps": 100,
              "experiments": experiments, "stability": stability_evidence(),
              "figure": {"filename": "optimizer_sensitivity.png", "format": "PNG",
                         "width_pixels": 1680, "height_pixels": 1400, "dpi": 140},
              "interpretation": "Deterministic fixed-start sensitivity, not random retry statistics; "
              "conditional crossing means exclude failures, and finite-horizon stays are not asymptotic guarantees."}
    figure, _ = sensitivity_figure(experiments)
    try:
        figure.savefig(output_dir / "optimizer_sensitivity.png", dpi=140)
    finally:
        plt.close(figure)
    (output_dir / "optimizer_sensitivity.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    (output_dir / "optimizer_sensitivity.md").write_text(report_markdown(report), encoding="utf-8", newline="\n")
    return report


def main(argv: list[str] | None = None) -> int:
    """Execute the fixed experiment CLI with output path and optional cache bypass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)
    report = run_experiment(args.output_dir, use_cache=not args.no_cache)
    for objective, methods in report["experiments"].items():
        for method, group in methods.items():
            summary = group["summary"]["remaining_inside_radius_0_1_through_budget"]
            print(f"{objective} / {method}: stay <=0.1 in {summary['success_count']}/5 starts; "
                  f"conditional mean={summary['mean_successful_updates']} updates")
    print(f"Sensitivity PNG, JSON and Markdown saved in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
