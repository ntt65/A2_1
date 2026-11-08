"""Independent fixed-quadratic fixtures for the pure optimizer report renderer."""

import copy
import json
from pathlib import Path
import unittest

import numpy as np

from scripts.report_optimizer import render_optimizer

__covers__ = ["scripts/report_optimizer.py"]


def _run_fixture(objective: str, method: str, lr: float, steps: int) -> dict:
    """Build a synthetic measurement fixture from closed-form coordinate solutions.

    GD uses powers of 1-lr*h. Momentum uses the real trigonometric solution
    of r²-(1+.9-lr*h)r+.9=0, independent of the iterative production code.
    Histories have shape (steps+1,2) and include the fixed start (5,5).
    """
    hessian = np.array([2.0, 2.0 if objective == "circle" else 20.0])
    times = np.arange(steps+1)[:, None]
    if method == "GD":
        path = 5.0*(1.0-lr*hessian)**times
    else:
        radius = np.sqrt(0.9)
        angle = np.arccos((1.9-lr*hessian)/(2.0*radius))
        coefficient = ((1.0-lr*hessian)/radius-np.cos(angle))/np.sin(angle)
        path = 5.0*radius**times*(np.cos(times*angle)+coefficient*np.sin(times*angle))
    losses = np.sum(path**2*(hessian/2.0), axis=1)
    norms = np.linalg.norm(path*hessian, axis=1)
    radii = np.linalg.norm(path, axis=1)
    hits, outside = np.flatnonzero(radii <= 0.1), np.flatnonzero(radii > 0.1)
    stay = int(outside[-1]+1) if outside.size else 0
    return {"objective": objective, "method": method, "lr": lr,
            "beta": 0.9 if method == "Momentum" else None, "steps": steps,
            "iterations": steps, "gradient_tolerance": None, "status": "budget_exhausted",
            "initial": [5.0, 5.0], "path": path.tolist(), "losses": losses.tolist(),
            "gradient_norms": norms.tolist(), "final_norm": float(radii[-1]),
            "first_radius_0_1": int(hits[0]) if hits.size else None,
            "remaining_inside_radius_0_1_through_budget": stay if stay < len(path) else None}


def _fixture() -> dict:
    """Return all seven fixed report records using only independent closed forms."""
    return {"seed": 42, "dtype": "float64", "comparisons": {
        name: {method: _run_fixture(name, method, rate, 100) for method in ("GD", "Momentum")}
        for name, rate in (("circle", 0.1), ("ellipse", 0.05))},
        "learning_rates": {str(rate): _run_fixture("circle", "GD", rate, 20) for rate in (0.5, 1.0, 1.1)}}


class OptimizerReportTests(unittest.TestCase):
    """Verify readable numbers, preservation and refusal of inconsistent evidence."""

    def test_independent_fixture_reports_all_conditions_and_checkpoints(self):
        """Known closed forms yield measured values, learning context and preserved figure links."""
        report = _fixture()
        original = copy.deepcopy(report)
        text = render_optimizer(report)
        assert report == original
        assert "1.440402e-09" in text and "2.074758e-18" in text
        assert "2.710878e+02" in text and "7.348858e+04" in text
        assert "| 원형 Momentum | 0.1 |" in text
        assert "| 10 | 77 |" in text
        assert "| 타원형 GD | 0.05 |" in text and "| 38 | 38 |" in text
        assert "| 0.5 | 0 | 0 |" in text and "| 1 | -1 | 1 |" in text
        assert "| 1.1 | -1.2 | 1.44 |" in text
        assert "seed=42, dtype=float64" in text and "shape=(101,2)" in text and "shape=(21,2)" in text
        assert "100회까지" in text and "보편적인 우월성" in text
        assert "미션 연결: #9~#12" in text
        for explanation in ("관성이 남아 진동", "왼쪽에서 등고선 위 경로", "각 패널이 자기 좌표"):
            assert explanation in text
        for label, image in (("원형 경로·손실·gradient", "optimizer_circle.png"),
                             ("타원형 경로·손실·gradient", "optimizer_ellipse.png"),
                             ("학습률별 경로", "optimizer_learning_rates.png")):
            assert f"[{label}]({image})" in text
            self.assertRegex(text, r"!\[[^\]]+\]\(" + image.replace(".", r"\.") + r"\)")
        for image in ("optimizer_circle.png", "optimizer_ellipse.png", "optimizer_learning_rates.png"):
            assert f"]({image})" in text
        for condition in ("원형 GD", "원형 Momentum", "타원형 GD", "타원형 Momentum"):
            for step in (0, 1, 2, 5, 10, 20, 50, 100):
                assert f"| {condition} | {step} | (" in text
        for rate in ("0.5", "1", "1.1"):
            for step in (0, 1, 5, 10, 20):
                assert f"| {rate} | {step} | (" in text

    def test_real_committed_measurements_render_without_mutation(self):
        """The actual generator schema is accepted and its raw input remains byte-equivalent."""
        path = Path(__file__).resolve().parents[1]/"outputs"/"optimizer_metrics.json"
        source = path.read_text(encoding="utf-8")
        report = json.loads(source)
        before = json.dumps(report, sort_keys=True)
        text = render_optimizer(report)
        assert "1.018518e-09" in text and "2.016252e-02" in text
        assert "1.328070e-04" in text and "1.987508e-02" in text
        assert before == json.dumps(report, sort_keys=True)
        assert path.read_text(encoding="utf-8") == source

    def test_injected_summary_errors_are_rejected(self):
        """Wrong final radius, first/stay indices and labels cannot become a successful report."""
        for field, value in (("final_norm", 0.0), ("first_radius_0_1", 0),
                             ("remaining_inside_radius_0_1_through_budget", 10),
                             ("steps", 99), ("iterations", 99), ("lr", 0.2),
                             ("beta", 0.8), ("status", "gradient_tolerance")):
            report = _fixture()
            report["comparisons"]["circle"]["Momentum"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                render_optimizer(report)

    def test_inconsistent_history_nonfinite_and_missing_data_are_rejected(self):
        """Altered losses, norms, paths, shape or metadata fail instead of printing plausible results."""
        for field, value in (("losses", 99.0), ("gradient_norms", 99.0), ("losses", np.inf)):
            report = _fixture()
            report["comparisons"]["circle"]["GD"][field][1] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                render_optimizer(report)
        report = _fixture()
        report["learning_rates"]["1.1"]["path"] = [[5.0, 5.0]]
        with self.assertRaises(ValueError):
            render_optimizer(report)
        report = _fixture()
        del report["comparisons"]["ellipse"]["GD"]
        with self.assertRaises(ValueError):
            render_optimizer(report)
        for field, value in (("seed", 0), ("dtype", "float32")):
            report = _fixture()
            report[field] = value
            with self.assertRaises(ValueError):
                render_optimizer(report)
        report = _fixture()
        run = report["comparisons"]["circle"]["GD"]
        # Internally consistent but unsuccessful histories must not be labeled within radius 0.1.
        run["path"] = [[5.0, 5.0]]*101
        run["losses"] = [50.0]*101
        run["gradient_norms"] = [np.sqrt(200.0)]*101
        run["final_norm"] = np.sqrt(50.0)
        run["first_radius_0_1"] = None
        run["remaining_inside_radius_0_1_through_budget"] = None
        with self.assertRaisesRegex(ValueError, "final-radius bound"):
            render_optimizer(report)


if __name__ == "__main__":
    unittest.main()
