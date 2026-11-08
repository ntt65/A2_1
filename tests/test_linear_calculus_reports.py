"""Independent content and dishonest-PASS regression checks for report renderers."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np

from scripts.report_linear_calculus import render_calculus, render_linear_algebra
from src.utils.random import set_seed

__covers__ = ["scripts/report_linear_calculus.py"]


class LinearCalculusReportTests(unittest.TestCase):
    """Check saved measurements against independent constants without rerunning experiments."""

    @classmethod
    def setUpClass(cls):
        """Read the two delivered JSON dictionaries and seed once at this test entry."""
        set_seed(42)
        output = Path(__file__).resolve().parents[1] / "outputs"
        cls.linear = json.loads((output / "linear_algebra_metrics.json").read_text(encoding="utf-8"))
        cls.calculus = json.loads((output / "calculus_metrics.json").read_text(encoding="utf-8"))

    def test_linear_contains_measured_inputs_and_all_svd_ranks(self):
        """Known determinants, eigenvalue and storage counts must appear with all three figures linked."""
        report = render_linear_algebra(self.linear)
        np.testing.assert_allclose(
            [row["determinant"] for row in self.linear["transforms"].values()],
            [1, 1, 1, -6, 0], atol=1e-14, rtol=0,
        )
        np.testing.assert_allclose(self.linear["power_iteration"]["eigenvalue"],
                                   4.618033988749895, atol=1e-14, rtol=0)
        np.testing.assert_allclose(
            [row["relative_error"] for row in self.linear["svd"][1]["comparisons"]],
            [.1275184713562091, .05938687046053636, .01218300917179587],
            atol=1e-14, rtol=0,
        )
        for text in ("[[1,1],[0,1]]", "[[-2,0],[0,3]]", "4.618033988749896",
                     "0.31494140625", "1.568603515625", "64×64", "128×128",
                     "linear_transforms.png", "svd_64.png", "svd_128.png",
                     "linear_algebra_metrics.json", "python -m scripts.run_linear_algebra",
                     "python -m scripts.run_submission_reports"):
            self.assertIn(text, report)
        self.assertNotIn("| FAIL |", report)
        self.assertIn("미션 연결: #1~#4", report)
        for explanation in ("회전해도 윤곽이 겹치므로", "고유값 문제와 구분", "같은 밝기 기준"):
            self.assertIn(explanation, report)
        for label, filename in (("변환 PNG", "linear_transforms.png"),
                                ("64×64 PNG", "svd_64.png"), ("128×128 PNG", "svd_128.png")):
            self.assertIn(f"[{label}]({filename})", report)
            self.assertRegex(report, r"!\[[^\]]+\]\(" + filename.replace(".", r"\.") + r"\)")

    def test_calculus_distinguishes_fixed_check_from_sample_sweep(self):
        """Independent x² constants show one fixed PASS and 46/61 passing sampled h values."""
        report = render_calculus(self.calculus)
        study = self.calculus["step_size_study"]
        np.testing.assert_allclose(self.calculus["derivative"]["numerical"],
                                   6.000000000039306, atol=1e-14, rtol=0)
        self.assertEqual(sum(error <= 1e-4 for error in study["quadratic_error"]), 46)
        np.testing.assert_allclose(max(study["quadratic_error"]),
                                   .6709294817992486, atol=1e-14, rtol=0)
        for text in ("46/61", "15/61", "0.6709294817992486", "(9,2)",
                     "6.000000000039306", "3.29634097795406e-11", "연속 구간",
                     "절단오차", "gradient_geometry.png", "derivative_error.png",
                     "calculus_metrics.json", "python -m scripts.run_calculus",
                     "python -m scripts.run_submission_reports"):
            self.assertIn(text, report)
        self.assertLessEqual(len(report.splitlines()), 55)

    def test_consistent_failing_measurements_render_fail(self):
        """Stale true flags cannot turn actual area, eigenpair, SVD or derivative failures into PASS."""
        report = deepcopy(self.linear)
        scaling = report["transforms"]["scaling"]
        scaling.update(area_after=2 * scaling["area_before"], area_ratio=2.0, error=1.0)
        text = render_linear_algebra(report)
        self.assertIn("| FAIL |", next(line for line in text.splitlines() if line.startswith("| scaling |")))
        report = deepcopy(self.linear)
        eigenpair = report["power_iteration"]
        # For v=(1,0), lambda=4: A v-lambda v=(0,1), ||A||F=sqrt(27).
        relative = (4.618033988749895 - 4.) / 4.618033988749895
        residual = 1. / np.sqrt(27.)
        eigenpair.update(eigenvalue=4., eigenvector=[1., 0.], relative_error=relative,
                         relative_residual=residual)
        eigenpair["numpy_eig_comparison"].update(relative_eigenvalue_error=relative,
                                                independent_relative_residual=residual)
        text = render_linear_algebra(report)
        self.assertIn("≤ 0.05: FAIL", text)
        self.assertIn("≤ 1e-6: FAIL", text)
        report = deepcopy(self.linear)
        report["svd"][0]["full_rank_relative_error"] = .01
        report["svd"][0]["comparisons"][-1]["relative_error"] = .01
        self.assertIn("완전 rank 오차=0.01 ≤ 1e-10: FAIL", render_linear_algebra(report))
        report = deepcopy(self.calculus)
        report["derivative"].update(numerical=7.0, absolute_error=1.0)
        self.assertIn("| 고정 중심차분 | 7 | 6 | 1 | 0.0001 | FAIL |", render_calculus(report))

    def test_changed_h_error_splits_the_passing_sample_groups(self):
        """One injected interior failure changes both counts and prevents bridging the sample gap."""
        report = deepcopy(self.calculus)
        study = report["step_size_study"]
        study["quadratic_derivative"][30] = 6.01
        study["quadratic_error"][30] = abs(6.01 - 6.)
        text = render_calculus(report)
        self.assertIn("PASS 45/61, FAIL 16/61", text)
        self.assertIn("(15표본);", text)
        self.assertIn("(30표본)", text)

    def test_inconsistent_or_nonfinite_evidence_is_rejected(self):
        """Tampered summaries, tolerances, identities and shapes fail instead of publishing stale PASS."""
        cases = []
        for field, value in (("error", .2), ("tolerance", 1.0), ("determinant", 2.0)):
            report = deepcopy(self.linear)
            report["transforms"]["scaling"][field] = value
            cases.append((render_linear_algebra, report))
        report = deepcopy(self.linear)
        report["power_iteration"]["numpy_eig_comparison"]["relative_eigenvalue_error"] = 0.2
        cases.append((render_linear_algebra, report))
        report = deepcopy(self.linear)
        report["power_iteration"]["eigenvector"] = [1., 0.]
        cases.append((render_linear_algebra, report))
        report = deepcopy(self.linear)
        report["svd"][1]["comparisons"][2]["stored_over_original"] = 0.1
        cases.append((render_linear_algebra, report))
        for field, value in (("maximum_absolute_error", 0.0), ("numerical", [[0., 0.]]),
                             ("origin_direction", [1., 0.]), ("maximum_normalized_dot", np.nan)):
            report = deepcopy(self.calculus)
            report["gradient"][field] = value
            cases.append((render_calculus, report))
        report = deepcopy(self.calculus)
        report["step_size_study"]["quadratic_error"][0] = 0.
        cases.append((render_calculus, report))
        for renderer, report in cases:
            with self.subTest(renderer=renderer.__name__), self.assertRaises(ValueError):
                renderer(report)

    def test_rendering_preserves_inputs_rng_and_is_deterministic(self):
        """Pure rendering must neither mutate its dictionaries nor consume/reset global RNG."""
        for renderer, report in ((render_linear_algebra, self.linear), (render_calculus, self.calculus)):
            before = deepcopy(report)
            state = np.random.get_state()
            self.assertEqual(renderer(report), renderer(report))
            self.assertEqual(report, before)
            after = np.random.get_state()
            self.assertEqual(state[0], after[0])
            np.testing.assert_array_equal(state[1], after[1])
            self.assertEqual(state[2:], after[2:])
