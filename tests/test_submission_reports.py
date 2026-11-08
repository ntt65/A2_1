"""보고서 게시·새 실행과 제출 산출물 보존을 검증한다."""

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_submission_reports import (
    BEGIN, END, FINITE_DIFFERENCE_PATHS, REPORT_NAMES, ROOT,
    _check_approximation_accuracy, check_reproduction, run_reports, updated_readme,
)

__covers__ = ["scripts/run_submission_reports.py"]


class SubmissionReportTests(unittest.TestCase):
    """수치 변동을 거부하고 재현 근거를 안전하게 게시하는지 확인한다."""

    def test_reproduction_rejects_drift_shape_metadata_and_nonfinite(self):
        """바뀐 결과에 재현 성공 표시를 붙일 수 없는지 확인한다."""
        reference = {"values": [1., 2.], "passed": True, "numpy_version": "old"}
        check_reproduction({"values": [1. + 1e-13, 2.], "passed": True,
                            "numpy_version": "new"}, reference)
        for actual in (
            {**reference, "values": [1., 2.01]},
            {**reference, "values": [1.]},
            {**reference, "passed": 1},
            {**reference, "values": [float("nan"), 2.]},
            {**reference, "values": [float("inf"), 2.]},
            {"values": [1., 2.]},
        ):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                check_reproduction(actual, reference)

    def test_readme_block_is_idempotent_and_keeps_images(self):
        """정해진 블록 하나만 바뀌고 루트의 Markdown 경로는 유효하게 유지되는지 확인한다."""
        original = "# Mission\n\n![old](outputs/linear_transforms.png)\n\n## 검증 현황\n\n## 보너스\nKeep.\n"
        reports = dict.fromkeys(REPORT_NAMES, "# Result\n\n![plot](svd_128.png)\n[code](../src/linear_algebra.py)\n")
        updated = updated_readme(original, reports)
        self.assertEqual(updated_readme(updated, reports), updated)	# 같은 보고서를 반복 삽입해도 README 결과가 다시 바뀌지 않는다.
        self.assertEqual(updated.split(BEGIN)[0], original.split("## 검증 현황")[0])
        self.assertEqual(updated.split(END)[1], "\n\n## 검증 현황\n\n## 보너스\nKeep.\n")	# 관리 대상 블록 뒤의 기존 본문을 정확히 보존한다.
        self.assertIn("![old](outputs/linear_transforms.png)", updated)	# 기존 그림 링크가 상세 보고서 삽입 뒤에도 남아 있어야 한다.
        self.assertIn("![plot](outputs/svd_128.png)", updated)
        self.assertIn("[code](src/linear_algebra.py)", updated)

    def test_platform_difference_keeps_analytic_tolerance_and_inputs_strict(self):
        """실측 Linux·Windows 차이가 원래 해석적 허용오차 안에 있으며 입력 검사는 엄격한지 확인한다."""
        import copy
        import json
        from scripts.report_linear_calculus import render_calculus
        reference = json.loads((ROOT / "outputs/calculus_metrics.json").read_text(encoding="utf-8"))
        actual = copy.deepcopy(reference)
        actual["gradient"]["numerical"][1][0] = 2.8284271247347443
        actual["gradient"]["maximum_absolute_error"] = max(
            abs(left-right) for a, b in zip(actual["gradient"]["numerical"], actual["gradient"]["analytic"])
            for left, right in zip(a, b))
        self.assertLessEqual(actual["gradient"]["maximum_absolute_error"], 1e-8)	# 운영체제 수치 차이를 허용해도 원래 해석적 허용오차 1e-8은 유지한다.
        _check_approximation_accuracy("calculus", actual)
        render_calculus(actual)
        check_reproduction(actual, reference, "calculus", FINITE_DIFFERENCE_PATHS)
        for key, value in (("h", 1e-3), ("tolerance", 1.)):
            invalid = copy.deepcopy(actual)
            invalid["derivative"][key] = value
            with self.assertRaises(ValueError):
                check_reproduction(invalid, reference, "calculus", FINITE_DIFFERENCE_PATHS)
        for value in (float("nan"), [2.8]):
            invalid = copy.deepcopy(actual)
            invalid["gradient"]["numerical"][1][0] = value
            with self.assertRaises(ValueError):
                check_reproduction(invalid, reference, "calculus", FINITE_DIFFERENCE_PATHS)
        invalid = copy.deepcopy(actual)
        invalid["gradient"]["numerical"][1][0] += 1e-6
        with self.assertRaises(AssertionError):
            _check_approximation_accuracy("calculus", invalid)
        invalid = copy.deepcopy(actual)
        invalid["gradient"]["maximum_absolute_error"] = 0.
        with self.assertRaises(ValueError):
            render_calculus(invalid)
        invalid = copy.deepcopy(actual)
        invalid["gradient"]["analytic"][1][0] += 1e-6
        with self.assertRaises(ValueError):
            check_reproduction(invalid, reference, "calculus", FINITE_DIFFERENCE_PATHS)
        for key in ("quadratic_derivative", "sine_derivative"):
            invalid = copy.deepcopy(actual)
            invalid["step_size_study"][key][40] += .001
            with self.assertRaisesRegex(AssertionError, "finite-difference identity"):
                _check_approximation_accuracy("calculus", invalid)
        invalid = copy.deepcopy(actual)
        invalid["step_size_study"]["h"][0] *= 1.5
        with self.assertRaises(ValueError):
            check_reproduction(invalid, reference, "calculus", FINITE_DIFFERENCE_PATHS)

    def test_malformed_readme_fails_before_execution(self):
        """삽입 위치가 모호하면 실험이나 파일 기록을 시작하지 않는지 확인한다."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            readme = root / "README.md"
            for original in (BEGIN, END, END + BEGIN, BEGIN + BEGIN + END, "No anchor"):
                readme.write_text(original, encoding="utf-8")
                with patch("scripts.run_submission_reports.run_linear_algebra") as runner:
                    with self.assertRaises(ValueError):
                        run_reports(root / "out", readme)
                    runner.assert_not_called()	# 삽입 지점 오류는 실험을 호출하기 전에 검출한다.
                self.assertEqual(readme.read_text(encoding="utf-8"), original)
                self.assertFalse((root / "out").exists())

    def test_failed_experiment_publishes_no_reports(self):
        """실행 실패 후 성공한 것처럼 보이는 보고서를 남기지 않는지 확인한다."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            readme = root / "README.md"
            original = "# Mission\n## 검증 현황\n"
            readme.write_text(original, encoding="utf-8")
            with patch("scripts.run_submission_reports.run_linear_algebra", return_value={}), \
                 patch("scripts.run_submission_reports.run_calculus", side_effect=AssertionError("bad result")):
                with self.assertRaisesRegex(AssertionError, "bad result"):
                    run_reports(root / "out", readme)
            self.assertFalse((root / "out").exists())	# 실험 실패 후에는 성공 보고서처럼 보이는 출력 폴더도 만들지 않는다.
            self.assertEqual(readme.read_text(encoding="utf-8"), original)

    def test_fresh_execution_publishes_evidence_and_preserves_sources(self):
        """실험 세 개와 새 커널 두 개를 실행하여 실제 근거와 원본 산출물 불변을 확인한다."""
        paths = sorted((ROOT / "outputs").glob("*")) + sorted((ROOT / "notebooks").glob("*.ipynb"))
        hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if path.is_file()}
        (ROOT / ".cache").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="report-test-", dir=ROOT / ".cache") as folder:
            root = Path(folder)
            readme = root / "README.md"
            readme.write_text("# Mission\n\n## 검증 현황\n\n## 보너스\n", encoding="utf-8")
            reports = run_reports(root / "reports", readme)
            self.assertEqual(set(reports), set(REPORT_NAMES))
            self.assertEqual({path.name for path in (root / "reports").iterdir()}, set(REPORT_NAMES))
            text = readme.read_text(encoding="utf-8")
            for report in reports.values():
                self.assertIn("source SHA-256", report)
                self.assertIn("이번 실행 JSON SHA-256", report)	# 기존 원본과 이번 실제 실행의 JSON 해시를 구별하여 기록한다.
            for required in ("128×128", "고정 중심차분", "공통 네 자리 표시", "Momentum", "Gaussian"):
                self.assertTrue(required.lower() in text.lower(), f"missing evidence: {required}")
            self.assertLess(text.index(BEGIN), text.index("## 보너스"))
            self.assertIn("../../../outputs/", (root / "reports" / REPORT_NAMES[0]).read_text(encoding="utf-8"))
            self.assertEqual(text, updated_readme(text, reports, readme.parent))	# 실제 재실행 후 README도 같은 입력에서 결정적으로 유지된다.
        self.assertEqual(hashes, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in hashes})	# 원본 그림·JSON·노트북의 전체 바이트 해시가 모두 같아야 한다.


if __name__ == "__main__":
    unittest.main()
