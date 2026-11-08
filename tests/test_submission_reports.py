"""Check report publication, fresh execution and preservation of submitted assets."""

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_submission_reports import (
    BEGIN, END, REPORT_NAMES, ROOT, check_reproduction, run_reports, updated_readme,
)

__covers__ = ["scripts/run_submission_reports.py"]


class SubmissionReportTests(unittest.TestCase):
    """Verify drift rejection and safe publication of reproducible evidence."""

    def test_reproduction_rejects_drift_shape_metadata_and_nonfinite(self):
        """A changed result cannot acquire a reproduced-success label."""
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
        """Only one bounded block changes; Markdown paths remain valid at the root."""
        original = "# Mission\n\n![old](outputs/linear_transforms.png)\n\n## 검증 현황\n\n## 보너스\nKeep.\n"
        reports = dict.fromkeys(REPORT_NAMES, "# Result\n\n![plot](svd_128.png)\n[code](../src/linear_algebra.py)\n")
        updated = updated_readme(original, reports)
        self.assertEqual(updated_readme(updated, reports), updated)
        self.assertEqual(updated.split(BEGIN)[0], original.split("## 검증 현황")[0])
        self.assertEqual(updated.split(END)[1], "\n\n## 검증 현황\n\n## 보너스\nKeep.\n")
        self.assertIn("![old](outputs/linear_transforms.png)", updated)
        self.assertIn("![plot](outputs/svd_128.png)", updated)
        self.assertIn("[code](src/linear_algebra.py)", updated)

    def test_malformed_readme_fails_before_execution(self):
        """An ambiguous insertion point cannot trigger executions or writes."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            readme = root / "README.md"
            for original in (BEGIN, END, END + BEGIN, BEGIN + BEGIN + END, "No anchor"):
                readme.write_text(original, encoding="utf-8")
                with patch("scripts.run_submission_reports.run_linear_algebra") as runner:
                    with self.assertRaises(ValueError):
                        run_reports(root / "out", readme)
                    runner.assert_not_called()
                self.assertEqual(readme.read_text(encoding="utf-8"), original)
                self.assertFalse((root / "out").exists())

    def test_failed_experiment_publishes_no_reports(self):
        """A failed execution cannot leave an apparently successful report behind."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            readme = root / "README.md"
            original = "# Mission\n## 검증 현황\n"
            readme.write_text(original, encoding="utf-8")
            with patch("scripts.run_submission_reports.run_linear_algebra", return_value={}), \
                 patch("scripts.run_submission_reports.run_calculus", side_effect=AssertionError("bad result")):
                with self.assertRaisesRegex(AssertionError, "bad result"):
                    run_reports(root / "out", readme)
            self.assertFalse((root / "out").exists())
            self.assertEqual(readme.read_text(encoding="utf-8"), original)

    def test_fresh_execution_publishes_evidence_and_preserves_sources(self):
        """Rerun three experiments/two kernels; verify real evidence and no asset mutation."""
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
                self.assertIn("이번 실행 JSON SHA-256", report)
            for required in ("128×128", "6.000000000039", "공통 네 자리 표시", "Momentum", "Gaussian"):
                self.assertTrue(required.lower() in text.lower(), f"missing evidence: {required}")
            self.assertLess(text.index(BEGIN), text.index("## 보너스"))
            self.assertIn("../../../outputs/", (root / "reports" / REPORT_NAMES[0]).read_text(encoding="utf-8"))
            self.assertEqual(text, updated_readme(text, reports, readme.parent))
        self.assertEqual(hashes, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in hashes})


if __name__ == "__main__":
    unittest.main()
