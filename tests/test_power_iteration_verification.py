"""제출한 eig 비교 근거를 독립적으로 아는 값과 대조한다."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.verify_power_iteration import main, verify_mission_eigenpair

__covers__ = ["scripts/verify_power_iteration.py", "scripts/run_linear_algebra.py"]


class EigenpairEvidenceTests(unittest.TestCase):
    """기준 비교가 명확히 드러나고 수치적으로 맞으며 벡터 부호에 무관한지 확인한다."""

    def test_reference_matches_independent_closed_form_and_vector(self):
        """제출한 NumPy 기준값이 미션의 독립 닫힌형 고유쌍과 일치하는지 확인한다."""
        report = verify_mission_eigenpair()
        manual, reference = report["power_iteration"], report["numpy_eig"]
        expected = (7.0 + np.sqrt(5.0)) / 2.0	# 특성방정식으로 얻은 지배 고유값을 독립 기준으로 둔다.
        expected_vector = np.array([0.8506508083520399, 0.5257311121191336])
        assert abs(reference["eigenvalue"] - expected) <= 1e-14
        np.testing.assert_allclose(reference["eigenvector_sign_aligned"], expected_vector, atol=1e-14, rtol=0)	# 부호를 맞춘 벡터를 독립 좌표 상수와 절대오차 1e-14로 비교한다.
        measured_error = abs(manual["eigenvalue"] - reference["eigenvalue"]) / abs(reference["eigenvalue"])
        assert measured_error == manual["numpy_eig_comparison"]["relative_eigenvalue_error"]
        assert measured_error <= 0.05 and report["passed"] is True	# 실측 오차가 미션 기준을 만족할 때만 성공 표시를 허용한다.
        assert reference["selection"] == "argmax(abs(eigenvalues))"

    def test_reference_sign_is_irrelevant_and_wrong_value_is_rejected(self):
        """고유벡터의 반대 부호는 통과하지만 변조한 eig 기준값은 PASS가 될 수 없는지 확인한다."""
        values = np.array([4.618033988749895, 2.381966011250105])
        vectors = np.array([[0.8506508083520399, -0.5257311121191336],
                            [0.5257311121191336, 0.8506508083520399]])
        with patch("scripts.verify_power_iteration.np.linalg.eig", return_value=(values, -vectors)):
            report = verify_mission_eigenpair()
            assert report["passed"] is True
            assert report["numpy_eig"]["eigenvector_raw"][0] < 0
            assert report["numpy_eig"]["eigenvector_sign_aligned"][0] > 0	# 원래 부호가 반대여도 비교용 방향을 일관되게 맞춘다.
        with patch("scripts.verify_power_iteration.np.linalg.eig", return_value=(values + 1, vectors)):
            with self.assertRaisesRegex(AssertionError, "verification failed"):
                verify_mission_eigenpair()

    def test_cli_writes_readable_and_machine_readable_evidence(self):
        """문서의 명령이 두 방법을 모두 출력하고 정확한 보고서 값을 두 형식으로 저장하는지 확인한다."""
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as console:
            assert main(["--output-dir", directory]) == 0
            root = Path(directory)
            report = json.loads((root / "power_iteration_eig_comparison.json").read_text(encoding="utf-8"))
            evidence = (root / "power_iteration_eig_comparison.md").read_text(encoding="utf-8")
        assert report["power_iteration"]["matrix"] == [[4.0, 1.0], [1.0, 3.0]]
        assert report["power_iteration"]["initial"] == [1.0, 1.0]
        assert report["numpy_eig"]["eigenvalue"] == report["power_iteration"]["numpy_eig_comparison"]["numpy_eigenvalue"]	# 기계 판독 보고서의 두 위치에 같은 NumPy 기준값을 보존한다.
        for label in ("Power Iteration:", "np.linalg.eig:", "Relative eigenvalue error=", "5%", "PASS"):
            assert label in console.getvalue(), label
            assert label in evidence, label	# 콘솔뿐 아니라 저장한 문서에도 각 비교 근거를 남긴다.
