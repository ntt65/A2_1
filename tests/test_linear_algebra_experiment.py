"""영상 픽셀의 인코딩 대신 재현 가능한 데이터와 제출 명령을 검증한다."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.run_linear_algebra import generate_grayscale_image, main
from src.utils.random import set_seed

__covers__ = ["scripts/run_linear_algebra.py", "src/linear_algebra.py"]


class ExperimentTests(unittest.TestCase):
    """제출한 그림과 수치를 다시 생성할 수 있는지 확인한다."""

    def test_seeded_image_shape_range_and_detail(self):
        """고정 시드가 단순하지 않은 전체 랭크 회색조 영상을 재현하는지 확인한다."""
        set_seed()
        first = generate_grayscale_image(16)
        set_seed()
        second = generate_grayscale_image(16)
        assert first.shape == (16, 16) and first.dtype == np.float64
        np.testing.assert_array_equal(first, second)	# 동일 시드의 영상 데이터가 정확히 재현되는지 확인한다.
        assert np.all(np.isfinite(first)) and np.min(first) >= 0 and np.max(first) <= 1
        assert np.linalg.matrix_rank(first) == 16	# 단순 저랭크 영상이 아닌 전체 랭크 입력을 사용한다.
        for value in (True, 8.5, "16"):
            with self.assertRaises(TypeError): generate_grayscale_image(value)
        with self.assertRaises(ValueError): generate_grayscale_image(7)

    def test_delivery_command_outputs_and_assertions(self):
        """문서의 명령이 필수 PNG 전체와 실제 측정값을 생성하는지 확인한다."""
        with tempfile.TemporaryDirectory() as directory:
            assert main(["--output-dir", directory, "--no-cache"]) == 0
            output = Path(directory)
            report = json.loads((output / "linear_algebra_metrics.json").read_text(encoding="utf-8"))
            for filename in ("linear_transforms.png", "svd_64.png", "svd_128.png"):
                assert (output / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            assert report["seed"] == 42
            assert report["power_iteration"]["relative_error"] <= 0.05
            assert report["power_iteration"]["relative_residual"] <= 1e-6
            comparison = report["power_iteration"]["numpy_eig_comparison"]
            assert comparison["reference_method"] == "numpy.linalg.eig"
            assert comparison["relative_eigenvalue_error"] <= 0.05	# 실제 eig 비교에도 미션 고유값 상대오차 5%를 적용한다.
            assert comparison["passed"] is True
            reference = (7.0 + np.sqrt(5.0)) / 2.0
            assert abs(comparison["numpy_eigenvalue"] - reference) <= 1e-14	# NumPy 기준값 자체도 독립 닫힌형 값과 대조한다.
            assert (output / "power_iteration_eig_comparison.json").is_file()
            evidence = (output / "power_iteration_eig_comparison.md").read_text(encoding="utf-8")
            assert "np.linalg.eig" in evidence and "PASS" in evidence
            assert all(value["passed"] for value in report["transforms"].values())
            assert [[r["k"] for r in image["comparisons"]] for image in report["svd"]] == [[10, 50, 64], [10, 50, 100]]	# 64×64의 최대 랭크 64와 128×128의 랭크 100 실험을 구별한다.
            assert all(image["full_rank_relative_error"] <= 1e-10 for image in report["svd"])
            assert report["svd"][1]["comparisons"][2]["stored_over_original"] > 1	# 고랭크 근사가 원본보다 저장량이 커지는 경우도 숨기지 않는다.
