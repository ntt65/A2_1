"""확률분포와 우도 항등식을 새 커널에서 실행하여 검증한다."""

import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from scripts.run_notebooks import ROOT, execute_notebook

__covers__ = ["scripts/run_notebooks.py", "notebooks/probability_loss.ipynb"]


class ProbabilityNotebookTests(unittest.TestCase):
    """노트북 산출물을 독립적인 분포·손실 기준값과 비교한다."""

    def test_probability_fresh_kernel_and_independent_targets(self):
        """원본 수정 없이 모든 assert를 실행하고 PNG 두 개와 수치 근거를 확인한다."""
        path = ROOT / "notebooks" / "probability_loss.ipynb"
        original = path.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            notebook = execute_notebook(path, output_dir=output)
            metrics = json.loads((output / "probability_metrics.json").read_text(encoding="utf-8"))
            for filename in ("probability_normal.png", "probability_bernoulli.png"):
                png = (output / filename).read_bytes()
                assert png.startswith(b"\x89PNG\r\n\x1a\n"), filename
                assert png[12:16] == b"IHDR", filename
                width, height = struct.unpack(">II", png[16:24])
                assert width >= 400 and height >= 300, (filename, width, height)
        assert path.read_bytes() == original	# 새 커널 실행 후에도 원본 노트북 바이트는 보존한다.
        assert metrics["seed"] == 42 and metrics["dtype"] == "float64"
        assert metrics["all_assertions_passed"] is True

        normals = metrics["normals"]
        assert len(normals) == 2
        for row, mu, variance, peak in zip(
            normals, (0.0, 2.0), (1.0, 0.5),
            (0.3989422804014327, 0.5641895835477563),
        ):
            with self.subTest(normal=(mu, variance)):
                assert row["mu"] == mu and row["variance"] == variance
                assert row["grid_points"] == 10001
                assert abs(row["sigma"]**2 - variance) <= 2e-15	# 두 번째 분포 인자는 표준편차가 아니라 분산인지 확인한다.
                assert abs(row["integral"] - 1.0) <= 1e-4	# 수치 적분한 정규분포 밀도의 전체 면적에 1e-4 기준을 적용한다.
                assert abs(row["peak"] - peak) <= 1e-14
                assert np.isfinite(row["minimum_pdf"]) and row["minimum_pdf"] >= 0.0
        bernoulli = metrics["bernoulli"]
        assert len(bernoulli) == 2
        for row, p, expected in zip(bernoulli, (0.3, 0.7), ([0.7, 0.3], [0.3, 0.7])):
            with self.subTest(bernoulli=p):
                assert row["p"] == p and row["support"] == [0, 1]
                np.testing.assert_allclose(row["pmf"], expected, atol=1e-15, rtol=0)	# Bernoulli의 독립 확률 [1-p,p]와 비교한다.
                assert abs(sum(row["pmf"]) - 1.0) <= 1e-12

        softmax = metrics["softmax"]
        probabilities = np.asarray(softmax["probabilities"], dtype=np.float64)
        assert probabilities.shape == np.shape(softmax["logits"])
        assert probabilities.ndim == 2 and softmax["axis"] == -1
        assert np.all(np.isfinite(probabilities)) and np.all(probabilities >= 0.0)
        np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0, atol=1e-6, rtol=0)	# 클래스 축의 확률 합에 미션 허용오차 1e-6을 적용한다.
        assert softmax["column_sum_error"] <= 1e-6	# 별도 열 축 정규화 검증도 같은 기준을 만족해야 한다.

        gaussian = metrics["gaussian"]
        assert gaussian["n"] == 3 and gaussian["variance"] == 0.5
        assert abs(gaussian["sse"] - 1.5) <= 1e-14
        assert abs(gaussian["mse"] - 0.5) <= 1e-14
        assert abs(gaussian["nll"] - 3.21709482877410026) <= 1e-12	# Gaussian NLL의 독립 상수 3.21709482877410026과 비교한다.
        assert gaussian["identity_error"] <= 1e-12

        bce = metrics["bce"]
        assert bce["y"] == [0.0, 1.0]
        np.testing.assert_allclose(bce["probabilities"], [0.3, 0.7], atol=0, rtol=0)
        assert abs(bce["sum_loss"] - 0.71334988787746476) <= 1e-12	# Bernoulli 합 손실의 독립 상수와 절대오차 1e-12로 비교한다.
        assert abs(bce["mean_loss"] - 0.35667494393873238) <= 1e-12
        assert abs(bce["logit_sum_loss"] - bce["sum_loss"]) <= 1e-12
        assert bce["max_gradient_error"] <= 1e-6	# 로짓 BCE의 수치 기울기에 1e-6 기준을 적용한다.
        extreme_losses = np.asarray(bce["extreme_losses"])
        assert np.all(np.isfinite(extreme_losses)) and np.all(extreme_losses >= 0.0)
        assert 4.24e-18 < bce["positive_tail_at_40"] < 4.26e-18	# 양의 극단 로짓에서도 작은 양의 손실 꼬리를 0으로 없애지 않는다.

        categorical = metrics["categorical"]
        np.testing.assert_allclose(categorical["probabilities"], [0.2, 0.3, 0.5], atol=0, rtol=0)
        assert categorical["target"] == 2
        assert abs(categorical["probability_ce"] - 0.69314718055994531) <= 1e-12	# 관측 클래스 확률 0.5의 독립 손실 ln(2)와 비교한다.
        assert abs(categorical["logit_ce"] - 0.69314718055994531) <= 1e-12
        assert np.all(np.isfinite(categorical["extreme_ce"]))

        codes = [cell for cell in notebook.cells if cell.cell_type == "code" and cell.source.strip()]
        assert codes
        assert [cell.execution_count for cell in codes] == list(range(1, len(codes)+1))
        assert not any(output.output_type == "error" for cell in codes for output in cell.outputs)	# 성공 플래그뿐 아니라 실행 셀 출력에 오류가 없는지도 확인한다.
        assert all("execution" not in cell.metadata for cell in notebook.cells)


if __name__ == "__main__":
    unittest.main()
