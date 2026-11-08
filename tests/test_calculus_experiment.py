"""문서에 제시한 Step 3 명령과 실측 산출물을 통합 검증한다."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.run_calculus import main

__covers__ = ["scripts/run_calculus.py", "src/calculus.py"]


class CalculusExperimentTests(unittest.TestCase):
    """제출용 명령이 유한 측정값과 PNG를 생성하는지 확인한다."""

    def test_delivery_command_and_recorded_geometry(self):
        """기록한 데이터가 독립 도함수·기하 기준을 만족하며 원점 방향을 포함하지 않는지 확인한다."""
        with tempfile.TemporaryDirectory() as directory:
            assert main(["--output-dir", directory]) == 0
            output = Path(directory)
            report = json.loads((output / "calculus_metrics.json").read_text(encoding="utf-8"))
            for filename in ("gradient_geometry.png", "derivative_error.png"):
                assert (output / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            assert report["seed"] == 42
            assert abs(report["derivative"]["numerical"]-6.0) <= 1e-4	# 제출 명령의 실제 도함수를 독립 해석값 6과 비교한다.
            data = report["gradient"]
            points = np.asarray(data["points"])
            assert points.shape == (9, 2)
            np.testing.assert_allclose(data["numerical"], 2*points, atol=1e-8, rtol=0)	# 원점 포함 아홉 점의 독립 기울기 2p와 수치 결과를 대조한다.
            tangents = np.asarray(data["nonzero_tangents"])
            np.testing.assert_allclose(np.sum(points[:-1]*tangents, axis=1), 0, atol=1e-12, rtol=0)	# 등고선 접선과 점 방향의 내적이 0인지 확인한다.
            assert data["maximum_normalized_dot"] <= 1e-12
            assert data["origin_gradient"] == [0.0, 0.0]
            assert data["origin_direction"] is None	# 기울기가 0인 원점에는 정의되지 않은 방향을 임의로 만들지 않는다.
            assert len(report["step_size_study"]["h"]) > 10
