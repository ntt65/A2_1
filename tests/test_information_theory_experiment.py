"""보너스 CLI의 독립 기준값과 경계값의 정직한 직렬화를 검증한다."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np

from scripts.run_information_theory import _json_number, example_report, main

__covers__ = ["scripts/run_information_theory.py", "src/information_theory.py"]


class InformationTheoryExperimentTests(unittest.TestCase):
    """보고서의 재현성을 유지하며 양의 무한대를 유한한 잘라낸 값과 구별한다."""

    def tearDown(self):
        """그림 검증이 실패한 경우에도 Figure를 닫는다."""
        plt.close("all")

    def test_independent_examples_and_infinity_encoding(self):
        """알려진 유리수 분포의 정보량과 정확한 지지집합 경계가 보고서에 남는지 확인한다."""
        rows = {row["name"]: row for row in example_report()}
        assert len(rows) == 6 and all(row["passed"] for row in rows.values())
        assert abs(rows["rational_forward"]["kl_nats"] - .13081203594113697) <= 1e-14
        assert abs(rows["rational_reverse"]["kl_nats"] - .14384103622589045) <= 1e-14	# 역방향 KL의 독립 상수로 비대칭 사례를 확인한다.
        assert rows["support_mismatch"]["cross_entropy_nats"] == "+inf"	# JSON에서도 양의 무한대라는 경계 의미를 명시적으로 남긴다.
        assert rows["smallest_positive_q"]["cross_entropy_nats"] < 745.
        assert _json_number(np.inf) == "+inf"
        for value in (np.nan, -np.inf):
            with self.assertRaises(ValueError):
                _json_number(value)

    def test_real_cli_outputs_full_curves_and_true_png_size(self):
        """실제 CLI로 독립 중간값·지지집합 끝점·그림 메타데이터와 전체 곡선을 검증한다."""
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as console:
            assert main(["--output-dir", directory]) == 0
            root = Path(directory)
            report = json.loads((root / "information_theory.json").read_text(encoding="utf-8"))
            markdown = (root / "information_theory.md").read_text(encoding="utf-8")
            png = (root / "information_theory.png").read_bytes()
        assert "independent examples passed" in console.getvalue()
        assert report["units"] == "nats" and report["seed"] == 42
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        assert struct.unpack(">II", png[16:24]) == (1792, 686)	# 실제 PNG 헤더의 픽셀 크기를 기대값과 대조한다.
        assert report["figure"]["width_px"] == 1792 and report["figure"]["height_px"] == 686
        assert len(report["binary_entropy"]["entropy_nats"]) == 501
        assert abs(report["binary_entropy"]["entropy_nats"][250] - .6931471805599453) <= 1e-14	# 이진 균등분포의 최대 엔트로피 ln(2)를 확인한다.
        assert report["binary_entropy"]["entropy_nats"][0] == report["binary_entropy"]["entropy_nats"][-1] == 0.	# 확정 분포인 양 끝점의 엔트로피는 정확히 0이다.
        for field in ("cross_entropy_nats", "kl_nats"):
            assert report["varying_model"][field][0] == report["varying_model"][field][-1] == "+inf"	# 모델 지지집합이 맞지 않는 두 끝점은 모두 양의 무한대로 기록한다.
        assert abs(report["varying_model"]["cross_entropy_nats"][150] - .6108643020548935) <= 1e-14
        assert abs(report["varying_model"]["kl_nats"][150]) <= 1e-14
        assert report["curve_checks"]["finite_model_count"] == 499	# 무한대 두 끝점을 뺀 499개 표본만 유한 항등식 검증에 포함한다.
        assert report["curve_checks"]["maximum_identity_error"] <= 1e-12	# 유한 구간의 CE=H+KL 최대 오차에 1e-12 기준을 적용한다.
        assert "not a finite cap" in markdown and "not symmetric" in markdown


if __name__ == "__main__":
    unittest.main()
