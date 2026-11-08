"""평가 근거의 메타데이터·허용오차와 의도한 실패 주입을 독립적으로 검증한다."""

import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

import numpy as np

from scripts.run_evaluation_evidence import (
    difference_report, failure_diagnostics, png_metadata, run_experiment,
    softmax_summary, step_size_summary,
)

__covers__ = ["scripts/run_evaluation_evidence.py"]


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    """독립 CRC와 빅엔디언 길이를 사용하여 테스트 PNG 청크를 인코딩한다."""
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff)


class EvaluationEvidenceTests(unittest.TestCase):
    """주입한 실패를 회귀 오류로 오인하지 않고 실측 메타데이터와 진단을 확인한다."""

    def test_png_metadata_reads_independent_dimensions_and_units(self):
        """알려진 7×11 크기와 10000 픽셀/미터를 254 DPI로 해석하는지 확인한다."""
        data = b"\x89PNG\r\n\x1a\n"
        data += png_chunk(b"IHDR", struct.pack(">IIBBBBB", 7, 11, 8, 6, 0, 0, 0))
        data += png_chunk(b"pHYs", struct.pack(">IIB", 10000, 10000, 1))
        data += png_chunk(b"IEND", b"")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.png"
            path.write_bytes(data)
            result = png_metadata(path)
            assert (result["width_px"], result["height_px"]) == (7, 11)	# 합성 PNG의 독립 가로·세로 크기 7×11을 확인한다.
            np.testing.assert_allclose(result["dpi_xy"], [254, 254], atol=1e-12, rtol=0)	# 10000 픽셀/미터가 254 DPI라는 단위 환산을 대조한다.
            for bad in (b"not png", data[:-1], data[:20] + b"x" + data[21:]):
                path.write_bytes(bad)
                with self.assertRaises(ValueError):
                    png_metadata(path)

    def test_diagnostic_signed_difference_shape_and_failure(self):
        """부호 있는 차이가 불일치 위치를 드러내고 실패 값이 조용히 통과하지 않는지 확인한다."""
        result = difference_report([1., -2.], [1.25, -2.5], .3)
        np.testing.assert_array_equal(result["signed_difference"], [.25, -.5])	# 관측값과 기준값의 차이 부호까지 보존하여 실패 위치를 드러낸다.
        assert result["max_absolute_error"] == .5 and not result["passed"]
        assert difference_report(1., 1.125, .125)["passed"]
        for expected, observed, tolerance in (([1], [[1]], .1), ([], [], .1), ([1], [np.inf], .1), (1, 1, -1)):
            with self.assertRaises(ValueError):
                difference_report(expected, observed, tolerance)
        with self.assertRaises(TypeError):
            difference_report([True], [True], .1)

    def test_h_summary_does_not_bridge_failing_samples(self):
        """실패 표본을 사이에 둔 두 통과 영역을 하나의 검증된 구간으로 표시하지 않는지 확인한다."""
        result = step_size_summary([1., 2., 3., 4., 5.], [.1, .2, 1., .1, .2], .2)
        assert result["passing_h"] == [1., 2., 4., 5.]
        assert len(result["contiguous_sample_groups"]) == 2	# 중간 실패 표본을 건너뛰어 통과 영역을 하나로 합치지 않는다.
        assert result["minimizing_sampled_h"] == [1., 4.]
        assert step_size_summary([1.], [1.], .1)["contiguous_sample_groups"] == []
        for h, errors in (([2., 1.], [0., 0.]), ([0.], [0.]), ([1.], [-1.])):
            with self.assertRaises(ValueError):
                step_size_summary(h, errors)

    def test_real_examples_pass_and_both_injections_fail(self):
        """고정 단위 넓이 변환과 알려진 역전파 기울기는 통과하고 의도한 변조는 실패하는지 확인한다."""
        report = failure_diagnostics()
        assert report["area"]["genuine"]["area_ratio"] == 1.
        assert report["area"]["genuine"]["passed"]
        assert report["backprop"]["genuine"]["passed"]
        assert not report["area"]["injected_observation"]["passed"]
        assert not report["backprop"]["injected_observation"]["passed"]	# 잘못된 역전파 값을 주입한 사례는 성공으로 표시하지 않는다.
        np.testing.assert_allclose(report["backprop"]["injected_observation"]["signed_difference"],
                                   [[.001, 0], [0, 0]], atol=1e-15, rtol=0)

    def test_softmax_reports_distinct_axes_and_finite_error(self):
        """보고한 각 축을 실제로 정규화하며 합이 허용오차 안에서 1인지 확인한다."""
        rows = softmax_summary()
        assert len(rows) == 6 and {row["axis"] for row in rows} == {-1, 0, 1}
        assert all(row["sum_error_max"] <= 1e-6 and row["passed"] for row in rows)	# 각 축의 실제 확률 합 오차에 미션 기준 1e-6을 적용한다.
        assert rows[1]["sum_shape"] == [3] and rows[2]["sum_shape"] == [2]

    def test_delivery_preserves_originals_and_writes_metadata(self):
        """보고서를 두 형식으로 저장하면서 입력 PNG 파일 바이트를 그대로 보존하는지 확인한다."""
        source = Path(__file__).resolve().parents[1] / "outputs"
        originals = {path.name: path.read_bytes() for path in source.glob("*.png")}
        with tempfile.TemporaryDirectory() as directory:
            report = run_experiment(source, Path(directory), use_cache=False)
            saved = json.loads((Path(directory) / "evaluation_evidence.json").read_text(encoding="utf-8"))
            assert report == saved
            assert (Path(directory) / "evaluation_evidence.md").is_file()
            assert len(saved["figure_metadata"]) == 10
            assert [row["input_shape"] for row in saved["svd_panel_metadata"]] == [[64, 64], [128, 128]]	# 서로 다른 64×64·128×128 SVD 입력 형상을 기록한다.
            assert all(len(row["panels"]) == 4 for row in saved["svd_panel_metadata"])
        assert all((source / name).read_bytes() == data for name, data in originals.items())	# 근거 보고서 생성은 기존 그림 바이트를 바꾸지 않는다.
