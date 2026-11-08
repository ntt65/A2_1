"""민감도 근거의 점화식·미도달 처리·캐시를 독립적으로 검증한다."""

import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from scripts.run_optimizer_sensitivity import aggregate, main, radius_summary, stability_evidence, trajectory
from src.optimizer import run_optimization
from src.utils.cache_manager import CacheManager

__covers__ = ["scripts/run_optimizer_sensitivity.py"]


class OptimizerSensitivityTests(unittest.TestCase):
    """실패한 시행을 성공 표본으로 취급하지 않고 고정 초기점 통계를 검증한다."""

    def tearDown(self):
        """수치 검증이 실패해도 모든 Figure를 닫는다."""
        plt.close("all")

    def test_gd_matches_independent_closed_form(self):
        """서로 다른 두 초기점의 원형·타원형 경로가 좌표별 배율의 거듭제곱과 일치하는지 확인한다."""
        for name, rate, factors in (("circle", 0.1, [0.8, 0.8]), ("ellipse", 0.05, [0.9, 0.0])):
            for initial in ([1.0, -3.0], [-5.0, 5.0]):
                record = trajectory(name, "GD", initial, rate)
                expected = np.asarray(initial) * np.asarray(factors)[None, :] ** np.arange(101)[:, None]	# 좌표별 배율의 거듭제곱으로 독립 전체 경로를 계산한다.
                np.testing.assert_allclose(record["path"], expected, atol=1e-14, rtol=1e-12)	# 초기점별 원시 경로를 독립 해와 비교한다.
                assert record["path"].shape == (101, 2)
        origin = trajectory("circle", "Momentum", [0, 0], 0.1)
        assert origin["first_radius_0_1"] == origin["remaining_inside_radius_0_1_through_budget"] == 0	# 원점 시작은 0회에 진입하고 예산 끝까지 유지된다.
        np.testing.assert_array_equal(origin["path"], np.zeros((101, 2)))

    def test_crossings_reentry_and_missing_population(self):
        """첫 진입과 이후 유지가 다르며 미도달 사례는 성공 표본의 조건부 평균에서 제외되는지 확인한다."""
        missing = radius_summary([2.0, 1.0, 0.2])
        reentry = radius_summary([1.0, 0.09, 0.2, 0.05, 0.01])
        exits = radius_summary([0.0, 0.2, 0.3])
        assert missing["first_radius_0_1"] is None
        assert missing["remaining_inside_radius_0_1_through_budget"] is None
        assert reentry["first_radius_0_1"] == 1
        assert reentry["remaining_inside_radius_0_1_through_budget"] == 3	# 첫 진입 이후 나갔다가 다시 들어온 시점을 유지 시작으로 구별한다.
        assert exits["first_radius_0_1"] == 0
        assert exits["remaining_inside_radius_0_1_through_budget"] is None	# 예산 끝에 밖으로 나가 있으면 유지 시작은 없다고 기록한다.
        summary = aggregate([missing, reentry, exits])
        first = summary["first_radius_0_1"]
        assert first == {"success_count": 2, "failure_count": 1,
                         "mean_successful_updates": 0.5, "variance_successful_updates": 0.25}	# 미도달 사례를 제외한 성공 표본 수·평균·분산을 확인한다.
        stays = summary["remaining_inside_radius_0_1_through_budget"]
        assert stays["success_count"] == 1 and stays["failure_count"] == 2
        assert stays["mean_successful_updates"] == 3.0 and stays["variance_successful_updates"] == 0.0
        assert aggregate([missing])["first_radius_0_1"]["mean_successful_updates"] is None
        np.testing.assert_allclose(summary["final_radius_all_samples"]["mean"], 0.17, atol=1e-15, rtol=0)	# 최종 반경 평균에는 성공 여부와 무관하게 모든 표본을 포함한다.

    def test_cache_equivalence_input_separation_and_copy(self):
        """전체 초기 좌표·방법·학습률·예산으로 캐시를 구별하고 반환 복사본을 격리하는지 확인한다."""
        with tempfile.TemporaryDirectory() as directory:
            cache = CacheManager(directory)
            with patch("scripts.run_optimizer_sensitivity.run_optimization", wraps=run_optimization) as wrapped:
                first = trajectory("circle", "Momentum", [1.0, -3.0], 0.1, cache=cache)
                second = trajectory("circle", "Momentum", [1.0, -3.0], 0.1, cache=cache)
                assert wrapped.call_count == 1
                trajectory("circle", "Momentum", [-1.0, -3.0], 0.1, cache=cache)
                trajectory("circle", "Momentum", [1.0, -3.0], 0.05, cache=cache)
                trajectory("circle", "GD", [1.0, -3.0], 0.1, cache=cache)
                trajectory("circle", "Momentum", [1.0, -3.0], 0.1, steps=2, cache=cache)
                assert wrapped.call_count == 5
                uncached = trajectory("circle", "Momentum", [1.0, -3.0], 0.1,
                                      cache=CacheManager(directory, enabled=False))
                assert wrapped.call_count == 6
            for field in ("path", "losses", "gradient_norms"):
                np.testing.assert_array_equal(first[field], second[field])
                np.testing.assert_array_equal(first[field], uncached[field])	# 캐시 사용 여부가 각 원시 이력의 값을 바꾸지 않는다.
            first["path"][:] = 999
            restored = trajectory("circle", "Momentum", [1.0, -3.0], 0.1, cache=cache)
            np.testing.assert_array_equal(restored["path"], second["path"])

    def test_stability_uses_exact_factors(self):
        """학습률 .5는 수렴, 1은 진동, 1.1은 손실 배율 1.44를 보이는지 확인한다."""
        rows = stability_evidence()
        np.testing.assert_allclose([r["coordinate_multiplier"] for r in rows], [0, -1, -1.2], atol=1e-15, rtol=0)
        np.testing.assert_allclose([r["theoretical_loss_multiplier"] for r in rows], [0, 1, 1.44], atol=1e-15, rtol=0)	# 손실 배율은 좌표 배율의 제곱인 0·1·1.44이다.
        assert rows[0]["final_radius"] == 0 and rows[0]["observed_ratio_count"] == 1
        assert not rows[1]["observed_strict_loss_increase"]
        assert rows[2]["observed_strict_loss_increase"] and rows[2]["all_values_finite"]

    def test_input_contracts(self):
        """잘못된 형상·예산·이름·반경을 거부하고 불리언을 숫자로 강제 변환하지 않는지 확인한다."""
        for start in ([], [1], [[1, 2]], [1, np.inf]):
            with self.assertRaises(ValueError):
                trajectory("circle", "GD", start, 0.1)
        for budget in (True, 2.1):
            with self.assertRaises(TypeError):
                trajectory("circle", "GD", [1, 2], 0.1, steps=budget)
        for values in ([], [-1], [np.nan], [[1, 2]]):
            with self.assertRaises(ValueError):
                radius_summary(values)
        with self.assertRaises(ValueError):
            radius_summary([1.0], radius=0)
        with self.assertRaises(ValueError):
            aggregate([])

    def test_cli_delivers_declared_population_and_png_metadata(self):
        """실제 CLI가 방법별 다섯 초기점, 원시 경로와 실제 PNG 크기를 기록하는지 확인한다."""
        with tempfile.TemporaryDirectory() as directory:
            assert main(["--output-dir", directory, "--no-cache"]) == 0
            root = Path(directory)
            report = json.loads((root / "optimizer_sensitivity.json").read_text(encoding="utf-8"))
            png = (root / "optimizer_sensitivity.png").read_bytes()
            assert png[:8] == b"\x89PNG\r\n\x1a\n"
            assert struct.unpack(">II", png[16:24]) == (1680, 1400)	# 실제 PNG 헤더에서 가로·세로 픽셀 수를 읽어 확인한다.
            assert report["figure"]["width_pixels"] == 1680 and report["figure"]["height_pixels"] == 1400
            for methods in report["experiments"].values():
                for group in methods.values():
                    assert len(group["runs"]) == group["summary"]["sample_count"] == 5
                    assert all(np.asarray(r["path"]).shape == (101, 2) for r in group["runs"])	# 각 시행의 100회 갱신 경로는 초기점을 포함한 (101,2)이다.
            text = (root / "optimizer_sensitivity.md").read_text(encoding="utf-8")
            assert "not repeated random trials" in text and "first crossing" in text


if __name__ == "__main__":
    unittest.main()
