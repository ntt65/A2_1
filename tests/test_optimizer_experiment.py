"""최적화 실험의 재현성, 경로 캐시와 그림에 사용된 데이터를 검증한다."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.run_optimizer import cached_trajectory, main
from src.calculus import quadratic
from src.optimizer import optimizer_comparison_figure, run_optimization
from src.utils.cache_manager import CacheManager

__covers__ = ["scripts/run_optimizer.py", "src/optimizer.py"]


class OptimizerExperimentTests(unittest.TestCase):
    """동일 입력 재사용, 독립 수치 기준과 재사용 가능한 그림을 확인한다."""

    def tearDown(self):
        """검증이 실패해도 테스트가 만든 그림을 닫는다."""
        plt.close("all")

    def test_cache_matches_uncached_and_separates_parameters(self):
        """반복 실행은 캐시를 사용하고 학습률·목적함수 변경은 다시 계산하는지 확인한다."""
        with tempfile.TemporaryDirectory() as directory:
            cache = CacheManager(directory)
            with patch("scripts.run_optimizer.run_optimization", wraps=run_optimization) as wrapped:
                first = cached_trajectory("circle", "Momentum", 0.1, cache=cache)
                second = cached_trajectory("circle", "Momentum", 0.1, cache=cache)
                assert wrapped.call_count == 1	# 같은 입력의 반복 호출은 실제 계산을 한 번만 수행한다.
                changed = cached_trajectory("circle", "Momentum", 0.05, cache=cache)
                cached_trajectory("ellipse", "Momentum", 0.1, cache=cache)
                assert wrapped.call_count == 3
                uncached = cached_trajectory("circle", "Momentum", 0.1,
                                             cache=CacheManager(directory, enabled=False))
                assert wrapped.call_count == 4
            for field in ("path", "losses", "gradient_norms"):
                np.testing.assert_array_equal(first[field], second[field])
                np.testing.assert_array_equal(first[field], uncached[field])	# 캐시를 끈 계산과 원시 이력이 정확히 같아야 한다.
            assert not np.array_equal(first["path"], changed["path"])
            first["path"][:] = 999
            restored = cached_trajectory("circle", "Momentum", 0.1, cache=cache)
            np.testing.assert_array_equal(restored["path"], second["path"])

    def test_comparison_figure_retains_history_and_zero(self):
        """측정된 경로를 그리며 symlog 지표 축에서 정확한 0을 유지하는지 확인한다."""
        runs = {"GD": cached_trajectory("circle", "GD", 0.5, steps=3)}
        figure, axes = optimizer_comparison_figure(runs, quadratic, "Test comparison")
        assert axes.shape == (3,) and axes[0].get_aspect() == 1.0
        np.testing.assert_array_equal(axes[1].lines[0].get_ydata(), [50.0, 0.0, 0.0, 0.0])	# 손실 그래프의 정확한 0을 양의 바닥값으로 바꾸지 않는다.
        assert axes[1].get_yscale() == axes[2].get_yscale() == "symlog"
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for artist in [*figure.texts, *(a.title for a in axes), *(a.get_legend() for a in axes)]:
            box = artist.get_window_extent(renderer)
            assert box.x0 >= 0 and box.y0 >= 0
            assert box.x1 <= figure.bbox.width and box.y1 <= figure.bbox.height	# 제목과 설명의 경계가 실제 그림 캔버스 안에 있는지 확인한다.

    def test_cached_input_contract_matches_core(self):
        """NumPy 정수 예산은 정규화하고 잘못된 이름·예산은 명시적으로 거부하는지 확인한다."""
        with tempfile.TemporaryDirectory() as directory:
            record = cached_trajectory("circle", "GD", 0.1, steps=np.int64(2), cache=CacheManager(directory))
            assert record["path"].shape == (3, 2)
        for name, method in (("unknown", "GD"), ("circle", "unknown")):
            with self.assertRaises(ValueError):
                cached_trajectory(name, method, 0.1)
        for steps in (True, 1.5):
            with self.assertRaises(TypeError):
                cached_trajectory("circle", "GD", 0.1, steps=steps)
        with self.assertRaises(ValueError):
            cached_trajectory("circle", "GD", 0.1, steps=-1)

    def test_cli_outputs_and_stability_cases(self):
        """제출용 명령이 PNG 세 개와 독립적인 학습률 안정성 측정값을 기록하는지 확인한다."""
        with tempfile.TemporaryDirectory() as directory:
            assert main(["--output-dir", directory, "--no-cache"]) == 0
            output = Path(directory)
            report = json.loads((output / "optimizer_metrics.json").read_text(encoding="utf-8"))
            for name in ("circle", "ellipse", "learning_rates"):
                assert (output / f"optimizer_{name}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            circle = report["comparisons"]["circle"]["GD"]
            assert np.asarray(circle["path"]).shape == (101, 2)	# 기본 100회 갱신은 초기점 포함 101개 좌표를 기록한다.
            assert circle["final_norm"] <= 0.1
            assert circle["iterations"] == 100
            rates = report["learning_rates"]
            assert rates["0.5"]["final_norm"] == 0.0
            np.testing.assert_allclose(rates["1.0"]["final_norm"], np.sqrt(50), atol=1e-14, rtol=0)	# lr=1의 진동은 시작 반경 sqrt(50)을 그대로 유지한다.
            assert rates["1.1"]["final_norm"] > 100	# lr=1.1의 발산 사례가 큰 최종 반경을 보이는지 확인한다.


if __name__ == "__main__":
    unittest.main()
