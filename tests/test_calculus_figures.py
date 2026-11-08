"""미적분 그림의 기하학적 관계와 수치 표시를 독립적으로 검증한다."""

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.quiver import Quiver
import numpy as np

from src.calculus import derivative_error_figure, derivative_error_study, gradient_figure

__covers__ = ["src/calculus.py"]


class CalculusFigureTests(unittest.TestCase):
    """그림의 방향, 실제 오차와 읽을 수 있는 고정 크기 배치를 검증한다."""

    def tearDown(self):
        """검증 실패 뒤에도 테스트가 만든 모든 Figure를 해제한다."""
        plt.close("all")

    def test_gradient_arrows_follow_level_normals_and_skip_origin(self):
        """두 화살표 집합이 f=4에 있으며 기울기는 바깥 방향이고 접선과 수직인지 확인한다."""
        figure, axes = gradient_figure()
        assert axes.figure is figure
        assert axes.get_aspect() == 1.0
        arrows = [item for item in axes.collections if isinstance(item, Quiver)]
        assert len(arrows) == 2
        gradient, tangent = arrows
        for quiver in arrows:
            assert len(quiver.X) == 8
            assert np.all(np.hypot(quiver.X, quiver.Y) > 0)
            np.testing.assert_allclose(quiver.X**2 + quiver.Y**2, 4.0, atol=2e-15, rtol=0)	# 모든 화살표 시작점이 등고선 x²+y²=4 위에 있는지 확인한다.
        np.testing.assert_allclose(gradient.U, gradient.X / 2.0, atol=1e-15, rtol=0)
        np.testing.assert_allclose(gradient.V, gradient.Y / 2.0, atol=1e-15, rtol=0)
        np.testing.assert_allclose(gradient.U * tangent.U + gradient.V * tangent.V,
                                   np.zeros(8), atol=1e-15, rtol=0)	# 기울기와 접선 방향의 내적이 절대오차 1e-15 내에서 0인지 확인한다.
        assert any("zero gradient" in text.get_text() for text in axes.texts)

    def test_step_study_uses_independent_derivatives_and_retains_zero(self):
        """이차함수 기준값 6과 해석적으로 알려진 사인 절단오차로 간격 실험을 확인한다."""
        study = derivative_error_study()
        assert set(study) == {"h", "quadratic_derivative", "quadratic_error",
                              "sine_derivative", "sine_error"}
        assert all(value.shape == study["h"].shape for value in study.values())
        assert all(np.all(np.isfinite(value)) for value in study.values())
        assert np.all(np.diff(study["h"]) > 0)
        np.testing.assert_allclose(study["quadratic_error"],
                                   np.abs(study["quadratic_derivative"] - 6.0), atol=0, rtol=0)	# 보고한 오차가 기준 도함수 6과 실제 측정값의 차이인지 확인한다.
        index = int(np.argmin(np.abs(study["h"] - 1e-5)))
        assert study["quadratic_error"][index] <= 1e-4
        # h=1일 때 sin 함수의 정확한 중심차분은 cos(3)*sin(1)이다.
        expected = abs(np.cos(3.0) * (np.sin(1.0) - 1.0))	# 사인 덧셈정리로 h=1의 절단오차를 독립 계산한다.
        assert abs(study["sine_error"][-1] - expected) <= 3e-16	# 실험 마지막 표본을 해석적으로 계산한 절단오차와 비교한다.
        assert study["quadratic_error"][-1] == 0.0

    def test_error_plot_keeps_raw_zero_data(self):
        """symlog 축이 0을 양의 바닥값으로 바꾸지 않고 원시 데이터로 표시하는지 확인한다."""
        figure, axes = derivative_error_figure()
        assert axes.figure is figure
        assert axes.get_xscale() == "log"
        assert axes.get_yscale() == "symlog"
        assert axes.get_ylim()[0] == 0.0
        np.testing.assert_array_equal(axes.lines[0].get_ydata(), derivative_error_study()["quadratic_error"])
        assert axes.lines[0].get_ydata()[-1] == 0.0	# 정확한 0인 원시 오차가 그래프에도 그대로 남는다.
        assert "no truncation error" in axes.get_title()

    def test_titles_legends_and_scale_notes_fit_canvas(self):
        """최종 그림 두 개에서 제목·범례·축척 설명이 캔버스 안에 들어가는지 확인한다."""
        for factory in (gradient_figure, derivative_error_figure):
            with self.subTest(figure=factory.__name__):
                figure, axes = factory()
                figure.set_dpi(140)
                figure.canvas.draw()
                renderer = figure.canvas.get_renderer()
                canvas = figure.bbox
                artists = [axes.title, axes.xaxis.label, axes.yaxis.label, axes.get_legend(), *figure.texts]
                for artist in artists:
                    bounds = artist.get_window_extent(renderer)
                    assert bounds.x0 >= canvas.x0 and bounds.x1 <= canvas.x1
                    assert bounds.y0 >= canvas.y0 and bounds.y1 <= canvas.y1	# 설명 글자의 실제 세로 경계가 최종 캔버스 안에 있어야 한다.
                plt.close(figure)


if __name__ == "__main__":
    unittest.main()
