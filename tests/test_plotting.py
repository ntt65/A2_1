"""재사용 가능한 등고선 격자와 화면 없는 그림 생성을 검증한다."""

from pathlib import Path
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.plotting import contour_base, contour_grid

__covers__ = ["src/utils/plotting.py"]


class ContourTests(unittest.TestCase):
    """기하학적 배치, 함수 조건, 그림 소유권과 PNG 출력을 확인한다."""

    def test_mesh_orientation_and_values(self):
        """알려진 일차함수로 구현과 독립적으로 xy 인덱싱 방향과 격자값을 확인한다."""
        x_grid, y_grid, values = contour_grid(lambda x, y: x + 2 * y, (-2, 2), (-1, 1), 3)
        np.testing.assert_array_equal(x_grid, [[-2, 0, 2]] * 3)	# x가 열 방향으로 증가하는 xy 격자 관례를 확인한다.
        np.testing.assert_array_equal(y_grid, [[-1] * 3, [0] * 3, [1] * 3])
        np.testing.assert_array_equal(values, [[-4, -2, 0], [-2, 0, 2], [0, 2, 4]])	# 알려진 일차함수를 격자 각 좌표에 직접 대입한 표와 비교한다.
        for array in (x_grid, y_grid, values):
            assert array.shape == (3, 3)
            assert array.dtype == np.float64

    def test_contour_figure_can_be_saved(self):
        """재사용 가능한 기본 그림이 두 축 이름을 표시하고 창 없이 실제 PNG로 저장되는지 확인한다."""
        figure, axes = contour_base(lambda x, y: x + y, points=9, levels=5, title="Grid check")
        try:
            assert axes.figure is figure
            assert axes.get_xlabel() == "x"
            assert axes.get_ylabel() == "y"
            assert axes.get_title() == "Grid check"
            assert axes.get_aspect() == 1.0	# x와 y의 길이를 같은 축척으로 표시한다.
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "contours.png"
                figure.savefig(target)
                assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")	# 화면을 띄우지 않아도 실제 PNG 바이트가 생성되는지 확인한다.
                assert target.stat().st_size > 100
        finally:
            plt.close(figure)

    def test_invalid_grid_parameters(self):
        """잘못된 구간이나 점 개수를 격자·그림 할당 전에 거부하는지 확인한다."""
        for bounds in ((1, 1), (2, 1), (0, np.inf), (0,), (0, 1, 2)):
            with self.subTest(bounds=bounds):
                with self.assertRaises(ValueError):
                    contour_grid(lambda x, y: x, x_range=bounds)
        for points in (0, 1, -1):
            with self.assertRaises(ValueError):
                contour_grid(lambda x, y: x, points=points)
        for points in (2.5, True):
            with self.assertRaises(TypeError):
                contour_grid(lambda x, y: x, points=points)
        with self.assertRaises(TypeError):
            contour_grid(None)
        with self.assertRaises(ValueError):
            contour_grid(lambda x, y: x, x_range=(-1e308, 1e308))

    def test_invalid_function_output(self):
        """격자 함수가 정확한 격자 형상의 유한 실수 값을 제공해야 하는지 확인한다."""
        for function in (
            lambda x, y: 1.0,
            lambda x, y: x[0],
            lambda x, y: np.full_like(x, np.nan),
        ):
            with self.assertRaises(ValueError):
                contour_grid(function, points=3)
        with self.assertRaises(TypeError):
            contour_grid(lambda x, y: x.astype(complex), points=3)

    def test_invalid_levels_do_not_leak_figures(self):
        """잘못된 등고선 수준을 거부할 때 정리가 필요한 그림을 남기지 않는지 확인한다."""
        before = plt.get_fignums()
        for levels in (0, -1):
            with self.assertRaises(ValueError):
                contour_base(lambda x, y: x, levels=levels)
        for levels in (True, 2.5):
            with self.assertRaises(TypeError):
                contour_base(lambda x, y: x, levels=levels)
        assert plt.get_fignums() == before	# 잘못된 등고선 수준을 거부할 때 Figure가 누출되지 않는다.


if __name__ == "__main__":
    unittest.main()
