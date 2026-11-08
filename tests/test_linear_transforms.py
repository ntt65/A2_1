"""Step 2 선형변환의 기하학적 결과와 오류 조건을 독립적으로 검증한다."""

from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.linear_algebra import (
    determinant_area_check, polygon_area, rotation_matrix, scaling_matrix,
    shear_matrix, transform_points, transformation_figure, unit_circle,
)

__covers__ = ["src/linear_algebra.py"]


class TransformTests(unittest.TestCase):
    """행렬식 구현과 독립적으로 기하학적 변환 결과를 확인한다."""

    def test_known_matrices_and_column_convention(self):
        """90도 회전·축별 스케일·수평 전단이 알려진 점에 작용하는 결과와 열벡터 관례를 확인한다."""
        point = np.array([[1.0], [2.0]])
        np.testing.assert_allclose(rotation_matrix(np.pi/2), [[0, -1], [1, 0]], atol=1e-15, rtol=0)
        np.testing.assert_allclose(transform_points(rotation_matrix(np.pi/2), point), [[-2], [1]], atol=1e-15, rtol=0)	# 열벡터 (1,2)의 90도 회전 결과 (-2,1)을 직접 확인한다.
        np.testing.assert_array_equal(transform_points(scaling_matrix(2, 0.5), point), [[2], [1]])
        np.testing.assert_array_equal(transform_points(shear_matrix(3), point), [[7], [2]])	# 수평 전단은 x+3y=7, y=2를 보존한다.
        assert transform_points(np.eye(2), point).shape == (2, 1)

    def test_circle_shape_norm_and_polygon_approximation(self):
        """원 표본의 반경이 1이고 다각형 넓이가 pi에 가까워지는지 확인한다."""
        circle = unit_circle(400)
        assert circle.shape == (2, 400) and circle.dtype == np.float64
        np.testing.assert_allclose(np.sum(circle**2, axis=0), 1, atol=1e-15, rtol=0)
        assert abs(polygon_area(circle) - np.pi) / np.pi < 1e-4	# 표본 다각형의 넓이를 독립 상수 pi와 상대오차로 비교한다.

    def test_polygon_known_area_winding_and_translation(self):
        """3×2 직사각형의 넓이 6이 순서 반전과 평행이동에도 유지되는지 확인한다."""
        rectangle = np.array([[0, 3, 3, 0], [0, 0, 2, 2]], dtype=float)
        assert polygon_area(rectangle) == 6.0
        assert polygon_area(rectangle[:, ::-1]) == 6.0
        assert polygon_area(rectangle + 1e8) == 6.0	# 큰 평행이동 뒤에도 직사각형 넓이 6을 보존한다.
        assert polygon_area(np.column_stack((rectangle, rectangle[:, 0]))) == 6.0

    def test_required_area_tolerance_and_reflection(self):
        """음의 행렬식을 포함하여 독립 넓이비 1·6·6을 미션 허용오차로 확인한다."""
        for matrix, expected in ((rotation_matrix(0.7), 1), (scaling_matrix(2, 0.5), 1),
                                 (shear_matrix(1.5), 1), (scaling_matrix(2, 3), 6),
                                 (scaling_matrix(-2, 3), 6)):
            with self.subTest(matrix=matrix):
                result = determinant_area_check(matrix)
                assert abs(result["area_ratio"] - expected) <= 1e-12	# 변환의 독립 넓이비 1·6·6을 먼저 확인한다.
                assert result["error_kind"] == "relative"
                assert result["error"] <= 0.01 and result["passed"]	# 행렬식과 넓이비 비교의 미션 상대오차 1%를 적용한다.
        assert determinant_area_check(scaling_matrix(-2, 3))["determinant"] < 0

    def test_singular_transform_uses_absolute_error(self):
        """랭크 1 변환의 상은 직선이므로 행렬식으로 나누지 않고 넓이비 0을 확인한다."""
        for matrix in (np.zeros((2, 2)), np.array([[1, 1], [2, 2]]),
                       np.array([[7, 7], [35, 35]])):
            result = determinant_area_check(matrix)
            assert result["error_kind"] == "absolute"	# 행렬식 0에서는 상대오차로 나누지 않고 절대오차를 사용한다.
            assert result["error"] <= 1e-10 and result["passed"]

    def test_input_shapes_types_and_degenerate_polygon(self):
        """실수 입력 조건 위반과 원본 다각형의 넓이 0을 명확하게 거부하는지 확인한다."""
        for value in (True, 3.5, "4"):
            with self.assertRaises(TypeError): unit_circle(value)
        with self.assertRaises(ValueError): unit_circle(2)
        for value in (np.nan, np.inf, [1, 2]):
            with self.assertRaises(ValueError): rotation_matrix(value)
        with self.assertRaises(TypeError): shear_matrix(1j)
        with self.assertRaises(ValueError): transform_points(np.eye(3), [[1], [2]])
        with self.assertRaises(ValueError): transform_points(np.eye(2), np.ones((3, 2)))
        with self.assertRaises(ValueError): polygon_area(np.zeros((2, 2)))
        with self.assertRaises(ValueError): determinant_area_check(np.eye(2), np.zeros((2, 3)))
        with self.assertRaises(ValueError): determinant_area_check(np.eye(2), relative_tolerance=-1)
        with self.assertRaises(ValueError): determinant_area_check(np.eye(2), absolute_tolerance=np.inf)

    def test_overflow_is_not_a_silent_geometric_result(self):
        """표현 불가능한 좌표 곱이나 다각형 넓이에 ValueError를 내는지 확인한다."""
        with self.assertRaises(ValueError):
            transform_points(np.full((2, 2), 1e308), np.full((2, 3), 1e308))
        with self.assertRaises(ValueError):
            polygon_area([[0, 1e308, 0], [0, 0, 1e308]])

    def test_underflow_is_not_reported_as_a_singular_success(self):
        """float64 범위 밖의 0이 아닌 행렬식을 0으로 오인하여 성공시키지 않는지 확인한다."""
        with self.assertRaises(ValueError):
            determinant_area_check(scaling_matrix(1e-200, 1e-200))
        with self.assertRaises(ValueError):
            determinant_area_check([[1e-200, 1.0], [0.0, 1e-200]])

    def test_comparison_figure_and_reference_arrows(self):
        """이름 붙인 중첩 그림 세 개와 기준 화살표가 실제 PNG로 저장되는지 확인한다."""
        figure, axes = transformation_figure(100)
        try:
            assert len(axes) == 3
            assert all(axis.get_xlabel() == "x" and axis.get_ylabel() == "y" for axis in axes)
            assert all(len(axis.texts) >= 2 for axis in axes)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "transforms.png"
                figure.savefig(path)
                assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")	# 단순 파일 존재가 아니라 실제 PNG 시그니처를 확인한다.
        finally:
            plt.close(figure)
