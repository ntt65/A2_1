"""Independent geometry and error-contract checks for Step 2 transforms."""

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
    """Check geometric outcomes independently of determinant implementations."""

    def test_known_matrices_and_column_convention(self):
        """A quarter turn, independent scaling and a horizontal shear map known points."""
        point = np.array([[1.0], [2.0]])
        np.testing.assert_allclose(rotation_matrix(np.pi/2), [[0, -1], [1, 0]], atol=1e-15, rtol=0)
        np.testing.assert_allclose(transform_points(rotation_matrix(np.pi/2), point), [[-2], [1]], atol=1e-15, rtol=0)
        np.testing.assert_array_equal(transform_points(scaling_matrix(2, 0.5), point), [[2], [1]])
        np.testing.assert_array_equal(transform_points(shear_matrix(3), point), [[7], [2]])
        assert transform_points(np.eye(2), point).shape == (2, 1)

    def test_circle_shape_norm_and_polygon_approximation(self):
        """Circle samples have unit radius and a polygon area approaching pi."""
        circle = unit_circle(400)
        assert circle.shape == (2, 400) and circle.dtype == np.float64
        np.testing.assert_allclose(np.sum(circle**2, axis=0), 1, atol=1e-15, rtol=0)
        assert abs(polygon_area(circle) - np.pi) / np.pi < 1e-4

    def test_polygon_known_area_winding_and_translation(self):
        """A 3-by-2 rectangle has area six under winding reversal and translation."""
        rectangle = np.array([[0, 3, 3, 0], [0, 0, 2, 2]], dtype=float)
        assert polygon_area(rectangle) == 6.0
        assert polygon_area(rectangle[:, ::-1]) == 6.0
        assert polygon_area(rectangle + 1e8) == 6.0
        assert polygon_area(np.column_stack((rectangle, rectangle[:, 0]))) == 6.0

    def test_required_area_tolerance_and_reflection(self):
        """Known area ratios are one, six and six, including a negative determinant."""
        for matrix, expected in ((rotation_matrix(0.7), 1), (scaling_matrix(2, 0.5), 1),
                                 (shear_matrix(1.5), 1), (scaling_matrix(2, 3), 6),
                                 (scaling_matrix(-2, 3), 6)):
            with self.subTest(matrix=matrix):
                result = determinant_area_check(matrix)
                assert abs(result["area_ratio"] - expected) <= 1e-12
                assert result["error_kind"] == "relative"
                assert result["error"] <= 0.01 and result["passed"]
        assert determinant_area_check(scaling_matrix(-2, 3))["determinant"] < 0

    def test_singular_transform_uses_absolute_error(self):
        """A rank-one image is a line, so its area ratio is zero without division by det."""
        for matrix in (np.zeros((2, 2)), np.array([[1, 1], [2, 2]]),
                       np.array([[7, 7], [35, 35]])):
            result = determinant_area_check(matrix)
            assert result["error_kind"] == "absolute"
            assert result["error"] <= 1e-10 and result["passed"]

    def test_input_shapes_types_and_degenerate_polygon(self):
        """Invalid real-data contracts and zero source area fail clearly."""
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
        """Unrepresentable coordinate products or polygon areas raise ValueError."""
        with self.assertRaises(ValueError):
            transform_points(np.full((2, 2), 1e308), np.full((2, 3), 1e308))
        with self.assertRaises(ValueError):
            polygon_area([[0, 1e308, 0], [0, 0, 1e308]])

    def test_underflow_is_not_reported_as_a_singular_success(self):
        """A nonzero determinant outside float64 range must not silently pass as zero."""
        with self.assertRaises(ValueError):
            determinant_area_check(scaling_matrix(1e-200, 1e-200))
        with self.assertRaises(ValueError):
            determinant_area_check([[1e-200, 1.0], [0.0, 1e-200]])

    def test_comparison_figure_and_reference_arrows(self):
        """Three labeled overlays include reference arrows and export a real PNG."""
        figure, axes = transformation_figure(100)
        try:
            assert len(axes) == 3
            assert all(axis.get_xlabel() == "x" and axis.get_ylabel() == "y" for axis in axes)
            assert all(len(axis.texts) >= 2 for axis in axes)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "transforms.png"
                figure.savefig(path)
                assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        finally:
            plt.close(figure)
