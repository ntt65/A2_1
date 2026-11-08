"""Checks for reusable contour grids and headless plot construction."""

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
    """Validate geometry, function contracts, figure ownership, and PNG export."""

    def test_mesh_orientation_and_values(self):
        """A known affine function checks xy indexing independently of the grid implementation."""
        x_grid, y_grid, values = contour_grid(lambda x, y: x + 2 * y, (-2, 2), (-1, 1), 3)
        np.testing.assert_array_equal(x_grid, [[-2, 0, 2]] * 3)
        np.testing.assert_array_equal(y_grid, [[-1] * 3, [0] * 3, [1] * 3])
        np.testing.assert_array_equal(values, [[-4, -2, 0], [-2, 0, 2], [0, 2, 4]])
        for array in (x_grid, y_grid, values):
            assert array.shape == (3, 3)
            assert array.dtype == np.float64

    def test_contour_figure_can_be_saved(self):
        """The reusable base labels both axes and exports an actual PNG without showing a window."""
        figure, axes = contour_base(lambda x, y: x + y, points=9, levels=5, title="Grid check")
        try:
            assert axes.figure is figure
            assert axes.get_xlabel() == "x"
            assert axes.get_ylabel() == "y"
            assert axes.get_title() == "Grid check"
            assert axes.get_aspect() == 1.0
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "contours.png"
                figure.savefig(target)
                assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
                assert target.stat().st_size > 100
        finally:
            plt.close(figure)

    def test_invalid_grid_parameters(self):
        """Invalid ranges or point counts fail before allocating a mesh or figure."""
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
        """The grid function must provide finite real values with the exact grid shape."""
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
        """Reject invalid contour levels without creating a figure requiring cleanup."""
        before = plt.get_fignums()
        for levels in (0, -1):
            with self.assertRaises(ValueError):
                contour_base(lambda x, y: x, levels=levels)
        for levels in (True, 2.5):
            with self.assertRaises(TypeError):
                contour_base(lambda x, y: x, levels=levels)
        assert plt.get_fignums() == before


if __name__ == "__main__":
    unittest.main()
