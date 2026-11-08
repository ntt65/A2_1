"""Independent low-rank, reconstruction, cache, and figure checks for SVD."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.linear_algebra import (
    compress_image, reconstruct_svd, svd_comparison_figure, svd_decompose,
)
from src.utils.cache_manager import CacheManager
from src.utils.random import set_seed

__covers__ = ["src/linear_algebra.py"]


class SVDTests(unittest.TestCase):
    """Validate SVD against analytical matrices and explicit image contracts."""

    def setUp(self):
        """Start each independent numerical test at the required seed."""
        set_seed()

    def tearDown(self):
        """Close test-owned Matplotlib figures after every test."""
        plt.close("all")

    def test_diagonal_known_truncations(self):
        """A known diagonal matrix has exact singular values and rank approximations."""
        image = np.diag([9.0, 4.0, 1.0])
        u, singular_values, vt = svd_decompose(image)
        assert u.shape == (3, 3) and singular_values.shape == (3,) and vt.shape == (3, 3)
        np.testing.assert_allclose(singular_values, [9.0, 4.0, 1.0], rtol=0, atol=1e-12)
        np.testing.assert_allclose(reconstruct_svd((u, singular_values, vt), 1), np.diag([9.0, 0.0, 0.0]), rtol=0, atol=1e-12)
        np.testing.assert_allclose(compress_image(image, 2), np.diag([9.0, 4.0, 0.0]), rtol=0, atol=1e-12)

    def test_nonsquare_rank_one_and_factor_shapes(self):
        """An outer product of nonzero vectors reconstructs exactly at rank one."""
        image = np.outer([1.0, 2.0, -3.0], [4.0, 1.0, 0.0, -2.0, 3.0])
        factors = svd_decompose(image)
        assert tuple(value.shape for value in factors) == ((3, 3), (3,), (3, 5))
        assert all(value.dtype == np.float64 for value in factors)
        np.testing.assert_allclose(compress_image(image, 1), image, rtol=0, atol=1e-12)

    def test_full_rank_and_monotone_error(self):
        """Increasing rank never increases Frobenius error; full rank recovers pixels."""
        image = np.random.normal(size=(13, 9))
        factors = svd_decompose(image)
        errors = np.array([np.linalg.norm(image - reconstruct_svd(factors, rank), ord="fro") for rank in range(1, 10)])
        assert np.all(np.diff(errors) <= 1e-12)
        assert errors[-1] / np.linalg.norm(image, ord="fro") <= 1e-10
        for rank in (1, 3, 8):
            expected = np.sqrt(np.sum(factors[1][rank:] ** 2))
            assert abs(errors[rank-1] - expected) <= 1e-12

    def test_zero_and_single_pixel(self):
        """Zero matrices and one-pixel grayscale images remain finite and exact."""
        assert np.array_equal(compress_image(np.zeros((3, 5)), 3), np.zeros((3, 5)))
        np.testing.assert_allclose(compress_image([[7]], 1), [[7]], rtol=0, atol=1e-12)
        np.testing.assert_allclose(compress_image([[-7]], 1), [[-7]], rtol=0, atol=1e-12)

    def test_invalid_images(self):
        """Reject color/empty/scalar, nonfinite, and non-real image arrays."""
        for image in (1.0, [1, 2], np.zeros((2, 2, 3)), np.zeros((0, 2)), [[np.nan]], [[np.inf]]):
            with self.subTest(image=image), self.assertRaises(ValueError):
                svd_decompose(image)
        for image in ([[True]], [[1j]], [["2"]], np.array([[1]], dtype=object)):
            with self.subTest(image=image), self.assertRaises(TypeError):
                svd_decompose(image)

    def test_invalid_rank_fails_before_decomposition(self):
        """Reject invalid ranks instead of clipping or doing unnecessary SVD work."""
        with patch("numpy.linalg.svd", side_effect=AssertionError("SVD must not run")):
            for rank in (0, -1, 4):
                with self.subTest(rank=rank), self.assertRaises(ValueError):
                    compress_image(np.eye(3), rank)
            for rank in (True, np.bool_(False), 1.0, "1"):
                with self.subTest(rank=rank), self.assertRaises(TypeError):
                    compress_image(np.eye(3), rank)
        assert compress_image(np.eye(3), np.int64(2)).shape == (3, 3)

    def test_factor_contract(self):
        """Reject invalid factor shapes, nonfinite values, and invalid singular order."""
        good = (np.eye(2), np.array([2.0, 1.0]), np.eye(2))
        for factors in ((good[0], good[1]), None, np.eye(2)):
            with self.subTest(factors=factors), self.assertRaises(TypeError):
                reconstruct_svd(factors, 1)
        invalid = [
            (np.eye(2), [[2.0, 1.0]], np.eye(2)),
            (np.eye(2), [1.0], np.eye(2)),
            (np.empty((2, 0)), [], np.empty((0, 2))),
            (np.ones((1, 2)), [2.0, 1.0], np.ones((2, 1))),
            (np.eye(2), [2.0, -1.0], np.eye(2)),
            (np.eye(2), [1.0, 2.0], np.eye(2)),
            (np.eye(2), [np.inf, 1.0], np.eye(2)),
        ]
        for factors in invalid:
            with self.subTest(factors=factors), self.assertRaises(ValueError):
                reconstruct_svd(factors, 1)
        np.testing.assert_allclose(reconstruct_svd((np.ones((2, 1)), [2.0], np.ones((1, 3))), 1), np.full((2, 3), 2.0), rtol=0, atol=0)

    def test_reconstruction_is_not_clipped(self):
        """The numerical approximation preserves out-of-display-range values."""
        factors = (np.array([[1.0], [0.0]]), np.array([2.0]), np.array([[1.0, 0.0]]))
        assert reconstruct_svd(factors, 1)[0, 0] == 2.0

    def test_nonrepresentable_result_and_svd_failure(self):
        """Reject nonfinite factor products and preserve NumPy convergence failures."""
        with self.assertRaises(ValueError):
            reconstruct_svd(([[1e308]], [1e308], [[1.0]]), 1)
        with patch("numpy.linalg.svd", return_value=(np.eye(2), np.array([np.inf, 1.0]), np.eye(2))):
            with self.assertRaises(ValueError):
                svd_decompose(np.eye(2))
        with patch("numpy.linalg.svd", side_effect=np.linalg.LinAlgError("fixture failure")):
            with self.assertRaises(np.linalg.LinAlgError):
                svd_decompose(np.eye(2))

    def test_cache_reuses_full_factors_across_ranks(self):
        """One image factors once across ranks, including a fresh disk-only manager."""
        image = np.diag([9.0, 4.0, 1.0])
        with tempfile.TemporaryDirectory() as directory:
            cache = CacheManager(directory)
            with patch("numpy.linalg.svd", wraps=np.linalg.svd) as implementation:
                first = compress_image(image, 1, cache)
                second = compress_image(image, 2, cache)
                assert implementation.call_count == 1
                original_factors = svd_decompose(image, cache)
                original_factors[0][:] = 0
                assert np.array_equal(compress_image(image, 1, cache), first)
                disk_cache = CacheManager(directory, max_memory_entries=0)
                np.testing.assert_allclose(compress_image(image, 2, disk_cache), second, rtol=0, atol=0)
                assert implementation.call_count == 1
                compress_image(image * 2, 1, cache)
                assert implementation.call_count == 2

    def test_disabled_cache_matches_uncached(self):
        """Disabled caching computes every call while preserving reconstructed values."""
        image = np.array([[1.0, 4.0, 5.0], [2.0, 3.0, -1.0]])
        expected = compress_image(image, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unused"
            cache = CacheManager(path, enabled=False)
            with patch("numpy.linalg.svd", wraps=np.linalg.svd) as implementation:
                for unused in range(2):
                    np.testing.assert_allclose(compress_image(image, 1, cache), expected, rtol=0, atol=0)
                assert implementation.call_count == 2
            assert not path.exists()
        with self.assertRaises(TypeError):
            svd_decompose(image, cache={})

    def test_comparison_figure_has_shared_scale_and_metrics(self):
        """Render comparable panels with independent analytical error/storage labels."""
        figure, axes = svd_comparison_figure(np.diag([9.0, 4.0, 1.0]), [1, 3])
        assert axes.shape == (3,)
        assert all(axis.images[0].get_clim() == (0.0, 1.0) for axis in axes)
        assert "k=1" in axes[1].get_title()
        assert "stored/original=0.78x" in axes[1].get_title()
        assert "stored/original=2.33x" in axes[2].get_title()
        assert f"rel. error={np.sqrt(17.0 / 98.0):.3g}" in axes[1].get_title()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "svd.png"
            figure.savefig(path)
            assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    def test_zero_figure_and_invalid_figure_ranks(self):
        """Zero input has finite absolute-error labels; invalid ranks create no figures."""
        figure, axes = svd_comparison_figure(np.zeros((2, 3)), [1, 2])
        assert "abs. error=0" in axes[1].get_title()
        plt.close(figure)
        for ranks, exception in (([], TypeError), (None, TypeError), ([0], ValueError), ([4], ValueError), ([1.0], TypeError)):
            before = plt.get_fignums()
            with self.subTest(ranks=ranks), self.assertRaises(exception):
                svd_comparison_figure(np.eye(3), ranks)
            assert plt.get_fignums() == before

    def test_extreme_pixels_render_with_common_scaled_mapping(self):
        """Opposite finite extremes render without overflowing color normalization."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            figure, axes = svd_comparison_figure(np.array([[-1e308, 1e308]]), [1])
            figure.canvas.draw()
        assert all(axis.images[0].get_clim() == (-1.0, 1.0) for axis in axes)
        np.testing.assert_allclose(axes[0].images[0].get_array(), [[-1.0, 1.0]], rtol=0, atol=0)
        np.testing.assert_allclose(axes[1].images[0].get_array(), [[-1.0, 1.0]], rtol=0, atol=1e-14)

    def test_comparison_titles_remain_inside_figure(self):
        """Both required four-panel comparisons keep multiline titles in the canvas."""
        for size, ranks in ((64, [10, 50, 64]), (128, [10, 50, 100])):
            with self.subTest(size=size):
                image = np.diag(np.linspace(1.0, 0.1, size))
                figure, axes = svd_comparison_figure(image, ranks)
                figure.set_dpi(140)
                figure.canvas.draw()
                renderer = figure.canvas.get_renderer()
                bounds = figure.bbox
                for axis in axes:
                    title = axis.title.get_window_extent(renderer)
                    assert title.x0 >= bounds.x0 and title.x1 <= bounds.x1
                    assert title.y0 >= bounds.y0 and title.y1 <= bounds.y1
                plt.close(figure)


if __name__ == "__main__":
    unittest.main()
