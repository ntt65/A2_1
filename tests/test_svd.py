"""SVD의 저랭크 근사·복원·캐시·그림을 독립 기준으로 검증한다."""

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
    """해석적으로 아는 행렬과 명시적인 영상 입력 조건으로 SVD를 검증한다."""

    def setUp(self):
        """각 독립 수치 검증의 시작에서 필수 시드를 설정한다."""
        set_seed()

    def tearDown(self):
        """매 검증이 끝나면 테스트가 만든 Matplotlib 그림을 닫는다."""
        plt.close("all")

    def test_diagonal_known_truncations(self):
        """알려진 대각행렬의 정확한 특이값과 랭크별 근사를 확인한다."""
        image = np.diag([9.0, 4.0, 1.0])
        u, singular_values, vt = svd_decompose(image)
        assert u.shape == (3, 3) and singular_values.shape == (3,) and vt.shape == (3, 3)
        np.testing.assert_allclose(singular_values, [9.0, 4.0, 1.0], rtol=0, atol=1e-12)	# 대각행렬의 특이값 9·4·1을 독립 기대값으로 비교한다.
        np.testing.assert_allclose(reconstruct_svd((u, singular_values, vt), 1), np.diag([9.0, 0.0, 0.0]), rtol=0, atol=1e-12)	# 랭크 1 근사는 가장 큰 대각 성분만 남겨야 한다.
        np.testing.assert_allclose(compress_image(image, 2), np.diag([9.0, 4.0, 0.0]), rtol=0, atol=1e-12)

    def test_nonsquare_rank_one_and_factor_shapes(self):
        """0이 아닌 두 벡터의 외적을 랭크 1로 정확히 복원하고 인수 형상을 확인한다."""
        image = np.outer([1.0, 2.0, -3.0], [4.0, 1.0, 0.0, -2.0, 3.0])
        factors = svd_decompose(image)
        assert tuple(value.shape for value in factors) == ((3, 3), (3,), (3, 5))
        assert all(value.dtype == np.float64 for value in factors)
        np.testing.assert_allclose(compress_image(image, 1), image, rtol=0, atol=1e-12)	# 외적으로 만든 랭크 1 행렬을 절대오차 1e-12로 복원한다.

    def test_full_rank_and_monotone_error(self):
        """랭크 증가 시 Frobenius 오차가 증가하지 않으며 전체 랭크에서 픽셀을 복원하는지 확인한다."""
        image = np.random.normal(size=(13, 9))
        factors = svd_decompose(image)
        errors = np.array([np.linalg.norm(image - reconstruct_svd(factors, rank), ord="fro") for rank in range(1, 10)])
        assert np.all(np.diff(errors) <= 1e-12)	# 랭크를 늘려도 Frobenius 오차가 증가하지 않는지 확인한다.
        assert errors[-1] / np.linalg.norm(image, ord="fro") <= 1e-10	# 전체 랭크 복원의 상대오차 기준 1e-10을 확인한다.
        for rank in (1, 3, 8):
            expected = np.sqrt(np.sum(factors[1][rank:] ** 2))	# 버린 특이값 제곱합의 제곱근으로 독립 복원 오차를 구한다.
            assert abs(errors[rank-1] - expected) <= 1e-12

    def test_zero_and_single_pixel(self):
        """영행렬과 단일 픽셀 회색조 영상의 결과가 유한하고 정확한지 확인한다."""
        assert np.array_equal(compress_image(np.zeros((3, 5)), 3), np.zeros((3, 5)))
        np.testing.assert_allclose(compress_image([[7]], 1), [[7]], rtol=0, atol=1e-12)
        np.testing.assert_allclose(compress_image([[-7]], 1), [[-7]], rtol=0, atol=1e-12)

    def test_invalid_images(self):
        """컬러·빈 배열·스칼라 영상, 비유한 값과 실수가 아닌 입력을 거부하는지 확인한다."""
        for image in (1.0, [1, 2], np.zeros((2, 2, 3)), np.zeros((0, 2)), [[np.nan]], [[np.inf]]):
            with self.subTest(image=image), self.assertRaises(ValueError):
                svd_decompose(image)
        for image in ([[True]], [[1j]], [["2"]], np.array([[1]], dtype=object)):
            with self.subTest(image=image), self.assertRaises(TypeError):
                svd_decompose(image)

    def test_invalid_rank_fails_before_decomposition(self):
        """잘못된 랭크를 잘라 맞추거나 불필요하게 SVD를 계산하지 않고 먼저 거부하는지 확인한다."""
        with patch("numpy.linalg.svd", side_effect=AssertionError("SVD must not run")):
            for rank in (0, -1, 4):
                with self.subTest(rank=rank), self.assertRaises(ValueError):
                    compress_image(np.eye(3), rank)
            for rank in (True, np.bool_(False), 1.0, "1"):
                with self.subTest(rank=rank), self.assertRaises(TypeError):
                    compress_image(np.eye(3), rank)
        assert compress_image(np.eye(3), np.int64(2)).shape == (3, 3)

    def test_factor_contract(self):
        """잘못된 인수 형상, 비유한 값과 잘못된 특이값 순서를 거부하는지 확인한다."""
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
        """수치 근사값을 표시 범위로 잘라내지 않고 원래 값을 보존하는지 확인한다."""
        factors = (np.array([[1.0], [0.0]]), np.array([2.0]), np.array([[1.0, 0.0]]))
        assert reconstruct_svd(factors, 1)[0, 0] == 2.0	# 표시 범위를 벗어난 복원값 2를 임의로 잘라내지 않는다.

    def test_nonrepresentable_result_and_svd_failure(self):
        """비유한 인수 곱을 거부하고 NumPy의 수렴 실패를 그대로 전달하는지 확인한다."""
        with self.assertRaises(ValueError):
            reconstruct_svd(([[1e308]], [1e308], [[1.0]]), 1)
        with patch("numpy.linalg.svd", return_value=(np.eye(2), np.array([np.inf, 1.0]), np.eye(2))):
            with self.assertRaises(ValueError):
                svd_decompose(np.eye(2))
        with patch("numpy.linalg.svd", side_effect=np.linalg.LinAlgError("fixture failure")):
            with self.assertRaises(np.linalg.LinAlgError):
                svd_decompose(np.eye(2))

    def test_cache_reuses_full_factors_across_ranks(self):
        """새 디스크 전용 관리자에서도 같은 영상의 전체 인수를 한 번만 계산해 랭크별로 재사용하는지 확인한다."""
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
                assert implementation.call_count == 1	# 랭크를 바꾸고 관리자를 새로 만들어도 저장된 전체 인수를 재사용한다.
                compress_image(image * 2, 1, cache)
                assert implementation.call_count == 2

    def test_disabled_cache_matches_uncached(self):
        """캐시를 끄면 매번 계산하면서도 복원 결과는 같게 유지하는지 확인한다."""
        image = np.array([[1.0, 4.0, 5.0], [2.0, 3.0, -1.0]])
        expected = compress_image(image, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unused"
            cache = CacheManager(path, enabled=False)
            with patch("numpy.linalg.svd", wraps=np.linalg.svd) as implementation:
                for unused in range(2):
                    np.testing.assert_allclose(compress_image(image, 1, cache), expected, rtol=0, atol=0)	# 캐시 비활성화 결과가 캐시 없는 기준값과 정확히 같다.
                assert implementation.call_count == 2
            assert not path.exists()
        with self.assertRaises(TypeError):
            svd_decompose(image, cache={})

    def test_comparison_figure_has_shared_scale_and_metrics(self):
        """독립적으로 아는 오차·저장량 설명과 공통 축척으로 비교 패널을 그리는지 확인한다."""
        figure, axes = svd_comparison_figure(np.diag([9.0, 4.0, 1.0]), [1, 3])
        assert axes.shape == (3,)
        assert all(axis.images[0].get_clim() == (0.0, 1.0) for axis in axes)
        assert "k=1" in axes[1].get_title()
        assert "stored/original=0.78x" in axes[1].get_title()
        assert "stored/original=2.33x" in axes[2].get_title()
        assert f"rel. error={np.sqrt(17.0 / 98.0):.3g}" in axes[1].get_title()	# 대각행렬의 버린 성분으로 계산한 독립 상대오차를 그림에 표시한다.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "svd.png"
            figure.savefig(path)
            assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    def test_zero_figure_and_invalid_figure_ranks(self):
        """영입력에는 유한한 절대오차를 표시하고 잘못된 랭크에서는 그림을 만들지 않는지 확인한다."""
        figure, axes = svd_comparison_figure(np.zeros((2, 3)), [1, 2])
        assert "abs. error=0" in axes[1].get_title()	# 영행렬의 오차는 나눗셈 대신 절대오차 0으로 표시한다.
        plt.close(figure)
        for ranks, exception in (([], TypeError), (None, TypeError), ([0], ValueError), ([4], ValueError), ([1.0], TypeError)):
            before = plt.get_fignums()
            with self.subTest(ranks=ranks), self.assertRaises(exception):
                svd_comparison_figure(np.eye(3), ranks)
            assert plt.get_fignums() == before	# 잘못된 그림 요청이 열린 Figure를 남기지 않는다.

    def test_extreme_pixels_render_with_common_scaled_mapping(self):
        """양·음의 유한 극값 픽셀도 색 정규화 오버플로 없이 그리는지 확인한다."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            figure, axes = svd_comparison_figure(np.array([[-1e308, 1e308]]), [1])
            figure.canvas.draw()
        assert all(axis.images[0].get_clim() == (-1.0, 1.0) for axis in axes)
        np.testing.assert_allclose(axes[0].images[0].get_array(), [[-1.0, 1.0]], rtol=0, atol=0)	# 양·음 극값을 같은 축척으로 표시한 기준 픽셀을 확인한다.
        np.testing.assert_allclose(axes[1].images[0].get_array(), [[-1.0, 1.0]], rtol=0, atol=1e-14)

    def test_comparison_titles_remain_inside_figure(self):
        """필수 네 패널 비교 그림 두 개에서 여러 줄 제목이 캔버스 안에 들어가는지 확인한다."""
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
                    assert title.y0 >= bounds.y0 and title.y1 <= bounds.y1	# 여러 줄 제목의 세로 경계가 캔버스를 넘지 않는지 확인한다.
                plt.close(figure)


if __name__ == "__main__":
    unittest.main()
