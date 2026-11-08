"""직접 구현한 Power Iteration의 정확도와 실패 상태를 독립 기준으로 검증한다."""

from dataclasses import FrozenInstanceError
import unittest
from unittest.mock import patch

import numpy as np

from src.linear_algebra import PowerIterationResult, power_iteration
from src.utils.random import set_seed


__covers__ = ["src/linear_algebra.py"]


class PowerIterationTests(unittest.TestCase):
    """해석값 및 검증에만 허용한 eig 호출로 고유쌍을 대조한다."""

    def test_mission_matrix_matches_analytic_value_and_eig(self):
        """미션 행렬의 지배 고유쌍과 잔차를 해석값으로 확인한다."""
        matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
        result = power_iteration(matrix, initial=[1.0, 1.0])
        expected = (7.0 + np.sqrt(5.0)) / 2.0	# 2×2 특성방정식에서 얻은 지배 고유값을 독립 기준으로 둔다.
        values, vectors = np.linalg.eig(matrix)	# np.linalg.eig는 알고리즘 구현이 아닌 독립 비교 검증에만 사용한다.
        index = int(np.argmax(np.abs(values)))
        actual_residual = np.linalg.norm(matrix @ result.eigenvector - result.eigenvalue * result.eigenvector)
        actual_residual /= np.linalg.norm(matrix, ord="fro") * np.linalg.norm(result.eigenvector)
        assert isinstance(result, PowerIterationResult)
        assert result.converged and result.reason == "residual_tolerance"
        assert 0 < result.iterations <= 1000
        assert abs(result.eigenvalue - expected) / abs(expected) <= 0.05	# 미션 상대오차 기준 5%를 그대로 적용한다.
        assert abs(result.eigenvalue - values[index]) / abs(values[index]) <= 0.05
        assert abs(abs(np.dot(result.eigenvector, vectors[:, index])) - 1.0) <= 1e-8	# 고유벡터는 부호가 달라도 같은 방향이므로 내적 절댓값으로 비교한다.
        assert actual_residual <= 1e-6
        assert abs(actual_residual - result.residual) <= 1e-14	# 직접 계산한 잔차와 보고된 잔차가 일치해야 한다.
        assert abs(np.linalg.norm(result.eigenvector) - 1.0) <= 1e-14
        assert result.eigenvector.shape == (2,) and result.eigenvector.dtype == np.float64

    def test_negative_dominant_eigenvalue_preserves_sign(self):
        """고유벡터의 부호가 번갈아도 음의 지배 고유값에 대한 잔차 수렴을 막지 않는지 확인한다."""
        matrix = np.diag([-5.0, 2.0])
        result = power_iteration(matrix, initial=[1.0, 1.0])
        assert result.converged
        assert abs(result.eigenvalue + 5.0) <= 1e-9	# 지배 고유값이 음수인 경우에도 부호를 보존한다.
        assert abs(abs(result.eigenvector[0]) - 1.0) <= 1e-9
        assert result.residual <= 1e-10

    def test_zero_matrix_has_an_explicit_exact_result(self):
        """영행렬의 모든 단위 방향이 고유값 0의 고유벡터라는 정확한 결과를 확인한다."""
        result = power_iteration(np.zeros((3, 3)), initial=[1.0, 2.0, 2.0])
        assert result.converged and result.reason == "zero_matrix"
        assert result.iterations == 0 and result.eigenvalue == 0.0 and result.residual == 0.0	# 영행렬은 반복 없이 고유값·잔차가 모두 정확히 0이다.
        np.testing.assert_allclose(result.eigenvector, [1 / 3, 2 / 3, 2 / 3], atol=1e-15, rtol=0)

    def test_nullspace_and_nilpotent_breakdown_are_not_dominance_success(self):
        """0이 아닌 행렬에서 반복 벡터가 소실되면 지배 고유값 성공 대신 계산 중단으로 기록하는지 확인한다."""
        immediate = power_iteration(np.diag([3.0, 0.0]), initial=[0.0, 1.0])
        later = power_iteration([[0.0, 1.0], [0.0, 0.0]], initial=[1.0, 1.0])
        for result, iterations in ((immediate, 0), (later, 1)):
            assert not result.converged and result.reason == "nullspace_breakdown"	# 비영행렬의 영공간 진입을 지배 고유값 수렴으로 표시하지 않는다.
            assert result.eigenvalue == 0.0 and result.residual == 0.0
            assert result.iterations == iterations

    def test_equal_magnitude_and_complex_modes_can_fail_to_converge(self):
        """절댓값이 같은 고유값이나 실수 회전에서는 갱신 예산이 소진될 수 있음을 확인한다."""
        for matrix in (np.diag([-2.0, 2.0]), np.array([[0.0, -1.0], [1.0, 0.0]])):
            with self.subTest(matrix=matrix):
                result = power_iteration(matrix, initial=[1.0, 1.0], max_iter=9)
                assert not result.converged and result.reason == "max_iterations"	# 동일 절댓값 고유값이나 회전의 미수렴을 예산 소진으로 기록한다.
                assert result.iterations == 9 and result.residual > 0.1
        exact_tie = power_iteration(np.eye(2), initial=[1.0, 2.0])
        assert exact_tie.converged and abs(exact_tie.eigenvalue - 1.0) <= 1e-14

    def test_budget_and_tolerance_are_observable(self):
        """한 번 갱신한 근사 결과에 명시적인 미수렴 표시가 있는지 확인한다."""
        result = power_iteration([[4.0, 1.0], [1.0, 3.0]], initial=[1.0, 0.0], max_iter=1, tol=1e-15)
        assert result.iterations == 1
        assert not result.converged and result.reason == "max_iterations"
        assert result.residual > 1e-15

    def test_subdominant_initial_direction_documents_residual_limit(self):
        """작은 잔차가 고유쌍만 검증하며 지배 고유값까지 보장하지 않는 반례를 확인한다."""
        result = power_iteration(np.diag([5.0, 1.0]), initial=[0.0, 1.0])
        assert result.converged and result.iterations == 0
        assert result.eigenvalue == 1.0 and result.residual == 0.0	# 잔차 0이어도 작은 고유값 1일 수 있어 지배성은 보장하지 않는다.

    def test_uniform_scales_and_extreme_initial_vectors_are_supported(self):
        """표현 가능한 고유쌍의 노름이 스케일링으로 오버플로·언더플로하지 않는지 확인한다."""
        base = np.array([[4.0, 1.0], [1.0, 3.0]])
        expected = (7.0 + np.sqrt(5.0)) / 2.0
        with np.errstate(all="raise"):
            for scale in (1e-300, 1e300):
                for initial in ([1e-300, 1e-300], [1e300, 1e300]):
                    result = power_iteration(base * scale, initial=initial)
                    assert result.converged
                    assert abs(result.eigenvalue / scale - expected) / expected <= 1e-8	# 행렬의 스케일을 되돌린 고유값을 같은 독립 해와 비교한다.
                    assert np.all(np.isfinite(result.eigenvector)) and result.residual <= 1e-10

    def test_unrepresentable_eigenvalues_and_matrix_scaling_fail_explicitly(self):
        """표현 불가능한 고유값을 무한대로 돌려주거나 스케일링으로 비영 성분을 지우지 않는지 확인한다."""
        largest = np.finfo(np.float64).max
        with self.assertRaises(ValueError):
            power_iteration(np.full((2, 2), largest), initial=[1.0, 1.0])
        with self.assertRaises(ValueError):
            power_iteration(np.diag([1e308, 1e-308]), initial=[1.0, 1.0])

    def test_tiny_residual_is_not_mistaken_for_exact_convergence(self):
        """아주 작은 동일 절댓값 고유값 블록도 더 엄격한 허용오차에서는 순환을 계속하는지 확인한다."""
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1e-200], [0.0, 1e-200, 0.0]])
        result = power_iteration(matrix, initial=[0.0, 1.0, 0.0], tol=1e-250, max_iter=4)
        assert not result.converged and result.reason == "max_iterations"
        assert result.iterations == 4
        assert abs(result.residual / 1e-200 - 1.0) <= 1e-14	# 극소 잔차를 0으로 소실시키지 않고 원래 크기를 보존한다.

    def test_invalid_matrix_initial_and_control_inputs_fail(self):
        """맞지 않는 형상·비유한 값·영벡터 초기값과 잘못된 제어 매개변수를 거부하는지 확인한다."""
        for matrix in ([], [1, 2], np.zeros((0, 0)), np.zeros((2, 3)), [[1, np.nan], [0, 2]], [[1, np.inf], [0, 2]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                power_iteration(matrix)
        for matrix in ([[True, False], [False, True]], [[1j, 0], [0, 1]], [["1", "0"], ["0", "1"]]):
            with self.subTest(matrix=matrix), self.assertRaises(TypeError):
                power_iteration(matrix)
        for initial in ([0.0, 0.0], [1.0], [[1.0, 1.0]], [1.0, np.nan]):
            with self.subTest(initial=initial), self.assertRaises(ValueError):
                power_iteration(np.eye(2), initial=initial)
        for tolerance in (0.0, -1.0, np.nan, np.inf, [1e-10]):
            with self.subTest(tol=tolerance), self.assertRaises(ValueError):
                power_iteration(np.eye(2), tol=tolerance)
        with self.assertRaises(TypeError):
            power_iteration(np.eye(2), tol=True)
        for count in (0, -1, 1.5, True):
            with self.subTest(max_iter=count), self.assertRaises((TypeError, ValueError)):
                power_iteration(np.eye(2), max_iter=count)

    def test_external_seed_reproduces_random_start_without_reseeding(self):
        """난수는 호출자가 제어하며 초기 벡터를 지정하면 난수를 소비하지 않는지 확인한다."""
        state = np.random.get_state()
        try:
            set_seed()
            first = power_iteration([[4.0, 1.0], [1.0, 3.0]])
            set_seed()
            second = power_iteration([[4.0, 1.0], [1.0, 3.0]])
            assert first.eigenvalue == second.eigenvalue and first.iterations == second.iterations
            np.testing.assert_array_equal(first.eigenvector, second.eigenvector)	# 외부 시드를 같게 설정하면 전체 고유벡터 결과가 재현된다.
            with patch("numpy.random.normal", side_effect=AssertionError("unexpected draw")):
                power_iteration(np.eye(2), initial=[1.0, 1.0])
        finally:
            np.random.set_state(state)

    def test_inputs_are_preserved_and_eigensolvers_are_not_used(self):
        """수동 알고리즘이 입력을 바꾸거나 고유값 계산기에 위임하지 않는지 확인한다."""
        matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
        initial = np.array([1.0, 1.0])
        saved_matrix, saved_initial = matrix.copy(), initial.copy()
        with patch("numpy.linalg.eig", side_effect=AssertionError("forbidden eig")), patch(
            "numpy.linalg.eigvals", side_effect=AssertionError("forbidden eigvals")
        ), patch("numpy.linalg.eigh", side_effect=AssertionError("forbidden eigh")):
            result = power_iteration(matrix, initial=initial)
        np.testing.assert_array_equal(matrix, saved_matrix)
        np.testing.assert_array_equal(initial, saved_initial)	# 명시한 초기 벡터는 알고리즘 호출 후에도 원본을 유지한다.
        result.eigenvector[0] = 0.0
        np.testing.assert_array_equal(initial, saved_initial)
        with self.assertRaises(FrozenInstanceError):
            result.iterations = 0


if __name__ == "__main__":
    unittest.main()
