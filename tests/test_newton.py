"""Newton 갱신을 독립 이차함수·Hessian 기대값과 수치 실패 반례로 검증한다."""

import unittest

import numpy as np

from src.calculus import quadratic_gradient
from src.newton import NewtonMethod, circle_hessian, elliptic_hessian
from src.optimizer import elliptic_gradient, run_optimization
from src.utils.numerical import central_difference

__covers__ = ["src/newton.py", "src/optimizer.py"]


class NewtonStepTests(unittest.TestCase):
    """정확한 최소점, 감쇠, 입력 소유권과 Hessian의 조건을 검증한다."""

    def test_general_spd_quadratic_reaches_known_shifted_minimizer(self):
        """변수 간 결합이 있는 3차원 양의 정부호 이차함수가 독립적으로 아는 중심에 한 번에 도달하는지 확인한다."""
        matrix = np.array([[4., 1., 0.], [1., 3., 1.], [0., 1., 2.]])
        center = np.array([1., -2., .5])
        initial = np.array([5., 4., -3.])

        def objective(point):
            """최소점이 알려진 center (3,)를 기준으로 이동된 이차함수 값을 계산한다."""
            delta = point - center
            return .5 * float(delta @ matrix @ delta)

        def gradient(point):
            """변수 간 결합이 있는 이차함수의 기울기 (3,)를 반환한다."""
            return matrix @ (point - center)

        optimizer = NewtonMethod(lambda point: matrix)
        result = run_optimization(objective, gradient, initial, optimizer, steps=5,
                                  gradient_tolerance=1e-12)
        assert result.iterations == 1 and result.status == "gradient_tolerance"
        assert result.path.shape == (2, 3)
        np.testing.assert_allclose(result.path[-1], center, atol=1e-12, rtol=0)	# 독립적으로 아는 이차함수 중심과 절대오차 1e-12로 비교한다.
        assert abs(result.losses[-1]) <= 1e-24
        np.testing.assert_array_equal(initial, [5., 4., -3.])

    def test_fixed_damping_matches_closed_form_path(self):
        """감쇠 .25에서 theta_k=center+.75**k*(theta_0-center)인 닫힌형 경로와 비교한다."""
        matrix = np.array([[3., 1.], [1., 2.]])
        center, initial = np.array([-2., 3.]), np.array([6., -5.])
        result = run_optimization(
            lambda point: .5 * float((point-center) @ matrix @ (point-center)),
            lambda point: matrix @ (point-center), initial,
            NewtonMethod(lambda point: matrix, lr=.25), steps=8,
        )
        expected = center + .75**np.arange(9)[:, None] * (initial-center)	# 감쇠 .25의 오차 배율 .75를 반복 구현 없이 거듭제곱으로 계산한다.
        np.testing.assert_allclose(result.path, expected, atol=1e-12, rtol=0)	# 전체 감쇠 경로를 닫힌형 해와 비교한다.
        assert result.iterations == 8 and result.status == "budget_exhausted"

    def test_mission_quadratics_reach_zero_in_one_update(self):
        """반복 횟수 조정 없이 두 고정 Hessian이 알려진 최소점인 원점에 한 번에 도달하는지 확인한다."""
        for hessian, gradient in ((circle_hessian, quadratic_gradient),
                                  (elliptic_hessian, elliptic_gradient)):
            with self.subTest(hessian=hessian.__name__):
                point = np.array([5., -3.])
                actual = NewtonMethod(hessian).step(point, gradient(point))
                np.testing.assert_allclose(actual, [0., 0.], atol=1e-14, rtol=0)	# 두 미션 함수의 알려진 최소점인 원점과 비교한다.

    def test_hessian_callback_cannot_mutate_input_or_output(self):
        """콜백에 입력 복사본을 전달하고 기존 H를 반환해도 원본을 바꾸지 않는지 확인한다."""
        matrix = np.diag([2., 4.])
        theta, gradient = np.array([3., 2.]), np.array([2., 8.])

        def hessian(point):
            """전달받은 복사본을 의도적으로 바꾸면서 저장된 Hessian을 반환한다."""
            point[:] = 999
            return matrix

        actual = NewtonMethod(hessian).step(theta, gradient)
        np.testing.assert_allclose(actual, [2., 0.], atol=1e-15, rtol=0)	# H의 각 대각 성분으로 나눈 독립 갱신값을 확인한다.
        actual[:] = -100
        np.testing.assert_array_equal(theta, [3., 2.])
        np.testing.assert_array_equal(gradient, [2., 8.])
        np.testing.assert_array_equal(matrix, np.diag([2., 4.]))	# 콜백이 돌려준 원래 Hessian 배열은 변하지 않는다.

    def test_constructor_and_step_contracts(self):
        """잘못된 감쇠율, Hessian 콜백과 서로 맞지 않는 유한 벡터를 거부하는지 확인한다."""
        for rate in (0., -1., 1.01, np.inf, np.nan, [.5]):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                NewtonMethod(circle_hessian, lr=rate)
        for rate in (True, "1", 1j):
            with self.subTest(rate=rate), self.assertRaises(TypeError):
                NewtonMethod(circle_hessian, lr=rate)
        with self.assertRaises(TypeError):
            NewtonMethod(np.eye(2))
        optimizer = NewtonMethod(circle_hessian)
        for point, gradient in (([], []), ([[1., 2.]], [[1., 2.]]),
                                ([1., 2.], [1.]), ([np.inf, 2.], [1., 2.])):
            with self.subTest(point=point), self.assertRaises(ValueError):
                optimizer.step(point, gradient)
        with self.assertRaises(TypeError):
            optimizer.step([True, False], [1., 2.])

    def test_hessian_type_shape_symmetry_and_curvature(self):
        """실수가 아닌 값, 잘못된 크기, 비대칭·특이·부정부호 Hessian을 거부하는지 확인한다."""
        for matrix in (np.ones(2), np.eye(3), [[2., 1.], [0., 2.]],
                       np.diag([1., 0.]), np.diag([1., -1.]), [[1., 2.], [2., 1.]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                NewtonMethod(lambda point: matrix).step([1., 1.], [2., 2.])
        for matrix in ([[True, False], [False, True]], [[1j, 0], [0, 1j]],
                       [["1", "0"], ["0", "1"]]):
            with self.subTest(matrix=matrix), self.assertRaises(TypeError):
                NewtonMethod(lambda point: matrix).step([1., 1.], [2., 2.])

    def test_accepted_roundoff_asymmetry_uses_symmetric_part(self):
        """허용 범위의 작은 반대칭 잡음을 대칭 부분으로 바꾼 뒤 양의 정부호 선형계를 푸는지 확인한다."""
        matrix = np.array([[2., 1e-13], [-1e-13, 3.]])
        actual = NewtonMethod(lambda point: matrix).step([3., 4.], [2., 6.])
        np.testing.assert_allclose(actual, [2., 2.], atol=1e-14, rtol=0)	# 허용한 비대칭 성분을 제거한 대칭계의 해를 확인한다.
        near_symmetric_but_indefinite = np.array([[1e-20, 1e-12], [0., 1e-20]])
        with self.assertRaisesRegex(ValueError, "positive definite"):
            NewtonMethod(lambda point: near_symmetric_but_indefinite).step([0., 0.], [1., 1.])

    def test_exact_symmetric_subnormal_hessians_preserve_newton_direction(self):
        """H=[s], g=[s]에서 s가 최소 또는 홀수 배 비정규 값이어도 방향이 1인지 확인한다."""
        smallest = np.nextafter(0., 1.)
        for value in (smallest, 3 * smallest):
            with self.subTest(value=value):
                matrix = np.array([[value]])
                actual = NewtonMethod(lambda point: matrix).step([2.], [value])
                np.testing.assert_array_equal(actual, [1.])	# 최소 비정규 크기에서도 H*d=g의 방향 1을 정확히 보존한다.
                np.testing.assert_array_equal(matrix, [[value]])

    def test_accepted_asymmetric_subnormal_pairs_have_safe_midpoints(self):
        """허용된 작은 비대칭 쌍을 각각 반으로 나누지 않고 평균하여 2s 또는 s를 보존하는지 확인한다."""
        smallest = np.nextafter(0., 1.)
        for matrix, gradient in (
            (np.array([[5., 1.], [3., 5.]]) * smallest, [7 * smallest, 7 * smallest]),
            (np.array([[5., 3.], [-1., 5.]]) * smallest, [6 * smallest, 6 * smallest]),
        ):
            with self.subTest(matrix=matrix):
                actual = NewtonMethod(lambda point: matrix).step([2., 2.], gradient)
                np.testing.assert_allclose(actual, [1., 1.], atol=1e-14, rtol=0)	# 작은 비대칭 쌍의 안전한 평균으로 얻은 독립 방향과 비교한다.

    def test_large_finite_symmetric_hessian_does_not_overflow_on_averaging(self):
        """H=1e308*I, g=(1e308,-1e308)의 정확한 유한 방향 (1,-1)을 확인한다."""
        matrix = np.diag([1e308, 1e308])
        actual = NewtonMethod(lambda point: matrix).step([2., -2.], [1e308, -1e308])
        np.testing.assert_allclose(actual, [1., -1.], atol=1e-14, rtol=0)	# 큰 유한 Hessian의 평균 계산이 오버플로하지 않았는지 확인한다.
        np.testing.assert_array_equal(matrix, np.diag([1e308, 1e308]))

    def test_nonfinite_outputs_and_unrepresentable_updates_raise(self):
        """비유한 콜백, 방향의 오버플로와 후보점의 오버플로를 성공으로 처리하지 않는지 확인한다."""
        for matrix in (np.diag([np.inf, 1.]), np.diag([np.nan, 1.])):
            with self.subTest(matrix=matrix), self.assertRaises(FloatingPointError):
                NewtonMethod(lambda point: matrix).step([1., 1.], [2., 2.])
        with self.assertRaises(FloatingPointError):
            NewtonMethod(lambda point: [[1e-308]]).step([0.], [1e308])
        with self.assertRaises(FloatingPointError):
            NewtonMethod(lambda point: [[1.]]).step([1e308], [-1e308])

    def test_runner_rejects_nonfinite_newton_direction(self):
        """공통 실행기가 비유한 후보나 거부된 행을 경로에 기록하지 않는지 확인한다."""
        result = run_optimization(lambda point: 0., lambda point: np.array([1e308]),
                                  [0.], NewtonMethod(lambda point: [[1e-308]]), steps=3)
        assert result.status == "nonfinite" and result.iterations == 0	# 표현 불가능한 방향을 수락된 갱신으로 기록하지 않는다.
        np.testing.assert_array_equal(result.path, [[0.]])
        assert result.losses.shape == result.gradient_norms.shape == (1,)

    def test_callback_errors_propagate(self):
        """사용자 콜백 오류를 곡률 판정이나 수렴 성공으로 바꾸지 않고 전달하는지 확인한다."""
        def hessian(point):
            """예외 전달을 검증하기 위해 의도한 응용 함수 오류를 발생시킨다."""
            raise RuntimeError("user Hessian failure")

        with self.assertRaisesRegex(RuntimeError, "user Hessian failure"):
            NewtonMethod(hessian).step([1.], [2.])


class HessianTests(unittest.TestCase):
    """정확한 Hessian 성분을 기울기의 독립 수치미분과 비교한다."""

    def test_hessians_match_gradient_central_differences(self):
        """공통 스칼라 중심차분 도구로 기울기의 각 좌표를 미분하여 Hessian과 비교한다."""
        point = np.array([1.25, -2.5])
        for hessian, gradient, diagonal in ((circle_hessian, quadratic_gradient, [2., 2.]),
                                            (elliptic_hessian, elliptic_gradient, [2., 20.])):
            with self.subTest(hessian=hessian.__name__):
                numerical = np.empty((2, 2))
                for i in range(2):
                    for j in range(2):
                        def coordinate_gradient(value):
                            """한 좌표만 변화시켜 기울기의 한 성분을 계산한다."""
                            shifted = point.copy()
                            shifted[j] = value
                            return gradient(shifted)[i]

                        numerical[i, j] = central_difference(coordinate_gradient, point[j], h=1e-5)
                actual = hessian(point)
                np.testing.assert_array_equal(actual, np.diag(diagonal))	# 미션별 정확한 대각 Hessian을 기준으로 비교한다.
                np.testing.assert_allclose(actual, numerical, atol=1e-8, rtol=0)	# 기울기 성분의 중심차분과 절대오차 1e-8로 독립 대조한다.
                assert actual.shape == (2, 2) and actual.dtype == np.float64
                actual[:] = 0
                np.testing.assert_array_equal(hessian(point), np.diag(diagonal))

    def test_hessians_validate_points(self):
        """미션 Hessian이 상수여도 유한한 실수 (2,) 입력점을 요구하는지 확인한다."""
        for function in (circle_hessian, elliptic_hessian):
            for point in ([1.], [[1., 2.]], [np.nan, 0.], [np.inf, 0.]):
                with self.subTest(function=function.__name__, point=point), self.assertRaises(ValueError):
                    function(point)
            with self.assertRaises(TypeError):
                function([True, False])
