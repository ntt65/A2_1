"""미적분 API를 독립 해석식과 입력·출력 계약으로 검증한다."""

import unittest

import numpy as np

from src.calculus import (
    numerical_derivative,
    numerical_gradient,
    quadratic,
    quadratic_gradient,
)

__covers__ = ["src/calculus.py"]


class DerivativeTests(unittest.TestCase):
    """미션 허용오차, 절단오차의 변화와 스칼라 입력·출력 조건을 확인한다."""

    def test_required_quadratic_derivative(self):
        """x=3에서 x 제곱의 도함수가 6과 절대오차 1e-4 이내로 일치하는지 확인한다."""
        result = numerical_derivative(lambda x: x**2, 3.0, 1e-5)
        assert isinstance(result, float)
        assert abs(result - 6.0) <= 1e-4	# 미션의 독립 해석값 6과 절대오차 1e-4로 비교한다.

    def test_independent_analytic_references(self):
        """상수·일차·삼차 함수의 도함수를 독립 공식과 비교한다."""
        assert numerical_derivative(lambda x: 4.0, -3.0) == 0.0
        assert abs(numerical_derivative(lambda x: -7.0 * x + 9.0, 2.0) + 7.0) <= 1e-9
        assert abs(numerical_derivative(lambda x: x**3, -2.0) - 12.0) <= 1e-8	# 삼차 함수의 x=-2 도함수 12를 기준으로 비교한다.

    def test_sine_truncation_error(self):
        """매끄러운 비이차 함수에서 기대한 2차 절단오차가 나타나는지 확인한다."""
        exact = np.cos(3.0)
        coarse = abs(numerical_derivative(np.sin, 3.0, 0.1) - exact)
        fine = abs(numerical_derivative(np.sin, 3.0, 0.01) - exact)
        assert coarse > fine > 0
        assert 99.0 <= coarse / fine <= 101.0	# h를 10배 줄이면 2차 절단오차가 약 100배 줄어드는지 확인한다.

    def test_scalar_contract_and_invalid_steps(self):
        """잘못된 h, 구분되지 않는 표본점과 스칼라가 아닌 값을 공통 검사로 거부하는지 확인한다."""
        for step in (0.0, -1.0, np.inf, np.nan, 1e-300, [1e-5]):
            with self.subTest(step=step), self.assertRaises(ValueError):
                numerical_derivative(lambda x: x**2, 3.0, step)
        for value in ([2.0], np.nan, np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                numerical_derivative(lambda x: x, value)
        with self.assertRaises(ValueError):
            numerical_derivative(lambda x: [x], 3.0)
        with self.assertRaises(TypeError):
            numerical_derivative(lambda x: x, 3.0, True)
        with self.assertRaises(TypeError):
            numerical_derivative(None, 3.0)


class GradientTests(unittest.TestCase):
    """기울기의 좌표 순서, 표본 격리, 실패 처리와 기하학적 의미를 검증한다."""

    def test_asymmetric_three_variable_reference(self):
        """변수가 결합된 다항식으로 좌표 뒤바뀜과 다른 좌표로의 영향 누출을 검출한다."""
        point = np.array([2.0, -3.0, 0.5])
        actual = numerical_gradient(
            lambda p: p[0] ** 2 + 3 * p[0] * p[1] + 5 * p[1] ** 2 - 7 * p[2],
            point,
        )
        np.testing.assert_allclose(actual, [-5.0, -24.0, -7.0], atol=1e-8, rtol=0)	# 결합 다항식의 세 편미분을 손계산한 값과 비교한다.
        np.testing.assert_array_equal(point, [2.0, -3.0, 0.5])
        assert actual.shape == (3,)
        assert actual.dtype == np.float64

    def test_one_coordinate_and_constant(self):
        """1차원 입력의 축이 유지되고 상수 함수의 기울기가 0인지 확인한다."""
        result = numerical_gradient(lambda p: p[0] ** 2, [3])
        assert result.shape == (1,)
        np.testing.assert_allclose(result, [6.0], atol=1e-4, rtol=0)
        np.testing.assert_array_equal(numerical_gradient(lambda p: 8.0, [1.0, 2.0]), [0, 0])

    def test_each_objective_sample_is_isolated(self):
        """목적함수가 표본을 바꾸더라도 원본 입력이나 후속 표본이 바뀌지 않는지 확인한다."""
        point = np.array([2.0, 3.0])
        point.setflags(write=False)
        samples = []

        def mutating_objective(sample):
            """전용 표본을 제자리에서 바꾸기 전에 x+4y 값을 계산한다."""
            samples.append(sample)
            value = sample[0] + 4.0 * sample[1]
            sample[:] = 999.0
            return value

        actual = numerical_gradient(mutating_objective, point)
        np.testing.assert_allclose(actual, [1.0, 4.0], atol=1e-9, rtol=0)
        np.testing.assert_array_equal(point, [2.0, 3.0])
        assert len(samples) == 4
        assert len({id(sample) for sample in samples}) == 4	# 네 평가가 서로 다른 표본 복사본을 받는지 확인한다.

    def test_point_and_output_contracts(self):
        """기울기 계산이 유한하고 비어 있지 않은 실수 벡터와 스칼라 목적함수 출력을 요구하는지 확인한다."""
        for point in (3.0, [], [[1.0, 2.0]], [np.inf], [np.nan]):
            with self.subTest(point=point), self.assertRaises(ValueError):
                numerical_gradient(lambda p: 0.0, point)
        for point in ([1j], [True], ["1"]):
            with self.subTest(point=point), self.assertRaises(TypeError):
                numerical_gradient(lambda p: 0.0, point)
        for result in ([0.0], np.inf, np.nan):
            with self.subTest(result=result), self.assertRaises(ValueError):
                numerical_gradient(lambda p: result, [1.0, 2.0])
        with self.assertRaises(TypeError):
            numerical_gradient(lambda p: 1j, [1.0])
        with self.assertRaises(TypeError):
            numerical_gradient(None, [1.0])

    def test_invalid_steps_and_user_exception(self):
        """모든 좌표에 공통 h 검증을 적용하고 사용자 예외를 숨기지 않는지 확인한다."""
        for step in (0.0, -1e-5, np.inf, np.nan, 1e-300, [1e-5]):
            with self.subTest(step=step), self.assertRaises(ValueError):
                numerical_gradient(lambda p: np.sum(p), [3.0, 4.0], step)
        with self.assertRaises(ValueError):
            numerical_gradient(lambda p: 0.0, [0.0, 1e20], 1e-5)

        def failing_objective(sample):
            """변형 없이 전달되어야 하는 사용자 함수 실패를 재현한다."""
            raise RuntimeError("objective failure")

        with self.assertRaisesRegex(RuntimeError, "objective failure"):
            numerical_gradient(failing_objective, [1.0, 2.0])

    def test_quadratic_numerical_and_analytic_gradients(self):
        """비대칭 점·축 위 점·원점에서 기준 기울기 (2x,2y)와 비교한다."""
        points = np.array([[1.5, -2.0], [-3.0, 0.25], [0.0, 2.0], [0.0, 0.0]])
        expected = np.array([[3.0, -4.0], [-6.0, 0.5], [0.0, 4.0], [0.0, 0.0]])
        analytic = quadratic_gradient(points)
        numeric = np.array([numerical_gradient(lambda p: quadratic(p[0], p[1]), p) for p in points])
        np.testing.assert_allclose(analytic, expected, atol=0, rtol=0)
        np.testing.assert_allclose(numeric, expected, atol=1e-8, rtol=0)	# 원점·축·비대칭 점을 포함한 독립 기울기와 비교한다.
        tangent = np.column_stack((-points[:, 1], points[:, 0]))
        np.testing.assert_allclose(np.sum(analytic * tangent, axis=1), 0.0, atol=1e-12, rtol=0)	# 기울기와 접선의 내적이 0인지 확인한다.
        np.testing.assert_array_equal(analytic[-1], [0.0, 0.0])


class QuadraticTests(unittest.TestCase):
    """스칼라·브로드캐스트·배치 형상과 유한 값의 한계를 확인한다."""

    def test_scalar_and_broadcast_surface(self):
        """3-4-5 삼각형과 비대칭 격자에서 독립적으로 아는 정확한 함수값을 확인한다."""
        scalar = quadratic(3, -4)
        assert scalar.shape == ()
        assert scalar.dtype == np.float64
        assert float(scalar) == 25.0
        x = np.array([[1.0], [-2.0]])
        y = np.array([0.0, 3.0, 4.0])
        np.testing.assert_array_equal(quadratic(x, y), [[1.0, 10.0, 17.0], [4.0, 13.0, 20.0]])	# 브로드캐스트 격자의 각 x²+y² 값을 직접 계산한 표와 비교한다.
        np.testing.assert_array_equal(x, [[1.0], [-2.0]])
        np.testing.assert_array_equal(y, [0.0, 3.0, 4.0])

    def test_gradient_batch_shape_and_input_preservation(self):
        """한 점과 고차원 점 배치가 마지막 좌표 축을 보존하는지 확인한다."""
        np.testing.assert_array_equal(quadratic_gradient([3.0, -4.0]), [6.0, -8.0])
        batch = np.array([[[1.0, 2.0]], [[-3.0, 4.0]]])
        original = batch.copy()
        actual = quadratic_gradient(batch)
        assert actual.shape == (2, 1, 2)
        assert actual.dtype == np.float64
        np.testing.assert_array_equal(actual, [[[2.0, 4.0]], [[-6.0, 8.0]]])	# 배치의 마지막 좌표 축을 유지한 기울기와 비교한다.
        actual[:] = 0
        np.testing.assert_array_equal(batch, original)

    def test_surface_and_gradient_invalid_inputs(self):
        """비유한 값·실수가 아닌 값·잘못된 형상과 표현 불가능한 출력을 거부하는지 확인한다."""
        for value in (np.inf, np.nan, 1e308):
            with self.subTest(value=value), self.assertRaises(ValueError):
                quadratic(value, 0.0)
        with self.assertRaises(ValueError):
            quadratic(np.ones(2), np.ones(3))
        for value in (1j, True, "1"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                quadratic(value, 0.0)
        for point in (1.0, [], [1.0], [1.0, 2.0, 3.0], np.empty((0, 2)), [np.inf, 0.0], [1e308, 0.0]):
            with self.subTest(point=point), self.assertRaises(ValueError):
                quadratic_gradient(point)
        for point in ([True, False], [1j, 2j], ["1", "2"]):
            with self.subTest(point=point), self.assertRaises(TypeError):
                quadratic_gradient(point)


if __name__ == "__main__":
    unittest.main()
