"""공통 수학 유틸리티의 수치 결과와 경계 조건을 독립 기준으로 검증한다."""

import importlib
import unittest

import numpy as np

from src.utils.math import sigmoid, softmax
from src.utils.numerical import central_difference
from src.utils.random import set_seed

__covers__ = ["src/utils/math.py", "src/utils/numerical.py", "src/utils/random.py"]


class ActivationTests(unittest.TestCase):
    """알려진 확률, 형상, 수치 안정성과 잘못된 입력을 확인한다."""

    def test_sigmoid_known_probabilities_and_shape(self):
        """오즈 1/3·1·3의 로그를 넣었을 때 각각 0.25·0.5·0.75인지 확인한다."""
        values = np.array([[-np.log(3.0), 0.0, np.log(3.0)]])
        original = values.copy()
        actual = sigmoid(values)
        np.testing.assert_allclose(actual, [[0.25, 0.5, 0.75]], atol=1e-15, rtol=0)	# 알려진 오즈로부터 얻는 독립 확률과 절대오차 1e-15로 비교한다.
        np.testing.assert_array_equal(values, original)
        assert actual.shape == values.shape
        assert actual.dtype == np.float64
        assert sigmoid(0.0).shape == ()
        assert float(sigmoid(0.0)) == 0.5
        assert sigmoid(np.empty((0, 2))).shape == (0, 2)

    def test_sigmoid_extremes_without_overflow(self):
        """양·음의 float64 극값에서도 오버플로 없이 유한한 경계 확률이 되는지 확인한다."""
        with np.errstate(all="raise"):
            actual = sigmoid([-1e308, -1000.0, 0.0, 1000.0, 1e308])
        np.testing.assert_array_equal(actual, [0.0, 0.0, 0.5, 1.0, 1.0])	# 양·음 극값의 포화 결과와 중앙값 0.5를 정확히 확인한다.
        assert np.all(np.isfinite(actual))

    def test_softmax_known_probabilities(self):
        """가중치 1·2·3의 로그를 독립적인 유리수 확률로 정규화하는지 확인한다."""
        values = np.log([1.0, 2.0, 3.0])
        actual = softmax(values)
        np.testing.assert_allclose(actual, [1 / 6, 1 / 3, 1 / 2], atol=1e-15, rtol=0)	# 합이 6인 독립 가중치의 유리수 확률을 기준으로 비교한다.
        assert abs(float(actual.sum()) - 1.0) <= 1e-6
        assert actual.shape == (3,)
        assert actual.dtype == np.float64

    def test_softmax_axes_and_shift_invariance(self):
        """선택한 축의 각 구간이 정규화되고 공통 상수를 더해도 확률이 유지되는지 확인한다."""
        values = np.array([[0.0, 0.0, 0.0], [np.log(3.0)] * 3])
        columns = softmax(values, axis=0)
        np.testing.assert_allclose(columns, [[0.25] * 3, [0.75] * 3], atol=1e-15, rtol=0)
        rows = softmax(values, axis=-1)
        np.testing.assert_allclose(rows, np.full((2, 3), 1 / 3), atol=1e-15, rtol=0)
        cube = np.stack((values, values + 1000.0))
        actual = softmax(cube, axis=1)
        assert actual.shape == cube.shape
        assert np.max(np.abs(actual.sum(axis=1) - 1.0)) <= 1e-6	# 지정한 축의 확률 합에 미션 허용오차 1e-6을 적용한다.
        np.testing.assert_allclose(actual[0], actual[1], atol=1e-13, rtol=0)	# 공통 로짓 이동에 따른 확률 불변성을 확인한다.
        np.testing.assert_array_equal(softmax(np.ones((2, 1)), axis=1), np.ones((2, 1)))

    def test_softmax_extreme_logits(self):
        """차이 계산이 오버플로할 수 있는 유한 극값 로짓에서도 유한 분포를 반환하는지 확인한다."""
        with np.errstate(all="raise"):
            actual = softmax([[1e308, -1e308], [-1e308, -1e308]])
        np.testing.assert_array_equal(actual, [[1.0, 0.0], [0.5, 0.5]])
        assert np.all(np.isfinite(actual))
        assert np.all(actual >= 0)
        assert np.max(np.abs(actual.sum(axis=1) - 1.0)) <= 1e-6	# 극값 입력에서도 유한한 정규화 조건을 유지한다.

    def test_activation_input_rejection(self):
        """비유한 값·복소수·불리언·문자열·객체 입력을 거부하는지 확인한다."""
        for function in (sigmoid, softmax):
            for value in ([np.inf], [np.nan], [-np.inf]):
                with self.subTest(function=function.__name__, value=value):
                    with self.assertRaises(ValueError):
                        function(value)
            for value in ([1j], [True], ["1"], np.array([1], dtype=object)):
                with self.subTest(function=function.__name__, value=value):
                    with self.assertRaises(TypeError):
                        function(value)

    def test_softmax_axis_and_empty_rejection(self):
        """분포를 정의할 수 없는 스칼라·빈 배열과 잘못된 축을 명시적으로 거부하는지 확인한다."""
        for values in (0.0, [], np.empty((2, 0)), np.empty((0, 3))):
            with self.subTest(shape=np.shape(values)):
                with self.assertRaises(ValueError):
                    softmax(values)
        for axis in (2, -3):
            with self.assertRaises(ValueError):
                softmax(np.ones((2, 3)), axis=axis)
        for axis in (None, 0.0, True, "0"):
            with self.assertRaises(TypeError):
                softmax([1.0, 2.0], axis=axis)


class NumericalTests(unittest.TestCase):
    """중심차분을 해석 도함수와 비교하고 잘못된 입력·출력 조건을 거부하는지 확인한다."""

    def test_quadratic_required_accuracy(self):
        """미션의 x=3인 x 제곱 도함수가 6과 절대오차 1e-4 이내로 일치하는지 확인한다."""
        actual = central_difference(lambda x: x**2, 3.0)
        assert isinstance(actual, float)
        assert abs(actual - 6.0) <= 1e-4	# 중심차분의 미션 기준은 절대오차 1e-4이며 이를 바꾸지 않는다.

    def test_other_analytical_derivatives(self):
        """상수·일차·사인 함수의 독립 해석 도함수와 비교한다."""
        assert central_difference(lambda x: 7.0, 1.0) == 0.0
        assert abs(central_difference(lambda x: 3.0 * x - 4.0, -2.0) - 3.0) <= 1e-9
        assert abs(central_difference(np.sin, 0.0) - 1.0) <= 1e-9	# 사인 함수의 원점 도함수 1과 비교한다.

    def test_invalid_steps_and_sampling_points(self):
        """간격이 양수·유한해야 하며 두 표본점이 수치적으로 구분되어야 하는지 확인한다."""
        for step in (0.0, -1e-5, np.nan, np.inf, 1e-300):
            with self.subTest(step=step):
                with self.assertRaises(ValueError):
                    central_difference(lambda x: x**2, 3.0, h=step)
        with self.assertRaises(ValueError):
            central_difference(lambda x: x, 1e308, h=1e308)

    def test_scalar_function_contract(self):
        """스칼라가 아니거나 비유한 표본·출력을 묵시적으로 브로드캐스트하지 않고 거부하는지 확인한다."""
        for point in ([3.0], np.inf, np.nan):
            with self.subTest(point=point):
                with self.assertRaises(ValueError):
                    central_difference(lambda x: x, point)
        for result in ([1.0], np.nan, np.inf):
            with self.subTest(result=result):
                with self.assertRaises(ValueError):
                    central_difference(lambda x: result, 3.0)
        with self.assertRaises(TypeError):
            central_difference(lambda x: 1j, 3.0)
        with self.assertRaises(TypeError):
            central_difference(None, 3.0)
        with self.assertRaises(ValueError):
            central_difference(lambda x: np.sign(x) * 1e308, 0.0)


class SeedTests(unittest.TestCase):
    """명시적 시드 설정을 검증하고 유틸리티를 가져오기만 해서는 시드가 초기화되지 않는지 확인한다."""

    def test_seed_reproduces_known_numpy_sequence(self):
        """필수 시드가 알려진 RandomState의 첫 세 값을 재현하는지 확인한다."""
        previous = np.random.get_state()
        try:
            set_seed()
            actual = np.random.random(3)
            expected = [0.3745401188473625, 0.9507143064099162, 0.7319939418114051]	# 시드 42의 알려진 첫 세 난수를 독립 상수로 고정한다.
            np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)
            set_seed(42)
            np.testing.assert_array_equal(np.random.random(3), actual)
        finally:
            np.random.set_state(previous)

    def test_import_has_no_rng_side_effect(self):
        """유틸리티를 다시 가져와도 전역 난수를 소비하거나 초기화하지 않는지 확인한다."""
        previous = np.random.get_state()
        try:
            set_seed(42)
            reference = np.random.random(4)
            set_seed(42)
            first = np.random.random(2)
            importlib.reload(importlib.import_module("src.utils.random"))
            second = np.random.random(2)
            np.testing.assert_array_equal(np.concatenate((first, second)), reference)	# 가져오기 전후 난수열을 이어 붙여 원래 수열과 정확히 비교한다.
        finally:
            np.random.set_state(previous)

    def test_invalid_seed(self):
        """NumPy가 지원하는 32비트 시드 범위의 정수만 허용하는지 확인한다."""
        for seed in (-1, 2**32):
            with self.assertRaises(ValueError):
                set_seed(seed)
        for seed in (1.0, True, None, "42"):
            with self.assertRaises(TypeError):
                set_seed(seed)


if __name__ == "__main__":
    unittest.main()
