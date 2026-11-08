"""Adam의 독립 기대값과 최적화 실행기의 상태 복원 동작을 검증한다."""

import unittest

import numpy as np

from src.adam import Adam
from src.optimizer import Momentum, run_optimization

__covers__ = ["src/adam.py", "src/optimizer.py"]


class AdamStepTests(unittest.TestCase):
    """모멘트 편향 보정, 입력·상태 계약과 유한 범위의 산술 연산을 확인한다."""

    def test_first_two_steps_match_independent_hand_calculation(self):
        """beta1=beta2=.5일 때 m2=[1,0], v2=[1.5,6], 편향 보정 분모 3/4를 손계산과 비교한다."""
        optimizer = Adam(lr=0.3, beta1=0.5, beta2=0.5, epsilon=0.25)
        initial = np.array([1.0, -2.0])
        gradient = np.array([2.0, -4.0])
        first = optimizer.step(initial, gradient)
        expected_first = np.array([1.0 - 0.3*2.0/2.25, -2.0 + 0.3*4.0/4.25])	# 첫 갱신의 기대값을 편향 보정식으로 직접 계산한다.
        np.testing.assert_allclose(first, expected_first, atol=1e-15, rtol=0)	# 상대오차 없이 절대오차 1e-15로 첫 갱신을 비교한다.
        np.testing.assert_array_equal(optimizer.m, [1.0, -2.0])
        np.testing.assert_array_equal(optimizer.v, [2.0, 8.0])
        second = optimizer.step(first, [1.0, 2.0])
        # 손계산: m2 = .5*[1,-2]+.5*[1,2] = [1,0].
        # 손계산: v2 = .5*[2,8]+.5*[1,4] = [1.5,6].
        # 편향 보정 후 mhat2=[4/3,0], vhat2=[2,8], 제곱근은 [sqrt(2),sqrt(8)].
        expected_second = expected_first - np.array([0.3*(4.0/3.0)/(np.sqrt(2.0)+0.25), 0.0])
        np.testing.assert_allclose(second, expected_second, atol=1e-15, rtol=0)	# 서로 다른 두 번째 기울기의 손계산 기대값과 비교한다.
        np.testing.assert_array_equal(optimizer.m, [1.0, 0.0])
        np.testing.assert_array_equal(optimizer.v, [1.5, 6.0])
        assert optimizer.t == 2
        np.testing.assert_array_equal(initial, [1.0, -2.0])
        np.testing.assert_array_equal(gradient, [2.0, -4.0])

    def test_constant_gradient_bias_correction(self):
        """일정한 0이 아닌 g에 대해 매번 보정된 갱신량이 lr*g/(abs(g)+epsilon)인지 확인한다."""
        optimizer = Adam(lr=0.1, beta1=0.8, beta2=0.7, epsilon=0.2)
        point = np.array([2.0, -3.0])
        decrement = np.array([0.1*3.0/3.2, -0.1*4.0/4.2])
        for step in range(1, 11):
            point = optimizer.step(point, [3.0, -4.0])
            np.testing.assert_allclose(point, np.array([2.0, -3.0])-step*decrement, atol=1e-14, rtol=0)	# 일정한 기울기의 독립 누적 갱신식을 절대오차 1e-14로 확인한다.
        assert optimizer.t == 10

    def test_zero_betas_zero_gradient_and_epsilon_position(self):
        """beta=0이면 현재 기울기를 정규화하며 epsilon은 제곱근 밖에 더해지는지 검증한다."""
        optimizer = Adam(lr=0.2, beta1=0, beta2=0, epsilon=0.5)
        first = optimizer.step([1.0, 2.0], [0.0, 0.25])
        np.testing.assert_allclose(first, [1.0, 2.0-0.2/3.0], atol=1e-15, rtol=0)	# epsilon을 제곱근 밖에 더한 분모 3을 기준으로 비교한다.
        second = optimizer.step(first, [0.0, 0.0])
        np.testing.assert_array_equal(second, first)
        np.testing.assert_array_equal(optimizer.m, [0.0, 0.0])
        np.testing.assert_array_equal(optimizer.v, [0.0, 0.0])
        assert optimizer.t == 2

    def test_state_copy_shape_and_reset(self):
        """공개 상태의 복사본 격리, 형상 변경 전 reset 요구와 재시작의 재현성을 확인한다."""
        optimizer = Adam()
        assert optimizer.m is None and optimizer.v is None and optimizer.t == 0
        first = optimizer.step([1, 2], [3, 4])
        moment, variance = optimizer.m, optimizer.v
        moment[:] = variance[:] = 999
        np.testing.assert_allclose(optimizer.m, [0.3, 0.4], atol=1e-15, rtol=0)	# 외부에서 받은 상태 복사본을 바꿔도 내부 1차 모멘트는 유지된다.
        np.testing.assert_allclose(optimizer.v, [0.009, 0.016], atol=1e-15, rtol=0)
        with self.assertRaises(ValueError):
            optimizer.step([1], [2])
        assert optimizer.t == 1
        optimizer.reset()
        assert optimizer.m is None and optimizer.v is None and optimizer.t == 0	# reset 후 모멘트와 수락 횟수가 초기 상태인지 확인한다.
        np.testing.assert_array_equal(optimizer.step([1, 2], [3, 4]), first)
        optimizer.reset()
        assert optimizer.step([1], [1]).shape == (1,)

    def test_invalid_hyperparameters_and_inputs(self):
        """잘못된 스칼라, 비유한 값, 불리언과 벡터 형상 불일치를 거부하는지 확인한다."""
        for name in ("lr", "beta1", "beta2", "epsilon"):
            for value in (True, "0.1", 1+0j):
                with self.subTest(name=name, value=value), self.assertRaises(TypeError):
                    Adam(**{name: value})
            for value in (np.nan, np.inf, [0.1]):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    Adam(**{name: value})
        for name, value in (("lr", 0), ("lr", -1), ("epsilon", 0), ("epsilon", -1),
                            ("beta1", -0.1), ("beta1", 1), ("beta2", -0.1), ("beta2", 1)):
            with self.assertRaises(ValueError):
                Adam(**{name: value})
        optimizer = Adam()
        for point, gradient in (([], []), ([[1]], [[1]]), ([1, 2], [1]), ([np.inf], [1])):
            with self.assertRaises(ValueError):
                optimizer.step(point, gradient)
        with self.assertRaises(TypeError):
            optimizer.step([True], [1])
        assert optimizer.t == 0

    def test_overflow_preserves_every_state_component(self):
        """2차 모멘트나 갱신의 오버플로가 발생해도 m·v·t를 확정하지 않는지 확인한다."""
        optimizer = Adam(lr=0.1)
        optimizer.step([0.0], [1.0])
        moment, variance, count = optimizer.m, optimizer.v, optimizer.t
        with self.assertRaises(FloatingPointError):
            optimizer.step([0.0], [1e200])
        np.testing.assert_array_equal(optimizer.m, moment)
        np.testing.assert_array_equal(optimizer.v, variance)
        assert optimizer.t == count	# 실패한 갱신으로 수락 횟수가 증가하지 않았는지 확인한다.
        large = Adam(lr=1.7e308, beta1=0, beta2=0)
        with self.assertRaises(FloatingPointError):
            large.step([1.7e308], [-1.0])
        assert large.m is None and large.v is None and large.t == 0

    def test_moment_underflow_is_rejected_atomically(self):
        """기울기 제곱이 소실되어 정규화된 단위 갱신이 1e108로 바뀌는 반례를 거부한다."""
        optimizer = Adam(lr=1.0, beta1=0.0, beta2=0.0, epsilon=1e-308)
        with self.assertRaisesRegex(FloatingPointError, "second moment.*underflow"):
            optimizer.step([0.0], [1e-200])
        assert optimizer.m is None and optimizer.v is None and optimizer.t == 0
        optimizer.step([0.0], [1.0])
        moment, variance, count = optimizer.m, optimizer.v, optimizer.t
        with self.assertRaisesRegex(FloatingPointError, "second moment.*underflow"):
            optimizer.step([0.0], [1e-200])
        np.testing.assert_array_equal(optimizer.m, moment)
        np.testing.assert_array_equal(optimizer.v, variance)
        assert optimizer.t == count	# 언더플로 실패 후에도 마지막 수락 횟수를 보존한다.

    def test_weighted_and_inexact_subnormal_moments_are_rejected(self):
        """정확한 비정규 제곱은 허용하되 가중항·1차 모멘트의 소실 또는 부정확한 비정규 결과는 거부한다."""
        # g^2 = 2^-1060은 정확히 표현되는 비정규 값이며 1e-400의 소실과 구별한다.
        derivative = np.ldexp(1.0, -530)
        exact = Adam(lr=1.0, beta1=0.0, beta2=0.0, epsilon=1e-308)
        np.testing.assert_array_equal(exact.step([0.0], [derivative]), [-1.0])	# 정확히 표현되는 비정규 제곱은 단위 갱신을 유지한다.
        for beta in (np.nextafter(1.0, 0.0), 0.1):
            optimizer = Adam(lr=1.0, beta1=0.0, beta2=beta, epsilon=1e-308)
            # 첫 가중치는 0을 만들고 두 번째는 0이 아니지만 부정확한 비정규 값을 만든다.
            with self.subTest(beta=beta), self.assertRaisesRegex(FloatingPointError, "second moment.*underflow"):
                optimizer.step([0.0], [derivative])
            assert optimizer.t == 0 and optimizer.m is None and optimizer.v is None
        optimizer = Adam(beta1=0.5, beta2=0.0)
        with self.assertRaisesRegex(FloatingPointError, "first moment.*underflow"):
            optimizer.step([0.0], [np.nextafter(0.0, 1.0)])
        assert optimizer.t == 0 and optimizer.m is None and optimizer.v is None	# 가중항의 정밀도 소실은 어떤 Adam 상태도 확정하지 않는다.


class AdamRunnerTests(unittest.TestCase):
    """후보점 평가가 실패하면 모든 상태를 마지막으로 수락한 점의 상태로 유지하는지 검증한다."""

    def test_runner_resets_all_state_and_reproduces(self):
        """Adam을 재사용하면 모멘트와 t가 0에서 시작하고 경로·횟수가 수락한 갱신과 맞는지 확인한다."""
        optimizer = Adam(lr=0.1)
        first = run_optimization(lambda p: float(p@p), lambda p: 2*p, [2.0, -1.0], optimizer, steps=5)
        assert optimizer.t == first.iterations == 5
        moments = optimizer.m, optimizer.v
        second = run_optimization(lambda p: float(p@p), lambda p: 2*p, [2.0, -1.0], optimizer, steps=5)
        np.testing.assert_array_equal(first.path, second.path)	# 같은 최적화기를 다시 실행해도 전체 경로가 정확히 같다.
        np.testing.assert_array_equal(optimizer.m, moments[0])
        np.testing.assert_array_equal(optimizer.v, moments[1])

    def test_runner_stops_on_moment_underflow_without_committing(self):
        """모멘트 정밀도 소실을 거대한 갱신 대신 공통 비유한 종료 경로로 처리하는지 확인한다."""
        optimizer = Adam(lr=1.0, beta1=0.0, beta2=0.0, epsilon=1e-308)
        result = run_optimization(lambda p: 0.0, lambda p: [1e-200], [0.0], optimizer, steps=2)
        assert result.status == "nonfinite" and result.iterations == 0	# 모멘트 소실은 갱신을 수락하지 않은 비유한 종료로 기록한다.
        np.testing.assert_array_equal(result.path, [[0.0]])
        assert optimizer.t == 0 and optimizer.m is None and optimizer.v is None

    def test_nonfinite_callback_restores_last_accepted_adam_state(self):
        """목적함수가 두 번째 후보를 거부하면 첫 번째 모멘트와 횟수를 보존하는지 확인한다."""
        optimizer = Adam(lr=1.0, beta1=0.5, beta2=0.5, epsilon=0.25)
        result = run_optimization(lambda p: 0.0 if p[0] < 1.0 else np.inf,
                                  lambda p: [-1.0], [0.0], optimizer, steps=5)
        assert result.status == "nonfinite" and result.iterations == optimizer.t == 1
        np.testing.assert_allclose(result.path, [[0.0], [0.8]], atol=1e-15, rtol=0)	# 두 번째 후보 거부 후 초기점과 첫 수락점만 경로에 남긴다.
        np.testing.assert_array_equal(optimizer.m, [-0.5])
        np.testing.assert_array_equal(optimizer.v, [0.5])
        rejected = run_optimization(lambda p: 0.0 if p[0] == 0 else np.inf,
                                    lambda p: [-1.0], [0.0], optimizer, steps=2)
        assert rejected.iterations == optimizer.t == 0
        assert optimizer.m is None and optimizer.v is None

    def test_contract_errors_restore_state_then_propagate(self):
        """잘못된 기울기와 콜백 예외를 비유한 수렴으로 오인하지 않고 상태 복원 후 전달하는지 확인한다."""
        for optimizer in (Adam(lr=1.0, beta1=0.5, beta2=0.5, epsilon=0.25), Momentum(lr=0.8)):
            with self.assertRaisesRegex(ValueError, "shape"):
                run_optimization(lambda p: 0.0, lambda p: [-1.0] if p[0] < 1.0 else [-1.0, -1.0],
                                 [0.0], optimizer, steps=5)
            if isinstance(optimizer, Adam):
                assert optimizer.t == 1
                np.testing.assert_array_equal(optimizer.m, [-0.5])	# 잘못된 후속 콜백에도 마지막 수락 1차 모멘트를 복원한다.
                np.testing.assert_array_equal(optimizer.v, [0.5])
            else:
                np.testing.assert_array_equal(optimizer.velocity, [-1.0])
        optimizer = Adam()

        def broken_objective(point):
            """Adam 상태가 바뀐 후보점에서만 의도한 오류를 발생시킨다."""
            if point[0] != 0:
                raise RuntimeError("callback sentinel")
            return 0.0

        with self.assertRaisesRegex(RuntimeError, "callback sentinel"):
            run_optimization(broken_objective, lambda p: [1.0], [0.0], optimizer, steps=2)
        assert optimizer.t == 0 and optimizer.m is None and optimizer.v is None	# 사용자 예외가 전파된 경우에도 새 상태를 남기지 않는다.


if __name__ == "__main__":
    unittest.main()
