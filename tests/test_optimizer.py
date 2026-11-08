"""최적화 점화식·입력 계약·수치적 종료를 독립 기대값으로 검증한다."""

import unittest

import numpy as np

from src.calculus import quadratic, quadratic_gradient
from src.optimizer import Momentum, VanillaGD, elliptic, elliptic_gradient, run_optimization

__covers__ = ["src/optimizer.py"]


def circle(point):
    """두 좌표 벡터에서 공통 원형 이차함수 값을 계산한다."""
    return quadratic(point[0], point[1])


class OptimizerStepTests(unittest.TestCase):
    """경로 실행기와 독립적으로 개별 최적화 갱신을 검증한다."""

    def test_gd_step_preserves_inputs(self):
        """비대칭 벡터로 부호·학습률·입력 소유권이 올바른지 확인한다."""
        theta, gradient = np.array([1.0, 2.0]), np.array([3.0, -4.0])
        result = VanillaGD(0.1).step(theta, gradient)
        np.testing.assert_allclose(result, [0.7, 2.4], atol=1e-15, rtol=0)	# 비대칭 입력의 손계산 GD 갱신값과 비교한다.
        result[:] = 0
        np.testing.assert_array_equal(theta, [1.0, 2.0])
        np.testing.assert_array_equal(gradient, [3.0, -4.0])

    def test_momentum_first_two_updates_and_state_copy(self):
        """명시적인 v1=(3,-4), v2=(1.7,-1.6)으로 선택한 모멘텀 정의와 상태 복사를 확인한다."""
        optimizer = Momentum(0.1)
        assert optimizer.velocity is None
        first = optimizer.step([1, 2], [3, -4])
        second = optimizer.step(first, [-1, 2])
        np.testing.assert_allclose(first, [0.7, 2.4], atol=1e-15, rtol=0)
        np.testing.assert_allclose(second, [0.53, 2.56], atol=1e-15, rtol=0)	# 두 번째 속도 (1.7,-1.6)으로 계산한 좌표와 비교한다.
        state = optimizer.velocity
        state[:] = 999
        np.testing.assert_allclose(optimizer.velocity, [1.7, -1.6], atol=1e-15, rtol=0)	# 외부 상태 복사본의 변경이 내부 속도에 전파되지 않는다.
        optimizer.reset()
        assert optimizer.velocity is None
        np.testing.assert_allclose(optimizer.step([1, 2], [3, -4]), first, atol=0, rtol=0)

    def test_zero_beta_matches_gd(self):
        """beta=0이면 속도 기억이 사라져 같은 입력의 GD 갱신과 정확히 일치하는지 확인한다."""
        gd, momentum = VanillaGD(0.2), Momentum(0.2, beta=0)
        theta = np.array([1.0, -3.0])
        for derivative in ([4.0, 1.0], [-2.0, 5.0]):
            expected = gd.step(theta, derivative)
            np.testing.assert_array_equal(momentum.step(theta, derivative), expected)	# beta가 0인 Momentum이 GD와 정확히 같은지 확인한다.
            theta = expected

    def test_hyperparameter_and_shape_contracts(self):
        """학습률·beta·벡터 차원과 속도 형상을 검증하는지 확인한다."""
        for lr in (0, -1, np.inf, np.nan, [0.1]):
            with self.subTest(lr=lr), self.assertRaises(ValueError):
                VanillaGD(lr)
        with self.assertRaises(TypeError): VanillaGD(True)
        for beta in (-0.1, 1, np.inf, np.nan, [0.9]):
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                Momentum(0.1, beta)
        for theta, derivative in (([], []), ([[1]], [[2]]), ([1, 2], [1]), ([np.nan], [1])):
            with self.subTest(theta=theta), self.assertRaises(ValueError):
                VanillaGD(0.1).step(theta, derivative)
        with self.assertRaises(TypeError): VanillaGD(0.1).step([True], [1])
        momentum = Momentum(0.1)
        momentum.step([1, 2], [1, 2])
        with self.assertRaises(ValueError): momentum.step([1], [1])

    def test_overflow_does_not_commit_momentum_velocity(self):
        """속도 또는 갱신 계산이 실패하면 마지막 유효 상태를 보존하는지 확인한다."""
        momentum = Momentum(2.0)
        momentum.step([1.0], [1.0])
        before = momentum.velocity
        with self.assertRaises(FloatingPointError): momentum.step([1.0], [1e308])
        np.testing.assert_array_equal(momentum.velocity, before)	# 오버플로 후에도 이전 속도를 보존한다.
        with self.assertRaises(FloatingPointError): VanillaGD(2.0).step([0.0], [1e308])


class TrajectoryTests(unittest.TestCase):
    """닫힌형 경로, 동일한 갱신 예산과 유한한 종료 기록을 검증한다."""

    def test_circle_closed_form_and_required_accuracy(self):
        """lr=.1의 GD가 정확히 100회 동안 theta_k=5*.8**k를 따르는지 확인한다."""
        initial = np.array([5.0, 5.0])
        result = run_optimization(circle, quadratic_gradient, initial, VanillaGD(0.1))
        expected = np.repeat((5.0*0.8**np.arange(101))[:, None], 2, axis=1)	# 원형 GD의 독립 점화식 해 5*.8**k를 구성한다.
        assert result.path.shape == (101, 2)
        assert result.losses.shape == result.gradient_norms.shape == (101,)
        assert result.iterations == 100 and result.status == "budget_exhausted"
        np.testing.assert_allclose(result.path, expected, atol=1e-14, rtol=1e-14)	# 101개 점의 전체 경로를 명시한 절대·상대오차로 비교한다.
        np.testing.assert_allclose(result.losses, 50.0*0.8**(2*np.arange(101)), atol=1e-13, rtol=1e-13)
        assert np.linalg.norm(result.path[-1]) <= 0.1
        np.testing.assert_array_equal(initial, [5.0, 5.0])

    def test_circle_immediate_oscillation_and_divergence(self):
        """lr=.5는 원점 도달, 1은 부호 교대, 1.1은 배율 -1.2로 증가하는지 확인한다."""
        for rate, factor in ((0.5, 0.0), (1.0, -1.0), (1.1, -1.2)):
            result = run_optimization(circle, quadratic_gradient, [5, 5], VanillaGD(rate), steps=8)
            expected = 5*factor**np.arange(9)
            np.testing.assert_allclose(result.path[:, 0], expected, atol=1e-12, rtol=1e-14)	# 수렴·진동·발산을 좌표 배율의 독립 거듭제곱과 비교한다.
            assert result.iterations == 8

    def test_momentum_runner_resets_between_runs(self):
        """최적화기를 재사용해도 초기 속도가 0인 경로가 재현되는지 확인한다."""
        optimizer = Momentum(0.1)
        first = run_optimization(circle, quadratic_gradient, [5, 5], optimizer, steps=2)
        second = run_optimization(circle, quadratic_gradient, [5, 5], optimizer, steps=2)
        np.testing.assert_allclose(first.path, [[5, 5], [4, 4], [2.3, 2.3]], atol=1e-15, rtol=0)
        np.testing.assert_array_equal(first.path, second.path)	# reset을 거친 재실행은 같은 전체 경로를 반환한다.

    def test_elliptic_gd_reference_and_equal_budgets(self):
        """lr=.05에서 y는 한 번에 0이 되고 x는 .9배로 줄어들며 지그재그를 강제하지 않는지 확인한다."""
        gd = run_optimization(elliptic, elliptic_gradient, [5, 5], VanillaGD(0.05))
        momentum = run_optimization(elliptic, elliptic_gradient, [5, 5], Momentum(0.05))
        np.testing.assert_allclose(gd.path[:, 0], 5*.9**np.arange(101), atol=1e-14, rtol=1e-14)	# 타원형 x 좌표의 오차 배율 .9를 독립적으로 확인한다.
        np.testing.assert_array_equal(gd.path[1:, 1], np.zeros(100))	# 학습률 .05에서 y 좌표는 첫 갱신 이후 정확히 0이다.
        assert gd.iterations == momentum.iterations == 100	# 방법 간 비교에 같은 100회 갱신 예산을 사용했는지 확인한다.
        assert momentum.path.shape == gd.path.shape == (101, 2)
        assert np.all(np.isfinite(momentum.losses))

    def test_tolerance_and_zero_budget(self):
        """선택적인 조기 종료가 명시적으로 적용되고 초기점을 포함하는지 확인한다."""
        stopped = run_optimization(circle, quadratic_gradient, [5, 5], VanillaGD(0.5), gradient_tolerance=0)
        assert stopped.status == "gradient_tolerance" and stopped.iterations == 1
        initial = run_optimization(circle, quadratic_gradient, [0, 0], Momentum(.1), gradient_tolerance=0)
        assert initial.iterations == 0 and initial.path.shape == (1, 2)	# 처음부터 종료 조건을 만족하면 초기점 한 행만 남긴다.
        zero = run_optimization(circle, quadratic_gradient, [5, 5], VanillaGD(.1), steps=0)
        assert zero.status == "budget_exhausted" and zero.iterations == 0

    def test_nonfinite_objective_and_gradient_keep_last_finite_rows(self):
        """후속 평가가 거부되어도 길이를 맞춘 이력에 NaN·Inf가 들어가지 않는지 확인한다."""
        for kind in ("objective", "gradient"):
            objective = (lambda p: np.inf if p[0] > 1 else p[0]) if kind == "objective" else (lambda p: p[0])
            gradient = (lambda p: [np.nan] if p[0] > 1 else [-1.0]) if kind == "gradient" else (lambda p: [-1.0])
            result = run_optimization(objective, gradient, [0.0], VanillaGD(1.0), steps=10)
            assert result.status == "nonfinite" and result.iterations == 1
            np.testing.assert_array_equal(result.path, [[0.0], [1.0]])	# 거부된 비유한 후보를 빼고 마지막 유한점까지 기록한다.
            assert len(result.losses) == len(result.gradient_norms) == len(result.path)
            assert np.all(np.isfinite(result.losses))

    def test_rejected_momentum_candidate_restores_accepted_state(self):
        """목적함수가 거부한 후보에서 계산된 유효 속도도 마지막 수락 상태로 되돌리는지 확인한다."""
        optimizer = Momentum(1.0)
        result = run_optimization(lambda p: np.inf if p[0] > 1 else p[0], lambda p: [-1.0], [0.0], optimizer)
        assert result.status == "nonfinite" and result.iterations == 1
        np.testing.assert_array_equal(optimizer.velocity, [-1.0])	# 후보 거부 시 속도도 마지막 수락값으로 되돌린다.

    def test_step_overflow_and_stable_small_norm(self):
        """갱신 오버플로는 초기 행에서 멈추고 매우 작은 기울기는 0이 아닌 노름을 유지하는지 확인한다."""
        result = run_optimization(lambda p: 0.0, lambda p: [1e308], [0.0], VanillaGD(2.0))
        assert result.status == "nonfinite" and result.iterations == 0
        tiny = run_optimization(lambda p: 0.0, lambda p: [1e-300, 1e-300], [0.0, 0.0], VanillaGD(.1), steps=0)
        assert 1.4e-300 < tiny.gradient_norms[0] < 1.5e-300	# 작은 기울기의 노름이 언더플로로 0이 되지 않는지 확인한다.

    def test_input_and_callback_contracts(self):
        """초기 지표 오류는 거부하며 잘못된 콜백과 사용자 예외는 그대로 전달하는지 확인한다."""
        for objective, gradient in ((lambda p: np.inf, lambda p: [1.0]), (lambda p: 0.0, lambda p: [np.nan])):
            with self.assertRaises(ValueError): run_optimization(objective, gradient, [1.0], VanillaGD(.1))
        for steps in (-1,):
            with self.assertRaises(ValueError): run_optimization(circle, quadratic_gradient, [1, 1], VanillaGD(.1), steps)
        for steps in (True, 2.5):
            with self.assertRaises(TypeError): run_optimization(circle, quadratic_gradient, [1, 1], VanillaGD(.1), steps)
        for tolerance in (-1, np.inf):
            with self.assertRaises(ValueError): run_optimization(circle, quadratic_gradient, [1, 1], VanillaGD(.1), gradient_tolerance=tolerance)
        with self.assertRaises(ValueError): run_optimization(lambda p: [1.0], lambda p: [1.0], [1.0], VanillaGD(.1))
        with self.assertRaises(ValueError): run_optimization(lambda p: 1.0, lambda p: [1.0, 2.0], [1.0], VanillaGD(.1))

        def broken(point):
            """발산으로 잘못 분류하면 안 되는 사용자 예외를 재현한다."""
            raise RuntimeError("user failure")

        with self.assertRaisesRegex(RuntimeError, "user failure"):
            run_optimization(broken, lambda p: [1.0], [1.0], VanillaGD(.1))

    def test_callback_copies_preserve_point_and_each_other(self):
        """콜백이 인수를 바꾸더라도 경로나 다음 콜백 입력점을 훼손하지 않는지 확인한다."""
        def objective(point):
            """x의 제곱을 계산한 뒤 전달받은 전용 인수만 바꾼다."""
            value = point[0]**2
            point[:] = 999
            return value

        def gradient(point):
            """2x를 계산한 뒤 전달받은 전용 인수만 바꾼다."""
            value = 2*point
            point[:] = -999
            return value

        initial = np.array([5.0])
        result = run_optimization(objective, gradient, initial, VanillaGD(.1), steps=2)
        np.testing.assert_allclose(result.path[:, 0], [5, 4, 3.2], atol=1e-15, rtol=0)	# 입력을 바꾸는 콜백을 사용해도 독립 GD 경로를 유지한다.
        np.testing.assert_array_equal(initial, [5.0])


class EllipticTests(unittest.TestCase):
    """반복 계산과 독립적으로 고정된 비등방성 이차함수를 확인한다."""

    def test_known_values_and_failures(self):
        """(3,-4)의 손실 169·기울기 (6,-80)를 확인하고 잘못된 좌표 차원을 거부한다."""
        assert elliptic([3, -4]) == 169	# 반복 계산 없이 구한 이차함수값 9+160과 비교한다.
        np.testing.assert_array_equal(elliptic_gradient([3, -4]), [6, -80])
        for function in (elliptic, elliptic_gradient):
            with self.assertRaises(ValueError): function([1, 2, 3])
            with self.assertRaises(FloatingPointError): function([0, 1e308])


if __name__ == "__main__":
    unittest.main()
