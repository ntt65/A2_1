"""정보량 측정의 독립 상수, 지지집합 경계와 축 조건을 검증한다."""

import importlib
import unittest

import numpy as np

import src.information_theory as information_theory
from src.information_theory import cross_entropy, entropy, kl_divergence
from src.utils.random import set_seed

__covers__ = ["src/information_theory.py"]


class InformationTheoryTests(unittest.TestCase):
    """자연로그 기반 정보량을 독립적으로 정확히 아는 예제와 비교한다."""

    def test_uniform_and_point_mass_entropy(self):
        """네 결과의 균등분포 엔트로피는 ln(4), 확정된 결과의 엔트로피는 0인지 확인한다."""
        assert abs(entropy([.25, .25, .25, .25]) - 1.3862943611198906) <= 1e-14	# 네 결과 균등분포의 ln(4)를 독립 상수로 비교한다.
        assert entropy([0., 1., 0.]) == 0.
        assert entropy([1]) == 0.
        assert isinstance(entropy([.5, .5]), float)

    def test_rational_reference_values_and_asymmetry(self):
        """p=(1/4,3/4), q=(1/2,1/2)의 알려진 값으로 모든 식과 비대칭성을 독립 검증한다."""
        p, q = [.25, .75], [.5, .5]
        assert abs(entropy(p) - .5623351446188083) <= 1e-14
        assert abs(cross_entropy(p, q) - .6931471805599453) <= 1e-14
        assert abs(kl_divergence(p, q) - .13081203594113697) <= 1e-14	# 유리수 분포의 KL을 독립 계산 상수와 절대오차 1e-14로 비교한다.
        assert abs(kl_divergence(q, p) - .14384103622589045) <= 1e-14
        assert abs(kl_divergence(p, q) - kl_divergence(q, p)) > .01	# 방향을 바꾼 KL이 같은 값이 아닌 반례를 확인한다.

    def test_support_boundaries_avoid_zero_times_infinity(self):
        """질량 0인 항은 사라지고 양의 질량에 모델 확률 0을 주면 +inf인지 확인한다."""
        with np.errstate(all="raise"):
            assert cross_entropy([0., 1.], [0., 1.]) == 0.
            assert kl_divergence([0., 1.], [0., 1.]) == 0.
            assert np.isposinf(cross_entropy([1., 0.], [0., 1.]))	# 불가능한 사건에 대한 양의 질량은 유한 상한 대신 양의 무한대이다.
            assert np.isposinf(kl_divergence([1., 0.], [0., 1.]))
            np.testing.assert_array_equal(cross_entropy([[1., 0.], [0., 1.]], [[1., 0.], [1., 0.]]),
                                          [0., np.inf])

    def test_tiny_positive_reference_stays_finite(self):
        """최소 양수 float64에서도 1/q를 만들지 않고 -ln(q)=1074 ln(2)를 유한하게 계산하는지 확인한다."""
        smallest = np.nextafter(np.float64(0), np.float64(1))
        with np.errstate(all="raise"):
            measured_ce = cross_entropy([1., 0.], [smallest, 1.])
            measured_kl = kl_divergence([1., 0.], [smallest, 1.])
        assert np.isfinite(measured_ce) and np.isfinite(measured_kl)
        assert abs(measured_ce - 744.4400719213812) <= 2e-13
        assert abs(measured_kl - 1074 * np.log(2.)) <= 2e-13	# 최소 양수의 정보량을 1074*ln(2)라는 독립 식으로 확인한다.

    def test_arbitrary_axes_remove_only_class_dimension(self):
        """클래스 축을 옮겨도 정보량 값은 같고 지정한 클래스 차원만 제거되는지 확인한다."""
        p = np.array([[.25, .75], [.5, .5], [0., 1.]])
        q = np.full_like(p, .5)
        expected_entropy = np.array([.5623351446188083, .6931471805599453, 0.])
        np.testing.assert_allclose(entropy(p, -1), expected_entropy, atol=1e-14, rtol=0)
        np.testing.assert_allclose(entropy(p.T, 0), expected_entropy, atol=1e-14, rtol=0)	# 클래스 축을 바꿔도 같은 엔트로피를 반환해야 한다.
        tensor = np.stack((p, p), axis=1)
        assert entropy(tensor, 2).shape == (3, 2)
        assert entropy(np.swapaxes(tensor, 0, 2), 0).shape == (2, 3)
        for function in (cross_entropy, kl_divergence):
            np.testing.assert_allclose(function(p.T, q.T, axis=0), function(p, q, axis=1), atol=1e-14, rtol=0)
        assert entropy(np.empty((0, 3)), axis=1).shape == (0,)
        assert cross_entropy(np.empty((0, 3)), np.empty((0, 3)), axis=1).shape == (0,)

    def test_finite_support_identity_and_nonnegativity(self):
        """명시한 양의 모델 확률과 0인 목표 항에서 CE=H+KL 및 KL>=0이 성립하는지 확인한다."""
        p = np.array([[.1, .2, .7], [.5, .25, .25], [1., 0., 0.]])
        q = np.array([[.2, .2, .6], [.25, .5, .25], [.9, .05, .05]])
        divergence = kl_divergence(p, q)
        np.testing.assert_allclose(cross_entropy(p, q), entropy(p) + divergence, atol=1e-14, rtol=0)	# 유한 지지집합에서 CE=H+KL 항등식을 확인한다.
        assert np.all(divergence >= -1e-14)	# 허용한 부동소수점 오차 안에서 KL의 비음수성을 확인한다.
        np.testing.assert_array_equal(kl_divergence(p, p), np.zeros(3))

    def test_invalid_probabilities_axis_and_shape(self):
        """정규화되지 않은 값·실수가 아닌 값·잘못된 축·브로드캐스트를 입력 보정 없이 거부하는지 확인한다."""
        for value in ([], 1., [[.5, .5], [.2, .2]], [-.1, 1.1], [1. + 1e-13, 0.],
                      [.2, .7], [np.nan, 1.], [np.inf, 0.], np.empty((3, 0))):
            with self.assertRaises(ValueError):
                entropy(value)
        for value in ([True, False], [1+0j, 0+0j], ["0.5", "0.5"]):
            with self.assertRaises(TypeError):
                entropy(value)
        for axis in (True, 0.5, "0", None):
            with self.assertRaises(TypeError):
                entropy([.5, .5], axis=axis)
        for axis in (1, -2):
            with self.assertRaises(ValueError):
                entropy([.5, .5], axis=axis)
        for function in (cross_entropy, kl_divergence):
            with self.assertRaises(ValueError):
                function([[.5, .5], [.5, .5]], [.5, .5])
            with self.assertRaises(ValueError):
                function([.5, .5], [.3, .3])
            with self.assertRaises(TypeError):
                function([.5, .5], [True, False])

    def test_normalization_tolerance_does_not_renormalize(self):
        """허용한 반올림 오차를 재정규화하지 않으며 1e-12 범위 밖의 입력은 거부하는지 확인한다."""
        p = np.array([.25, .75 - 5e-13])
        expected = float(-np.sum(p * np.log(p)))
        assert abs(entropy(p) - expected) <= 1e-15	# 허용 범위 입력을 재정규화하지 않은 원래 식과 비교한다.
        with self.assertRaises(ValueError):
            entropy([.25, .75 - 2e-12])

    def test_inputs_and_global_rng_remain_unchanged(self):
        """가져오기와 수학 함수 호출이 호출자 배열이나 전역 난수 상태를 바꾸지 않는지 확인한다."""
        set_seed(42)
        state = np.random.get_state()
        p, q = np.array([.25, .75]), np.array([.5, .5])
        before = p.copy(), q.copy()
        importlib.reload(information_theory)
        entropy(p)
        cross_entropy(p, q)
        kl_divergence(p, q)
        np.testing.assert_array_equal(p, before[0])
        np.testing.assert_array_equal(q, before[1])
        after = np.random.get_state()
        assert state[0] == after[0] and state[2:] == after[2:]
        np.testing.assert_array_equal(state[1], after[1])	# 가져오기와 계산이 전역 난수 상태 배열을 바꾸지 않는다.


if __name__ == "__main__":
    unittest.main()
