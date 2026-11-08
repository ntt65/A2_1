"""Independent constants, support boundaries and axis contracts for information measures."""

import importlib
import unittest

import numpy as np

import src.information_theory as information_theory
from src.information_theory import cross_entropy, entropy, kl_divergence
from src.utils.random import set_seed

__covers__ = ["src/information_theory.py"]


class InformationTheoryTests(unittest.TestCase):
    """Check natural-log information measures against independent exact examples."""

    def test_uniform_and_point_mass_entropy(self):
        """Uniform four-way entropy is ln(4), while a deterministic outcome has zero."""
        assert abs(entropy([.25, .25, .25, .25]) - 1.3862943611198906) <= 1e-14
        assert entropy([0., 1., 0.]) == 0.
        assert entropy([1]) == 0.
        assert isinstance(entropy([.5, .5]), float)

    def test_rational_reference_values_and_asymmetry(self):
        """Known p=(1/4,3/4), q=(1/2,1/2) values independently anchor all formulas."""
        p, q = [.25, .75], [.5, .5]
        assert abs(entropy(p) - .5623351446188083) <= 1e-14
        assert abs(cross_entropy(p, q) - .6931471805599453) <= 1e-14
        assert abs(kl_divergence(p, q) - .13081203594113697) <= 1e-14
        assert abs(kl_divergence(q, p) - .14384103622589045) <= 1e-14
        assert abs(kl_divergence(p, q) - kl_divergence(q, p)) > .01

    def test_support_boundaries_avoid_zero_times_infinity(self):
        """Zero-mass terms vanish; positive mass assigned zero model probability gives +inf."""
        with np.errstate(all="raise"):
            assert cross_entropy([0., 1.], [0., 1.]) == 0.
            assert kl_divergence([0., 1.], [0., 1.]) == 0.
            assert np.isposinf(cross_entropy([1., 0.], [0., 1.]))
            assert np.isposinf(kl_divergence([1., 0.], [0., 1.]))
            np.testing.assert_array_equal(cross_entropy([[1., 0.], [0., 1.]], [[1., 0.], [1., 0.]]),
                                          [0., np.inf])

    def test_tiny_positive_reference_stays_finite(self):
        """The least positive float64 has -ln(q)=1074 ln(2), without forming 1/q."""
        smallest = np.nextafter(np.float64(0), np.float64(1))
        with np.errstate(all="raise"):
            measured_ce = cross_entropy([1., 0.], [smallest, 1.])
            measured_kl = kl_divergence([1., 0.], [smallest, 1.])
        assert np.isfinite(measured_ce) and np.isfinite(measured_kl)
        assert abs(measured_ce - 744.4400719213812) <= 2e-13
        assert abs(measured_kl - 1074 * np.log(2.)) <= 2e-13

    def test_arbitrary_axes_remove_only_class_dimension(self):
        """Transposed classes produce unchanged information with explicit output shapes."""
        p = np.array([[.25, .75], [.5, .5], [0., 1.]])
        q = np.full_like(p, .5)
        expected_entropy = np.array([.5623351446188083, .6931471805599453, 0.])
        np.testing.assert_allclose(entropy(p, -1), expected_entropy, atol=1e-14, rtol=0)
        np.testing.assert_allclose(entropy(p.T, 0), expected_entropy, atol=1e-14, rtol=0)
        tensor = np.stack((p, p), axis=1)
        assert entropy(tensor, 2).shape == (3, 2)
        assert entropy(np.swapaxes(tensor, 0, 2), 0).shape == (2, 3)
        for function in (cross_entropy, kl_divergence):
            np.testing.assert_allclose(function(p.T, q.T, axis=0), function(p, q, axis=1), atol=1e-14, rtol=0)
        assert entropy(np.empty((0, 3)), axis=1).shape == (0,)
        assert cross_entropy(np.empty((0, 3)), np.empty((0, 3)), axis=1).shape == (0,)

    def test_finite_support_identity_and_nonnegativity(self):
        """CE=H+KL and KL>=0 hold on declared positive models including zero target terms."""
        p = np.array([[.1, .2, .7], [.5, .25, .25], [1., 0., 0.]])
        q = np.array([[.2, .2, .6], [.25, .5, .25], [.9, .05, .05]])
        divergence = kl_divergence(p, q)
        np.testing.assert_allclose(cross_entropy(p, q), entropy(p) + divergence, atol=1e-14, rtol=0)
        assert np.all(divergence >= -1e-14)
        np.testing.assert_array_equal(kl_divergence(p, p), np.zeros(3))

    def test_invalid_probabilities_axis_and_shape(self):
        """Reject unnormalized, nonreal, invalid axes and broadcasting without repairing inputs."""
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
        """Accepted roundoff is left untouched and values beyond 1e-12 are rejected."""
        p = np.array([.25, .75 - 5e-13])
        expected = float(-np.sum(p * np.log(p)))
        assert abs(entropy(p) - expected) <= 1e-15
        with self.assertRaises(ValueError):
            entropy([.25, .75 - 2e-12])

    def test_inputs_and_global_rng_remain_unchanged(self):
        """Import and mathematical calls neither mutate caller arrays nor reseed global RNG."""
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
        np.testing.assert_array_equal(state[1], after[1])


if __name__ == "__main__":
    unittest.main()
