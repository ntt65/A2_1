"""Independent Adam expectations and transactional optimizer runner checks."""

import unittest

import numpy as np

from src.adam import Adam
from src.optimizer import Momentum, run_optimization

__covers__ = ["src/adam.py", "src/optimizer.py"]


class AdamStepTests(unittest.TestCase):
    """Check moment corrections, input/state contracts and finite arithmetic."""

    def test_first_two_steps_match_independent_hand_calculation(self):
        """beta1=beta2=.5 gives m2=[1,0], v2=[1.5,6], with correction 3/4."""
        optimizer = Adam(lr=0.3, beta1=0.5, beta2=0.5, epsilon=0.25)
        initial = np.array([1.0, -2.0])
        gradient = np.array([2.0, -4.0])
        first = optimizer.step(initial, gradient)
        expected_first = np.array([1.0 - 0.3*2.0/2.25, -2.0 + 0.3*4.0/4.25])
        np.testing.assert_allclose(first, expected_first, atol=1e-15, rtol=0)
        np.testing.assert_array_equal(optimizer.m, [1.0, -2.0])
        np.testing.assert_array_equal(optimizer.v, [2.0, 8.0])
        second = optimizer.step(first, [1.0, 2.0])
        # m2 = .5*[1,-2]+.5*[1,2] = [1,0].
        # v2 = .5*[2,8]+.5*[1,4] = [1.5,6].
        # mhat2=[4/3,0], vhat2=[2,8], sqrt(vhat2)=[sqrt(2),sqrt(8)].
        expected_second = expected_first - np.array([0.3*(4.0/3.0)/(np.sqrt(2.0)+0.25), 0.0])
        np.testing.assert_allclose(second, expected_second, atol=1e-15, rtol=0)
        np.testing.assert_array_equal(optimizer.m, [1.0, 0.0])
        np.testing.assert_array_equal(optimizer.v, [1.5, 6.0])
        assert optimizer.t == 2
        np.testing.assert_array_equal(initial, [1.0, -2.0])
        np.testing.assert_array_equal(gradient, [2.0, -4.0])

    def test_constant_gradient_bias_correction(self):
        """For constant nonzero g, every corrected update is lr*g/(abs(g)+epsilon)."""
        optimizer = Adam(lr=0.1, beta1=0.8, beta2=0.7, epsilon=0.2)
        point = np.array([2.0, -3.0])
        decrement = np.array([0.1*3.0/3.2, -0.1*4.0/4.2])
        for step in range(1, 11):
            point = optimizer.step(point, [3.0, -4.0])
            np.testing.assert_allclose(point, np.array([2.0, -3.0])-step*decrement, atol=1e-14, rtol=0)
        assert optimizer.t == 10

    def test_zero_betas_zero_gradient_and_epsilon_position(self):
        """beta=0 normalizes the current gradient and epsilon remains outside sqrt."""
        optimizer = Adam(lr=0.2, beta1=0, beta2=0, epsilon=0.5)
        first = optimizer.step([1.0, 2.0], [0.0, 0.25])
        np.testing.assert_allclose(first, [1.0, 2.0-0.2/3.0], atol=1e-15, rtol=0)
        second = optimizer.step(first, [0.0, 0.0])
        np.testing.assert_array_equal(second, first)
        np.testing.assert_array_equal(optimizer.m, [0.0, 0.0])
        np.testing.assert_array_equal(optimizer.v, [0.0, 0.0])
        assert optimizer.t == 2

    def test_state_copy_shape_and_reset(self):
        """Public state is isolated, shape changes require reset, and restart is reproducible."""
        optimizer = Adam()
        assert optimizer.m is None and optimizer.v is None and optimizer.t == 0
        first = optimizer.step([1, 2], [3, 4])
        moment, variance = optimizer.m, optimizer.v
        moment[:] = variance[:] = 999
        np.testing.assert_allclose(optimizer.m, [0.3, 0.4], atol=1e-15, rtol=0)
        np.testing.assert_allclose(optimizer.v, [0.009, 0.016], atol=1e-15, rtol=0)
        with self.assertRaises(ValueError):
            optimizer.step([1], [2])
        assert optimizer.t == 1
        optimizer.reset()
        assert optimizer.m is None and optimizer.v is None and optimizer.t == 0
        np.testing.assert_array_equal(optimizer.step([1, 2], [3, 4]), first)
        optimizer.reset()
        assert optimizer.step([1], [1]).shape == (1,)

    def test_invalid_hyperparameters_and_inputs(self):
        """Malformed scalars, nonfinite values, booleans and vector shape mismatches fail."""
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
        """Overflowing second moments or updates never commit m, v or t."""
        optimizer = Adam(lr=0.1)
        optimizer.step([0.0], [1.0])
        moment, variance, count = optimizer.m, optimizer.v, optimizer.t
        with self.assertRaises(FloatingPointError):
            optimizer.step([0.0], [1e200])
        np.testing.assert_array_equal(optimizer.m, moment)
        np.testing.assert_array_equal(optimizer.v, variance)
        assert optimizer.t == count
        large = Adam(lr=1.7e308, beta1=0, beta2=0)
        with self.assertRaises(FloatingPointError):
            large.step([1.7e308], [-1.0])
        assert large.m is None and large.v is None and large.t == 0

    def test_moment_underflow_is_rejected_atomically(self):
        """A lost squared gradient must never turn the normalized unit step into 1e108."""
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
        assert optimizer.t == count

    def test_weighted_and_inexact_subnormal_moments_are_rejected(self):
        """Exact subnormal squares may work; lost/inexact weighting and first moments fail."""
        # g^2 = 2^-1060 is an exactly representable subnormal, unlike 1e-400.
        derivative = np.ldexp(1.0, -530)
        exact = Adam(lr=1.0, beta1=0.0, beta2=0.0, epsilon=1e-308)
        np.testing.assert_array_equal(exact.step([0.0], [derivative]), [-1.0])
        for beta in (np.nextafter(1.0, 0.0), 0.1):
            optimizer = Adam(lr=1.0, beta1=0.0, beta2=beta, epsilon=1e-308)
            # The first weight produces zero; the second a nonzero but inexact subnormal.
            with self.subTest(beta=beta), self.assertRaisesRegex(FloatingPointError, "second moment.*underflow"):
                optimizer.step([0.0], [derivative])
            assert optimizer.t == 0 and optimizer.m is None and optimizer.v is None
        optimizer = Adam(beta1=0.5, beta2=0.0)
        with self.assertRaisesRegex(FloatingPointError, "first moment.*underflow"):
            optimizer.step([0.0], [np.nextafter(0.0, 1.0)])
        assert optimizer.t == 0 and optimizer.m is None and optimizer.v is None


class AdamRunnerTests(unittest.TestCase):
    """Keep all state at the last accepted point after candidate evaluation failures."""

    def test_runner_resets_all_state_and_reproduces(self):
        """Reusing Adam begins at zero moments/t, with path and count tied to accepted updates."""
        optimizer = Adam(lr=0.1)
        first = run_optimization(lambda p: float(p@p), lambda p: 2*p, [2.0, -1.0], optimizer, steps=5)
        assert optimizer.t == first.iterations == 5
        moments = optimizer.m, optimizer.v
        second = run_optimization(lambda p: float(p@p), lambda p: 2*p, [2.0, -1.0], optimizer, steps=5)
        np.testing.assert_array_equal(first.path, second.path)
        np.testing.assert_array_equal(optimizer.m, moments[0])
        np.testing.assert_array_equal(optimizer.v, moments[1])

    def test_runner_stops_on_moment_underflow_without_committing(self):
        """Moment precision loss follows the shared nonfinite failure path, not a huge update."""
        optimizer = Adam(lr=1.0, beta1=0.0, beta2=0.0, epsilon=1e-308)
        result = run_optimization(lambda p: 0.0, lambda p: [1e-200], [0.0], optimizer, steps=2)
        assert result.status == "nonfinite" and result.iterations == 0
        np.testing.assert_array_equal(result.path, [[0.0]])
        assert optimizer.t == 0 and optimizer.m is None and optimizer.v is None

    def test_nonfinite_callback_restores_last_accepted_adam_state(self):
        """Rejecting the second objective candidate leaves the first moments/count intact."""
        optimizer = Adam(lr=1.0, beta1=0.5, beta2=0.5, epsilon=0.25)
        result = run_optimization(lambda p: 0.0 if p[0] < 1.0 else np.inf,
                                  lambda p: [-1.0], [0.0], optimizer, steps=5)
        assert result.status == "nonfinite" and result.iterations == optimizer.t == 1
        np.testing.assert_allclose(result.path, [[0.0], [0.8]], atol=1e-15, rtol=0)
        np.testing.assert_array_equal(optimizer.m, [-0.5])
        np.testing.assert_array_equal(optimizer.v, [0.5])
        rejected = run_optimization(lambda p: 0.0 if p[0] == 0 else np.inf,
                                    lambda p: [-1.0], [0.0], optimizer, steps=2)
        assert rejected.iterations == optimizer.t == 0
        assert optimizer.m is None and optimizer.v is None

    def test_contract_errors_restore_state_then_propagate(self):
        """Malformed gradients and callback exceptions are not mislabeled nonfinite convergence."""
        for optimizer in (Adam(lr=1.0, beta1=0.5, beta2=0.5, epsilon=0.25), Momentum(lr=0.8)):
            with self.assertRaisesRegex(ValueError, "shape"):
                run_optimization(lambda p: 0.0, lambda p: [-1.0] if p[0] < 1.0 else [-1.0, -1.0],
                                 [0.0], optimizer, steps=5)
            if isinstance(optimizer, Adam):
                assert optimizer.t == 1
                np.testing.assert_array_equal(optimizer.m, [-0.5])
                np.testing.assert_array_equal(optimizer.v, [0.5])
            else:
                np.testing.assert_array_equal(optimizer.velocity, [-1.0])
        optimizer = Adam()

        def broken_objective(point):
            """Raise a sentinel error only on a candidate, after its Adam state changed."""
            if point[0] != 0:
                raise RuntimeError("callback sentinel")
            return 0.0

        with self.assertRaisesRegex(RuntimeError, "callback sentinel"):
            run_optimization(broken_objective, lambda p: [1.0], [0.0], optimizer, steps=2)
        assert optimizer.t == 0 and optimizer.m is None and optimizer.v is None


if __name__ == "__main__":
    unittest.main()
