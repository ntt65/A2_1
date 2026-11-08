"""Independent recurrence, contract, and numerical-termination optimizer checks."""

import unittest

import numpy as np

from src.calculus import quadratic, quadratic_gradient
from src.optimizer import Momentum, VanillaGD, elliptic, elliptic_gradient, run_optimization

__covers__ = ["src/optimizer.py"]


def circle(point):
    """Evaluate the shared circular quadratic on a two-coordinate vector."""
    return quadratic(point[0], point[1])


class OptimizerStepTests(unittest.TestCase):
    """Verify updates independently of the trajectory runner."""

    def test_gd_step_preserves_inputs(self):
        """A nonsymmetric vector checks signs, learning rate, and input ownership."""
        theta, gradient = np.array([1.0, 2.0]), np.array([3.0, -4.0])
        result = VanillaGD(0.1).step(theta, gradient)
        np.testing.assert_allclose(result, [0.7, 2.4], atol=1e-15, rtol=0)
        result[:] = 0
        np.testing.assert_array_equal(theta, [1.0, 2.0])
        np.testing.assert_array_equal(gradient, [3.0, -4.0])

    def test_momentum_first_two_updates_and_state_copy(self):
        """Explicit v1=(3,-4),v2=(1.7,-1.6) checks the chosen momentum convention."""
        optimizer = Momentum(0.1)
        assert optimizer.velocity is None
        first = optimizer.step([1, 2], [3, -4])
        second = optimizer.step(first, [-1, 2])
        np.testing.assert_allclose(first, [0.7, 2.4], atol=1e-15, rtol=0)
        np.testing.assert_allclose(second, [0.53, 2.56], atol=1e-15, rtol=0)
        state = optimizer.velocity
        state[:] = 999
        np.testing.assert_allclose(optimizer.velocity, [1.7, -1.6], atol=1e-15, rtol=0)
        optimizer.reset()
        assert optimizer.velocity is None
        np.testing.assert_allclose(optimizer.step([1, 2], [3, -4]), first, atol=0, rtol=0)

    def test_zero_beta_matches_gd(self):
        """Beta zero removes velocity memory exactly for identical steps."""
        gd, momentum = VanillaGD(0.2), Momentum(0.2, beta=0)
        theta = np.array([1.0, -3.0])
        for derivative in ([4.0, 1.0], [-2.0, 5.0]):
            expected = gd.step(theta, derivative)
            np.testing.assert_array_equal(momentum.step(theta, derivative), expected)
            theta = expected

    def test_hyperparameter_and_shape_contracts(self):
        """Learning rate, beta, vector ranks, and velocity shape are validated."""
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
        """A failed velocity/update computation preserves the last valid state."""
        momentum = Momentum(2.0)
        momentum.step([1.0], [1.0])
        before = momentum.velocity
        with self.assertRaises(FloatingPointError): momentum.step([1.0], [1e308])
        np.testing.assert_array_equal(momentum.velocity, before)
        with self.assertRaises(FloatingPointError): VanillaGD(2.0).step([0.0], [1e308])


class TrajectoryTests(unittest.TestCase):
    """Verify closed-form paths, fair budgets, and finite termination records."""

    def test_circle_closed_form_and_required_accuracy(self):
        """GD at lr=.1 follows theta_k=5*.8**k for exactly 100 updates."""
        initial = np.array([5.0, 5.0])
        result = run_optimization(circle, quadratic_gradient, initial, VanillaGD(0.1))
        expected = np.repeat((5.0*0.8**np.arange(101))[:, None], 2, axis=1)
        assert result.path.shape == (101, 2)
        assert result.losses.shape == result.gradient_norms.shape == (101,)
        assert result.iterations == 100 and result.status == "budget_exhausted"
        np.testing.assert_allclose(result.path, expected, atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(result.losses, 50.0*0.8**(2*np.arange(101)), atol=1e-13, rtol=1e-13)
        assert np.linalg.norm(result.path[-1]) <= 0.1
        np.testing.assert_array_equal(initial, [5.0, 5.0])

    def test_circle_immediate_oscillation_and_divergence(self):
        """lr=.5 reaches zero, 1 alternates, and 1.1 grows with factor -1.2."""
        for rate, factor in ((0.5, 0.0), (1.0, -1.0), (1.1, -1.2)):
            result = run_optimization(circle, quadratic_gradient, [5, 5], VanillaGD(rate), steps=8)
            expected = 5*factor**np.arange(9)
            np.testing.assert_allclose(result.path[:, 0], expected, atol=1e-12, rtol=1e-14)
            assert result.iterations == 8

    def test_momentum_runner_resets_between_runs(self):
        """Reusing an optimizer reproduces zero-initial-velocity trajectories."""
        optimizer = Momentum(0.1)
        first = run_optimization(circle, quadratic_gradient, [5, 5], optimizer, steps=2)
        second = run_optimization(circle, quadratic_gradient, [5, 5], optimizer, steps=2)
        np.testing.assert_allclose(first.path, [[5, 5], [4, 4], [2.3, 2.3]], atol=1e-15, rtol=0)
        np.testing.assert_array_equal(first.path, second.path)

    def test_elliptic_gd_reference_and_equal_budgets(self):
        """lr=.05 eliminates y once; x contracts .9 per update without a required zigzag."""
        gd = run_optimization(elliptic, elliptic_gradient, [5, 5], VanillaGD(0.05))
        momentum = run_optimization(elliptic, elliptic_gradient, [5, 5], Momentum(0.05))
        np.testing.assert_allclose(gd.path[:, 0], 5*.9**np.arange(101), atol=1e-14, rtol=1e-14)
        np.testing.assert_array_equal(gd.path[1:, 1], np.zeros(100))
        assert gd.iterations == momentum.iterations == 100
        assert momentum.path.shape == gd.path.shape == (101, 2)
        assert np.all(np.isfinite(momentum.losses))

    def test_tolerance_and_zero_budget(self):
        """Early stopping is explicit, includes the initial point, and is optional."""
        stopped = run_optimization(circle, quadratic_gradient, [5, 5], VanillaGD(0.5), gradient_tolerance=0)
        assert stopped.status == "gradient_tolerance" and stopped.iterations == 1
        initial = run_optimization(circle, quadratic_gradient, [0, 0], Momentum(.1), gradient_tolerance=0)
        assert initial.iterations == 0 and initial.path.shape == (1, 2)
        zero = run_optimization(circle, quadratic_gradient, [5, 5], VanillaGD(.1), steps=0)
        assert zero.status == "budget_exhausted" and zero.iterations == 0

    def test_nonfinite_objective_and_gradient_keep_last_finite_rows(self):
        """Rejected later evaluations never introduce NaN/Inf into aligned histories."""
        for kind in ("objective", "gradient"):
            objective = (lambda p: np.inf if p[0] > 1 else p[0]) if kind == "objective" else (lambda p: p[0])
            gradient = (lambda p: [np.nan] if p[0] > 1 else [-1.0]) if kind == "gradient" else (lambda p: [-1.0])
            result = run_optimization(objective, gradient, [0.0], VanillaGD(1.0), steps=10)
            assert result.status == "nonfinite" and result.iterations == 1
            np.testing.assert_array_equal(result.path, [[0.0], [1.0]])
            assert len(result.losses) == len(result.gradient_norms) == len(result.path)
            assert np.all(np.isfinite(result.losses))

    def test_rejected_momentum_candidate_restores_accepted_state(self):
        """Valid velocity from a rejected objective point is also rolled back."""
        optimizer = Momentum(1.0)
        result = run_optimization(lambda p: np.inf if p[0] > 1 else p[0], lambda p: [-1.0], [0.0], optimizer)
        assert result.status == "nonfinite" and result.iterations == 1
        np.testing.assert_array_equal(optimizer.velocity, [-1.0])

    def test_step_overflow_and_stable_small_norm(self):
        """Update overflow stops at the initial row; tiny gradients retain nonzero norm."""
        result = run_optimization(lambda p: 0.0, lambda p: [1e308], [0.0], VanillaGD(2.0))
        assert result.status == "nonfinite" and result.iterations == 0
        tiny = run_optimization(lambda p: 0.0, lambda p: [1e-300, 1e-300], [0.0, 0.0], VanillaGD(.1), steps=0)
        assert 1.4e-300 < tiny.gradient_norms[0] < 1.5e-300

    def test_input_and_callback_contracts(self):
        """Initial invalid metrics fail, while malformed callbacks and user errors propagate."""
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
            """Simulate a user exception that must not be mislabeled as divergence."""
            raise RuntimeError("user failure")

        with self.assertRaisesRegex(RuntimeError, "user failure"):
            run_optimization(broken, lambda p: [1.0], [1.0], VanillaGD(.1))

    def test_callback_copies_preserve_point_and_each_other(self):
        """Mutating callbacks cannot corrupt the path or the next callback's point."""
        def objective(point):
            """Compute x squared then mutate only the private argument."""
            value = point[0]**2
            point[:] = 999
            return value

        def gradient(point):
            """Compute 2x then mutate only the private argument."""
            value = 2*point
            point[:] = -999
            return value

        initial = np.array([5.0])
        result = run_optimization(objective, gradient, initial, VanillaGD(.1), steps=2)
        np.testing.assert_allclose(result.path[:, 0], [5, 4, 3.2], atol=1e-15, rtol=0)
        np.testing.assert_array_equal(initial, [5.0])


class EllipticTests(unittest.TestCase):
    """Check the fixed anisotropic quadratic independently of iteration."""

    def test_known_values_and_failures(self):
        """At (3,-4), loss is169 and gradient(6,-80); invalid dimensions fail."""
        assert elliptic([3, -4]) == 169
        np.testing.assert_array_equal(elliptic_gradient([3, -4]), [6, -80])
        for function in (elliptic, elliptic_gradient):
            with self.assertRaises(ValueError): function([1, 2, 3])
            with self.assertRaises(FloatingPointError): function([0, 1e308])


if __name__ == "__main__":
    unittest.main()
