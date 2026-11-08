"""Independent quadratic, Hessian and numerical-failure checks for Newton steps."""

import unittest

import numpy as np

from src.calculus import quadratic_gradient
from src.newton import NewtonMethod, circle_hessian, elliptic_hessian
from src.optimizer import elliptic_gradient, run_optimization
from src.utils.numerical import central_difference

__covers__ = ["src/newton.py", "src/optimizer.py"]


class NewtonStepTests(unittest.TestCase):
    """Verify exact minimizers, damping, input ownership and Hessian contracts."""

    def test_general_spd_quadratic_reaches_known_shifted_minimizer(self):
        """A coupled 3D SPD quadratic reaches its independent center in one full step."""
        matrix = np.array([[4., 1., 0.], [1., 3., 1.], [0., 1., 2.]])
        center = np.array([1., -2., .5])
        initial = np.array([5., 4., -3.])

        def objective(point):
            """Evaluate the shifted quadratic with known minimizer center (3,)."""
            delta = point - center
            return .5 * float(delta @ matrix @ delta)

        def gradient(point):
            """Return the coupled quadratic gradient (3,)."""
            return matrix @ (point - center)

        optimizer = NewtonMethod(lambda point: matrix)
        result = run_optimization(objective, gradient, initial, optimizer, steps=5,
                                  gradient_tolerance=1e-12)
        assert result.iterations == 1 and result.status == "gradient_tolerance"
        assert result.path.shape == (2, 3)
        np.testing.assert_allclose(result.path[-1], center, atol=1e-12, rtol=0)
        assert abs(result.losses[-1]) <= 1e-24
        np.testing.assert_array_equal(initial, [5., 4., -3.])

    def test_fixed_damping_matches_closed_form_path(self):
        """Damping .25 gives theta_k=center+.75**k*(theta_0-center)."""
        matrix = np.array([[3., 1.], [1., 2.]])
        center, initial = np.array([-2., 3.]), np.array([6., -5.])
        result = run_optimization(
            lambda point: .5 * float((point-center) @ matrix @ (point-center)),
            lambda point: matrix @ (point-center), initial,
            NewtonMethod(lambda point: matrix, lr=.25), steps=8,
        )
        expected = center + .75**np.arange(9)[:, None] * (initial-center)
        np.testing.assert_allclose(result.path, expected, atol=1e-12, rtol=0)
        assert result.iterations == 8 and result.status == "budget_exhausted"

    def test_mission_quadratics_reach_zero_in_one_update(self):
        """Both fixed Hessians give the known origin minimizer without iteration tuning."""
        for hessian, gradient in ((circle_hessian, quadratic_gradient),
                                  (elliptic_hessian, elliptic_gradient)):
            with self.subTest(hessian=hessian.__name__):
                point = np.array([5., -3.])
                actual = NewtonMethod(hessian).step(point, gradient(point))
                np.testing.assert_allclose(actual, [0., 0.], atol=1e-14, rtol=0)

    def test_hessian_callback_cannot_mutate_input_or_output(self):
        """A callback receives an isolated point and returning an existing H is safe."""
        matrix = np.diag([2., 4.])
        theta, gradient = np.array([3., 2.]), np.array([2., 8.])

        def hessian(point):
            """Deliberately mutate the received copy while returning a stored Hessian."""
            point[:] = 999
            return matrix

        actual = NewtonMethod(hessian).step(theta, gradient)
        np.testing.assert_allclose(actual, [2., 0.], atol=1e-15, rtol=0)
        actual[:] = -100
        np.testing.assert_array_equal(theta, [3., 2.])
        np.testing.assert_array_equal(gradient, [2., 8.])
        np.testing.assert_array_equal(matrix, np.diag([2., 4.]))

    def test_constructor_and_step_contracts(self):
        """Reject invalid damping, Hessian callback and nonmatching finite vectors."""
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
        """Non-real, wrong-sized, asymmetric, singular and indefinite Hessians fail."""
        for matrix in (np.ones(2), np.eye(3), [[2., 1.], [0., 2.]],
                       np.diag([1., 0.]), np.diag([1., -1.]), [[1., 2.], [2., 1.]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                NewtonMethod(lambda point: matrix).step([1., 1.], [2., 2.])
        for matrix in ([[True, False], [False, True]], [[1j, 0], [0, 1j]],
                       [["1", "0"], ["0", "1"]]):
            with self.subTest(matrix=matrix), self.assertRaises(TypeError):
                NewtonMethod(lambda point: matrix).step([1., 1.], [2., 2.])

    def test_accepted_roundoff_asymmetry_uses_symmetric_part(self):
        """Tiny accepted antisymmetric noise is removed before the SPD solve."""
        matrix = np.array([[2., 1e-13], [-1e-13, 3.]])
        actual = NewtonMethod(lambda point: matrix).step([3., 4.], [2., 6.])
        np.testing.assert_allclose(actual, [2., 2.], atol=1e-14, rtol=0)
        near_symmetric_but_indefinite = np.array([[1e-20, 1e-12], [0., 1e-20]])
        with self.assertRaisesRegex(ValueError, "positive definite"):
            NewtonMethod(lambda point: near_symmetric_but_indefinite).step([0., 0.], [1., 1.])

    def test_exact_symmetric_subnormal_hessians_preserve_newton_direction(self):
        """For H=[s], g=[s], direction=1 even for the minimum or odd subnormal s."""
        smallest = np.nextafter(0., 1.)
        for value in (smallest, 3 * smallest):
            with self.subTest(value=value):
                matrix = np.array([[value]])
                actual = NewtonMethod(lambda point: matrix).step([2.], [value])
                np.testing.assert_array_equal(actual, [1.])
                np.testing.assert_array_equal(matrix, [[value]])

    def test_accepted_asymmetric_subnormal_pairs_have_safe_midpoints(self):
        """Tiny accepted pairs average to 2s or s without separately halving each entry."""
        smallest = np.nextafter(0., 1.)
        for matrix, gradient in (
            (np.array([[5., 1.], [3., 5.]]) * smallest, [7 * smallest, 7 * smallest]),
            (np.array([[5., 3.], [-1., 5.]]) * smallest, [6 * smallest, 6 * smallest]),
        ):
            with self.subTest(matrix=matrix):
                actual = NewtonMethod(lambda point: matrix).step([2., 2.], gradient)
                np.testing.assert_allclose(actual, [1., 1.], atol=1e-14, rtol=0)

    def test_large_finite_symmetric_hessian_does_not_overflow_on_averaging(self):
        """H=1e308*I and g=(1e308,-1e308) have the exact finite direction (1,-1)."""
        matrix = np.diag([1e308, 1e308])
        actual = NewtonMethod(lambda point: matrix).step([2., -2.], [1e308, -1e308])
        np.testing.assert_allclose(actual, [1., -1.], atol=1e-14, rtol=0)
        np.testing.assert_array_equal(matrix, np.diag([1e308, 1e308]))

    def test_nonfinite_outputs_and_unrepresentable_updates_raise(self):
        """Nonfinite callbacks, overflow directions and overflow candidates cannot pass."""
        for matrix in (np.diag([np.inf, 1.]), np.diag([np.nan, 1.])):
            with self.subTest(matrix=matrix), self.assertRaises(FloatingPointError):
                NewtonMethod(lambda point: matrix).step([1., 1.], [2., 2.])
        with self.assertRaises(FloatingPointError):
            NewtonMethod(lambda point: [[1e-308]]).step([0.], [1e308])
        with self.assertRaises(FloatingPointError):
            NewtonMethod(lambda point: [[1.]]).step([1e308], [-1e308])

    def test_runner_rejects_nonfinite_newton_direction(self):
        """The shared runner records no nonfinite candidate or rejected row."""
        result = run_optimization(lambda point: 0., lambda point: np.array([1e308]),
                                  [0.], NewtonMethod(lambda point: [[1e-308]]), steps=3)
        assert result.status == "nonfinite" and result.iterations == 0
        np.testing.assert_array_equal(result.path, [[0.]])
        assert result.losses.shape == result.gradient_norms.shape == (1,)

    def test_callback_errors_propagate(self):
        """A user callback error is not converted to curvature or convergence success."""
        def hessian(point):
            """Raise a sentinel application error for exception-propagation verification."""
            raise RuntimeError("user Hessian failure")

        with self.assertRaisesRegex(RuntimeError, "user Hessian failure"):
            NewtonMethod(hessian).step([1.], [2.])


class HessianTests(unittest.TestCase):
    """Check exact Hessian entries and independent numerical gradient derivatives."""

    def test_hessians_match_gradient_central_differences(self):
        """Differentiate each gradient coordinate with the shared scalar difference tool."""
        point = np.array([1.25, -2.5])
        for hessian, gradient, diagonal in ((circle_hessian, quadratic_gradient, [2., 2.]),
                                            (elliptic_hessian, elliptic_gradient, [2., 20.])):
            with self.subTest(hessian=hessian.__name__):
                numerical = np.empty((2, 2))
                for i in range(2):
                    for j in range(2):
                        def coordinate_gradient(value):
                            """Vary one coordinate and evaluate one gradient component."""
                            shifted = point.copy()
                            shifted[j] = value
                            return gradient(shifted)[i]

                        numerical[i, j] = central_difference(coordinate_gradient, point[j], h=1e-5)
                actual = hessian(point)
                np.testing.assert_array_equal(actual, np.diag(diagonal))
                np.testing.assert_allclose(actual, numerical, atol=1e-8, rtol=0)
                assert actual.shape == (2, 2) and actual.dtype == np.float64
                actual[:] = 0
                np.testing.assert_array_equal(hessian(point), np.diag(diagonal))

    def test_hessians_validate_points(self):
        """Mission Hessians require finite real (2,) points even though they are constant."""
        for function in (circle_hessian, elliptic_hessian):
            for point in ([1.], [[1., 2.]], [np.nan, 0.], [np.inf, 0.]):
                with self.subTest(function=function.__name__, point=point), self.assertRaises(ValueError):
                    function(point)
            with self.assertRaises(TypeError):
                function([True, False])
