"""Independent analytic references and contracts for the calculus APIs."""

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
    """Check the mission tolerance, truncation behavior, and scalar contracts."""

    def test_required_quadratic_derivative(self):
        """The derivative of x squared at x=3 is six within absolute error 1e-4."""
        result = numerical_derivative(lambda x: x**2, 3.0, 1e-5)
        assert isinstance(result, float)
        assert abs(result - 6.0) <= 1e-4

    def test_independent_analytic_references(self):
        """Constant, affine, and cubic derivatives agree with independent formulas."""
        assert numerical_derivative(lambda x: 4.0, -3.0) == 0.0
        assert abs(numerical_derivative(lambda x: -7.0 * x + 9.0, 2.0) + 7.0) <= 1e-9
        assert abs(numerical_derivative(lambda x: x**3, -2.0) - 12.0) <= 1e-8

    def test_sine_truncation_error(self):
        """A smooth nonquadratic function exhibits the expected second-order error."""
        exact = np.cos(3.0)
        coarse = abs(numerical_derivative(np.sin, 3.0, 0.1) - exact)
        fine = abs(numerical_derivative(np.sin, 3.0, 0.01) - exact)
        assert coarse > fine > 0
        assert 99.0 <= coarse / fine <= 101.0

    def test_scalar_contract_and_invalid_steps(self):
        """Shared checks reject invalid h, unresolved samples, and nonscalar values."""
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
    """Verify coordinate order, sample isolation, failure handling, and geometry."""

    def test_asymmetric_three_variable_reference(self):
        """A coupled polynomial detects coordinate swaps and coordinate leakage."""
        point = np.array([2.0, -3.0, 0.5])
        actual = numerical_gradient(
            lambda p: p[0] ** 2 + 3 * p[0] * p[1] + 5 * p[1] ** 2 - 7 * p[2],
            point,
        )
        np.testing.assert_allclose(actual, [-5.0, -24.0, -7.0], atol=1e-8, rtol=0)
        np.testing.assert_array_equal(point, [2.0, -3.0, 0.5])
        assert actual.shape == (3,)
        assert actual.dtype == np.float64

    def test_one_coordinate_and_constant(self):
        """One-dimensional inputs retain their axis, and constant gradients are zero."""
        result = numerical_gradient(lambda p: p[0] ** 2, [3])
        assert result.shape == (1,)
        np.testing.assert_allclose(result, [6.0], atol=1e-4, rtol=0)
        np.testing.assert_array_equal(numerical_gradient(lambda p: 8.0, [1.0, 2.0]), [0, 0])

    def test_each_objective_sample_is_isolated(self):
        """An objective mutating its sample cannot change the input or later samples."""
        point = np.array([2.0, 3.0])
        point.setflags(write=False)
        samples = []

        def mutating_objective(sample):
            """Evaluate x+4y before mutating the private sample in place."""
            samples.append(sample)
            value = sample[0] + 4.0 * sample[1]
            sample[:] = 999.0
            return value

        actual = numerical_gradient(mutating_objective, point)
        np.testing.assert_allclose(actual, [1.0, 4.0], atol=1e-9, rtol=0)
        np.testing.assert_array_equal(point, [2.0, 3.0])
        assert len(samples) == 4
        assert len({id(sample) for sample in samples}) == 4

    def test_point_and_output_contracts(self):
        """Gradients require a finite nonempty real vector and scalar objective output."""
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
        """Every coordinate uses shared h validation; user exceptions are not hidden."""
        for step in (0.0, -1e-5, np.inf, np.nan, 1e-300, [1e-5]):
            with self.subTest(step=step), self.assertRaises(ValueError):
                numerical_gradient(lambda p: np.sum(p), [3.0, 4.0], step)
        with self.assertRaises(ValueError):
            numerical_gradient(lambda p: 0.0, [0.0, 1e20], 1e-5)

        def failing_objective(sample):
            """Represent a user-side function failure that must propagate unchanged."""
            raise RuntimeError("objective failure")

        with self.assertRaisesRegex(RuntimeError, "objective failure"):
            numerical_gradient(failing_objective, [1.0, 2.0])

    def test_quadratic_numerical_and_analytic_gradients(self):
        """At asymmetric, axis, and origin points the reference derivative is (2x,2y)."""
        points = np.array([[1.5, -2.0], [-3.0, 0.25], [0.0, 2.0], [0.0, 0.0]])
        expected = np.array([[3.0, -4.0], [-6.0, 0.5], [0.0, 4.0], [0.0, 0.0]])
        analytic = quadratic_gradient(points)
        numeric = np.array([numerical_gradient(lambda p: quadratic(p[0], p[1]), p) for p in points])
        np.testing.assert_allclose(analytic, expected, atol=0, rtol=0)
        np.testing.assert_allclose(numeric, expected, atol=1e-8, rtol=0)
        tangent = np.column_stack((-points[:, 1], points[:, 0]))
        np.testing.assert_allclose(np.sum(analytic * tangent, axis=1), 0.0, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(analytic[-1], [0.0, 0.0])


class QuadraticTests(unittest.TestCase):
    """Check scalar/broadcast/batch shapes and finite-value limits."""

    def test_scalar_and_broadcast_surface(self):
        """The 3-4-5 triangle and an asymmetric grid give exact independent values."""
        scalar = quadratic(3, -4)
        assert scalar.shape == ()
        assert scalar.dtype == np.float64
        assert float(scalar) == 25.0
        x = np.array([[1.0], [-2.0]])
        y = np.array([0.0, 3.0, 4.0])
        np.testing.assert_array_equal(quadratic(x, y), [[1.0, 10.0, 17.0], [4.0, 13.0, 20.0]])
        np.testing.assert_array_equal(x, [[1.0], [-2.0]])
        np.testing.assert_array_equal(y, [0.0, 3.0, 4.0])

    def test_gradient_batch_shape_and_input_preservation(self):
        """One point and higher-rank point batches preserve their coordinate axis."""
        np.testing.assert_array_equal(quadratic_gradient([3.0, -4.0]), [6.0, -8.0])
        batch = np.array([[[1.0, 2.0]], [[-3.0, 4.0]]])
        original = batch.copy()
        actual = quadratic_gradient(batch)
        assert actual.shape == (2, 1, 2)
        assert actual.dtype == np.float64
        np.testing.assert_array_equal(actual, [[[2.0, 4.0]], [[-6.0, 8.0]]])
        actual[:] = 0
        np.testing.assert_array_equal(batch, original)

    def test_surface_and_gradient_invalid_inputs(self):
        """Nonfinite, nonreal, malformed shapes, and unrepresentable outputs fail."""
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
