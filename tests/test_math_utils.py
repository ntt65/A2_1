"""Independent numerical and boundary checks for the shared math utilities."""

import importlib
import unittest

import numpy as np

from src.utils.math import sigmoid, softmax
from src.utils.numerical import central_difference
from src.utils.random import set_seed

__covers__ = ["src/utils/math.py", "src/utils/numerical.py", "src/utils/random.py"]


class ActivationTests(unittest.TestCase):
    """Check known probabilities, shapes, numerical stability, and invalid inputs."""

    def test_sigmoid_known_probabilities_and_shape(self):
        """Log-odds of one third, one, and three yield 0.25, 0.5, and 0.75."""
        values = np.array([[-np.log(3.0), 0.0, np.log(3.0)]])
        original = values.copy()
        actual = sigmoid(values)
        np.testing.assert_allclose(actual, [[0.25, 0.5, 0.75]], atol=1e-15, rtol=0)
        np.testing.assert_array_equal(values, original)
        assert actual.shape == values.shape
        assert actual.dtype == np.float64
        assert sigmoid(0.0).shape == ()
        assert float(sigmoid(0.0)) == 0.5
        assert sigmoid(np.empty((0, 2))).shape == (0, 2)

    def test_sigmoid_extremes_without_overflow(self):
        """Opposite float64 extremes remain finite and saturate at valid probabilities."""
        with np.errstate(all="raise"):
            actual = sigmoid([-1e308, -1000.0, 0.0, 1000.0, 1e308])
        np.testing.assert_array_equal(actual, [0.0, 0.0, 0.5, 1.0, 1.0])
        assert np.all(np.isfinite(actual))

    def test_softmax_known_probabilities(self):
        """Log weights 1, 2, 3 must normalize to their independent rational probabilities."""
        values = np.log([1.0, 2.0, 3.0])
        actual = softmax(values)
        np.testing.assert_allclose(actual, [1 / 6, 1 / 3, 1 / 2], atol=1e-15, rtol=0)
        assert abs(float(actual.sum()) - 1.0) <= 1e-6
        assert actual.shape == (3,)
        assert actual.dtype == np.float64

    def test_softmax_axes_and_shift_invariance(self):
        """Each selected slice is normalized; adding a common offset preserves probabilities."""
        values = np.array([[0.0, 0.0, 0.0], [np.log(3.0)] * 3])
        columns = softmax(values, axis=0)
        np.testing.assert_allclose(columns, [[0.25] * 3, [0.75] * 3], atol=1e-15, rtol=0)
        rows = softmax(values, axis=-1)
        np.testing.assert_allclose(rows, np.full((2, 3), 1 / 3), atol=1e-15, rtol=0)
        cube = np.stack((values, values + 1000.0))
        actual = softmax(cube, axis=1)
        assert actual.shape == cube.shape
        assert np.max(np.abs(actual.sum(axis=1) - 1.0)) <= 1e-6
        np.testing.assert_allclose(actual[0], actual[1], atol=1e-13, rtol=0)
        np.testing.assert_array_equal(softmax(np.ones((2, 1)), axis=1), np.ones((2, 1)))

    def test_softmax_extreme_logits(self):
        """Finite extreme logits yield finite distributions even if subtraction overflows."""
        with np.errstate(all="raise"):
            actual = softmax([[1e308, -1e308], [-1e308, -1e308]])
        np.testing.assert_array_equal(actual, [[1.0, 0.0], [0.5, 0.5]])
        assert np.all(np.isfinite(actual))
        assert np.all(actual >= 0)
        assert np.max(np.abs(actual.sum(axis=1) - 1.0)) <= 1e-6

    def test_activation_input_rejection(self):
        """Nonfinite, complex, boolean, string, and object-valued inputs are rejected."""
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
        """Scalar and empty arrays have no supported distribution; invalid axes fail explicitly."""
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
    """Compare central differences with analytical derivatives and reject invalid contracts."""

    def test_quadratic_required_accuracy(self):
        """The mission's x squared derivative at three is six within absolute error 1e-4."""
        actual = central_difference(lambda x: x**2, 3.0)
        assert isinstance(actual, float)
        assert abs(actual - 6.0) <= 1e-4

    def test_other_analytical_derivatives(self):
        """Constant, affine, and sine functions give independent reference derivatives."""
        assert central_difference(lambda x: 7.0, 1.0) == 0.0
        assert abs(central_difference(lambda x: 3.0 * x - 4.0, -2.0) - 3.0) <= 1e-9
        assert abs(central_difference(np.sin, 0.0) - 1.0) <= 1e-9

    def test_invalid_steps_and_sampling_points(self):
        """Steps must be positive, finite, and numerically distinguish the two samples."""
        for step in (0.0, -1e-5, np.nan, np.inf, 1e-300):
            with self.subTest(step=step):
                with self.assertRaises(ValueError):
                    central_difference(lambda x: x**2, 3.0, h=step)
        with self.assertRaises(ValueError):
            central_difference(lambda x: x, 1e308, h=1e308)

    def test_scalar_function_contract(self):
        """Nonscalar or nonfinite samples and outputs fail instead of broadcasting silently."""
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
    """Verify explicit seeding and ensure merely importing utilities does not reseed."""

    def test_seed_reproduces_known_numpy_sequence(self):
        """The required seed reproduces the established first three RandomState values."""
        previous = np.random.get_state()
        try:
            set_seed()
            actual = np.random.random(3)
            expected = [0.3745401188473625, 0.9507143064099162, 0.7319939418114051]
            np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)
            set_seed(42)
            np.testing.assert_array_equal(np.random.random(3), actual)
        finally:
            np.random.set_state(previous)

    def test_import_has_no_rng_side_effect(self):
        """Reloading the utility does not consume or reset global random values."""
        previous = np.random.get_state()
        try:
            set_seed(42)
            reference = np.random.random(4)
            set_seed(42)
            first = np.random.random(2)
            importlib.reload(importlib.import_module("src.utils.random"))
            second = np.random.random(2)
            np.testing.assert_array_equal(np.concatenate((first, second)), reference)
        finally:
            np.random.set_state(previous)

    def test_invalid_seed(self):
        """Only integer values in NumPy's supported 32-bit seed range are accepted."""
        for seed in (-1, 2**32):
            with self.assertRaises(ValueError):
                set_seed(seed)
        for seed in (1.0, True, None, "42"):
            with self.assertRaises(TypeError):
                set_seed(seed)


if __name__ == "__main__":
    unittest.main()
