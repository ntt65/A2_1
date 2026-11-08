"""Independent accuracy and failure-state checks for manual Power Iteration."""

from dataclasses import FrozenInstanceError
import unittest
from unittest.mock import patch

import numpy as np

from src.linear_algebra import PowerIterationResult, power_iteration
from src.utils.random import set_seed


__covers__ = ["src/linear_algebra.py"]


class PowerIterationTests(unittest.TestCase):
    """Check eigenpairs against analytic values and verification-only eig calls."""

    def test_mission_matrix_matches_analytic_value_and_eig(self):
        """The mission matrix reaches its analytic dominant eigenpair and residual."""
        matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
        result = power_iteration(matrix, initial=[1.0, 1.0])
        expected = (7.0 + np.sqrt(5.0)) / 2.0
        values, vectors = np.linalg.eig(matrix)  # Verification only.
        index = int(np.argmax(np.abs(values)))
        actual_residual = np.linalg.norm(matrix @ result.eigenvector - result.eigenvalue * result.eigenvector)
        actual_residual /= np.linalg.norm(matrix, ord="fro") * np.linalg.norm(result.eigenvector)
        assert isinstance(result, PowerIterationResult)
        assert result.converged and result.reason == "residual_tolerance"
        assert 0 < result.iterations <= 1000
        assert abs(result.eigenvalue - expected) / abs(expected) <= 0.05
        assert abs(result.eigenvalue - values[index]) / abs(values[index]) <= 0.05
        assert abs(abs(np.dot(result.eigenvector, vectors[:, index])) - 1.0) <= 1e-8
        assert actual_residual <= 1e-6
        assert abs(actual_residual - result.residual) <= 1e-14
        assert abs(np.linalg.norm(result.eigenvector) - 1.0) <= 1e-14
        assert result.eigenvector.shape == (2,) and result.eigenvector.dtype == np.float64

    def test_negative_dominant_eigenvalue_preserves_sign(self):
        """Sign alternation of the vector does not prevent residual convergence."""
        matrix = np.diag([-5.0, 2.0])
        result = power_iteration(matrix, initial=[1.0, 1.0])
        assert result.converged
        assert abs(result.eigenvalue + 5.0) <= 1e-9
        assert abs(abs(result.eigenvector[0]) - 1.0) <= 1e-9
        assert result.residual <= 1e-10

    def test_zero_matrix_has_an_explicit_exact_result(self):
        """All unit directions are zero-eigenvalue eigenvectors of the zero matrix."""
        result = power_iteration(np.zeros((3, 3)), initial=[1.0, 2.0, 2.0])
        assert result.converged and result.reason == "zero_matrix"
        assert result.iterations == 0 and result.eigenvalue == 0.0 and result.residual == 0.0
        np.testing.assert_allclose(result.eigenvector, [1 / 3, 2 / 3, 2 / 3], atol=1e-15, rtol=0)

    def test_nullspace_and_nilpotent_breakdown_are_not_dominance_success(self):
        """A vanishing iterate in a nonzero matrix is reported as a breakdown."""
        immediate = power_iteration(np.diag([3.0, 0.0]), initial=[0.0, 1.0])
        later = power_iteration([[0.0, 1.0], [0.0, 0.0]], initial=[1.0, 1.0])
        for result, iterations in ((immediate, 0), (later, 1)):
            assert not result.converged and result.reason == "nullspace_breakdown"
            assert result.eigenvalue == 0.0 and result.residual == 0.0
            assert result.iterations == iterations

    def test_equal_magnitude_and_complex_modes_can_fail_to_converge(self):
        """A spectral tie or real rotation can exhaust the update budget."""
        for matrix in (np.diag([-2.0, 2.0]), np.array([[0.0, -1.0], [1.0, 0.0]])):
            with self.subTest(matrix=matrix):
                result = power_iteration(matrix, initial=[1.0, 1.0], max_iter=9)
                assert not result.converged and result.reason == "max_iterations"
                assert result.iterations == 9 and result.residual > 0.1
        exact_tie = power_iteration(np.eye(2), initial=[1.0, 2.0])
        assert exact_tie.converged and abs(exact_tie.eigenvalue - 1.0) <= 1e-14

    def test_budget_and_tolerance_are_observable(self):
        """One update gives an approximate result with an explicit nonconverged flag."""
        result = power_iteration([[4.0, 1.0], [1.0, 3.0]], initial=[1.0, 0.0], max_iter=1, tol=1e-15)
        assert result.iterations == 1
        assert not result.converged and result.reason == "max_iterations"
        assert result.residual > 1e-15

    def test_subdominant_initial_direction_documents_residual_limit(self):
        """A small residual certifies an eigenpair, not a dominant eigenvalue."""
        result = power_iteration(np.diag([5.0, 1.0]), initial=[0.0, 1.0])
        assert result.converged and result.iterations == 0
        assert result.eigenvalue == 1.0 and result.residual == 0.0

    def test_uniform_scales_and_extreme_initial_vectors_are_supported(self):
        """Scaling prevents norm overflow/underflow for representable eigenpairs."""
        base = np.array([[4.0, 1.0], [1.0, 3.0]])
        expected = (7.0 + np.sqrt(5.0)) / 2.0
        with np.errstate(all="raise"):
            for scale in (1e-300, 1e300):
                for initial in ([1e-300, 1e-300], [1e300, 1e300]):
                    result = power_iteration(base * scale, initial=initial)
                    assert result.converged
                    assert abs(result.eigenvalue / scale - expected) / expected <= 1e-8
                    assert np.all(np.isfinite(result.eigenvector)) and result.residual <= 1e-10

    def test_unrepresentable_eigenvalues_and_matrix_scaling_fail_explicitly(self):
        """Do not silently return infinity or erase nonzero matrix entries by scaling."""
        largest = np.finfo(np.float64).max
        with self.assertRaises(ValueError):
            power_iteration(np.full((2, 2), largest), initial=[1.0, 1.0])
        with self.assertRaises(ValueError):
            power_iteration(np.diag([1e308, 1e-308]), initial=[1.0, 1.0])

    def test_tiny_residual_is_not_mistaken_for_exact_convergence(self):
        """A tiny equal-magnitude block keeps cycling when tolerance is tighter."""
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1e-200], [0.0, 1e-200, 0.0]])
        result = power_iteration(matrix, initial=[0.0, 1.0, 0.0], tol=1e-250, max_iter=4)
        assert not result.converged and result.reason == "max_iterations"
        assert result.iterations == 4
        assert abs(result.residual / 1e-200 - 1.0) <= 1e-14

    def test_invalid_matrix_initial_and_control_inputs_fail(self):
        """Reject incompatible shapes, nonfinite values, zero starts and bad controls."""
        for matrix in ([], [1, 2], np.zeros((0, 0)), np.zeros((2, 3)), [[1, np.nan], [0, 2]], [[1, np.inf], [0, 2]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                power_iteration(matrix)
        for matrix in ([[True, False], [False, True]], [[1j, 0], [0, 1]], [["1", "0"], ["0", "1"]]):
            with self.subTest(matrix=matrix), self.assertRaises(TypeError):
                power_iteration(matrix)
        for initial in ([0.0, 0.0], [1.0], [[1.0, 1.0]], [1.0, np.nan]):
            with self.subTest(initial=initial), self.assertRaises(ValueError):
                power_iteration(np.eye(2), initial=initial)
        for tolerance in (0.0, -1.0, np.nan, np.inf, [1e-10]):
            with self.subTest(tol=tolerance), self.assertRaises(ValueError):
                power_iteration(np.eye(2), tol=tolerance)
        with self.assertRaises(TypeError):
            power_iteration(np.eye(2), tol=True)
        for count in (0, -1, 1.5, True):
            with self.subTest(max_iter=count), self.assertRaises((TypeError, ValueError)):
                power_iteration(np.eye(2), max_iter=count)

    def test_external_seed_reproduces_random_start_without_reseeding(self):
        """The caller controls randomness; supplying an initial vector uses no draws."""
        state = np.random.get_state()
        try:
            set_seed()
            first = power_iteration([[4.0, 1.0], [1.0, 3.0]])
            set_seed()
            second = power_iteration([[4.0, 1.0], [1.0, 3.0]])
            assert first.eigenvalue == second.eigenvalue and first.iterations == second.iterations
            np.testing.assert_array_equal(first.eigenvector, second.eigenvector)
            with patch("numpy.random.normal", side_effect=AssertionError("unexpected draw")):
                power_iteration(np.eye(2), initial=[1.0, 1.0])
        finally:
            np.random.set_state(state)

    def test_inputs_are_preserved_and_eigensolvers_are_not_used(self):
        """The manual algorithm neither mutates inputs nor delegates to eigensolvers."""
        matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
        initial = np.array([1.0, 1.0])
        saved_matrix, saved_initial = matrix.copy(), initial.copy()
        with patch("numpy.linalg.eig", side_effect=AssertionError("forbidden eig")), patch(
            "numpy.linalg.eigvals", side_effect=AssertionError("forbidden eigvals")
        ), patch("numpy.linalg.eigh", side_effect=AssertionError("forbidden eigh")):
            result = power_iteration(matrix, initial=initial)
        np.testing.assert_array_equal(matrix, saved_matrix)
        np.testing.assert_array_equal(initial, saved_initial)
        result.eigenvector[0] = 0.0
        np.testing.assert_array_equal(initial, saved_initial)
        with self.assertRaises(FrozenInstanceError):
            result.iterations = 0


if __name__ == "__main__":
    unittest.main()
