"""Check explicit submitted eig evidence against independent known values."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.verify_power_iteration import main, verify_mission_eigenpair

__covers__ = ["scripts/verify_power_iteration.py", "scripts/run_linear_algebra.py"]


class EigenpairEvidenceTests(unittest.TestCase):
    """Keep reference comparison visible, numerically correct, and sign invariant."""

    def test_reference_matches_independent_closed_form_and_vector(self):
        """The submitted NumPy reference agrees with the independent mission eigenpair."""
        report = verify_mission_eigenpair()
        manual, reference = report["power_iteration"], report["numpy_eig"]
        expected = (7.0 + np.sqrt(5.0)) / 2.0
        expected_vector = np.array([0.8506508083520399, 0.5257311121191336])
        assert abs(reference["eigenvalue"] - expected) <= 1e-14
        np.testing.assert_allclose(reference["eigenvector_sign_aligned"], expected_vector, atol=1e-14, rtol=0)
        measured_error = abs(manual["eigenvalue"] - reference["eigenvalue"]) / abs(reference["eigenvalue"])
        assert measured_error == manual["numpy_eig_comparison"]["relative_eigenvalue_error"]
        assert measured_error <= 0.05 and report["passed"] is True
        assert reference["selection"] == "argmax(abs(eigenvalues))"

    def test_reference_sign_is_irrelevant_and_wrong_value_is_rejected(self):
        """Opposite eigenvector signs pass, while a perturbed eig reference cannot claim PASS."""
        values = np.array([4.618033988749895, 2.381966011250105])
        vectors = np.array([[0.8506508083520399, -0.5257311121191336],
                            [0.5257311121191336, 0.8506508083520399]])
        with patch("scripts.verify_power_iteration.np.linalg.eig", return_value=(values, -vectors)):
            report = verify_mission_eigenpair()
            assert report["passed"] is True
            assert report["numpy_eig"]["eigenvector_raw"][0] < 0
            assert report["numpy_eig"]["eigenvector_sign_aligned"][0] > 0
        with patch("scripts.verify_power_iteration.np.linalg.eig", return_value=(values + 1, vectors)):
            with self.assertRaisesRegex(AssertionError, "verification failed"):
                verify_mission_eigenpair()

    def test_cli_writes_readable_and_machine_readable_evidence(self):
        """The documented command prints both methods and persists the exact report values."""
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as console:
            assert main(["--output-dir", directory]) == 0
            root = Path(directory)
            report = json.loads((root / "power_iteration_eig_comparison.json").read_text(encoding="utf-8"))
            evidence = (root / "power_iteration_eig_comparison.md").read_text(encoding="utf-8")
        assert report["power_iteration"]["matrix"] == [[4.0, 1.0], [1.0, 3.0]]
        assert report["power_iteration"]["initial"] == [1.0, 1.0]
        assert report["numpy_eig"]["eigenvalue"] == report["power_iteration"]["numpy_eig_comparison"]["numpy_eigenvalue"]
        for label in ("Power Iteration:", "np.linalg.eig:", "Relative eigenvalue error=", "5%", "PASS"):
            assert label in console.getvalue(), label
            assert label in evidence, label
