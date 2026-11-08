"""Exercise the bonus CLI, independent anchors and truthful boundary serialization."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np

from scripts.run_information_theory import _json_number, example_report, main

__covers__ = ["scripts/run_information_theory.py", "src/information_theory.py"]


class InformationTheoryExperimentTests(unittest.TestCase):
    """Keep reports reproducible and distinguish positive infinity from finite clipping."""

    def tearDown(self):
        """Release figures if a plotting assertion fails."""
        plt.close("all")

    def test_independent_examples_and_infinity_encoding(self):
        """Known rational information values and exact boundaries remain visible in the report."""
        rows = {row["name"]: row for row in example_report()}
        assert len(rows) == 6 and all(row["passed"] for row in rows.values())
        assert abs(rows["rational_forward"]["kl_nats"] - .13081203594113697) <= 1e-14
        assert abs(rows["rational_reverse"]["kl_nats"] - .14384103622589045) <= 1e-14
        assert rows["support_mismatch"]["cross_entropy_nats"] == "+inf"
        assert rows["smallest_positive_q"]["cross_entropy_nats"] < 745.
        assert _json_number(np.inf) == "+inf"
        for value in (np.nan, -np.inf):
            with self.assertRaises(ValueError):
                _json_number(value)

    def test_real_cli_outputs_full_curves_and_true_png_size(self):
        """Run the actual CLI and verify independent midpoints, support endpoints and image metadata."""
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as console:
            assert main(["--output-dir", directory]) == 0
            root = Path(directory)
            report = json.loads((root / "information_theory.json").read_text(encoding="utf-8"))
            markdown = (root / "information_theory.md").read_text(encoding="utf-8")
            png = (root / "information_theory.png").read_bytes()
        assert "independent examples passed" in console.getvalue()
        assert report["units"] == "nats" and report["seed"] == 42
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        assert struct.unpack(">II", png[16:24]) == (1792, 686)
        assert report["figure"]["width_px"] == 1792 and report["figure"]["height_px"] == 686
        assert len(report["binary_entropy"]["entropy_nats"]) == 501
        assert abs(report["binary_entropy"]["entropy_nats"][250] - .6931471805599453) <= 1e-14
        assert report["binary_entropy"]["entropy_nats"][0] == report["binary_entropy"]["entropy_nats"][-1] == 0.
        for field in ("cross_entropy_nats", "kl_nats"):
            assert report["varying_model"][field][0] == report["varying_model"][field][-1] == "+inf"
        assert abs(report["varying_model"]["cross_entropy_nats"][150] - .6108643020548935) <= 1e-14
        assert abs(report["varying_model"]["kl_nats"][150]) <= 1e-14
        assert report["curve_checks"]["finite_model_count"] == 499
        assert report["curve_checks"]["maximum_identity_error"] <= 1e-12
        assert "not a finite cap" in markdown and "not symmetric" in markdown


if __name__ == "__main__":
    unittest.main()
