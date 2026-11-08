"""Integration check of the documented Step 3 command and measured artifacts."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.run_calculus import main

__covers__ = ["scripts/run_calculus.py", "src/calculus.py"]


class CalculusExperimentTests(unittest.TestCase):
    """Verify the deliverable command produces finite measurements and PNGs."""

    def test_delivery_command_and_recorded_geometry(self):
        """Written data meets independent derivative/geometry targets and includes no origin direction."""
        with tempfile.TemporaryDirectory() as directory:
            assert main(["--output-dir", directory]) == 0
            output = Path(directory)
            report = json.loads((output / "calculus_metrics.json").read_text(encoding="utf-8"))
            for filename in ("gradient_geometry.png", "derivative_error.png"):
                assert (output / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            assert report["seed"] == 42
            assert abs(report["derivative"]["numerical"]-6.0) <= 1e-4
            data = report["gradient"]
            points = np.asarray(data["points"])
            assert points.shape == (9, 2)
            np.testing.assert_allclose(data["numerical"], 2*points, atol=1e-8, rtol=0)
            tangents = np.asarray(data["nonzero_tangents"])
            np.testing.assert_allclose(np.sum(points[:-1]*tangents, axis=1), 0, atol=1e-12, rtol=0)
            assert data["maximum_normalized_dot"] <= 1e-12
            assert data["origin_gradient"] == [0.0, 0.0]
            assert data["origin_direction"] is None
            assert len(report["step_size_study"]["h"]) > 10
