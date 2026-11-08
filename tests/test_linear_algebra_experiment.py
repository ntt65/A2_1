"""Check the reproducible data and delivery command, not image pixel encodings."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.run_linear_algebra import generate_grayscale_image, main
from src.utils.random import set_seed

__covers__ = ["scripts/run_linear_algebra.py", "src/linear_algebra.py"]


class ExperimentTests(unittest.TestCase):
    """Verify that the submitted figures and numbers can be regenerated."""

    def test_seeded_image_shape_range_and_detail(self):
        """A fixed seed reproduces a nontrivial full-rank grayscale image."""
        set_seed()
        first = generate_grayscale_image(16)
        set_seed()
        second = generate_grayscale_image(16)
        assert first.shape == (16, 16) and first.dtype == np.float64
        np.testing.assert_array_equal(first, second)
        assert np.all(np.isfinite(first)) and np.min(first) >= 0 and np.max(first) <= 1
        assert np.linalg.matrix_rank(first) == 16
        for value in (True, 8.5, "16"):
            with self.assertRaises(TypeError): generate_grayscale_image(value)
        with self.assertRaises(ValueError): generate_grayscale_image(7)

    def test_delivery_command_outputs_and_assertions(self):
        """The documented command produces all required PNGs and honest measurements."""
        with tempfile.TemporaryDirectory() as directory:
            assert main(["--output-dir", directory, "--no-cache"]) == 0
            output = Path(directory)
            report = json.loads((output / "linear_algebra_metrics.json").read_text(encoding="utf-8"))
            for filename in ("linear_transforms.png", "svd_64.png", "svd_128.png"):
                assert (output / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            assert report["seed"] == 42
            assert report["power_iteration"]["relative_error"] <= 0.05
            assert report["power_iteration"]["relative_residual"] <= 1e-6
            assert all(value["passed"] for value in report["transforms"].values())
            assert [[r["k"] for r in image["comparisons"]] for image in report["svd"]] == [[10, 50, 64], [10, 50, 100]]
            assert all(image["full_rank_relative_error"] <= 1e-10 for image in report["svd"])
            assert report["svd"][1]["comparisons"][2]["stored_over_original"] > 1
