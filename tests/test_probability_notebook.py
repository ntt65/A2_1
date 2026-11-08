"""Fresh-kernel validation of probability distributions and likelihood identities."""

import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from scripts.run_notebooks import ROOT, execute_notebook

__covers__ = ["scripts/run_notebooks.py", "notebooks/probability_loss.ipynb"]


class ProbabilityNotebookTests(unittest.TestCase):
    """Compare notebook artifacts with independent distribution and loss targets."""

    def test_probability_fresh_kernel_and_independent_targets(self):
        """Execute all assertions without source edits and check both PNGs and numerical evidence."""
        path = ROOT / "notebooks" / "probability_loss.ipynb"
        original = path.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            notebook = execute_notebook(path, output_dir=output)
            metrics = json.loads((output / "probability_metrics.json").read_text(encoding="utf-8"))
            for filename in ("probability_normal.png", "probability_bernoulli.png"):
                png = (output / filename).read_bytes()
                assert png.startswith(b"\x89PNG\r\n\x1a\n"), filename
                assert png[12:16] == b"IHDR", filename
                width, height = struct.unpack(">II", png[16:24])
                assert width >= 400 and height >= 300, (filename, width, height)
        assert path.read_bytes() == original
        assert metrics["seed"] == 42 and metrics["dtype"] == "float64"
        assert metrics["all_assertions_passed"] is True

        normals = metrics["normals"]
        assert len(normals) == 2
        for row, mu, variance, peak in zip(
            normals, (0.0, 2.0), (1.0, 0.5),
            (0.3989422804014327, 0.5641895835477563),
        ):
            with self.subTest(normal=(mu, variance)):
                assert row["mu"] == mu and row["variance"] == variance
                assert row["grid_points"] == 10001
                assert abs(row["sigma"]**2 - variance) <= 2e-15
                assert abs(row["integral"] - 1.0) <= 1e-4
                assert abs(row["peak"] - peak) <= 1e-14
                assert np.isfinite(row["minimum_pdf"]) and row["minimum_pdf"] >= 0.0
        bernoulli = metrics["bernoulli"]
        assert len(bernoulli) == 2
        for row, p, expected in zip(bernoulli, (0.3, 0.7), ([0.7, 0.3], [0.3, 0.7])):
            with self.subTest(bernoulli=p):
                assert row["p"] == p and row["support"] == [0, 1]
                np.testing.assert_allclose(row["pmf"], expected, atol=1e-15, rtol=0)
                assert abs(sum(row["pmf"]) - 1.0) <= 1e-12

        softmax = metrics["softmax"]
        probabilities = np.asarray(softmax["probabilities"], dtype=np.float64)
        assert probabilities.shape == np.shape(softmax["logits"])
        assert probabilities.ndim == 2 and softmax["axis"] == -1
        assert np.all(np.isfinite(probabilities)) and np.all(probabilities >= 0.0)
        np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0, atol=1e-6, rtol=0)
        assert softmax["column_sum_error"] <= 1e-6

        gaussian = metrics["gaussian"]
        assert gaussian["n"] == 3 and gaussian["variance"] == 0.5
        assert abs(gaussian["sse"] - 1.5) <= 1e-14
        assert abs(gaussian["mse"] - 0.5) <= 1e-14
        assert abs(gaussian["nll"] - 3.21709482877410026) <= 1e-12
        assert gaussian["identity_error"] <= 1e-12

        bce = metrics["bce"]
        assert bce["y"] == [0.0, 1.0]
        np.testing.assert_allclose(bce["probabilities"], [0.3, 0.7], atol=0, rtol=0)
        assert abs(bce["sum_loss"] - 0.71334988787746476) <= 1e-12
        assert abs(bce["mean_loss"] - 0.35667494393873238) <= 1e-12
        assert abs(bce["logit_sum_loss"] - bce["sum_loss"]) <= 1e-12
        assert bce["max_gradient_error"] <= 1e-6
        extreme_losses = np.asarray(bce["extreme_losses"])
        assert np.all(np.isfinite(extreme_losses)) and np.all(extreme_losses >= 0.0)
        assert 4.24e-18 < bce["positive_tail_at_40"] < 4.26e-18

        categorical = metrics["categorical"]
        np.testing.assert_allclose(categorical["probabilities"], [0.2, 0.3, 0.5], atol=0, rtol=0)
        assert categorical["target"] == 2
        assert abs(categorical["probability_ce"] - 0.69314718055994531) <= 1e-12
        assert abs(categorical["logit_ce"] - 0.69314718055994531) <= 1e-12
        assert np.all(np.isfinite(categorical["extreme_ce"]))

        codes = [cell for cell in notebook.cells if cell.cell_type == "code" and cell.source.strip()]
        assert codes
        assert [cell.execution_count for cell in codes] == list(range(1, len(codes)+1))
        assert not any(output.output_type == "error" for cell in codes for output in cell.outputs)
        assert all("execution" not in cell.metadata for cell in notebook.cells)


if __name__ == "__main__":
    unittest.main()
