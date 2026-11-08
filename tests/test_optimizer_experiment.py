"""Check optimizer experiment reproducibility, trajectory cache, and figure data."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.run_optimizer import cached_trajectory, main
from src.calculus import quadratic
from src.optimizer import optimizer_comparison_figure, run_optimization
from src.utils.cache_manager import CacheManager

__covers__ = ["scripts/run_optimizer.py", "src/optimizer.py"]


class OptimizerExperimentTests(unittest.TestCase):
    """Verify same-input reuse, independent numeric targets, and reusable pictures."""

    def tearDown(self):
        """Close test figures even after a failed check."""
        plt.close("all")

    def test_cache_matches_uncached_and_separates_parameters(self):
        """A repeated run hits the cache while rate/objective changes recompute."""
        with tempfile.TemporaryDirectory() as directory:
            cache = CacheManager(directory)
            with patch("scripts.run_optimizer.run_optimization", wraps=run_optimization) as wrapped:
                first = cached_trajectory("circle", "Momentum", 0.1, cache=cache)
                second = cached_trajectory("circle", "Momentum", 0.1, cache=cache)
                assert wrapped.call_count == 1
                changed = cached_trajectory("circle", "Momentum", 0.05, cache=cache)
                cached_trajectory("ellipse", "Momentum", 0.1, cache=cache)
                assert wrapped.call_count == 3
                uncached = cached_trajectory("circle", "Momentum", 0.1,
                                             cache=CacheManager(directory, enabled=False))
                assert wrapped.call_count == 4
            for field in ("path", "losses", "gradient_norms"):
                np.testing.assert_array_equal(first[field], second[field])
                np.testing.assert_array_equal(first[field], uncached[field])
            assert not np.array_equal(first["path"], changed["path"])
            first["path"][:] = 999
            restored = cached_trajectory("circle", "Momentum", 0.1, cache=cache)
            np.testing.assert_array_equal(restored["path"], second["path"])

    def test_comparison_figure_retains_history_and_zero(self):
        """Plots use the measured paths and keep exact zero with symlog metric axes."""
        runs = {"GD": cached_trajectory("circle", "GD", 0.5, steps=3)}
        figure, axes = optimizer_comparison_figure(runs, quadratic, "Test comparison")
        assert axes.shape == (3,) and axes[0].get_aspect() == 1.0
        np.testing.assert_array_equal(axes[1].lines[0].get_ydata(), [50.0, 0.0, 0.0, 0.0])
        assert axes[1].get_yscale() == axes[2].get_yscale() == "symlog"
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for artist in [*figure.texts, *(a.title for a in axes), *(a.get_legend() for a in axes)]:
            box = artist.get_window_extent(renderer)
            assert box.x0 >= 0 and box.y0 >= 0
            assert box.x1 <= figure.bbox.width and box.y1 <= figure.bbox.height

    def test_cached_input_contract_matches_core(self):
        """NumPy integer budgets are normalized; invalid names and budgets fail explicitly."""
        with tempfile.TemporaryDirectory() as directory:
            record = cached_trajectory("circle", "GD", 0.1, steps=np.int64(2), cache=CacheManager(directory))
            assert record["path"].shape == (3, 2)
        for name, method in (("unknown", "GD"), ("circle", "unknown")):
            with self.assertRaises(ValueError):
                cached_trajectory(name, method, 0.1)
        for steps in (True, 1.5):
            with self.assertRaises(TypeError):
                cached_trajectory("circle", "GD", 0.1, steps=steps)
        with self.assertRaises(ValueError):
            cached_trajectory("circle", "GD", 0.1, steps=-1)

    def test_cli_outputs_and_stability_cases(self):
        """The delivery command writes three PNGs and independent stability measurements."""
        with tempfile.TemporaryDirectory() as directory:
            assert main(["--output-dir", directory, "--no-cache"]) == 0
            output = Path(directory)
            report = json.loads((output / "optimizer_metrics.json").read_text(encoding="utf-8"))
            for name in ("circle", "ellipse", "learning_rates"):
                assert (output / f"optimizer_{name}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            circle = report["comparisons"]["circle"]["GD"]
            assert np.asarray(circle["path"]).shape == (101, 2)
            assert circle["final_norm"] <= 0.1
            assert circle["iterations"] == 100
            rates = report["learning_rates"]
            assert rates["0.5"]["final_norm"] == 0.0
            np.testing.assert_allclose(rates["1.0"]["final_norm"], np.sqrt(50), atol=1e-14, rtol=0)
            assert rates["1.1"]["final_norm"] > 100


if __name__ == "__main__":
    unittest.main()
