"""Independent recurrence, missing-crossing and cache checks for sensitivity evidence."""

import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from scripts.run_optimizer_sensitivity import aggregate, main, radius_summary, stability_evidence, trajectory
from src.optimizer import run_optimization
from src.utils.cache_manager import CacheManager

__covers__ = ["scripts/run_optimizer_sensitivity.py"]


class OptimizerSensitivityTests(unittest.TestCase):
    """Verify fixed-start statistics without treating failures as successful runs."""

    def tearDown(self):
        """Close all Figures even when numerical assertions fail."""
        plt.close("all")

    def test_gd_matches_independent_closed_form(self):
        """Circle/ellipse paths equal coordinatewise powers from two distinct starts."""
        for name, rate, factors in (("circle", 0.1, [0.8, 0.8]), ("ellipse", 0.05, [0.9, 0.0])):
            for initial in ([1.0, -3.0], [-5.0, 5.0]):
                record = trajectory(name, "GD", initial, rate)
                expected = np.asarray(initial) * np.asarray(factors)[None, :] ** np.arange(101)[:, None]
                np.testing.assert_allclose(record["path"], expected, atol=1e-14, rtol=1e-12)
                assert record["path"].shape == (101, 2)
        origin = trajectory("circle", "Momentum", [0, 0], 0.1)
        assert origin["first_radius_0_1"] == origin["remaining_inside_radius_0_1_through_budget"] == 0
        np.testing.assert_array_equal(origin["path"], np.zeros((101, 2)))

    def test_crossings_reentry_and_missing_population(self):
        """Crossing and staying differ; failed cases never enter conditional averages."""
        missing = radius_summary([2.0, 1.0, 0.2])
        reentry = radius_summary([1.0, 0.09, 0.2, 0.05, 0.01])
        exits = radius_summary([0.0, 0.2, 0.3])
        assert missing["first_radius_0_1"] is None
        assert missing["remaining_inside_radius_0_1_through_budget"] is None
        assert reentry["first_radius_0_1"] == 1
        assert reentry["remaining_inside_radius_0_1_through_budget"] == 3
        assert exits["first_radius_0_1"] == 0
        assert exits["remaining_inside_radius_0_1_through_budget"] is None
        summary = aggregate([missing, reentry, exits])
        first = summary["first_radius_0_1"]
        assert first == {"success_count": 2, "failure_count": 1,
                         "mean_successful_updates": 0.5, "variance_successful_updates": 0.25}
        stays = summary["remaining_inside_radius_0_1_through_budget"]
        assert stays["success_count"] == 1 and stays["failure_count"] == 2
        assert stays["mean_successful_updates"] == 3.0 and stays["variance_successful_updates"] == 0.0
        assert aggregate([missing])["first_radius_0_1"]["mean_successful_updates"] is None
        np.testing.assert_allclose(summary["final_radius_all_samples"]["mean"], 0.17, atol=1e-15, rtol=0)

    def test_cache_equivalence_input_separation_and_copy(self):
        """Whole initial coordinates, method/rate/budget distinguish entries; returns are isolated."""
        with tempfile.TemporaryDirectory() as directory:
            cache = CacheManager(directory)
            with patch("scripts.run_optimizer_sensitivity.run_optimization", wraps=run_optimization) as wrapped:
                first = trajectory("circle", "Momentum", [1.0, -3.0], 0.1, cache=cache)
                second = trajectory("circle", "Momentum", [1.0, -3.0], 0.1, cache=cache)
                assert wrapped.call_count == 1
                trajectory("circle", "Momentum", [-1.0, -3.0], 0.1, cache=cache)
                trajectory("circle", "Momentum", [1.0, -3.0], 0.05, cache=cache)
                trajectory("circle", "GD", [1.0, -3.0], 0.1, cache=cache)
                trajectory("circle", "Momentum", [1.0, -3.0], 0.1, steps=2, cache=cache)
                assert wrapped.call_count == 5
                uncached = trajectory("circle", "Momentum", [1.0, -3.0], 0.1,
                                      cache=CacheManager(directory, enabled=False))
                assert wrapped.call_count == 6
            for field in ("path", "losses", "gradient_norms"):
                np.testing.assert_array_equal(first[field], second[field])
                np.testing.assert_array_equal(first[field], uncached[field])
            first["path"][:] = 999
            restored = trajectory("circle", "Momentum", [1.0, -3.0], 0.1, cache=cache)
            np.testing.assert_array_equal(restored["path"], second["path"])

    def test_stability_uses_exact_factors(self):
        """A half-step converges, unit-step oscillates, and 1.1 has loss multiplier 1.44."""
        rows = stability_evidence()
        np.testing.assert_allclose([r["coordinate_multiplier"] for r in rows], [0, -1, -1.2], atol=1e-15, rtol=0)
        np.testing.assert_allclose([r["theoretical_loss_multiplier"] for r in rows], [0, 1, 1.44], atol=1e-15, rtol=0)
        assert rows[0]["final_radius"] == 0 and rows[0]["observed_ratio_count"] == 1
        assert not rows[1]["observed_strict_loss_increase"]
        assert rows[2]["observed_strict_loss_increase"] and rows[2]["all_values_finite"]

    def test_input_contracts(self):
        """Invalid shape, budgets, names and radii fail without coercing booleans."""
        for start in ([], [1], [[1, 2]], [1, np.inf]):
            with self.assertRaises(ValueError):
                trajectory("circle", "GD", start, 0.1)
        for budget in (True, 2.1):
            with self.assertRaises(TypeError):
                trajectory("circle", "GD", [1, 2], 0.1, steps=budget)
        for values in ([], [-1], [np.nan], [[1, 2]]):
            with self.assertRaises(ValueError):
                radius_summary(values)
        with self.assertRaises(ValueError):
            radius_summary([1.0], radius=0)
        with self.assertRaises(ValueError):
            aggregate([])

    def test_cli_delivers_declared_population_and_png_metadata(self):
        """The real CLI writes five starts per method, raw histories and true PNG dimensions."""
        with tempfile.TemporaryDirectory() as directory:
            assert main(["--output-dir", directory, "--no-cache"]) == 0
            root = Path(directory)
            report = json.loads((root / "optimizer_sensitivity.json").read_text(encoding="utf-8"))
            png = (root / "optimizer_sensitivity.png").read_bytes()
            assert png[:8] == b"\x89PNG\r\n\x1a\n"
            assert struct.unpack(">II", png[16:24]) == (1680, 1400)
            assert report["figure"]["width_pixels"] == 1680 and report["figure"]["height_pixels"] == 1400
            for methods in report["experiments"].values():
                for group in methods.values():
                    assert len(group["runs"]) == group["summary"]["sample_count"] == 5
                    assert all(np.asarray(r["path"]).shape == (101, 2) for r in group["runs"])
            text = (root / "optimizer_sensitivity.md").read_text(encoding="utf-8")
            assert "not repeated random trials" in text and "first crossing" in text


if __name__ == "__main__":
    unittest.main()
