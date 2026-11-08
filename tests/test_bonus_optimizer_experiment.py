"""Independent trajectory and cache checks for the submitted optimizer bonus."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.run_bonus_optimizers import main, trajectory
from src.optimizer import run_optimization
from src.utils.cache_manager import CacheManager

__covers__ = ["scripts/run_bonus_optimizers.py"]


class BonusExperimentTests(unittest.TestCase):
    """Check declared budgets, independent endpoints and honest success criteria."""

    def test_independent_quadratic_and_newton_stop(self):
        """Closed-form GD and one-step Newton have different accepted path lengths."""
        gd = trajectory("ellipse", "GD", .05, steps=3, initial=[2., -3.])
        np.testing.assert_allclose(gd["path"], [[2,-3],[1.8,0],[1.62,0],[1.458,0]], atol=1e-14, rtol=0)
        newton = trajectory("ellipse", "Newton", 1., initial=[2., -3.], gradient_tolerance=1e-8)
        assert newton["iterations"] == 1 and newton["status"] == "gradient_tolerance"
        assert newton["path"].shape == (2,2)
        np.testing.assert_allclose(newton["path"][-1], [0,0], atol=1e-12, rtol=0)

    def test_cache_reuse_configuration_and_copy(self):
        """Initial, budget, rate and stopping policy distinguish cache identities."""
        with tempfile.TemporaryDirectory() as directory:
            cache = CacheManager(directory)
            with patch("scripts.run_bonus_optimizers.run_optimization", wraps=run_optimization) as compute:
                first = trajectory("circle", "Adam", .1, steps=3, cache=cache)
                repeated = trajectory("circle", "Adam", .1, steps=3, cache=cache)
                assert compute.call_count == 1
                for kwargs in ({"initial":[1.,2.]},{"steps":4},{"gradient_tolerance":1e-8}):
                    parameters={"steps":3, **kwargs}
                    trajectory("circle", "Adam", .1, cache=cache, **parameters)
                trajectory("circle", "Adam", .05, steps=3, cache=cache)
                assert compute.call_count == 5
                uncached = trajectory("circle", "Adam", .1, steps=3, cache=CacheManager(directory, enabled=False))
                assert compute.call_count == 6
            for field in ("path","losses","gradient_norms"):
                np.testing.assert_array_equal(first[field], repeated[field])
                np.testing.assert_array_equal(first[field], uncached[field])
            first["path"][:] = 999
            restored = trajectory("circle", "Adam", .1, steps=3, cache=cache)
            np.testing.assert_array_equal(restored["path"], repeated["path"])

    def test_invalid_configuration(self):
        """Malformed names, shapes, budgets and thresholds fail before any caching."""
        for args in (("bad","Adam",.1),("circle","bad",.1)):
            with self.assertRaises(ValueError):
                trajectory(*args)
        for kwargs in ({"initial":[1.]},{"steps":-1},{"gradient_tolerance":-1}):
            with self.assertRaises(ValueError):
                trajectory("circle","GD",.1,**kwargs)
        with self.assertRaises(TypeError):
            trajectory("circle","GD",.1,steps=True)

    def test_cli_reports_all_three_bonus_optimizer_comparisons(self):
        """The CLI includes raw histories and actual PNGs without an Adam-wins assertion."""
        with tempfile.TemporaryDirectory() as directory:
            assert main(["--output-dir",directory,"--no-cache"]) == 0
            root = Path(directory)
            report=json.loads((root/"bonus_optimizer_metrics.json").read_text(encoding="utf-8"))
            assert report["all_assertions_passed"] is True
            for methods in report["adam_comparisons"].values():
                assert set(methods)=={"GD","Momentum","Adam"}
                for record in methods.values():
                    assert np.asarray(record["path"]).shape == (201,2)
                    assert record["steps_budget"] == record["iterations"] == 200
            assert report["newton_comparison"]["Newton"]["iterations"] == 1
            assert report["newton_comparison"]["GD"]["iterations"] > 1
            for figure in report["figures"]:
                assert (root/figure["filename"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            assert "Update counts are not runtime comparisons" in (root/"bonus_optimizer_report.md").read_text(encoding="utf-8")
