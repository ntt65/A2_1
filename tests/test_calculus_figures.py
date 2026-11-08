"""Independent geometry and numerical-display checks for calculus figures."""

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.quiver import Quiver
import numpy as np

from src.calculus import derivative_error_figure, derivative_error_study, gradient_figure

__covers__ = ["src/calculus.py"]


class CalculusFigureTests(unittest.TestCase):
    """Verify plotted directions, genuine errors, and readable fixed-size layouts."""

    def tearDown(self):
        """Release every test-owned Figure even after a failed assertion."""
        plt.close("all")

    def test_gradient_arrows_follow_level_normals_and_skip_origin(self):
        """Both arrow sets lie on f=4; gradient is outward and tangent perpendicular."""
        figure, axes = gradient_figure()
        assert axes.figure is figure
        assert axes.get_aspect() == 1.0
        arrows = [item for item in axes.collections if isinstance(item, Quiver)]
        assert len(arrows) == 2
        gradient, tangent = arrows
        for quiver in arrows:
            assert len(quiver.X) == 8
            assert np.all(np.hypot(quiver.X, quiver.Y) > 0)
            np.testing.assert_allclose(quiver.X**2 + quiver.Y**2, 4.0, atol=2e-15, rtol=0)
        np.testing.assert_allclose(gradient.U, gradient.X / 2.0, atol=1e-15, rtol=0)
        np.testing.assert_allclose(gradient.V, gradient.Y / 2.0, atol=1e-15, rtol=0)
        np.testing.assert_allclose(gradient.U * tangent.U + gradient.V * tangent.V,
                                   np.zeros(8), atol=1e-15, rtol=0)
        assert any("zero gradient" in text.get_text() for text in axes.texts)

    def test_step_study_uses_independent_derivatives_and_retains_zero(self):
        """The quadratic reference is six and sine truncation is known analytically."""
        study = derivative_error_study()
        assert set(study) == {"h", "quadratic_derivative", "quadratic_error",
                              "sine_derivative", "sine_error"}
        assert all(value.shape == study["h"].shape for value in study.values())
        assert all(np.all(np.isfinite(value)) for value in study.values())
        assert np.all(np.diff(study["h"]) > 0)
        np.testing.assert_allclose(study["quadratic_error"],
                                   np.abs(study["quadratic_derivative"] - 6.0), atol=0, rtol=0)
        index = int(np.argmin(np.abs(study["h"] - 1e-5)))
        assert study["quadratic_error"][index] <= 1e-4
        # At h=1, the exact central difference of sin is cos(3)*sin(1).
        expected = abs(np.cos(3.0) * (np.sin(1.0) - 1.0))
        assert abs(study["sine_error"][-1] - expected) <= 3e-16
        assert study["quadratic_error"][-1] == 0.0

    def test_error_plot_keeps_raw_zero_data(self):
        """The symlog axis can display zero without replacing it by a positive floor."""
        figure, axes = derivative_error_figure()
        assert axes.figure is figure
        assert axes.get_xscale() == "log"
        assert axes.get_yscale() == "symlog"
        assert axes.get_ylim()[0] == 0.0
        np.testing.assert_array_equal(axes.lines[0].get_ydata(), derivative_error_study()["quadratic_error"])
        assert axes.lines[0].get_ydata()[-1] == 0.0
        assert "no truncation error" in axes.get_title()

    def test_titles_legends_and_scale_notes_fit_canvas(self):
        """All explanatory labels fit inside both final figure canvases."""
        for factory in (gradient_figure, derivative_error_figure):
            with self.subTest(figure=factory.__name__):
                figure, axes = factory()
                figure.set_dpi(140)
                figure.canvas.draw()
                renderer = figure.canvas.get_renderer()
                canvas = figure.bbox
                artists = [axes.title, axes.xaxis.label, axes.yaxis.label, axes.get_legend(), *figure.texts]
                for artist in artists:
                    bounds = artist.get_window_extent(renderer)
                    assert bounds.x0 >= canvas.x0 and bounds.x1 <= canvas.x1
                    assert bounds.y0 >= canvas.y0 and bounds.y1 <= canvas.y1
                plt.close(figure)


if __name__ == "__main__":
    unittest.main()
