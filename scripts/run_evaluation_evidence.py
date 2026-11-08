"""Report existing figure metadata and reproducible numerical diagnostics.

This supplemental experiment does not overwrite the original plots or metrics.
It distinguishes genuine measurements from deliberately injected failures.
"""

import argparse
import json
from pathlib import Path
import struct
import sys
import zlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.run_linear_algebra import generate_grayscale_image
from src.calculus import derivative_error_study
from src.linear_algebra import determinant_area_check, svd_comparison_figure
from src.utils.cache_manager import CacheManager
from src.utils.math import sigmoid, softmax
from src.utils.random import set_seed

__concepts__ = {
    "Evaluation_Evidence": ["run_experiment", "main"],
    "Figure_Metadata": ["png_metadata", "svd_panel_metadata"],
    "Numerical_Diagnostics": ["difference_report", "step_size_summary", "softmax_summary"],
}

FIGURES = (
    "linear_transforms.png", "svd_64.png", "svd_128.png",
    "gradient_geometry.png", "derivative_error.png",
    "optimizer_circle.png", "optimizer_ellipse.png", "optimizer_learning_rates.png",
    "probability_normal.png", "probability_bernoulli.png",
)


def png_metadata(path: Path) -> dict:
    """Read PNG canvas pixels and physical DPI from IHDR/pHYs without decoding pixels.

    Returns scalar width/height, format and a two-element DPI list (or None
    when absent). Chunk bounds and CRC are checked; malformed metadata raises
    ValueError and I/O errors propagate. No panel geometry is inferred here.
    """
    data = Path(path).read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("not a PNG file")
    offset, dimensions, dpi, ended = 8, None, None, False
    while offset + 12 <= len(data):
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        end = offset + 8 + length
        if end + 4 > len(data):
            raise ValueError("truncated PNG chunk")
        payload = data[offset + 8:end]
        crc = struct.unpack(">I", data[end:end + 4])[0]
        if zlib.crc32(kind + payload) & 0xffffffff != crc:
            raise ValueError("PNG chunk CRC mismatch")
        if offset == 8 and kind != b"IHDR":
            raise ValueError("PNG must start with IHDR")
        if kind == b"IHDR":
            if dimensions is not None or length != 13:
                raise ValueError("invalid PNG IHDR")
            width, height = struct.unpack(">II", payload[:8])
            if width == 0 or height == 0:
                raise ValueError("PNG dimensions must be positive")
            dimensions = (width, height)
        elif kind == b"pHYs":
            if length != 9:
                raise ValueError("invalid PNG pHYs")
            x, y, unit = struct.unpack(">IIB", payload)
            if unit not in (0, 1):
                raise ValueError("invalid PNG physical unit")
            if unit == 1:
                dpi = [x * 0.0254, y * 0.0254]
        offset = end + 4
        if kind == b"IEND":
            if length != 0 or offset != len(data):
                raise ValueError("invalid PNG end")
            ended = True
            break
    if dimensions is None or not ended:
        raise ValueError("incomplete PNG metadata")
    return {"filename": Path(path).name, "format": "PNG", "width_px": dimensions[0],
            "height_px": dimensions[1], "dpi_xy": dpi,
            "measurement": "stored PNG canvas; not the input array or one panel"}


def svd_panel_metadata(cache: CacheManager | None = None) -> list[dict]:
    """Render the two seeded SVD figures at 140 DPI and measure image Axes extents.

    Caller sets seed 42 once before invocation. Each input is (size,size),
    matching the original experiment's generation order 64 then 128. Returned
    fractional pixel bboxes describe recreated image panels, excluding titles;
    they are not integer crops recovered from the stored PNG. No image is saved.
    """
    records = []
    for size, ranks in ((64, [10, 50, 64]), (128, [10, 50, 100])):
        image = generate_grayscale_image(size)
        figure, axes = svd_comparison_figure(image, ranks, cache)
        try:
            figure.set_dpi(140)
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            panels = []
            for label, axis in zip(["original"] + [f"rank {k}" for k in ranks], axes):
                bounds = axis.get_window_extent(renderer).bounds
                panels.append({"label": label, "input_shape": [size, size],
                               "bbox_xywh_px": [float(value) for value in bounds]})
            records.append({"filename": f"svd_{size}.png", "input_shape": [size, size],
                            "role": "main" if size == 64 else "rank-100 supplement",
                            "figure_inches": figure.get_size_inches().tolist(),
                            "render_dpi": 140, "panels": panels,
                            "measurement": "recreated current-renderer image Axes; titles excluded"})
        finally:
            plt.close(figure)
    return records


def difference_report(expected, observed, atol: float) -> dict:
    """Return signed observed-minus-expected errors for equally shaped real arrays.

    Scalars have shape (). Inputs must be nonempty finite real arrays and atol
    finite/nonnegative. No broadcasting occurs. Type/shape/value violations
    raise TypeError/ValueError. The returned passed flag uses rtol=0 explicitly.
    """
    arrays = [np.asarray(value) for value in (expected, observed)]
    if any(value.dtype.kind not in "iuf" for value in arrays):
        raise TypeError("expected and observed must be real numeric values")
    if arrays[0].shape != arrays[1].shape or not arrays[0].size:
        raise ValueError("expected and observed must have the same nonempty shape")
    if (isinstance(atol, (bool, np.bool_)) or not np.isscalar(atol)
            or np.asarray(atol).dtype.kind not in "iuf"):
        raise TypeError("atol must be a real scalar")
    try:
        tolerance = float(atol)
    except (ValueError, TypeError) as error:
        raise TypeError("atol must be a real scalar") from error
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("atol must be finite and nonnegative")
    arrays = [value.astype(np.float64) for value in arrays]
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("expected and observed must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        difference = arrays[1] - arrays[0]
    if not np.all(np.isfinite(difference)):
        raise ValueError("difference exceeds float64 range")
    return {"expected": arrays[0].tolist(), "observed": arrays[1].tolist(),
            "shape": list(arrays[0].shape), "signed_difference": difference.tolist(),
            "max_absolute_error": float(np.max(np.abs(difference))),
            "atol": tolerance, "rtol": 0.0,
            "passed": bool(np.all(np.abs(difference) <= tolerance))}


def step_size_summary(steps, errors, tolerance: float = 1e-4) -> dict:
    """Summarize a sampled h/error grid (n,) without asserting continuous validity.

    h must be strictly increasing, finite and positive; errors and tolerance
    must be finite/nonnegative. Returns every passing h, contiguous *sample*
    groups, and every tied minimum. Invalid arrays/tolerance raise ValueError.
    """
    steps = np.asarray(steps, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    if (steps.ndim != 1 or errors.shape != steps.shape or steps.size == 0
            or not np.all(np.isfinite(steps)) or not np.all(np.isfinite(errors))
            or np.any(steps <= 0) or np.any(np.diff(steps) <= 0) or np.any(errors < 0)
            or not np.isscalar(tolerance) or not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError("require increasing positive h, nonnegative errors/tolerance")
    indices = np.flatnonzero(errors <= tolerance)
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1) if indices.size else []
    minimum = float(np.min(errors))
    return {"sample_count": int(steps.size), "tolerance": float(tolerance),
            "passing_count": int(indices.size), "passing_h": steps[indices].tolist(),
            "contiguous_sample_groups": [
                {"first_h": float(steps[g[0]]), "last_h": float(steps[g[-1]]), "sample_count": int(g.size)}
                for g in groups],
            "minimum_observed_error": minimum, "minimizing_sampled_h": steps[errors == minimum].tolist(),
            "scope": "only tested h values; gaps and intervening real h are not certified"}


def softmax_summary() -> list[dict]:
    """Evaluate fixed float64 logits on axes -1, 0 and 1, returning sum-error rows.

    Tests vector (3,), matrix (2,3), tensor (2,3,4), including large shifts.
    Every probability must be finite/nonnegative; row sums use tolerance 1e-6.
    Assertions fail if the existing shape-preserving implementation regresses.
    """
    cases = [(np.array([1000.0, 1001.0, 1002.0]), -1),
             (np.array([[0., 1., 2.], [3., 4., 5.]]), 0),
             (np.array([[0., 1., 2.], [3., 4., 5.]]), 1),
             (np.arange(24, dtype=np.float64).reshape(2, 3, 4) - 1000, -1),
             (np.arange(24, dtype=np.float64).reshape(2, 3, 4) - 1000, 0),
             (np.arange(24, dtype=np.float64).reshape(2, 3, 4) - 1000, 1)]
    rows = []
    for logits, axis in cases:
        probabilities = softmax(logits, axis=axis)
        error = np.abs(np.sum(probabilities, axis=axis) - 1.0)
        assert probabilities.shape == logits.shape
        assert np.all(np.isfinite(probabilities)) and np.all(probabilities >= 0)
        assert np.max(error) <= 1e-6
        rows.append({"shape": list(logits.shape), "axis": axis, "input": logits.tolist(),
                     "sum_shape": list(error.shape), "sum_error_max": float(np.max(error)),
                     "sum_error_mean": float(np.mean(error)), "tolerance": 1e-6, "passed": True})
    return rows


def failure_diagnostics() -> dict:
    """Compute real fixed examples and explicitly inject incorrect observations.

    Area uses square points (2,4) and diag(2,.5). Backprop uses the mission
    W1(2,2), x(2,), W2(2,) and a hard-coded independent gradient reference.
    The perturbed records are intentional diagnostic examples, not failures
    measured in the submitted algorithms. Assertions require their rejection.
    """
    matrix = np.diag([2.0, 0.5])
    points = np.array([[0., 1., 1., 0.], [0., 0., 1., 1.]])
    area = determinant_area_check(matrix, points)
    assert area["passed"] and abs(area["expected_ratio"] - 1.0) <= 1e-15
    injected_ratio = area["area_ratio"] + 0.02
    area_failure = difference_report(area["expected_ratio"], injected_ratio, 0.01)
    assert not area_failure["passed"]
    x = np.array([1., 0.])
    w1 = np.array([[.1, .2], [.3, .4]])
    w2 = np.array([.5, .6])
    activation = sigmoid(w1 @ x)
    predicted = float(sigmoid(w2 @ activation))
    gradient = np.outer((predicted - 1) * w2 * activation * (1 - activation), x)
    reference = np.array([[-0.043978515808697015, 0.], [-0.05173350439092561, 0.]])
    genuine = difference_report(reference, gradient, 5e-5)
    perturbed = gradient.copy()
    perturbed[0, 0] += 0.001
    failure = difference_report(reference, perturbed, 5e-5)
    assert genuine["passed"] and not failure["passed"]
    return {
        "area": {"matrix": matrix.tolist(), "points": points.tolist(), "genuine": area,
                 "injected_observation": area_failure,
                 "injection": "add 0.02 to area ratio; expected ratio=1 so abs error equals relative error",
                 "diagnostic": "REJECTED: measured ratio differs by 2%, exceeding 1% tolerance"},
        "backprop": {"x": x.tolist(), "W1": w1.tolist(), "b1": [0., 0.],
                     "W2": w2.tolist(), "b2": 0., "y_true": 1.,
                     "gradient_name": "dL/dW1", "genuine": genuine,
                     "injected_observation": failure, "injection": "add 0.001 to dL/dW1[0,0]",
                     "diagnostic": "REJECTED: absolute error exceeds 5e-5; signed difference identifies component"},
        "scope": "injected incorrect observations demonstrate diagnostics; genuine computations pass",
    }


def run_experiment(input_dir: Path, output_dir: Path, use_cache: bool = True) -> dict:
    """Write supplemental JSON/Markdown from existing PNGs and newly computed checks.

    input_dir contains ten original figures. output_dir receives only two new
    evaluation_evidence files. Seed is set once here. Missing files, malformed
    metadata, numerical failures and I/O errors propagate; originals are read-only.
    """
    set_seed(42)
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    metadata = [png_metadata(input_dir / filename) for filename in FIGURES]
    cache = CacheManager(".cache/linear_algebra", enabled=use_cache)
    panels = svd_panel_metadata(cache)
    sweep = derivative_error_study()
    report = {
        "seed": 42, "dtype": "float64", "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__, "figure_metadata": metadata,
        "svd_panel_metadata": panels,
        "step_size": {"function": "x**2", "x": 3.0, "analytic_derivative": 6.0,
                      "h": sweep["h"].tolist(), "absolute_errors": sweep["quadratic_error"].tolist(),
                      **step_size_summary(sweep["h"], sweep["quadratic_error"]),
                      "interpretation": "central differences are exact for x**2 in exact arithmetic; sampled roundoff minima are not universal optimal h"},
        "softmax": softmax_summary(), "diagnostics": failure_diagnostics(),
        "notes": ["64x64 limits the main SVD input, not the entire Figure or rendered panels.",
                  "128x128 is the documented separate supplement required for rank 100.",
                  "Existing plots use nominal 140 DPI; preserve readable axes rather than shrinking every plot to 64px.",
                  "PNG pHYs stores integer pixels/metre, so recovered DPI can differ slightly from nominal 140.",
                  "Rendered panel geometry depends on renderer/fonts; it excludes titles and is not embedded PNG metadata."]}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_evidence.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    lines = ["# Supplementary evaluation evidence", "",
             "Reproduce: `python -m scripts.run_evaluation_evidence --no-cache`.", "",
             "## Stored PNG metadata", "",
             "The 64x64 guide applies to the main SVD input, not entire plotted Figures.", "",
             "| PNG | Canvas pixels | Actual DPI (x/y) |", "|---|---:|---:|"]
    for row in metadata:
        dpi = row["dpi_xy"]
        lines.append(f"| {row['filename']} | {row['width_px']} x {row['height_px']} | {dpi} |")
    lines += ["", "## SVD input and rendered panels", "",
              "Image Axes dimensions measured after recreating the original seeded experiment at 140 DPI.",
              "These fractional pixel extents exclude titles and are not PNG crop metadata.", "",
              "| File | Input shape | Image panel | Width x height (pixels) |", "|---|---|---|---:|"]
    for row in panels:
        for panel in row["panels"]:
            _, _, width, height = panel["bbox_xywh_px"]
            lines.append(f"| {row['filename']} | {row['input_shape']} | {panel['label']} | {width:.3f} x {height:.3f} |")
    h_summary = report["step_size"]
    lines += ["", "## Sampled derivative h", "",
              f"x**2 at x=3, float64; {h_summary['passing_count']}/{h_summary['sample_count']} sampled h satisfy error <= 1e-4.",
              f"Passing consecutive sample groups: `{h_summary['contiguous_sample_groups']}`.",
              f"Minimum observed error: {h_summary['minimum_observed_error']:.6e}; sampled minimizers: `{h_summary['minimizing_sampled_h']}`.",
              "No continuous interval or universal optimal h is claimed. Quadratic central differences have zero truncation error in exact arithmetic.", "",
              "## Softmax axes", "", "| Input shape | Axis | Maximum sum error | Mean sum error |", "|---|---:|---:|---:|"]
    for row in report["softmax"]:
        lines.append(f"| {row['shape']} | {row['axis']} | {row['sum_error_max']:.6e} | {row['sum_error_mean']:.6e} |")
    lines += ["", "All six cases pass the 1e-6 tolerance; complete inputs are in the JSON.", "",
              "## Intentional failure diagnostics", "",
              "Both genuine fixed calculations pass. These are explicitly injected wrong observations:", "",
              "- Area: diag(2,0.5), unit square; adding 0.02 to ratio 1 yields 2% error > 1%: REJECTED.",
              "- Backprop: mission input, add 0.001 to dL/dW1[0,0]; max error > 5e-5: REJECTED.",
              f"- Signed gradient difference (observed minus independent reference): `{report['diagnostics']['backprop']['injected_observation']['signed_difference']}`.", "",
              "The JSON records full reproduction inputs, expected/observed values, shapes, differences and tolerances.", ""]
    (output_dir / "evaluation_evidence.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return report


def main(argv: list[str] | None = None) -> int:
    """Run metadata/diagnostics CLI; optional input/output directories do not alter originals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--no-cache", action="store_true")
    arguments = parser.parse_args(argv)
    report = run_experiment(arguments.input_dir, arguments.output_dir, not arguments.no_cache)
    print(f"PNG metadata: {len(report['figure_metadata'])}; softmax axis cases: {len(report['softmax'])}; injected failures rejected: 2")
    print(f"Derivative h samples passing 1e-4: {report['step_size']['passing_count']}/61; evidence saved in {arguments.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
