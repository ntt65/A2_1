"""Independent metadata, tolerance and injected-failure checks for evaluation evidence."""

import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

import numpy as np

from scripts.run_evaluation_evidence import (
    difference_report, failure_diagnostics, png_metadata, run_experiment,
    softmax_summary, step_size_summary,
)

__covers__ = ["scripts/run_evaluation_evidence.py"]


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    """Encode a test PNG chunk with an independent CRC and big-endian length."""
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff)


class EvaluationEvidenceTests(unittest.TestCase):
    """Verify measured metadata and diagnostics without treating injected failures as regressions."""

    def test_png_metadata_reads_independent_dimensions_and_units(self):
        """Known 7x11 metadata and 10000 px/metre must decode as 254 DPI."""
        data = b"\x89PNG\r\n\x1a\n"
        data += png_chunk(b"IHDR", struct.pack(">IIBBBBB", 7, 11, 8, 6, 0, 0, 0))
        data += png_chunk(b"pHYs", struct.pack(">IIB", 10000, 10000, 1))
        data += png_chunk(b"IEND", b"")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.png"
            path.write_bytes(data)
            result = png_metadata(path)
            assert (result["width_px"], result["height_px"]) == (7, 11)
            np.testing.assert_allclose(result["dpi_xy"], [254, 254], atol=1e-12, rtol=0)
            for bad in (b"not png", data[:-1], data[:20] + b"x" + data[21:]):
                path.write_bytes(bad)
                with self.assertRaises(ValueError):
                    png_metadata(path)

    def test_diagnostic_signed_difference_shape_and_failure(self):
        """The signs locate a mismatch and failed values cannot silently pass."""
        result = difference_report([1., -2.], [1.25, -2.5], .3)
        np.testing.assert_array_equal(result["signed_difference"], [.25, -.5])
        assert result["max_absolute_error"] == .5 and not result["passed"]
        assert difference_report(1., 1.125, .125)["passed"]
        for expected, observed, tolerance in (([1], [[1]], .1), ([], [], .1), ([1], [np.inf], .1), (1, 1, -1)):
            with self.assertRaises(ValueError):
                difference_report(expected, observed, tolerance)
        with self.assertRaises(TypeError):
            difference_report([True], [True], .1)

    def test_h_summary_does_not_bridge_failing_samples(self):
        """Two passing regions must not be reported as one certified interval."""
        result = step_size_summary([1., 2., 3., 4., 5.], [.1, .2, 1., .1, .2], .2)
        assert result["passing_h"] == [1., 2., 4., 5.]
        assert len(result["contiguous_sample_groups"]) == 2
        assert result["minimizing_sampled_h"] == [1., 4.]
        assert step_size_summary([1.], [1.], .1)["contiguous_sample_groups"] == []
        for h, errors in (([2., 1.], [0., 0.]), ([0.], [0.]), ([1.], [-1.])):
            with self.assertRaises(ValueError):
                step_size_summary(h, errors)

    def test_real_examples_pass_and_both_injections_fail(self):
        """Fixed unit-area map and known backprop gradient pass, injected changes fail."""
        report = failure_diagnostics()
        assert report["area"]["genuine"]["area_ratio"] == 1.
        assert report["area"]["genuine"]["passed"]
        assert report["backprop"]["genuine"]["passed"]
        assert not report["area"]["injected_observation"]["passed"]
        assert not report["backprop"]["injected_observation"]["passed"]
        np.testing.assert_allclose(report["backprop"]["injected_observation"]["signed_difference"],
                                   [[.001, 0], [0, 0]], atol=1e-15, rtol=0)

    def test_softmax_reports_distinct_axes_and_finite_error(self):
        """Every reported axis is actually normalized and agrees with 1 within tolerance."""
        rows = softmax_summary()
        assert len(rows) == 6 and {row["axis"] for row in rows} == {-1, 0, 1}
        assert all(row["sum_error_max"] <= 1e-6 and row["passed"] for row in rows)
        assert rows[1]["sum_shape"] == [3] and rows[2]["sum_shape"] == [2]

    def test_delivery_preserves_originals_and_writes_metadata(self):
        """Report generation leaves its input PNG files byte-identical and writes both formats."""
        source = Path(__file__).resolve().parents[1] / "outputs"
        originals = {path.name: path.read_bytes() for path in source.glob("*.png")}
        with tempfile.TemporaryDirectory() as directory:
            report = run_experiment(source, Path(directory), use_cache=False)
            saved = json.loads((Path(directory) / "evaluation_evidence.json").read_text(encoding="utf-8"))
            assert report == saved
            assert (Path(directory) / "evaluation_evidence.md").is_file()
            assert len(saved["figure_metadata"]) == 10
            assert [row["input_shape"] for row in saved["svd_panel_metadata"]] == [[64, 64], [128, 128]]
            assert all(len(row["panels"]) == 4 for row in saved["svd_panel_metadata"])
        assert all((source / name).read_bytes() == data for name, data in originals.items())
