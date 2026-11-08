"""Fresh-kernel execution checks for the backprop notebook and its runner."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from nbclient.exceptions import CellExecutionError
import nbformat
import numpy as np

from scripts.run_notebooks import ROOT, execute_notebook

__covers__ = ["scripts/run_notebooks.py", "notebooks/backprop_derivation.ipynb"]


class BackpropNotebookTests(unittest.TestCase):
    """Check independent numerical targets and successful/error persistence rules."""

    def test_backprop_fresh_kernel_and_source_preservation(self):
        """The canonical notebook verifies all nine parameters without changing source bytes."""
        path = ROOT / "notebooks" / "backprop_derivation.ipynb"
        original = path.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            notebook = execute_notebook(path, output_dir=directory)
            metrics = json.loads((Path(directory) / "backprop_metrics.json").read_text(encoding="utf-8"))
        assert path.read_bytes() == original
        assert metrics["parameter_count"] == 9
        assert metrics["max_hand_absolute_error"] <= 5e-5
        assert metrics["max_numerical_gradient_error"] <= 1e-6
        assert metrics["four_decimal_match"] is True
        assert metrics["all_assertions_passed"] is True
        np.testing.assert_allclose(metrics["forward"]["y_pred"], 0.6472915699946178, atol=5e-13, rtol=0)
        codes = [cell for cell in notebook.cells if cell.cell_type == "code" and cell.source.strip()]
        assert codes
        assert [cell.execution_count for cell in codes] == list(range(1, len(codes)+1))
        assert not any(output.output_type == "error" for cell in codes for output in cell.outputs)
        assert all("execution" not in cell.metadata for cell in notebook.cells)

    def test_runner_clears_stale_state_and_saves_only_on_request(self):
        """Two isolated runs use this interpreter/root, remove timing, and preserve stable cell IDs."""
        previous_override = os.environ.get("A2_1_OUTPUT_DIR")
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            # Exercise a path alias on every platform; Windows CI also uses
            # 8.3 names such as RUNNER~1 that resolve to a different spelling.
            (temporary / "alias").mkdir()
            requested_output = temporary / "alias" / ".." / "artifacts"
            path = temporary / "runner_check.ipynb"
            code = (
                "import os, sys\nfrom pathlib import Path\n"
                "assert 'prior_run_variable' not in globals()\nprior_run_variable = 1\n"
                f"assert sys.executable == {sys.executable!r}\n"
                f"assert Path.cwd() == Path({str(ROOT)!r})\n"
                f"assert Path(os.environ['A2_1_OUTPUT_DIR']).samefile(Path({str(requested_output)!r}))\n"
                "print('fresh current interpreter')"
            )
            cell = nbformat.v4.new_code_cell(code, id="runner-check")
            cell.execution_count = 99
            cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text="STALE")]
            cell.metadata["execution"] = {"iopub.status.busy": "old timestamp"}
            nbformat.write(nbformat.v4.new_notebook(cells=[cell]), path)
            original = path.read_bytes()
            first = execute_notebook(path, requested_output)
            assert path.read_bytes() == original
            assert first.cells[0].execution_count == 1
            assert first.cells[0].id == "runner-check"
            assert "execution" not in first.cells[0].metadata
            assert first.cells[0].outputs[0].text == "fresh current interpreter\n"
            second = execute_notebook(path, requested_output, write=True)
            assert path.read_bytes() != original
            stored = nbformat.read(path, as_version=4)
            assert stored == second
            assert first == second
        assert os.environ.get("A2_1_OUTPUT_DIR") == previous_override

    def test_failed_cell_never_overwrites_notebook(self):
        """Even raises-exception tags cannot hide a failed assertion or write failed output."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "must_fail.ipynb"
            cell = nbformat.v4.new_code_cell("assert False, 'intentional runner test'", id="must-fail")
            cell.metadata["tags"] = ["raises-exception"]
            nbformat.write(nbformat.v4.new_notebook(cells=[cell]), path)
            original = path.read_bytes()
            with self.assertRaises(CellExecutionError):
                execute_notebook(path, directory, write=True)
            assert path.read_bytes() == original


if __name__ == "__main__":
    unittest.main()
