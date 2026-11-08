"""Smoke checks for the shared Python, plotting, and notebook environment."""

import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


class EnvironmentTests(unittest.TestCase):
    """Verify that the environment can run the project's required workflows."""

    def test_python_and_required_imports(self):
        """Use the project's Python minor version and import every direct dependency."""
        assert sys.version_info[:2] == (3, 11), sys.version
        for name in ("numpy", "matplotlib", "seaborn", "nbformat", "nbclient", "ipykernel"):
            with self.subTest(module=name):
                assert importlib.import_module(name) is not None

    def test_headless_png_rendering(self):
        """Render an actual PNG without requiring a graphical desktop."""
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "environment.png"
            figure, axes = plt.subplots()
            try:
                sns.lineplot(x=np.array([0.0, 1.0]), y=np.array([0.0, 1.0]), ax=axes)
                figure.savefig(output)
            finally:
                plt.close(figure)
            assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            assert output.stat().st_size > 100

    def test_notebook_execution(self):
        """Execute a notebook using this interpreter without installing a user kernel."""
        from jupyter_client import AsyncKernelManager
        from jupyter_client.kernelspec import KernelSpecManager
        from nbclient import NotebookClient
        import nbformat

        with tempfile.TemporaryDirectory() as temporary_directory:
            kernels = Path(temporary_directory) / "kernels"
            kernel = kernels / "a2_1"
            kernel.mkdir(parents=True)
            (kernel / "kernel.json").write_text(
                json.dumps(
                    {
                        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                        "display_name": "A2_1 environment check",
                        "language": "python",
                    }
                ),
                encoding="utf-8",
            )
            manager = AsyncKernelManager(
                kernel_name="a2_1",
                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]),
            )
            notebook = nbformat.v4.new_notebook(
                cells=[
                    nbformat.v4.new_code_cell(
                        "import sys\nimport numpy as np\n"
                        f"assert sys.executable == {sys.executable!r}\n"
                        "np.random.seed(42)\n"
                        "assert np.isfinite(np.random.random())\n"
                        "print('A2_1 notebook execution OK')"
                    )
                ]
            )
            NotebookClient(notebook, km=manager, timeout=30, startup_timeout=30).execute(cleanup_kc=True)
            output = notebook.cells[0].outputs
            assert any(
                item.output_type == "stream" and "A2_1 notebook execution OK" in item.text
                for item in output
            ), output


if __name__ == "__main__":
    unittest.main()
