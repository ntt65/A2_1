"""Execute project notebooks in isolated kernels using this Python interpreter."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient
import nbformat

ROOT = Path(__file__).resolve().parents[1]
__concepts__ = {"Notebook_Execution": ["execute_notebook", "main"]}


def _clear_execution(notebook: nbformat.NotebookNode) -> None:
    """Remove prior code outputs/counts and execution timing metadata in place.

    Notebook source, stable cell IDs, titles, and concept annotations are
    retained. Empty outputs and counts force validation to rely on this run.
    """
    for cell in notebook.cells:
        cell.metadata.pop("execution", None)
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None


def execute_notebook(
    path: str | Path, output_dir: str | Path | None = None, write: bool = False,
) -> nbformat.NotebookNode:
    """Run a notebook from a fresh kernel at the repository root and return it.

    The kernel uses sys.executable without installing a user/global kernel.
    Prior outputs/counts are cleared and timing metadata is disabled. The
    notebook can use A2_1_OUTPUT_DIR for artifacts; the default is ROOT/outputs.
    Source bytes are unchanged unless write=True and every cell succeeds.
    Kernel runtime/config files are temporary under ROOT/.cache. The caller's
    environment is not mutated. Notebook errors, timeouts, and file errors
    propagate; even cells tagged raises-exception must pass. Returned notebook
    cells include this run's outputs and stable original IDs.
    """
    source = Path(path).resolve()
    if source.suffix != ".ipynb":
        raise ValueError("path must identify an .ipynb notebook")
    notebook = nbformat.read(source, as_version=4)
    _clear_execution(notebook)
    destination = (ROOT / "outputs") if output_dir is None else Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    temporary_root = ROOT / ".cache"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="notebook-kernel-", dir=temporary_root) as directory:
        temporary = Path(directory)
        kernels = temporary / "kernels"
        kernel = kernels / "a2_1"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(
            json.dumps({
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}",
                         "--HistoryManager.hist_file=:memory:"],
                "display_name": "A2_1 isolated execution", "language": "python",
            }), encoding="utf-8",
        )
        environment = os.environ.copy()
        environment.update({
            "A2_1_OUTPUT_DIR": str(destination), "MPLBACKEND": "Agg",
            "IPYTHONDIR": str(temporary / "ipython"),
            "JUPYTER_RUNTIME_DIR": str(temporary / "runtime"),
            "MPLCONFIGDIR": str(temporary / "matplotlib"),
        })
        manager = AsyncKernelManager(
            kernel_name="a2_1",
            connection_file=str(temporary / "connection.json"),
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]),
        )
        NotebookClient(
            notebook, km=manager, timeout=120, startup_timeout=60,
            record_timing=False, allow_errors=False, force_raise_errors=True,
            resources={"metadata": {"path": str(ROOT)}},
        ).execute(cleanup_kc=True, cwd=str(ROOT), env=environment)
    for cell in notebook.cells:
        cell.metadata.pop("execution", None)
    if write:
        source.write_text(nbformat.writes(notebook) + "\n", encoding="utf-8", newline="\n")
    return notebook


def main(argv: list[str] | None = None) -> int:
    """Execute supplied paths or all project notebooks; --write saves successful runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--write", action="store_true", help="save successful executed notebooks")
    arguments = parser.parse_args(argv)
    paths = arguments.paths or sorted((ROOT / "notebooks").glob("*.ipynb"))
    if not paths:
        parser.error("no notebooks found")
    for path in paths:
        notebook = execute_notebook(path, arguments.output_dir, arguments.write)
        count = sum(cell.cell_type == "code" and bool(cell.source.strip()) for cell in notebook.cells)
        print(f"Executed {path.name}: {count} code cells passed; "
              f"{'saved' if arguments.write else 'source unchanged'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
