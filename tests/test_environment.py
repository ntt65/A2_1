"""공통 Python·그림·노트북 실행 환경의 기본 동작을 검증한다."""

import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


class EnvironmentTests(unittest.TestCase):
    """프로젝트에 필요한 실행 흐름을 현재 환경에서 수행할 수 있는지 확인한다."""

    def test_python_and_required_imports(self):
        """프로젝트의 Python 부버전과 모든 직접 의존성의 가져오기를 확인한다."""
        assert sys.version_info[:2] == (3, 11), sys.version	# 잠금 환경의 Python 부버전 3.11을 정확히 확인한다.
        for name in ("numpy", "matplotlib", "seaborn", "nbformat", "nbclient", "ipykernel"):
            with self.subTest(module=name):
                assert importlib.import_module(name) is not None	# 직접 의존성 각각을 실제로 가져와 설치 상태를 확인한다.

    def test_headless_png_rendering(self):
        """그래픽 데스크톱 없이 실제 PNG를 렌더하는지 확인한다."""
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
            assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")	# 파일명만이 아니라 렌더된 PNG 시그니처를 확인한다.
            assert output.stat().st_size > 100

    def test_notebook_execution(self):
        """사용자 커널을 설치하지 않고 현재 인터프리터로 노트북을 실행하는지 확인한다."""
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
            ), output	# 현재 인터프리터를 사용하는 실제 노트북 실행의 표준출력을 확인한다.


if __name__ == "__main__":
    unittest.main()
