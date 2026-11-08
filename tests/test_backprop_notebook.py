"""역전파 노트북과 실행기를 새 커널에서 실행하여 검증한다."""

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
    """독립 수치 기준과 성공·실패 시 저장 규칙을 확인한다."""

    def test_backprop_fresh_kernel_and_source_preservation(self):
        """원본 바이트를 바꾸지 않고 기준 노트북의 매개변수 아홉 개를 모두 검증하는지 확인한다."""
        path = ROOT / "notebooks" / "backprop_derivation.ipynb"
        original = path.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            notebook = execute_notebook(path, output_dir=directory)
            metrics = json.loads((Path(directory) / "backprop_metrics.json").read_text(encoding="utf-8"))
        assert path.read_bytes() == original	# 새 커널 검증이 기준 노트북 원본 바이트를 바꾸지 않는다.
        assert metrics["parameter_count"] == 9
        assert metrics["max_hand_absolute_error"] <= 5e-5	# 독립 손계산 기준의 최대 절대오차 5e-5를 적용한다.
        assert metrics["max_numerical_gradient_error"] <= 1e-6	# 매개변수 아홉 개의 수치 기울기 오차에 1e-6 기준을 적용한다.
        assert metrics["four_decimal_match"] is True
        assert metrics["all_assertions_passed"] is True
        np.testing.assert_allclose(metrics["forward"]["y_pred"], 0.6472915699946178, atol=5e-13, rtol=0)	# 순전파 출력의 독립 상수와 절대오차 5e-13로 비교한다.
        codes = [cell for cell in notebook.cells if cell.cell_type == "code" and cell.source.strip()]
        assert codes
        assert [cell.execution_count for cell in codes] == list(range(1, len(codes)+1))	# 이전 실행 상태를 이어받지 않고 첫 코드 셀부터 순서대로 실행한다.
        assert not any(output.output_type == "error" for cell in codes for output in cell.outputs)	# 성공 플래그뿐 아니라 실제 출력에 오류가 없는지 확인한다.
        assert all("execution" not in cell.metadata for cell in notebook.cells)

    def test_runner_clears_stale_state_and_saves_only_on_request(self):
        """격리된 두 실행이 현재 인터프리터·루트를 사용하고 실행 시간은 지우며 셀 ID는 보존하는지 확인한다."""
        previous_override = os.environ.get("A2_1_OUTPUT_DIR")
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            # 모든 운영체제에서 경로 별칭을 시험한다. Windows CI의
            # RUNNER~1 같은 8.3 이름도 실제 경로와 다른 표기로 해석될 수 있다.
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
            assert first == second	# 격리된 새 실행 두 번의 저장 결과가 결정적으로 같아야 한다.
        assert os.environ.get("A2_1_OUTPUT_DIR") == previous_override	# 임시 출력 경로 설정을 호출 전 환경 상태로 복원한다.

    def test_failed_cell_never_overwrites_notebook(self):
        """raises-exception 태그로도 실패한 assert를 숨기거나 실패 출력을 저장할 수 없는지 확인한다."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "must_fail.ipynb"
            cell = nbformat.v4.new_code_cell("assert False, 'intentional runner test'", id="must-fail")
            cell.metadata["tags"] = ["raises-exception"]
            nbformat.write(nbformat.v4.new_notebook(cells=[cell]), path)
            original = path.read_bytes()
            with self.assertRaises(CellExecutionError):
                execute_notebook(path, directory, write=True)
            assert path.read_bytes() == original	# 셀 실패가 발생하면 기존 노트북 파일을 덮어쓰지 않는다.


if __name__ == "__main__":
    unittest.main()
