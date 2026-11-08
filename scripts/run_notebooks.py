"""현재 Python 인터프리터의 격리된 커널에서 프로젝트 노트북을 실행한다."""

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
    """기존 코드 출력·실행 횟수·실행 시간 메타데이터를 제자리에서 제거한다.

    노트북 원문, 안정된 셀 ID, 제목과 개념 표기는 보존한다.
    출력과 실행 횟수를 비워 검증이 이번 실행에만 의존하도록 한다.
    """
    for cell in notebook.cells:
        cell.metadata.pop("execution", None)	# 직전 실행의 시간 기록을 제거한다.
        if cell.cell_type == "code":
            cell.outputs = []	# 이전 성공 출력을 이번 실행 결과로 오인하지 않게 비운다.
            cell.execution_count = None	# 모든 코드 셀이 이번 커널에서 다시 실행되어야 한다.


def execute_notebook(
    path: str | Path, output_dir: str | Path | None = None, write: bool = False,
) -> nbformat.NotebookNode:
    """저장소 루트에서 새 커널로 노트북을 실행하고 결과를 반환한다.

    사용자·전역 커널을 설치하지 않고 sys.executable을 커널에 사용한다.
    기존 출력·실행 횟수를 지우고 시간 메타데이터 기록을 끈다. 산출물은
    A2_1_OUTPUT_DIR을 사용할 수 있으며 기본값은 ROOT/outputs다.
    write=True이고 모든 셀이 성공한 경우에만 원본 파일 바이트를 변경한다.
    커널 실행·설정 파일은 ROOT/.cache 아래 임시 디렉터리에 둔다.
    호출자 환경은 변경하지 않는다. 노트북 오류·시간 초과·파일 오류는
    그대로 전달하며 raises-exception 표기가 있는 셀도 성공해야 한다.
    반환한 셀에는 이번 실행 출력과 안정된 원래 ID가 포함된다.
    """
    source = Path(path).resolve()	# 실행할 노트북의 실제 절대 경로를 확정한다.
    if source.suffix != ".ipynb":
        raise ValueError("path must identify an .ipynb notebook")
    notebook = nbformat.read(source, as_version=4)	# 원본은 먼저 메모리로 읽고 파일에 바로 쓰지 않는다.
    _clear_execution(notebook)	# 새 검증을 위해 기존 출력과 실행 기록을 초기화한다.
    destination = (ROOT / "outputs") if output_dir is None else Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    temporary_root = ROOT / ".cache"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="notebook-kernel-", dir=temporary_root) as directory:	# 커널 관련 임시 설정은 실행 뒤 자동으로 정리한다.
        temporary = Path(directory)
        kernels = temporary / "kernels"
        kernel = kernels / "a2_1"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(	# 현재 인터프리터를 가리키는 격리된 커널 명세를 만든다.
            json.dumps({
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}",
                         "--HistoryManager.hist_file=:memory:"],
                "display_name": "A2_1 isolated execution", "language": "python",
            }), encoding="utf-8",
        )
        environment = os.environ.copy()	# 환경 사본만 수정하여 호출자의 프로세스 환경을 보존한다.
        environment.update({	# 그림·산출물·런타임 설정을 이번 실행 경로로 분리한다.
            "A2_1_OUTPUT_DIR": str(destination), "MPLBACKEND": "Agg",
            "IPYTHONDIR": str(temporary / "ipython"),
            "JUPYTER_RUNTIME_DIR": str(temporary / "runtime"),
            "MPLCONFIGDIR": str(temporary / "matplotlib"),
        })
        manager = AsyncKernelManager(	# 설치된 전역 커널 대신 임시 명세를 사용한다.
            kernel_name="a2_1",
            connection_file=str(temporary / "connection.json"),
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]),
        )
        NotebookClient(	# 새 커널에서 모든 셀을 순서대로 실행하고 오류를 강제로 전달한다.
            notebook, km=manager, timeout=120, startup_timeout=60,
            record_timing=False, allow_errors=False, force_raise_errors=True,
            resources={"metadata": {"path": str(ROOT)}},
        ).execute(cleanup_kc=True, cwd=str(ROOT), env=environment)
    for cell in notebook.cells:
        cell.metadata.pop("execution", None)	# 재현 결과에 비결정적인 실행 시간 메타데이터가 남지 않게 한다.
    if write:	# 명시적으로 요청한 경우에만 성공한 노트북을 저장한다.
        source.write_text(nbformat.writes(notebook) + "\n", encoding="utf-8", newline="\n")	# 모든 셀 성공 후 원래 셀 ID와 이번 출력을 함께 기록한다.
    return notebook


def main(argv: list[str] | None = None) -> int:
    """지정한 경로나 모든 프로젝트 노트북을 실행하며 --write는 성공한 실행을 저장한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--write", action="store_true", help="save successful executed notebooks")
    arguments = parser.parse_args(argv)	# 호출자가 제공한 명령행 인자를 해석한다.
    paths = arguments.paths or sorted((ROOT / "notebooks").glob("*.ipynb"))	# 경로를 생략하면 프로젝트 노트북 목록을 정렬해 실행한다.
    if not paths:
        parser.error("no notebooks found")
    for path in paths:
        notebook = execute_notebook(path, arguments.output_dir, arguments.write)	# 각 노트북을 별도의 새 커널에서 검증한다.
        count = sum(cell.cell_type == "code" and bool(cell.source.strip()) for cell in notebook.cells)	# 실제로 실행 대상인 비어 있지 않은 코드 셀 수를 보고한다.
        print(f"Executed {path.name}: {count} code cells passed; "
              f"{'saved' if arguments.write else 'source unchanged'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())	# 명령행 실행 결과를 프로세스 종료 코드로 전달한다.
