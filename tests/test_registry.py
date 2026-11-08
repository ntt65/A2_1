"""독립 검증 자료로 레지스트리의 제한된 탐색 범위와 정합성 검사를 확인한다."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.update_registry import build_registry, check_registry, write_registry


__covers__ = ["scripts/update_registry.py"]


class RegistryTests(unittest.TestCase):
    """내용이 알려진 격리 저장소를 사용해 소스 색인을 검증한다."""

    def setUp(self):
        """임시 저장소와 개념 주석이 붙은 작은 합성 소스 자료를 만든다."""
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = (
            "import numpy as np\n"
            "from .helpers import helper\n"
            "__concepts__ = {'Softmax': ['softmax'], 'Estimator': ['Model', 'Model.fit']}\n"
            "\n"
            "def softmax(x):\n"
            "    return x\n"
            "\n"
            "class Model:\n"
            "    def fit(self):\n"
            "        return None\n"
        )
        self.write("src/math.py", self.source)
        self.write("tests/test_math.py", "__covers__ = ['src/math.py']\n")

    def tearDown(self):
        """이 테스트가 만든 임시 저장소만 정리한다."""
        self.temporary.cleanup()

    def write(self, relative, content):
        """안정적인 LF 줄바꿈의 UTF-8 검증 자료를 기록한다."""
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")

    def test_forward_and_reverse_locations(self):
        """이름·줄 위치·의존성·해시·테스트 연결을 정확한 기대값과 비교한다."""
        registry = build_registry(self.root)
        entry = registry["forward"]["src/math.py"]
        assert entry["sha256"] == hashlib.sha256(self.source.encode()).hexdigest()	# 문자열 검증 자료 자체의 SHA-256과 색인 해시를 정확히 비교한다.
        assert [(item["name"], item["kind"], item["line"]) for item in entry["symbols"]] == [
            ("softmax", "function", 5), ("Model", "class", 8), ("Model.fit", "method", 9)
        ]	# 합성 소스의 고정 줄 5·8·9를 바꾸지 않고 위치를 대조한다.
        assert entry["dependencies"] == [".helpers", "numpy"]
        assert entry["tests"] == ["tests/test_math.py"]
        assert registry["reverse"]["Softmax"] == [{
            "path": "src/math.py", "symbol": "softmax", "line": 5,
            "dependencies": [".helpers", "numpy"], "tests": ["tests/test_math.py"],
        }]	# 정방향과 역방향의 경로·심볼·줄·의존성·테스트 연결을 맞춘다.
        assert [item["symbol"] for item in registry["reverse"]["Estimator"]] == ["Model", "Model.fit"]

    def test_unannotated_symbols_and_bounded_discovery(self):
        """일반 정의를 포함하되 다른 루트와 캐시는 탐색에서 제외하는지 확인한다."""
        self.write("scripts/tool.py", "async def work():\n    return 1\n")
        for path in (".venv/bad.py", "src/.cache/bad.py", "src/__pycache__/bad.py", "misc/bad.py", "notebooks/bad.py"):
            self.write(path, "invalid syntax!")
        registry = build_registry(self.root)
        assert set(registry["forward"]) == {"scripts/tool.py", "src/math.py", "tests/test_math.py"}	# 허용한 세 경로만 탐색하여 다른 루트와 캐시를 제외한다.
        assert registry["forward"]["scripts/tool.py"]["symbols"][0]["kind"] == "async_function"

    def test_deterministic_update_and_byte_hash(self):
        """재생성이 바이트 단위로 같고 원시 줄바꿈 변경은 해시를 무효화하는지 확인한다."""
        path = write_registry(self.root)
        first = path.read_bytes()
        write_registry(self.root)
        assert path.read_bytes() == first	# 같은 입력으로 레지스트리를 다시 만들면 바이트가 정확히 같다.
        assert check_registry(self.root) == []
        (self.root / "src/math.py").write_bytes(self.source.replace("\n", "\r\n").encode())
        assert any("src/math.py" in message for message in check_registry(self.root))	# 논리적으로 같은 코드라도 원시 줄바꿈이 바뀌면 해시 변경을 검출한다.

    def test_changed_added_and_deleted_files_are_detected(self):
        """해시 변경과 범위 내 파일 추가·삭제를 검출하고 색인을 다시 맞추는지 확인한다."""
        write_registry(self.root)
        self.write("src/math.py", self.source + "# edited\n")
        assert check_registry(self.root)
        write_registry(self.root)
        self.write("src/new.py", "def added():\n    pass\n")
        assert any("src/new.py" in message for message in check_registry(self.root))
        write_registry(self.root)
        (self.root / "src/new.py").unlink()
        assert any("src/new.py" in message for message in check_registry(self.root))
        write_registry(self.root)
        assert check_registry(self.root) == []	# 추가·삭제를 반영해 갱신한 뒤에는 오래된 항목이 없어야 한다.

    def test_missing_corrupt_and_inconsistent_registry(self):
        """누락된 JSON·잘못된 JSON과 변조한 역방향 색인을 거부하는지 확인한다."""
        assert "Missing" in check_registry(self.root)[0]
        path = write_registry(self.root)
        path.write_text("{broken", encoding="utf-8")
        assert "Invalid" in check_registry(self.root)[0]
        registry = build_registry(self.root)
        registry["reverse"]["Softmax"][0]["line"] = 999
        write_registry(self.root, registry)
        assert any("reverse" in message for message in check_registry(self.root))	# 역방향 줄 위치만 변조해도 정합성 오류를 검출한다.

    def test_annotations_must_name_real_symbols_and_targets(self):
        """존재하지 않는 개념·파일·동적 메타데이터를 조용히 색인에 넣지 않는지 확인한다."""
        invalid_sources = [
            "__concepts__={'Absent':['missing']}\n",
            "__concepts__={'Empty':[]}\n",
            "__concepts__=dict()\n",
        ]
        for source in invalid_sources:
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.write("src/math.py", source)
                build_registry(self.root)
        self.write("src/math.py", self.source)
        for target in ("src/missing.py", "../outside.py", "src\\math.py"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.write("tests/test_math.py", f"__covers__ = [{target!r}]\n")
                build_registry(self.root)

    def test_notebook_cells_have_stable_concept_locations(self):
        """마크다운 수식과 코드 심볼을 안정적인 셀 ID·제목으로 색인하는지 확인한다."""
        notebook = {"cells": [
            {"id": "loss-theory", "cell_type": "markdown", "source": ["# BCE derivation\n", "L=-log(p)"],
             "metadata": {"concepts": ["BCE_Loss"]}},
            {"id": "loss-code", "cell_type": "code", "source": "import numpy as np\n__concepts__={'BCE_Loss':['loss']}\ndef loss(p):\n    return -np.log(p)\n",
             "metadata": {"title": "BCE check"}},
        ]}
        self.write("notebooks/loss.ipynb", json.dumps(notebook))
        self.write("tests/test_notebook.py", "__covers__=['notebooks/loss.ipynb']\n")
        registry = build_registry(self.root)
        locations = registry["reverse"]["BCE_Loss"]
        assert len(locations) == 2
        code = next(item for item in locations if item["symbol"] == "loss")
        assert code["cell_id"] == "loss-code" and code["line"] == 3	# 노트북 코드는 파일 줄 대신 안정적 셀 ID와 셀 내부 줄로 찾는다.
        assert code["cell_title"] == "BCE check"
        assert code["dependencies"] == ["numpy"]
        assert code["tests"] == ["tests/test_notebook.py"]
        prose = next(item for item in locations if item["symbol"] is None)
        assert prose["cell_id"] == "loss-theory" and prose["cell_title"] == "BCE derivation"	# 설명 셀도 고정 ID와 제목으로 근거 위치를 보존한다.
        for bad_id in (None, "loss-theory"):
            notebook["cells"][1]["id"] = bad_id
            self.write("notebooks/loss.ipynb", json.dumps(notebook))
            with self.assertRaises(ValueError):
                build_registry(self.root)

    def test_cli_exit_codes(self):
        """CLI가 누락·오래된 색인과 잘못된 소스를 서로 다른 종료 코드로 구별하는지 확인한다."""
        script = Path(__file__).resolve().parents[1] / "scripts/update_registry.py"
        command = [sys.executable, str(script), "--root", str(self.root)]
        missing = subprocess.run(command + ["--check"], capture_output=True, text=True)
        assert missing.returncode == 1 and "Missing" in missing.stderr	# 누락된 색인의 종료 코드 1과 메시지를 구별한다.
        updated = subprocess.run(command + ["--update"], capture_output=True, text=True)
        assert updated.returncode == 0, updated.stderr
        checked = subprocess.run(command + ["--check"], capture_output=True, text=True)
        assert checked.returncode == 0, checked.stderr
        self.write("src/math.py", "not valid Python!")
        invalid = subprocess.run(command + ["--check"], capture_output=True, text=True)
        assert invalid.returncode == 2 and "Registry error" in invalid.stderr	# 잘못된 Python 소스는 색인 누락과 다른 종료 코드 2를 반환한다.


if __name__ == "__main__":
    unittest.main()
