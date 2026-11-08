"""Independent fixture checks for bounded registry discovery and validation."""

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
    """Exercise source indexes using isolated repositories with known content."""

    def setUp(self):
        """Create a temporary repository and a small annotated source fixture."""
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
        """Remove only this test's temporary repository."""
        self.temporary.cleanup()

    def write(self, relative, content):
        """Write a UTF-8 fixture with stable LF line endings."""
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")

    def test_forward_and_reverse_locations(self):
        """Compare exact names, lines, dependencies, hashes, and test links."""
        registry = build_registry(self.root)
        entry = registry["forward"]["src/math.py"]
        assert entry["sha256"] == hashlib.sha256(self.source.encode()).hexdigest()
        assert [(item["name"], item["kind"], item["line"]) for item in entry["symbols"]] == [
            ("softmax", "function", 5), ("Model", "class", 8), ("Model.fit", "method", 9)
        ]
        assert entry["dependencies"] == [".helpers", "numpy"]
        assert entry["tests"] == ["tests/test_math.py"]
        assert registry["reverse"]["Softmax"] == [{
            "path": "src/math.py", "symbol": "softmax", "line": 5,
            "dependencies": [".helpers", "numpy"], "tests": ["tests/test_math.py"],
        }]
        assert [item["symbol"] for item in registry["reverse"]["Estimator"]] == ["Model", "Model.fit"]

    def test_unannotated_symbols_and_bounded_discovery(self):
        """Include ordinary definitions while excluding other roots and caches."""
        self.write("scripts/tool.py", "async def work():\n    return 1\n")
        for path in (".venv/bad.py", "src/.cache/bad.py", "src/__pycache__/bad.py", "misc/bad.py", "notebooks/bad.py"):
            self.write(path, "invalid syntax!")
        registry = build_registry(self.root)
        assert set(registry["forward"]) == {"scripts/tool.py", "src/math.py", "tests/test_math.py"}
        assert registry["forward"]["scripts/tool.py"]["symbols"][0]["kind"] == "async_function"

    def test_deterministic_update_and_byte_hash(self):
        """Regeneration is byte-identical and raw newline changes invalidate hashes."""
        path = write_registry(self.root)
        first = path.read_bytes()
        write_registry(self.root)
        assert path.read_bytes() == first
        assert check_registry(self.root) == []
        (self.root / "src/math.py").write_bytes(self.source.replace("\n", "\r\n").encode())
        assert any("src/math.py" in message for message in check_registry(self.root))

    def test_changed_added_and_deleted_files_are_detected(self):
        """Detect hash edits and scoped file additions/removals, then recover."""
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
        assert check_registry(self.root) == []

    def test_missing_corrupt_and_inconsistent_registry(self):
        """Reject missing JSON, malformed JSON, and a tampered reverse index."""
        assert "Missing" in check_registry(self.root)[0]
        path = write_registry(self.root)
        path.write_text("{broken", encoding="utf-8")
        assert "Invalid" in check_registry(self.root)[0]
        registry = build_registry(self.root)
        registry["reverse"]["Softmax"][0]["line"] = 999
        write_registry(self.root, registry)
        assert any("reverse" in message for message in check_registry(self.root))

    def test_annotations_must_name_real_symbols_and_targets(self):
        """Do not silently index nonexistent concepts, files, or dynamic metadata."""
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
        """Index markdown formulas and code symbols by cell ID and title."""
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
        assert code["cell_id"] == "loss-code" and code["line"] == 3
        assert code["cell_title"] == "BCE check"
        assert code["dependencies"] == ["numpy"]
        assert code["tests"] == ["tests/test_notebook.py"]
        prose = next(item for item in locations if item["symbol"] is None)
        assert prose["cell_id"] == "loss-theory" and prose["cell_title"] == "BCE derivation"
        for bad_id in (None, "loss-theory"):
            notebook["cells"][1]["id"] = bad_id
            self.write("notebooks/loss.ipynb", json.dumps(notebook))
            with self.assertRaises(ValueError):
                build_registry(self.root)

    def test_cli_exit_codes(self):
        """The CLI distinguishes missing/stale indexes from invalid source."""
        script = Path(__file__).resolve().parents[1] / "scripts/update_registry.py"
        command = [sys.executable, str(script), "--root", str(self.root)]
        missing = subprocess.run(command + ["--check"], capture_output=True, text=True)
        assert missing.returncode == 1 and "Missing" in missing.stderr
        updated = subprocess.run(command + ["--update"], capture_output=True, text=True)
        assert updated.returncode == 0, updated.stderr
        checked = subprocess.run(command + ["--check"], capture_output=True, text=True)
        assert checked.returncode == 0, checked.stderr
        self.write("src/math.py", "not valid Python!")
        invalid = subprocess.run(command + ["--check"], capture_output=True, text=True)
        assert invalid.returncode == 2 and "Registry error" in invalid.stderr


if __name__ == "__main__":
    unittest.main()
