"""탐색 범위를 제한한 결정론적 소스 레지스트리를 생성하고 검사한다.

src/, tests/, scripts/의 Python 파일은 import하지 않고 구문 분석한다.
모듈 수준 __concepts__ = {concept: [qualified_symbol, ...]}은 수식과
심볼을 연결하며 테스트 모듈은 __covers__ = [repository_path, ...]를
선언할 수 있다. 노트북 셀에는 안정된 ID가 필요하다. 셀 메타데이터에는
title과 문자열 목록인 concepts를 둘 수 있고 코드 셀에서도
__concepts__를 선언할 수 있다. 노트북 코드는 유효한 Python이어야 하며
IPython 매직 문법은 구문 분석하지 않는다.

해시는 노트북 출력을 포함한 파일 원시 바이트 전체를 대상으로 한다.
코드 셀의 줄 번호는 해당 셀 안의 상대 위치다. 심볼릭 링크와 숨김·캐시
디렉터리는 따라가지 않는다. 소스를 실행하지 않으며 출력에는 타임스탬프가 없다.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from typing import Any


SCHEMA_VERSION = 1
PYTHON_ROOTS = ("src", "tests", "scripts")
SKIP_DIRECTORIES = {"__pycache__", "venv", "env", "node_modules"}
REGISTRY_PATH = Path(".agent/registry.json")
__concepts__ = {"Registry": ["build_registry", "check_registry", "write_registry"]}


def _scoped_files(root: Path) -> list[Path]:
    """명시한 루트에서 지원 파일만 반환하며 심볼릭 링크는 따라가지 않는다."""
    result = []
    for folder in (*PYTHON_ROOTS, "notebooks"):	# 허용된 소스 루트와 노트북 디렉터리만 탐색한다.
        start = root / folder
        if not start.is_dir() or start.is_symlink():	# 루트가 없거나 심볼릭 링크이면 탐색하지 않는다.
            continue
        suffix = ".ipynb" if folder == "notebooks" else ".py"
        for directory, subdirs, files in os.walk(start, followlinks=False):	# 디렉터리 링크를 따라가며 범위 밖으로 이동하지 않는다.
            subdirs[:] = sorted(	# 숨김·캐시·가상환경·링크 디렉터리를 탐색 목록에서 제외한다.
                name for name in subdirs
                if not name.startswith(".")
                and name not in SKIP_DIRECTORIES
                and not (Path(directory) / name).is_symlink()
            )
            for name in sorted(files):
                path = Path(directory) / name
                if path.suffix == suffix and not path.is_symlink():
                    result.append(path)
    return sorted(result, key=lambda path: path.relative_to(root).as_posix())	# 실행 순서와 무관한 안정된 파일 순서를 반환한다.


def _string_list(value: Any, label: str) -> list[str]:
    """비어 있지 않은 문자열의 메타데이터 목록을 검증하고 중복을 제거한다."""
    if not isinstance(value, list) or any(	# 메타데이터가 빈 항목 없는 문자열 목록인지 검증한다.
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{label} must be a list of nonempty strings")
    return sorted(set(value))	# 중복을 제거한 뒤 결정론적으로 정렬한다.


def _metadata(tree: ast.Module, name: str, default: Any) -> Any:
    """소스를 실행하지 않고 최상위의 리터럴 메타데이터 대입 하나를 읽는다."""
    matches = []
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):	# 요청한 이름의 최상위 대입만 수집한다.
            matches.append(node.value)
    if len(matches) > 1:	# 여러 번 대입된 메타데이터는 의미가 모호하므로 거부한다.
        raise ValueError(f"{name} must be assigned at most once")
    if not matches:
        return default
    try:
        return ast.literal_eval(matches[0])	# 일반 eval 대신 리터럴만 해석하여 소스 실행을 막는다.
    except (ValueError, TypeError, SyntaxError) as error:
        raise ValueError(f"{name} must be a literal value") from error


class _SymbolVisitor(ast.NodeVisitor):
    """렉시컬 범위의 함수·클래스·메서드를 한정된 전체 이름으로 수집한다."""

    def __init__(self) -> None:
        """상위 심볼이 없는 상태로 방문자를 초기화한다."""
        self.symbols: list[dict[str, Any]] = []
        self.parents: list[tuple[str, str]] = []

    def _visit_symbol(self, node: Any, kind: str) -> None:
        """정의 하나를 기록하고 그 렉시컬 하위 정의를 재귀적으로 방문한다."""
        name = ".".join([parent[0] for parent in self.parents] + [node.name])	# 중첩 클래스·함수의 전체 이름으로 심볼 충돌을 구분한다.
        if kind in {"function", "async_function"} and self.parents and self.parents[-1][1] == "class":	# 클래스에 직접 속한 함수를 메서드로 분류한다.
            kind = "async_method" if kind == "async_function" else "method"
        self.symbols.append({
            "name": name, "kind": kind, "line": node.lineno,
            "end_line": node.end_lineno, "concepts": [],
        })
        self.parents.append((node.name, kind))	# 하위 정의를 방문하는 동안 상위 범위를 보존한다.
        self.generic_visit(node)
        self.parents.pop()	# 현재 정의 방문을 마친 뒤 상위 범위를 복원한다.

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """동기 함수나 메서드 정의를 방문한다."""
        self._visit_symbol(node, "function")	# 동기 함수 정의를 공통 수집기로 전달한다.

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """비동기 함수나 메서드 정의를 방문한다."""
        self._visit_symbol(node, "async_function")	# 비동기 함수 정의를 공통 수집기로 전달한다.

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """클래스와 그 멤버 정의를 방문한다."""
        self._visit_symbol(node, "class")	# 클래스 정의와 그 하위 심볼을 공통 수집기로 전달한다.


def _python_entry(source: str, label: str) -> dict[str, Any]:
    """소스 하나에서 Python 심볼·import·명시적 메타데이터를 구문 분석한다."""
    tree = ast.parse(source, filename=label)	# 파일을 import하거나 실행하지 않고 AST만 만든다.
    visitor = _SymbolVisitor()
    visitor.visit(tree)
    concepts = _metadata(tree, "__concepts__", {})	# 개념과 실제 심볼의 명시적 연결 정보를 읽는다.
    if not isinstance(concepts, dict) or any(
        not isinstance(key, str) or not key.strip() for key in concepts
    ):
        raise ValueError(f"{label}: __concepts__ must map concept names to symbol lists")
    by_name = {symbol["name"]: symbol for symbol in visitor.symbols}
    for concept, value in sorted(concepts.items()):
        names = _string_list(value, f"{label}: concept {concept}")
        if not names:
            raise ValueError(f"{label}: concept {concept} has no symbols")
        for name in names:
            if name not in by_name:	# 존재하지 않는 심볼을 구현된 것처럼 등록하지 않는다.
                raise ValueError(f"{label}: concept {concept} names missing symbol {name}")
            by_name[name]["concepts"].append(concept)
    dependencies = set()
    for node in ast.walk(tree):	# 실행 없이 AST의 import 구문에서 의존성을 수집한다.
        if isinstance(node, ast.Import):
            dependencies.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            if node.module:
                dependencies.add(prefix + node.module)
            else:
                dependencies.update(prefix + alias.name for alias in node.names)
    return {
        "symbols": visitor.symbols,
        "dependencies": sorted(dependencies),
        "concepts": sorted(concepts),
        "covers": _string_list(_metadata(tree, "__covers__", []), f"{label}: __covers__"),
    }


def _cell_source(cell: dict[str, Any], label: str) -> str:
    """노트북 셀 원문의 문자열 또는 줄 문자열 목록 표현을 읽는다."""
    source = cell.get("source", "")
    if isinstance(source, list) and all(isinstance(line, str) for line in source):
        return "".join(source)	# 목록 형태의 셀 원문을 문자 변경 없이 결합한다.
    if isinstance(source, str):
        return source
    raise ValueError(f"{label}: cell source must be a string or a list of strings")


def _notebook_entry(raw: bytes, label: str) -> dict[str, Any]:
    """안정된 셀 ID와 제목을 사용해 노트북 심볼·개념을 색인한다."""
    notebook = json.loads(raw.decode("utf-8-sig"))	# UTF-8 BOM을 허용하여 노트북 JSON을 읽는다.
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise ValueError(f"{label}: notebook must contain a cells list")
    symbols, cells, concepts, dependencies = [], [], set(), set()
    ids = set()
    for cell in notebook["cells"]:
        if not isinstance(cell, dict):
            raise ValueError(f"{label}: every cell must be an object")
        cell_id = cell.get("id")
        if not isinstance(cell_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cell_id) or cell_id in ids:	# 유효하고 고유한 셀 ID를 반드시 요구한다.
            raise ValueError(f"{label}: every cell requires a unique stable id")
        ids.add(cell_id)
        cell_type = cell.get("cell_type")
        if cell_type not in {"code", "markdown", "raw"}:
            raise ValueError(f"{label}: unsupported cell type {cell_type!r}")
        metadata = cell.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"{label}: cell metadata must be an object")
        source = _cell_source(cell, label)	# 셀 원문의 두 직렬화 표현을 공통 문자열로 읽는다.
        heading = next((line.lstrip("# ").strip() for line in source.splitlines() if line.startswith("#")), "")
        title = metadata.get("title", heading or f"Cell {cell_id}")	# 명시적 제목을 우선하고 없으면 첫 제목이나 셀 ID를 사용한다.
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"{label}: cell title must be a nonempty string")
        cell_concepts = _string_list(metadata.get("concepts", []), f"{label}:{cell_id} concepts")
        cells.append({"id": cell_id, "title": title, "cell_type": cell_type, "concepts": cell_concepts})
        concepts.update(cell_concepts)
        if cell_type == "code":
            entry = _python_entry(source, f"{label}:{cell_id}")	# 코드 셀도 Python 소스와 같은 규칙으로 색인한다.
            if entry["covers"]:
                raise ValueError(f"{label}: __covers__ is reserved for Python test files")
            for symbol in entry["symbols"]:
                symbol.update({"cell_id": cell_id, "cell_title": title})	# 노트북 심볼에 안정된 셀 ID·제목을 함께 기록한다.
            symbols.extend(entry["symbols"])
            concepts.update(entry["concepts"])
            dependencies.update(entry["dependencies"])
    return {"symbols": symbols, "dependencies": sorted(dependencies),
            "concepts": sorted(concepts), "covers": [], "cells": cells}


def build_registry(root: Path | str) -> dict[str, Any]:
    """제한된 소스 루트에서 정방향·역방향 색인을 만들며 잘못된 표기는 거부한다.

    결과의 forward[file]에는 원시 바이트 해시, 심볼, import한 모듈 이름,
    개념, 검사 대상과 관련 테스트 경로를 기록한다.
    reverse[concept]에는 같은 의존성·테스트 연결을 가진 위치를 기록한다.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Repository root does not exist: {root}")
    forward = {}
    for path in _scoped_files(root):	# 허용된 파일 목록 안에서만 레지스트리를 구성한다.
        relative = path.relative_to(root).as_posix()
        raw = path.read_bytes()	# 출력까지 포함한 원시 바이트를 해시의 기준으로 읽는다.
        entry = _notebook_entry(raw, relative) if path.suffix == ".ipynb" else _python_entry(raw.decode("utf-8-sig"), relative)
        entry.update({"sha256": hashlib.sha256(raw).hexdigest(), "tests": [],	# 파일 내용과 색인 사이의 정합성을 SHA-256으로 추적한다.
                      "language": "notebook" if path.suffix == ".ipynb" else "python"})
        forward[relative] = entry
    for test_path, entry in forward.items():
        if entry["covers"] and not test_path.startswith("tests/"):	# 검사 대상 선언은 테스트 디렉터리에서만 허용한다.
            raise ValueError(f"{test_path}: __covers__ is reserved for tests/")
        for target in entry["covers"]:
            normalized = PurePosixPath(target)
            if normalized.is_absolute() or ".." in normalized.parts or "\\" in target or normalized.as_posix() != target or target not in forward:	# 절대·상위·비정규 경로나 범위 밖 검사 대상을 거부한다.
                raise ValueError(f"{test_path}: covered target does not exist in scope: {target}")
            forward[target]["tests"].append(test_path)	# 실제 대상 파일에 관련 테스트를 연결한다.
    reverse: dict[str, list[dict[str, Any]]] = {}
    for path, entry in forward.items():
        entry["tests"].sort()
        common = {"path": path, "dependencies": entry["dependencies"], "tests": entry["tests"]}	# 정방향·역방향 색인이 같은 의존성·테스트 정보를 공유한다.
        for symbol in entry["symbols"]:
            location = {**common, "symbol": symbol["name"], "line": symbol["line"]}
            if "cell_id" in symbol:
                location.update({"cell_id": symbol["cell_id"], "cell_title": symbol["cell_title"]})
            for concept in symbol["concepts"]:
                reverse.setdefault(concept, []).append(location)	# 개념에서 실제 심볼 위치를 찾는 역방향 항목을 만든다.
        for cell in entry.get("cells", []):
            location = {**common, "symbol": None, "cell_id": cell["id"], "cell_title": cell["title"]}
            for concept in cell["concepts"]:
                reverse.setdefault(concept, []).append(location)	# 코드 심볼이 없는 개념 셀도 안정된 ID로 연결한다.
    return {"schema_version": SCHEMA_VERSION,
            "scope": {"python": list(PYTHON_ROOTS), "notebooks": ["notebooks"]},
            "forward": forward, "reverse": dict(sorted(reverse.items()))}


def write_registry(root: Path | str, registry: dict[str, Any] | None = None) -> Path:
    """.agent/registry.json을 결정론적 UTF-8 색인으로 원자적으로 교체한다."""
    root = Path(root).resolve()
    if registry is None:
        registry = build_registry(root)
    target = root / REGISTRY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=target.parent,	# 목적 파일과 같은 디렉터리에 임시 색인 파일을 만든다.
                                         prefix="registry-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(registry, handle, ensure_ascii=False, indent=2, sort_keys=True)	# 키 정렬과 고정 UTF-8 표현으로 재현 가능한 JSON을 쓴다.
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())	# 교체 전에 파일 내용을 디스크에 반영한다.
        os.replace(temporary, target)	# 완성한 임시 파일로 레지스트리를 원자적으로 교체한다.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)	# 실패하거나 남은 임시 파일을 정리한다.
    return target


def check_registry(root: Path | str) -> list[str]:
    """조치 가능한 차이 목록을 반환하며 빈 목록은 두 색인이 일치한다는 뜻이다."""
    root = Path(root).resolve()
    expected = build_registry(root)	# 현재 소스에서 기대 색인을 다시 구성해 비교한다.
    target = root / REGISTRY_PATH
    try:
        actual = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ["Missing .agent/registry.json; run --update."]
    except (json.JSONDecodeError, UnicodeError):
        return ["Invalid registry JSON; rebuild with --update."]
    if actual == expected:	# 스키마·정방향·역방향 모두 같으면 문제 없음으로 반환한다.
        return []
    if not isinstance(actual, dict):
        return ["Registry must be an object; rebuild with --update."]
    problems = []
    for name in ("schema_version", "scope", "reverse"):	# 스키마·탐색 범위·역방향 연결의 차이를 구분한다.
        if actual.get(name) != expected[name]:
            problems.append(f"{name} differs from current sources.")
    observed = actual.get("forward", {})
    if not isinstance(observed, dict):
        problems.append("forward must be an object.")
    else:
        for path in sorted(set(observed) | set(expected["forward"])):	# 삭제·추가·변경 파일 모두 비교 대상에 포함한다.
            if observed.get(path) != expected["forward"].get(path):
                problems.append(f"Missing, stale, or obsolete file entry: {path}")
    if not problems:
        problems.append("Unexpected registry fields differ from the schema.")
    return problems


def main(argv: list[str] | None = None) -> int:
    """--update 또는 --check를 실행하며 오래되거나 잘못된 입력에는 0이 아닌 종료 코드를 반환한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)	# 갱신과 검사 옵션을 동시에 요청하지 못하게 한다.
    modes.add_argument("--update", action="store_true", help="rebuild the scoped registry")
    modes.add_argument("--check", action="store_true", help="fail if the registry is stale")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)	# 호출자가 제공한 명령행 인자를 해석한다.
    try:
        if args.update:
            registry = build_registry(args.root)	# 명시적 갱신 모드에서만 새 색인을 구성한다.
            write_registry(args.root, registry)	# 갱신 모드에서 검증한 색인을 실제 파일에 저장한다.
            print(f"Registry updated: {len(registry['forward'])} files, {len(registry['reverse'])} concepts.")
            return 0
        problems = check_registry(args.root)	# 검사 모드는 파일을 쓰지 않고 차이만 반환한다.
        if problems:
            print("Registry check failed:\n- " + "\n- ".join(problems), file=sys.stderr)
            return 1
        print("Registry check passed.")
        return 0
    except (OSError, ValueError, SyntaxError) as error:
        print(f"Registry error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())	# 명령행 실행 결과를 프로세스 종료 코드로 전달한다.
