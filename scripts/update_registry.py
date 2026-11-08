"""Build and check the bounded, deterministic source registry.

Python files in src/, tests/, and scripts/ are parsed without importing them.
Module-level ``__concepts__ = {concept: [qualified_symbol, ...]}`` links formulas
to symbols; test modules may declare ``__covers__ = [repository_path, ...]``.
Notebook cells require stable IDs. Their metadata may contain a ``title`` and
``concepts`` (a list of strings); code cells may also declare __concepts__.
Notebook code must be valid Python (IPython magics are not parsed).

Hashes cover raw file bytes, including notebook outputs. Line numbers in code
cells are relative to that cell. Symlinks and hidden/cache directories are not
followed. No source is executed, and the output contains no timestamps.
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
    """Return only supported files from explicit roots, without following links."""
    result = []
    for folder in (*PYTHON_ROOTS, "notebooks"):
        start = root / folder
        if not start.is_dir() or start.is_symlink():
            continue
        suffix = ".ipynb" if folder == "notebooks" else ".py"
        for directory, subdirs, files in os.walk(start, followlinks=False):
            subdirs[:] = sorted(
                name for name in subdirs
                if not name.startswith(".")
                and name not in SKIP_DIRECTORIES
                and not (Path(directory) / name).is_symlink()
            )
            for name in sorted(files):
                path = Path(directory) / name
                if path.suffix == suffix and not path.is_symlink():
                    result.append(path)
    return sorted(result, key=lambda path: path.relative_to(root).as_posix())


def _string_list(value: Any, label: str) -> list[str]:
    """Validate a metadata list of nonempty strings and remove duplicates."""
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{label} must be a list of nonempty strings")
    return sorted(set(value))


def _metadata(tree: ast.Module, name: str, default: Any) -> Any:
    """Read one top-level literal metadata assignment without executing source."""
    matches = []
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            matches.append(node.value)
    if len(matches) > 1:
        raise ValueError(f"{name} must be assigned at most once")
    if not matches:
        return default
    try:
        return ast.literal_eval(matches[0])
    except (ValueError, TypeError, SyntaxError) as error:
        raise ValueError(f"{name} must be a literal value") from error


class _SymbolVisitor(ast.NodeVisitor):
    """Collect lexical functions, classes, and methods with qualified names."""

    def __init__(self) -> None:
        """Start a visitor with no enclosing symbols."""
        self.symbols: list[dict[str, Any]] = []
        self.parents: list[tuple[str, str]] = []

    def _visit_symbol(self, node: Any, kind: str) -> None:
        """Record one definition and recursively visit its lexical children."""
        name = ".".join([parent[0] for parent in self.parents] + [node.name])
        if kind in {"function", "async_function"} and self.parents and self.parents[-1][1] == "class":
            kind = "async_method" if kind == "async_function" else "method"
        self.symbols.append({
            "name": name, "kind": kind, "line": node.lineno,
            "end_line": node.end_lineno, "concepts": [],
        })
        self.parents.append((node.name, kind))
        self.generic_visit(node)
        self.parents.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a synchronous function or method definition."""
        self._visit_symbol(node, "function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit an asynchronous function or method definition."""
        self._visit_symbol(node, "async_function")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class and its member definitions."""
        self._visit_symbol(node, "class")


def _python_entry(source: str, label: str) -> dict[str, Any]:
    """Parse Python symbols, imports, and explicit metadata from one source."""
    tree = ast.parse(source, filename=label)
    visitor = _SymbolVisitor()
    visitor.visit(tree)
    concepts = _metadata(tree, "__concepts__", {})
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
            if name not in by_name:
                raise ValueError(f"{label}: concept {concept} names missing symbol {name}")
            by_name[name]["concepts"].append(concept)
    dependencies = set()
    for node in ast.walk(tree):
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
    """Read the string or line-list form of a notebook cell's source."""
    source = cell.get("source", "")
    if isinstance(source, list) and all(isinstance(line, str) for line in source):
        return "".join(source)
    if isinstance(source, str):
        return source
    raise ValueError(f"{label}: cell source must be a string or a list of strings")


def _notebook_entry(raw: bytes, label: str) -> dict[str, Any]:
    """Index notebook symbols and concepts using stable cell IDs and titles."""
    notebook = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise ValueError(f"{label}: notebook must contain a cells list")
    symbols, cells, concepts, dependencies = [], [], set(), set()
    ids = set()
    for cell in notebook["cells"]:
        if not isinstance(cell, dict):
            raise ValueError(f"{label}: every cell must be an object")
        cell_id = cell.get("id")
        if not isinstance(cell_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cell_id) or cell_id in ids:
            raise ValueError(f"{label}: every cell requires a unique stable id")
        ids.add(cell_id)
        cell_type = cell.get("cell_type")
        if cell_type not in {"code", "markdown", "raw"}:
            raise ValueError(f"{label}: unsupported cell type {cell_type!r}")
        metadata = cell.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"{label}: cell metadata must be an object")
        source = _cell_source(cell, label)
        heading = next((line.lstrip("# ").strip() for line in source.splitlines() if line.startswith("#")), "")
        title = metadata.get("title", heading or f"Cell {cell_id}")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"{label}: cell title must be a nonempty string")
        cell_concepts = _string_list(metadata.get("concepts", []), f"{label}:{cell_id} concepts")
        cells.append({"id": cell_id, "title": title, "cell_type": cell_type, "concepts": cell_concepts})
        concepts.update(cell_concepts)
        if cell_type == "code":
            entry = _python_entry(source, f"{label}:{cell_id}")
            if entry["covers"]:
                raise ValueError(f"{label}: __covers__ is reserved for Python test files")
            for symbol in entry["symbols"]:
                symbol.update({"cell_id": cell_id, "cell_title": title})
            symbols.extend(entry["symbols"])
            concepts.update(entry["concepts"])
            dependencies.update(entry["dependencies"])
    return {"symbols": symbols, "dependencies": sorted(dependencies),
            "concepts": sorted(concepts), "covers": [], "cells": cells}


def build_registry(root: Path | str) -> dict[str, Any]:
    """Build both indexes from bounded source roots; reject invalid annotations.

    The result maps ``forward[file]`` to its raw-byte hash, symbols, imported
    module names, concepts, covered targets, and related test paths.
    ``reverse[concept]`` contains locations with the same dependency/test links.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Repository root does not exist: {root}")
    forward = {}
    for path in _scoped_files(root):
        relative = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        entry = _notebook_entry(raw, relative) if path.suffix == ".ipynb" else _python_entry(raw.decode("utf-8-sig"), relative)
        entry.update({"sha256": hashlib.sha256(raw).hexdigest(), "tests": [],
                      "language": "notebook" if path.suffix == ".ipynb" else "python"})
        forward[relative] = entry
    for test_path, entry in forward.items():
        if entry["covers"] and not test_path.startswith("tests/"):
            raise ValueError(f"{test_path}: __covers__ is reserved for tests/")
        for target in entry["covers"]:
            normalized = PurePosixPath(target)
            if normalized.is_absolute() or ".." in normalized.parts or "\\" in target or normalized.as_posix() != target or target not in forward:
                raise ValueError(f"{test_path}: covered target does not exist in scope: {target}")
            forward[target]["tests"].append(test_path)
    reverse: dict[str, list[dict[str, Any]]] = {}
    for path, entry in forward.items():
        entry["tests"].sort()
        common = {"path": path, "dependencies": entry["dependencies"], "tests": entry["tests"]}
        for symbol in entry["symbols"]:
            location = {**common, "symbol": symbol["name"], "line": symbol["line"]}
            if "cell_id" in symbol:
                location.update({"cell_id": symbol["cell_id"], "cell_title": symbol["cell_title"]})
            for concept in symbol["concepts"]:
                reverse.setdefault(concept, []).append(location)
        for cell in entry.get("cells", []):
            location = {**common, "symbol": None, "cell_id": cell["id"], "cell_title": cell["title"]}
            for concept in cell["concepts"]:
                reverse.setdefault(concept, []).append(location)
    return {"schema_version": SCHEMA_VERSION,
            "scope": {"python": list(PYTHON_ROOTS), "notebooks": ["notebooks"]},
            "forward": forward, "reverse": dict(sorted(reverse.items()))}


def write_registry(root: Path | str, registry: dict[str, Any] | None = None) -> Path:
    """Atomically replace .agent/registry.json with a deterministic UTF-8 index."""
    root = Path(root).resolve()
    if registry is None:
        registry = build_registry(root)
    target = root / REGISTRY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=target.parent,
                                         prefix="registry-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(registry, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def check_registry(root: Path | str) -> list[str]:
    """Return actionable differences; an empty list means both indexes match."""
    root = Path(root).resolve()
    expected = build_registry(root)
    target = root / REGISTRY_PATH
    try:
        actual = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ["Missing .agent/registry.json; run --update."]
    except (json.JSONDecodeError, UnicodeError):
        return ["Invalid registry JSON; rebuild with --update."]
    if actual == expected:
        return []
    if not isinstance(actual, dict):
        return ["Registry must be an object; rebuild with --update."]
    problems = []
    for name in ("schema_version", "scope", "reverse"):
        if actual.get(name) != expected[name]:
            problems.append(f"{name} differs from current sources.")
    observed = actual.get("forward", {})
    if not isinstance(observed, dict):
        problems.append("forward must be an object.")
    else:
        for path in sorted(set(observed) | set(expected["forward"])):
            if observed.get(path) != expected["forward"].get(path):
                problems.append(f"Missing, stale, or obsolete file entry: {path}")
    if not problems:
        problems.append("Unexpected registry fields differ from the schema.")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Run --update or --check, returning nonzero for stale or invalid input."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--update", action="store_true", help="rebuild the scoped registry")
    modes.add_argument("--check", action="store_true", help="fail if the registry is stale")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        if args.update:
            registry = build_registry(args.root)
            write_registry(args.root, registry)
            print(f"Registry updated: {len(registry['forward'])} files, {len(registry['reverse'])} concepts.")
            return 0
        problems = check_registry(args.root)
        if problems:
            print("Registry check failed:\n- " + "\n- ".join(problems), file=sys.stderr)
            return 1
        print("Registry check passed.")
        return 0
    except (OSError, ValueError, SyntaxError) as error:
        print(f"Registry error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
