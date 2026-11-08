"""Bounded local caches for explicitly selected expensive NumPy computations."""

from collections import OrderedDict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable
import zipfile

import numpy as np

__concepts__ = {"Cache_Key": ["make_cache_key"], "Cache_Manager": ["CacheManager"]}

_FORMAT_VERSION = 1
_KEY_PATTERN = re.compile(r"[0-9a-f]{64}")
_SCALAR_TYPES = (type(None), bool, int, float, str)


def _pack_value(value: Any, arrays: dict[str, np.ndarray]) -> dict:
    """Encode supported values as tagged JSON nodes and copied numeric arrays.

    Arrays preserve shape and dtype. Supported containers are lists, tuples and
    dictionaries with string keys; leaves are JSON scalars or numeric ndarrays.
    Unsupported objects and nonfinite JSON floats raise TypeError/ValueError.
    """
    if type(value) is np.ndarray:
        if value.dtype.kind not in "biufc":
            raise TypeError("cache arrays must have boolean or numeric dtype")
        name = f"array_{len(arrays)}"
        arrays[name] = np.array(value, copy=True, order="C", subok=False)
        return {"kind": "array", "name": name}
    if type(value) in _SCALAR_TYPES:
        if type(value) is float and not math.isfinite(value):
            raise ValueError("JSON scalar floats must be finite")
        return {"kind": "scalar", "value": value}
    if type(value) in (list, tuple):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "items": [_pack_value(item, arrays) for item in value],
        }
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError("cache dictionary keys must be strings")
        return {
            "kind": "dict",
            "items": [[key, _pack_value(value[key], arrays)] for key in sorted(value)],
        }
    raise TypeError(f"unsupported cache value type: {type(value).__name__}")


def _unpack_value(node: dict, archive: Any) -> Any:
    """Decode validated tagged nodes, copying numeric arrays with original shape.

    Malformed nodes or unsupported array dtypes raise TypeError/ValueError and
    are handled as cache misses by the caller.
    """
    if not isinstance(node, dict):
        raise ValueError("invalid cache node")
    kind = node.get("kind")
    if kind == "array":
        name = node["name"]
        if not isinstance(name, str) or re.fullmatch(r"array_[0-9]+", name) is None:
            raise ValueError("invalid cache array name")
        value = archive[name]
        if type(value) is not np.ndarray:
            raise ValueError("cached array member must contain an NPY array")
        if value.dtype.kind not in "biufc":
            raise ValueError("unsupported cached array dtype")
        return np.array(value, copy=True, subok=False)
    if kind == "scalar":
        value = node["value"]
        if type(value) not in _SCALAR_TYPES:
            raise ValueError("invalid cached scalar")
        if type(value) is float and not math.isfinite(value):
            raise ValueError("nonfinite cached JSON float")
        return value
    if kind in ("list", "tuple"):
        items = node["items"]
        if not isinstance(items, list):
            raise ValueError("invalid cached sequence")
        values = [_unpack_value(item, archive) for item in items]
        return tuple(values) if kind == "tuple" else values
    if kind == "dict":
        items = node["items"]
        if not isinstance(items, list):
            raise ValueError("invalid cached dictionary")
        result = {}
        for item in items:
            if not isinstance(item, list) or len(item) != 2 or type(item[0]) is not str:
                raise ValueError("invalid cached dictionary entry")
            if item[0] in result:
                raise ValueError("duplicate cached dictionary key")
            result[item[0]] = _unpack_value(item[1], archive)
        return result
    raise ValueError("unknown cache node kind")


def _json_bytes(value: Any) -> bytes:
    """Serialize metadata deterministically; reject nonfinite JSON numbers."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def make_cache_key(
    namespace: str, version: str, inputs: Any, parameters: Any = None
) -> str:
    """Return a deterministic SHA-256 key for inputs and algorithm parameters.

    ``namespace`` and ``version`` must be nonempty strings. Inputs/parameters
    support JSON values, numeric ndarrays of any shape, and nested tuples.
    Array content, shape and dtype are included; memory layout is ignored.
    Only plain ndarrays and built-in containers are accepted, not subclasses.
    Scalar/container types remain distinct. Include seed in parameters when
    the computation depends on randomness. Unsupported values raise TypeError.
    """
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace must be a nonempty string")
    if not isinstance(version, str) or not version:
        raise ValueError("version must be a nonempty string")
    arrays: dict[str, np.ndarray] = {}
    payload = {
        "format": _FORMAT_VERSION,
        "namespace": namespace,
        "version": version,
        "inputs": _pack_value(inputs, arrays),
        "parameters": _pack_value(parameters, arrays),
    }
    payload["arrays"] = {
        name: {
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
        for name, array in arrays.items()
    }
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


class CacheManager:
    """Cache supported values in bounded memory and optional safe NPZ files.

    Memory uses LRU eviction. Disk evicts the oldest saved entries, counting
    files rather than bytes; choose modest input sizes and call ``clear`` when
    needed. A zero limit disables that tier. Storage is created lazily.

    Returned values are deep copies. Numeric ndarray shapes/dtypes and tuple
    containers are preserved. Array/container subclasses, object arrays,
    arbitrary objects and nonfinite
    JSON scalar floats are rejected. Disk read/write failures are cache misses
    or skipped writes, so an unavailable cache does not prevent computation.

    Use one manager per owning thread. Atomic replacement prevents partial
    files, but this class provides no cross-process locking, single-flight
    computation, or concurrent eviction/clear guarantees. Share a directory
    across processes only with external coordination.
    """

    def __init__(
        self,
        cache_dir: str | Path = ".cache",
        max_memory_entries: int = 32,
        max_disk_entries: int = 128,
        enabled: bool = True,
    ) -> None:
        """Configure cache tiers; noninteger/negative limits raise ValueError."""
        for limit in (max_memory_entries, max_disk_entries):
            if type(limit) is not int or limit < 0:
                raise ValueError("cache entry limits must be nonnegative integers")
        if type(enabled) is not bool:
            raise TypeError("enabled must be bool")
        self.cache_dir = Path(cache_dir)
        self.max_memory_entries = max_memory_entries
        self.max_disk_entries = max_disk_entries
        self.enabled = enabled
        self._memory: OrderedDict[str, Any] = OrderedDict()

    def get_or_compute(self, key: str, compute: Callable[[], Any]) -> Any:
        """Return a deep copy of the cached result or call ``compute`` once.

        ``key`` must be a lowercase 64-character SHA-256 hex string. The callback
        takes no arguments; its exceptions propagate without caching. Disabled
        mode computes every call without reading/writing either cache tier.
        Unsupported return types raise TypeError/ValueError in either mode.
        """
        if not isinstance(key, str) or _KEY_PATTERN.fullmatch(key) is None:
            raise ValueError("key must be a lowercase SHA-256 hex digest")
        if not callable(compute):
            raise TypeError("compute must be callable")
        if self.enabled and key in self._memory:
            self._memory.move_to_end(key)
            return copy.deepcopy(self._memory[key])
        if self.enabled and self.max_disk_entries:
            found, value = self._read_disk(key)
            if found:
                self._remember(key, value)
                return copy.deepcopy(value)
        value = compute()
        _pack_value(value, {})
        owned = copy.deepcopy(value)
        if self.enabled:
            self._remember(key, owned)
            if self.max_disk_entries:
                self._write_disk(key, owned)
        return copy.deepcopy(owned)

    def _remember(self, key: str, value: Any) -> None:
        """Keep an owned value and evict least-recently-used memory entries."""
        if not self.max_memory_entries:
            return
        self._memory[key] = value
        self._memory.move_to_end(key)
        while len(self._memory) > self.max_memory_entries:
            self._memory.popitem(last=False)

    def _read_disk(self, key: str) -> tuple[bool, Any]:
        """Load one NPZ without pickle; missing, stale and malformed files miss."""
        try:
            with np.load(self.cache_dir / f"{key}.npz", allow_pickle=False) as archive:
                metadata_array = archive["__metadata__"]
                if type(metadata_array) is not np.ndarray:
                    raise ValueError("cache metadata member must contain an NPY array")
                if metadata_array.dtype != np.uint8 or metadata_array.ndim != 1:
                    return False, None
                metadata = json.loads(metadata_array.tobytes().decode("utf-8"))
                if not isinstance(metadata, dict):
                    return False, None
                if metadata.get("format") != _FORMAT_VERSION or metadata.get("key") != key:
                    return False, None
                return True, _unpack_value(metadata["value"], archive)
        except (OSError, ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile):
            return False, None

    def _entry_paths(self) -> list[Path]:
        """List only this cache's hash-named NPZ files, ignoring unrelated files."""
        return [
            path
            for path in self.cache_dir.glob("*.npz")
            if _KEY_PATTERN.fullmatch(path.stem) and path.is_file()
        ]

    def _write_disk(self, key: str, value: Any) -> None:
        """Atomically save one safe archive and evict oldest disk entries."""
        temporary_path = None
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            arrays: dict[str, np.ndarray] = {}
            metadata = {"format": _FORMAT_VERSION, "key": key, "value": _pack_value(value, arrays)}
            arrays["__metadata__"] = np.frombuffer(_json_bytes(metadata), dtype=np.uint8)
            with tempfile.NamedTemporaryFile(
                dir=self.cache_dir, prefix=f".{key}-", suffix=".tmp", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                np.savez_compressed(temporary, **arrays)
            os.replace(temporary_path, self.cache_dir / f"{key}.npz")
            paths = sorted(self._entry_paths(), key=lambda path: (path.stat().st_mtime_ns, path.name))
            for path in paths[: max(0, len(paths) - self.max_disk_entries)]:
                path.unlink()
        except OSError:
            pass
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def clear(self) -> None:
        """Clear memory and this manager's hash-named NPZ files, leaving others.

        Unlike automatic cache writes, explicit cleanup reports filesystem
        errors to the caller. No recursive deletion is performed.
        """
        self._memory.clear()
        for path in self._entry_paths():
            path.unlink()
