"""Behavior checks for content-addressed, safe local NumPy caching."""

import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from src.utils.cache_manager import CacheManager, make_cache_key

__covers__ = ["src/utils/cache_manager.py"]


class CacheKeyTests(unittest.TestCase):
    """Verify key identity is based on logical input data and explicit versions."""

    def test_determinism_and_dictionary_order(self):
        """Equal values produce 64-character digests regardless of mapping order."""
        array = np.array([[1.0, 2.0], [3.0, 4.0]])
        first = make_cache_key("svd", "1", {"image": array, "mode": "gray"}, {"k": 1, "seed": 42})
        second = make_cache_key("svd", "1", {"mode": "gray", "image": array.copy()}, {"seed": 42, "k": 1})
        assert first == second
        assert len(first) == 64
        assert int(first, 16) >= 0
        assert first == make_cache_key("svd", "1", {"image": np.asfortranarray(array), "mode": "gray"}, {"k": 1, "seed": 42})

    def test_all_computation_inputs_affect_identity(self):
        """Values, shape, dtype, namespace, version and parameters change keys."""
        array = np.array([1, 2], dtype=np.int64)
        keys = [
            make_cache_key("svd", "1", array, {"k": 1}),
            make_cache_key("svd", "1", np.array([1, 3], dtype=np.int64), {"k": 1}),
            make_cache_key("svd", "1", array.reshape(1, 2), {"k": 1}),
            make_cache_key("svd", "1", array.astype(np.float64), {"k": 1}),
            make_cache_key("mesh", "1", array, {"k": 1}),
            make_cache_key("svd", "2", array, {"k": 1}),
            make_cache_key("svd", "1", array, {"k": 2}),
        ]
        assert len(set(keys)) == len(keys)

    def test_scalar_and_container_types_are_distinct(self):
        """Python bool/int/float and list/tuple do not collide."""
        values = [True, 1, 1.0, "1", [1], (1,), np.array([1]), np.array(1)]
        keys = [make_cache_key("typed", "1", value) for value in values]
        assert len(set(keys)) == len(values)

    def test_invalid_key_inputs_fail(self):
        """Arbitrary objects and unsafe dtypes are rejected before persistence."""
        for value in [
            object(),
            np.array([object()], dtype=object),
            np.array([(1, 2.0)], dtype=[("x", "i4"), ("y", "f8")]),
            np.array(["not numeric"]),
            np.ma.array([1.0, 2.0], mask=[False, True]),
            {1: "value"},
        ]:
            with self.subTest(value=type(value).__name__), self.assertRaises(TypeError):
                make_cache_key("test", "1", value)
        with self.assertRaises(ValueError):
            make_cache_key("test", "1", float("nan"))
        for namespace, version in [("", "1"), ("test", ""), (None, "1")]:
            with self.subTest(namespace=namespace, version=version), self.assertRaises(ValueError):
                make_cache_key(namespace, version, [1])


class CacheManagerTests(unittest.TestCase):
    """Exercise memory, disk, invalidation and callback behavior in isolation."""

    def setUp(self):
        """Create one isolated temporary cache directory and fixed key."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "cache"
        self.key = make_cache_key("example", "1", np.array([1.0, 2.0]))

    def test_memory_hit_and_mutation_isolation(self):
        """Only one computation runs and caller mutations cannot alter cache data."""
        manager = CacheManager(self.directory)
        calls = []
        original = {"arrays": (np.array([3.0, 4.0]),), "labels": ["original"]}

        def compute():
            """Return a mutable result whose ownership is deliberately external."""
            calls.append(1)
            return original

        first = manager.get_or_compute(self.key, compute)
        first["arrays"][0][0] = -100
        first["labels"][0] = "changed"
        original["arrays"][0][1] = -200
        second = manager.get_or_compute(self.key, compute)
        assert calls == [1]
        np.testing.assert_array_equal(second["arrays"][0], [3.0, 4.0])
        assert second["labels"] == ["original"]
        assert isinstance(second["arrays"], tuple)

    def test_disk_restart_preserves_shapes_dtypes_and_json(self):
        """A new manager restores nested tuple/JSON data without recomputing."""
        value = (np.arange(6, dtype=np.float32).reshape(2, 3), {"empty": [], "flag": True, "none": None})
        CacheManager(self.directory).get_or_compute(self.key, lambda: value)

        def fail():
            """Reject a cache miss in a test that requires disk restoration."""
            raise AssertionError("expected a disk hit")

        restored = CacheManager(self.directory).get_or_compute(self.key, fail)
        assert isinstance(restored, tuple)
        assert restored[0].shape == (2, 3)
        assert restored[0].dtype == np.float32
        np.testing.assert_array_equal(restored[0], value[0])
        assert restored[1] == value[1]
        with np.load(self.directory / f"{self.key}.npz", allow_pickle=False) as archive:
            assert all(archive[name].dtype.kind != "O" for name in archive.files)

    def test_none_is_a_valid_hit(self):
        """Cached None is distinguishable from a missing entry."""
        CacheManager(self.directory).get_or_compute(self.key, lambda: None)
        calls = []
        result = CacheManager(self.directory).get_or_compute(self.key, lambda: calls.append(1))
        assert result is None
        assert calls == []

    def test_corrupt_or_wrong_version_archive_is_recomputed(self):
        """Invalid ZIP bytes and old metadata trigger recomputation and repair."""
        self.directory.mkdir()
        target = self.directory / f"{self.key}.npz"
        for contents in [b"broken archive", "old-version"]:
            with self.subTest(contents=contents):
                if isinstance(contents, bytes):
                    target.write_bytes(contents)
                else:
                    metadata = {"format": -1, "key": self.key, "value": {"kind": "scalar", "value": -1}}
                    np.savez(target, __metadata__=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8))
                calls = []

                def compute():
                    """Record repair work and return a known safe replacement."""
                    calls.append(1)
                    return np.array([7, 8])

                result = CacheManager(self.directory).get_or_compute(self.key, compute)
                np.testing.assert_array_equal(result, [7, 8])
                assert calls == [1]
                np.testing.assert_array_equal(CacheManager(self.directory).get_or_compute(self.key, lambda: None), [7, 8])

    def test_object_array_archive_is_never_loaded_with_pickle(self):
        """A pickle-containing entry is rejected and replaced with numeric data."""
        self.directory.mkdir()
        metadata = {"format": 1, "key": self.key, "value": {"kind": "array", "name": "array_0"}}
        np.savez(
            self.directory / f"{self.key}.npz",
            __metadata__=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8),
            array_0=np.array([{"unsafe": True}], dtype=object),
        )
        result = CacheManager(self.directory).get_or_compute(self.key, lambda: np.array([42]))
        np.testing.assert_array_equal(result, [42])

    def _assert_raw_member_archive_is_repaired(self, members):
        """Store deliberately malformed ZIP members and verify one repair call."""
        self.directory.mkdir()
        target = self.directory / f"{self.key}.npz"
        with zipfile.ZipFile(target, "w") as archive:
            for name, content in members.items():
                archive.writestr(name, content)
        calls = []

        def compute():
            """Record exactly one recovery calculation with a known result."""
            calls.append(1)
            return np.array([17.0, 19.0])

        actual = CacheManager(self.directory).get_or_compute(self.key, compute)
        np.testing.assert_array_equal(actual, [17.0, 19.0])
        assert calls == [1]
        restored = CacheManager(self.directory).get_or_compute(self.key, compute)
        np.testing.assert_array_equal(restored, [17.0, 19.0])
        assert calls == [1]

    def test_non_npy_metadata_member_is_recomputed(self):
        """A ZIP metadata member containing raw bytes is a recoverable miss."""
        self._assert_raw_member_archive_is_repaired({"__metadata__.npy": b"not an NPY array"})

    def test_non_npy_result_member_is_recomputed(self):
        """A ZIP result member containing raw bytes is a recoverable miss."""
        metadata = {"format": 1, "key": self.key, "value": {"kind": "array", "name": "array_0"}}
        metadata_stream = io.BytesIO()
        np.save(metadata_stream, np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8))
        self._assert_raw_member_archive_is_repaired(
            {"__metadata__.npy": metadata_stream.getvalue(), "array_0.npy": b"not an NPY array"}
        )

    def test_disabled_mode_never_reads_or_writes_cache(self):
        """Disabling caching recomputes every time without creating directories."""
        manager = CacheManager(self.directory, enabled=False)
        calls = []

        def compute():
            """Return the invocation number so caching would be observable."""
            calls.append(1)
            return len(calls)

        assert manager.get_or_compute(self.key, compute) == 1
        assert manager.get_or_compute(self.key, compute) == 2
        assert not self.directory.exists()
        assert not manager._memory

    def test_memory_lru_capacity_without_disk(self):
        """A hit refreshes recency and the least recently used entry is evicted."""
        manager = CacheManager(self.directory, max_memory_entries=2, max_disk_entries=0)
        keys = [make_cache_key("lru", "1", i) for i in range(3)]
        for index in (0, 1):
            manager.get_or_compute(keys[index], lambda index=index: index)
        assert manager.get_or_compute(keys[0], lambda: -1) == 0
        manager.get_or_compute(keys[2], lambda: 2)
        assert manager.get_or_compute(keys[0], lambda: -1) == 0
        assert manager.get_or_compute(keys[1], lambda: 99) == 99
        assert len(manager._memory) == 2
        assert not self.directory.exists()

    def test_disk_capacity_and_clear_preserve_unrelated_files(self):
        """Disk eviction removes oldest saved hashes and clear leaves other files."""
        manager = CacheManager(self.directory, max_memory_entries=0, max_disk_entries=2)
        keys = [make_cache_key("disk", "1", i) for i in range(3)]
        manager.get_or_compute(keys[0], lambda: 0)
        os.utime(self.directory / f"{keys[0]}.npz", ns=(1, 1))
        manager.get_or_compute(keys[1], lambda: 1)
        manager.get_or_compute(keys[2], lambda: 2)
        assert not (self.directory / f"{keys[0]}.npz").exists()
        assert len(list(self.directory.glob("*.npz"))) == 2
        unrelated = self.directory / "user-notes.npz"
        unrelated.write_bytes(b"preserve")
        manager.clear()
        assert unrelated.read_bytes() == b"preserve"
        assert list(self.directory.glob("*.npz")) == [unrelated]

    def test_atomic_write_failure_keeps_old_file_and_computation_result(self):
        """A failed atomic replace does not leave partial files or fail compute."""
        CacheManager(self.directory).get_or_compute(self.key, lambda: 10)
        original_bytes = (self.directory / f"{self.key}.npz").read_bytes()
        replacement_key = make_cache_key("new", "1", 1)
        manager = CacheManager(self.directory)
        with patch("src.utils.cache_manager.os.replace", side_effect=OSError("simulated")):
            assert manager.get_or_compute(replacement_key, lambda: 20) == 20
        assert (self.directory / f"{self.key}.npz").read_bytes() == original_bytes
        assert not (self.directory / f"{replacement_key}.npz").exists()
        assert not list(self.directory.glob("*.tmp"))

    def test_invalid_keys_limits_and_results_fail(self):
        """Traversal keys, invalid limits and unsupported callback values fail."""
        manager = CacheManager(self.directory)
        for key in ["../escape", "a" * 63, "A" * 64, 42]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                manager.get_or_compute(key, lambda: 0)
        for limit in [-1, True, 1.2]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                CacheManager(self.directory, max_memory_entries=limit)
        for enabled in (True, False):
            manager.enabled = enabled
            for value in [object(), np.array([object()], dtype=object), np.ma.array([1.0], mask=[True])]:
                with self.subTest(enabled=enabled, value=type(value).__name__), self.assertRaises(TypeError):
                    manager.get_or_compute(self.key, lambda value=value: value)
        assert not self.directory.exists()

    def test_callback_exceptions_do_not_create_entries(self):
        """Computation failures propagate and are never saved as cache results."""
        manager = CacheManager(self.directory)

        def fail():
            """Raise a deliberate numerical failure."""
            raise ArithmeticError("test failure")

        with self.assertRaises(ArithmeticError):
            manager.get_or_compute(self.key, fail)
        assert not self.directory.exists()
        assert not manager._memory

    def test_failed_temporary_cleanup_does_not_break_computation(self):
        """Automatic persistence remains optional even if temporary cleanup fails."""
        manager = CacheManager(self.directory)
        with patch("src.utils.cache_manager.os.replace", side_effect=OSError("replace denied")):
            with patch("src.utils.cache_manager.Path.unlink", side_effect=OSError("cleanup denied")):
                assert manager.get_or_compute(self.key, lambda: 23) == 23
        assert manager.get_or_compute(self.key, lambda: -1) == 23


if __name__ == "__main__":
    unittest.main()
