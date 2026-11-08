"""입력 내용으로 키를 만드는 안전한 로컬 NumPy 캐시의 동작을 검증한다."""

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
    """논리적 입력값과 명시한 버전으로 캐시 키가 결정되는지 검증한다."""

    def test_determinism_and_dictionary_order(self):
        """같은 값은 사전 순서와 무관하게 64자리 해시를 만드는지 확인한다."""
        array = np.array([[1.0, 2.0], [3.0, 4.0]])
        first = make_cache_key("svd", "1", {"image": array, "mode": "gray"}, {"k": 1, "seed": 42})
        second = make_cache_key("svd", "1", {"mode": "gray", "image": array.copy()}, {"seed": 42, "k": 1})
        assert first == second	# 사전 삽입 순서가 달라도 논리적으로 같은 입력은 같은 키를 만든다.
        assert len(first) == 64
        assert int(first, 16) >= 0
        assert first == make_cache_key("svd", "1", {"image": np.asfortranarray(array), "mode": "gray"}, {"k": 1, "seed": 42})	# 메모리 배치 순서가 달라도 전체 배열 값이 같으면 같은 키를 사용한다.

    def test_all_computation_inputs_affect_identity(self):
        """값·형상·자료형·네임스페이스·버전·매개변수가 각각 키를 바꾸는지 확인한다."""
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
        assert len(set(keys)) == len(keys)	# 각 계산 입력 요소를 바꾼 키가 서로 충돌하지 않는지 확인한다.

    def test_scalar_and_container_types_are_distinct(self):
        """Python의 bool·int·float 및 list·tuple이 서로 같은 키가 되지 않는지 확인한다."""
        values = [True, 1, 1.0, "1", [1], (1,), np.array([1]), np.array(1)]
        keys = [make_cache_key("typed", "1", value) for value in values]
        assert len(set(keys)) == len(values)	# 값이 비슷해도 자료형이나 컨테이너 형식이 다르면 구별한다.

    def test_invalid_key_inputs_fail(self):
        """임의 객체와 안전하지 않은 자료형을 저장 전에 거부하는지 확인한다."""
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
    """메모리·디스크·무효화·콜백의 동작을 격리된 환경에서 검증한다."""

    def setUp(self):
        """격리된 임시 캐시 폴더 하나와 고정 키를 만든다."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "cache"
        self.key = make_cache_key("example", "1", np.array([1.0, 2.0]))

    def test_memory_hit_and_mutation_isolation(self):
        """계산은 한 번만 실행하고 호출자의 변경이 캐시 데이터에 영향을 주지 않는지 확인한다."""
        manager = CacheManager(self.directory)
        calls = []
        original = {"arrays": (np.array([3.0, 4.0]),), "labels": ["original"]}

        def compute():
            """소유권이 의도적으로 외부에 있는 변경 가능한 결과를 반환한다."""
            calls.append(1)
            return original

        first = manager.get_or_compute(self.key, compute)
        first["arrays"][0][0] = -100
        first["labels"][0] = "changed"
        original["arrays"][0][1] = -200
        second = manager.get_or_compute(self.key, compute)
        assert calls == [1]	# 호출자 변형 후에도 재계산 없이 캐시 복사본을 반환한다.
        np.testing.assert_array_equal(second["arrays"][0], [3.0, 4.0])	# 호출자가 수정한 배열이 저장된 캐시 값을 오염시키지 않는다.
        assert second["labels"] == ["original"]
        assert isinstance(second["arrays"], tuple)

    def test_disk_restart_preserves_shapes_dtypes_and_json(self):
        """새 관리자가 중첩 tuple·JSON 데이터를 재계산 없이 복원하는지 확인한다."""
        value = (np.arange(6, dtype=np.float32).reshape(2, 3), {"empty": [], "flag": True, "none": None})
        CacheManager(self.directory).get_or_compute(self.key, lambda: value)

        def fail():
            """디스크 복원이 필수인 검증에서 캐시 누락이 발생하면 실패시킨다."""
            raise AssertionError("expected a disk hit")

        restored = CacheManager(self.directory).get_or_compute(self.key, fail)
        assert isinstance(restored, tuple)
        assert restored[0].shape == (2, 3)
        assert restored[0].dtype == np.float32	# 디스크 복원 뒤에도 원래 float32 자료형을 유지한다.
        np.testing.assert_array_equal(restored[0], value[0])
        assert restored[1] == value[1]
        with np.load(self.directory / f"{self.key}.npz", allow_pickle=False) as archive:
            assert all(archive[name].dtype.kind != "O" for name in archive.files)	# 저장된 어떤 배열도 pickle이 필요한 객체 자료형이 아니어야 한다.

    def test_none_is_a_valid_hit(self):
        """캐시된 None을 항목이 없는 상태와 구별하는지 확인한다."""
        CacheManager(self.directory).get_or_compute(self.key, lambda: None)
        calls = []
        result = CacheManager(self.directory).get_or_compute(self.key, lambda: calls.append(1))
        assert result is None
        assert calls == []	# 저장된 None을 누락으로 오인하여 콜백을 실행하지 않는다.

    def test_corrupt_or_wrong_version_archive_is_recomputed(self):
        """잘못된 ZIP 바이트와 이전 버전 메타데이터를 재계산하여 복구하는지 확인한다."""
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
                    """복구 계산 횟수를 기록하고 알려진 안전한 대체값을 반환한다."""
                    calls.append(1)
                    return np.array([7, 8])

                result = CacheManager(self.directory).get_or_compute(self.key, compute)
                np.testing.assert_array_equal(result, [7, 8])
                assert calls == [1]	# 손상된 항목마다 복구 계산은 정확히 한 번 수행한다.
                np.testing.assert_array_equal(CacheManager(self.directory).get_or_compute(self.key, lambda: None), [7, 8])

    def test_object_array_archive_is_never_loaded_with_pickle(self):
        """pickle을 포함한 항목을 거부하고 수치 데이터로 교체하는지 확인한다."""
        self.directory.mkdir()
        metadata = {"format": 1, "key": self.key, "value": {"kind": "array", "name": "array_0"}}
        np.savez(
            self.directory / f"{self.key}.npz",
            __metadata__=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8),
            array_0=np.array([{"unsafe": True}], dtype=object),
        )
        result = CacheManager(self.directory).get_or_compute(self.key, lambda: np.array([42]))
        np.testing.assert_array_equal(result, [42])	# pickle 항목을 안전한 수치 결과로 교체했는지 확인한다.

    def _assert_raw_member_archive_is_repaired(self, members):
        """의도적으로 잘못된 ZIP 항목을 저장하고 복구 계산이 한 번 실행되는지 확인한다."""
        self.directory.mkdir()
        target = self.directory / f"{self.key}.npz"
        with zipfile.ZipFile(target, "w") as archive:
            for name, content in members.items():
                archive.writestr(name, content)
        calls = []

        def compute():
            """복구 계산을 정확히 한 번 기록하고 알려진 결과를 반환한다."""
            calls.append(1)
            return np.array([17.0, 19.0])

        actual = CacheManager(self.directory).get_or_compute(self.key, compute)
        np.testing.assert_array_equal(actual, [17.0, 19.0])
        assert calls == [1]
        restored = CacheManager(self.directory).get_or_compute(self.key, compute)
        np.testing.assert_array_equal(restored, [17.0, 19.0])
        assert calls == [1]	# 복구 이후 새 관리자는 디스크 값을 읽어 추가 계산하지 않는다.

    def test_non_npy_metadata_member_is_recomputed(self):
        """원시 바이트가 든 ZIP 메타데이터 항목을 복구 가능한 캐시 누락으로 처리하는지 확인한다."""
        self._assert_raw_member_archive_is_repaired({"__metadata__.npy": b"not an NPY array"})

    def test_non_npy_result_member_is_recomputed(self):
        """원시 바이트가 든 ZIP 결과 항목을 복구 가능한 캐시 누락으로 처리하는지 확인한다."""
        metadata = {"format": 1, "key": self.key, "value": {"kind": "array", "name": "array_0"}}
        metadata_stream = io.BytesIO()
        np.save(metadata_stream, np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8))
        self._assert_raw_member_archive_is_repaired(
            {"__metadata__.npy": metadata_stream.getvalue(), "array_0.npy": b"not an NPY array"}
        )

    def test_disabled_mode_never_reads_or_writes_cache(self):
        """캐시를 끄면 폴더를 만들거나 읽고 쓰지 않고 매번 다시 계산하는지 확인한다."""
        manager = CacheManager(self.directory, enabled=False)
        calls = []

        def compute():
            """캐시 사용 여부를 관측할 수 있도록 호출 횟수를 반환한다."""
            calls.append(1)
            return len(calls)

        assert manager.get_or_compute(self.key, compute) == 1
        assert manager.get_or_compute(self.key, compute) == 2
        assert not self.directory.exists()	# 캐시 비활성화는 디스크 폴더도 만들지 않는다.
        assert not manager._memory

    def test_memory_lru_capacity_without_disk(self):
        """캐시 적중 시 최근 사용 순서를 갱신하고 가장 오래 사용하지 않은 항목을 제거하는지 확인한다."""
        manager = CacheManager(self.directory, max_memory_entries=2, max_disk_entries=0)
        keys = [make_cache_key("lru", "1", i) for i in range(3)]
        for index in (0, 1):
            manager.get_or_compute(keys[index], lambda index=index: index)
        assert manager.get_or_compute(keys[0], lambda: -1) == 0
        manager.get_or_compute(keys[2], lambda: 2)
        assert manager.get_or_compute(keys[0], lambda: -1) == 0
        assert manager.get_or_compute(keys[1], lambda: 99) == 99	# 가장 오래 사용하지 않은 항목만 제거되어 해당 키를 다시 계산한다.
        assert len(manager._memory) == 2
        assert not self.directory.exists()

    def test_disk_capacity_and_clear_preserve_unrelated_files(self):
        """디스크 한도 초과 시 가장 오래 저장한 해시를 제거하고 clear가 무관한 파일을 보존하는지 확인한다."""
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
        assert unrelated.read_bytes() == b"preserve"	# 캐시 삭제 작업이 관련 없는 파일의 바이트를 보존한다.
        assert list(self.directory.glob("*.npz")) == [unrelated]

    def test_atomic_write_failure_keeps_old_file_and_computation_result(self):
        """원자적 교체 실패가 부분 파일을 남기거나 계산 자체를 실패시키지 않는지 확인한다."""
        CacheManager(self.directory).get_or_compute(self.key, lambda: 10)
        original_bytes = (self.directory / f"{self.key}.npz").read_bytes()
        replacement_key = make_cache_key("new", "1", 1)
        manager = CacheManager(self.directory)
        with patch("src.utils.cache_manager.os.replace", side_effect=OSError("simulated")):
            assert manager.get_or_compute(replacement_key, lambda: 20) == 20
        assert (self.directory / f"{self.key}.npz").read_bytes() == original_bytes	# 원자적 저장 실패가 기존 정상 파일을 바꾸지 않는다.
        assert not (self.directory / f"{replacement_key}.npz").exists()
        assert not list(self.directory.glob("*.tmp"))	# 저장 실패 뒤 부분 임시 파일이 남지 않는지 확인한다.

    def test_invalid_keys_limits_and_results_fail(self):
        """상위 경로로 벗어나는 키, 잘못된 한도와 지원하지 않는 콜백 결과를 거부하는지 확인한다."""
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
        """계산 실패가 그대로 전달되며 캐시 결과로 저장되지 않는지 확인한다."""
        manager = CacheManager(self.directory)

        def fail():
            """의도한 수치 계산 실패를 발생시킨다."""
            raise ArithmeticError("test failure")

        with self.assertRaises(ArithmeticError):
            manager.get_or_compute(self.key, fail)
        assert not self.directory.exists()
        assert not manager._memory	# 콜백 실패를 메모리 캐시에도 저장하지 않는다.

    def test_failed_temporary_cleanup_does_not_break_computation(self):
        """임시 파일 정리가 실패해도 선택적인 자동 저장 때문에 계산 결과가 훼손되지 않는지 확인한다."""
        manager = CacheManager(self.directory)
        with patch("src.utils.cache_manager.os.replace", side_effect=OSError("replace denied")):
            with patch("src.utils.cache_manager.Path.unlink", side_effect=OSError("cleanup denied")):
                assert manager.get_or_compute(self.key, lambda: 23) == 23
        assert manager.get_or_compute(self.key, lambda: -1) == 23	# 임시 정리 실패 후에도 유효한 계산 결과를 재사용한다.


if __name__ == "__main__":
    unittest.main()
