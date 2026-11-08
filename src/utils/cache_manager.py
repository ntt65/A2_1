"""명시적으로 선택한 비용이 큰 NumPy 계산에 크기가 제한된 로컬 캐시를 제공한다."""

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
    """지원 값을 태그가 있는 JSON 노드와 복사된 숫자 배열로 인코딩한다.

    배열의 shape와 dtype을 보존한다. 지원 컨테이너는 리스트, 튜플과 문자열
    키를 가진 딕셔너리이며 말단 값은 JSON 스칼라 또는 숫자 ndarray다.
    지원하지 않는 객체와 비유한 JSON 실수는 TypeError 또는 ValueError를
    발생시킨다.
    """
    if type(value) is np.ndarray:	# 사용자 정의 배열 하위 클래스의 동작이 직렬화에 끼어들지 않게 한다.
        if value.dtype.kind not in "biufc":	# 객체 배열 대신 불리언·숫자 배열만 안전하게 저장한다.
            raise TypeError("cache arrays must have boolean or numeric dtype")
        name = f"array_{len(arrays)}"
        arrays[name] = np.array(value, copy=True, order="C", subok=False)	# 호출자와 메모리를 분리하고 일정한 메모리 순서로 배열을 저장한다.
        return {"kind": "array", "name": name}
    if type(value) in _SCALAR_TYPES:
        if type(value) is float and not math.isfinite(value):	# JSON에 비표준 NaN·Infinity가 들어가지 않도록 한다.
            raise ValueError("JSON scalar floats must be finite")
        return {"kind": "scalar", "value": value}
    if type(value) in (list, tuple):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",	# 직렬화 이후에도 리스트와 튜플을 구분할 수 있게 태그를 둔다.
            "items": [_pack_value(item, arrays) for item in value],	# 중첩 컨테이너에도 동일한 자료형·배열 정책을 적용한다.
        }
    if type(value) is dict:
        if any(type(key) is not str for key in value):	# JSON 객체의 문자열 키 계약을 명확히 한다.
            raise TypeError("cache dictionary keys must be strings")
        return {
            "kind": "dict",
            "items": [[key, _pack_value(value[key], arrays)] for key in sorted(value)],	# 딕셔너리 삽입 순서가 해시에 영향을 주지 않도록 정렬한다.
        }
    raise TypeError(f"unsupported cache value type: {type(value).__name__}")


def _unpack_value(node: dict, archive: Any) -> Any:
    """검증된 태그 노드를 디코딩하고 원래 shape의 숫자 배열을 복사한다.

    잘못된 노드나 지원하지 않는 배열 dtype은 TypeError 또는 ValueError를
    발생시키며, 호출자는 이를 캐시 미스로 처리한다.
    """
    if not isinstance(node, dict):	# 저장된 메타데이터를 신뢰하지 않고 노드 구조부터 확인한다.
        raise ValueError("invalid cache node")
    kind = node.get("kind")
    if kind == "array":
        name = node["name"]
        if not isinstance(name, str) or re.fullmatch(r"array_[0-9]+", name) is None:	# 정해진 내부 배열 이름만 압축 파일에서 조회한다.
            raise ValueError("invalid cache array name")
        value = archive[name]
        if type(value) is not np.ndarray:	# 압축 항목이 실제 NPY 배열인지 확인한다.
            raise ValueError("cached array member must contain an NPY array")
        if value.dtype.kind not in "biufc":	# 객체 배열 대신 불리언·숫자 배열만 안전하게 저장한다.
            raise ValueError("unsupported cached array dtype")
        return np.array(value, copy=True, subok=False)	# 읽은 배열을 압축 파일 객체와 독립적으로 반환한다.
    if kind == "scalar":
        value = node["value"]
        if type(value) not in _SCALAR_TYPES:
            raise ValueError("invalid cached scalar")
        if type(value) is float and not math.isfinite(value):	# JSON에 비표준 NaN·Infinity가 들어가지 않도록 한다.
            raise ValueError("nonfinite cached JSON float")
        return value
    if kind in ("list", "tuple"):
        items = node["items"]
        if not isinstance(items, list):
            raise ValueError("invalid cached sequence")
        values = [_unpack_value(item, archive) for item in items]
        return tuple(values) if kind == "tuple" else values	# 저장 전에 사용한 컨테이너 자료형을 복원한다.
    if kind == "dict":
        items = node["items"]
        if not isinstance(items, list):
            raise ValueError("invalid cached dictionary")
        result = {}
        for item in items:
            if not isinstance(item, list) or len(item) != 2 or type(item[0]) is not str:	# 딕셔너리 항목의 키·값 구조를 검증한다.
                raise ValueError("invalid cached dictionary entry")
            if item[0] in result:	# 중복 키로 앞선 데이터가 조용히 덮이지 않게 한다.
                raise ValueError("duplicate cached dictionary key")
            result[item[0]] = _unpack_value(item[1], archive)	# 하위 노드까지 같은 검증 절차로 복원한다.
        return result
    raise ValueError("unknown cache node kind")


def _json_bytes(value: Any) -> bytes:
    """메타데이터를 결정적으로 직렬화하고 비유한 JSON 숫자는 거부한다."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False	# 키 순서·공백·숫자 규약을 고정해 같은 값의 바이트 표현을 일정하게 한다.
    ).encode("utf-8")


def make_cache_key(
    namespace: str, version: str, inputs: Any, parameters: Any = None
) -> str:
    """입력과 알고리즘 매개변수에 대한 결정적인 SHA-256 키를 반환한다.

    ``namespace``와 ``version``은 비어 있지 않은 문자열이어야 한다.
    입력·매개변수는 JSON 값, 임의 shape의 숫자 ndarray와 중첩 튜플을 지원한다.
    배열 내용·shape·dtype은 키에 포함하고 메모리 배치는 구별하지 않는다.
    일반 ndarray와 내장 컨테이너만 받고 하위 클래스는 받지 않는다.
    스칼라·컨테이너의 자료형은 서로 구분한다. 계산이 난수에 의존하면
    매개변수에 seed를 포함해야 한다. 지원하지 않는 값은 TypeError를 발생시킨다.
    """
    if not isinstance(namespace, str) or not namespace:	# 서로 다른 계산 영역의 키가 섞이지 않도록 이름을 필수로 받는다.
        raise ValueError("namespace must be a nonempty string")
    if not isinstance(version, str) or not version:	# 알고리즘 변경 시 이전 결과를 구분할 버전을 요구한다.
        raise ValueError("version must be a nonempty string")
    arrays: dict[str, np.ndarray] = {}
    payload = {
        "format": _FORMAT_VERSION,
        "namespace": namespace,
        "version": version,
        "inputs": _pack_value(inputs, arrays),	# 입력 전체와 그 자료형 구조를 키에 포함한다.
        "parameters": _pack_value(parameters, arrays),	# 알고리즘 매개변수와 필요한 시드도 같은 규칙으로 포함한다.
    }
    payload["arrays"] = {
        name: {
            "shape": list(array.shape),	# 같은 바이트라도 배열 구조가 다르면 다른 키를 만들도록 한다.
            "dtype": array.dtype.str,	# 자료형과 바이트 순서의 차이를 해시에 반영한다.
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),	# 세션에 따라 바뀌는 내장 hash 대신 배열 내용의 안정된 해시를 구한다.
        }
        for name, array in arrays.items()
    }
    return hashlib.sha256(_json_bytes(payload)).hexdigest()	# 메타데이터와 배열 해시를 합쳐 재현 가능한 최종 키를 만든다.


class CacheManager:
    """지원 값을 크기가 제한된 메모리와 선택적인 안전한 NPZ 파일에 캐시한다.

    메모리는 LRU 방식으로 오래 사용하지 않은 항목을 제거한다. 디스크는
    바이트가 아닌 파일 수를 기준으로 가장 오래 저장된 항목부터 제거하므로
    입력 크기를 적절히 정하고 필요하면 ``clear``를 호출한다. 제한이 0이면
    해당 저장 계층을 끈다. 저장 공간은 실제로 필요할 때 만든다.

    반환값은 깊은 복사본이다. 숫자 ndarray의 shape·dtype과 튜플 컨테이너를
    보존한다. 배열·컨테이너의 하위 클래스, 객체 배열, 임의 객체와 비유한
    JSON 실수 스칼라는 거부한다. 디스크 읽기·쓰기 실패는 캐시 미스나 쓰기
    생략으로 처리하므로 캐시를 사용할 수 없어도 계산을 막지 않는다.

    관리 객체 하나는 소유 스레드 하나에서 사용한다. 원자적 교체는 불완전한
    파일을 방지하지만, 프로세스 간 잠금·중복 계산 합치기·동시 제거 및 정리의
    일관성은 제공하지 않는다. 여러 프로세스가 디렉터리를 공유하려면 외부에서
    접근을 조정해야 한다.
    """

    def __init__(
        self,
        cache_dir: str | Path = ".cache",
        max_memory_entries: int = 32,
        max_disk_entries: int = 128,
        enabled: bool = True,
    ) -> None:
        """캐시 계층을 설정하며 정수가 아니거나 음수인 제한값은 ValueError로 거부한다."""
        for limit in (max_memory_entries, max_disk_entries):
            if type(limit) is not int or limit < 0:	# 불리언을 포함한 비정수 제한과 음수 항목 수를 거부한다.
                raise ValueError("cache entry limits must be nonnegative integers")
        if type(enabled) is not bool:	# 캐시 활성화 여부를 참·거짓으로 명확히 제한한다.
            raise TypeError("enabled must be bool")
        self.cache_dir = Path(cache_dir)
        self.max_memory_entries = max_memory_entries
        self.max_disk_entries = max_disk_entries
        self.enabled = enabled
        self._memory: OrderedDict[str, Any] = OrderedDict()	# 접근 순서를 유지하여 LRU 제거에 사용한다.

    def get_or_compute(self, key: str, compute: Callable[[], Any]) -> Any:
        """캐시 결과의 깊은 복사본을 반환하거나 ``compute``를 한 번 호출한다.

        ``key``는 소문자 64자리 SHA-256 16진 문자열이어야 한다. 콜백은 인수 없이
        호출하며 발생한 예외는 캐시하지 않고 그대로 전달한다. 비활성 모드는
        두 캐시 계층을 읽거나 쓰지 않고 매번 계산한다. 어느 모드에서든 지원하지
        않는 반환 자료형은 TypeError 또는 ValueError를 발생시킨다.
        """
        if not isinstance(key, str) or _KEY_PATTERN.fullmatch(key) is None:	# 해시 형식 검증으로 임의 경로를 키로 사용하는 것을 막는다.
            raise ValueError("key must be a lowercase SHA-256 hex digest")
        if not callable(compute):	# 캐시 미스가 났을 때 실제 계산을 호출할 수 있는지 확인한다.
            raise TypeError("compute must be callable")
        if self.enabled and key in self._memory:
            self._memory.move_to_end(key)	# 최근 접근 또는 저장 항목을 LRU 순서의 끝으로 옮긴다.
            return copy.deepcopy(self._memory[key])	# 반환값 변경으로 메모리 캐시가 오염되지 않게 한다.
        if self.enabled and self.max_disk_entries:
            found, value = self._read_disk(key)	# 메모리 미스일 때만 디스크 저장 결과를 확인한다.
            if found:
                self._remember(key, value)	# 디스크에서 읽은 결과를 다음 접근을 위해 메모리에도 보관한다.
                return copy.deepcopy(value)	# 캐시가 소유한 값과 호출자의 값을 분리한다.
        value = compute()	# 두 계층에서 결과를 찾지 못했거나 캐시가 꺼져 있으면 실제로 계산한다.
        _pack_value(value, {})	# 비활성 모드에서도 동일한 반환 자료형 계약을 검증한다.
        owned = copy.deepcopy(value)	# 계산 콜백이 가진 객체와 캐시가 저장할 객체를 분리한다.
        if self.enabled:
            self._remember(key, owned)
            if self.max_disk_entries:
                self._write_disk(key, owned)	# 활성화된 디스크 계층에만 검증된 결과를 저장한다.
        return copy.deepcopy(owned)	# 계산 직후 반환하는 값도 캐시 상태와 독립적으로 만든다.

    def _remember(self, key: str, value: Any) -> None:
        """독립적으로 소유한 값을 유지하고 가장 오래 사용하지 않은 메모리 항목을 제거한다."""
        if not self.max_memory_entries:	# 메모리 제한이 0이면 해당 계층을 사용하지 않는다.
            return
        self._memory[key] = value
        self._memory.move_to_end(key)	# 최근 접근 또는 저장 항목을 LRU 순서의 끝으로 옮긴다.
        while len(self._memory) > self.max_memory_entries:
            self._memory.popitem(last=False)	# 가장 오래 사용하지 않은 항목부터 제거해 항목 수를 제한한다.

    def _read_disk(self, key: str) -> tuple[bool, Any]:
        """pickle 없이 NPZ 하나를 읽으며 누락·구버전·손상 파일은 캐시 미스로 처리한다."""
        try:
            with np.load(self.cache_dir / f"{key}.npz", allow_pickle=False) as archive:	# 객체 역직렬화 없이 NumPy 압축 배열만 읽는다.
                metadata_array = archive["__metadata__"]
                if type(metadata_array) is not np.ndarray:
                    raise ValueError("cache metadata member must contain an NPY array")
                if metadata_array.dtype != np.uint8 or metadata_array.ndim != 1:	# 메타데이터를 UTF-8 바이트 벡터로 해석할 수 있어야 한다.
                    return False, None
                metadata = json.loads(metadata_array.tobytes().decode("utf-8"))	# 저장된 바이트를 JSON 구조로 복원한다.
                if not isinstance(metadata, dict):
                    return False, None
                if metadata.get("format") != _FORMAT_VERSION or metadata.get("key") != key:	# 형식 버전이나 키가 다르면 낡은 결과를 사용하지 않는다.
                    return False, None
                return True, _unpack_value(metadata["value"], archive)	# 검증된 구조와 독립 배열만 캐시 적중 결과로 반환한다.
        except (OSError, ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile):	# 누락·손상·읽기 실패는 재계산 가능한 캐시 미스로 처리한다.
            return False, None

    def _entry_paths(self) -> list[Path]:
        """이 캐시의 해시 이름을 가진 NPZ 파일만 나열하고 무관한 파일은 제외한다."""
        return [
            path
            for path in self.cache_dir.glob("*.npz")
            if _KEY_PATTERN.fullmatch(path.stem) and path.is_file()	# 정리 작업이 다른 이름의 파일에 영향을 주지 않게 범위를 제한한다.
        ]

    def _write_disk(self, key: str, value: Any) -> None:
        """안전한 압축 파일 하나를 원자적으로 저장하고 오래된 디스크 항목을 제거한다."""
        temporary_path = None
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)	# 실제로 저장할 때만 캐시 디렉터리를 만든다.
            arrays: dict[str, np.ndarray] = {}
            metadata = {"format": _FORMAT_VERSION, "key": key, "value": _pack_value(value, arrays)}
            arrays["__metadata__"] = np.frombuffer(_json_bytes(metadata), dtype=np.uint8)	# JSON 메타데이터를 pickle 없는 숫자 배열에 담는다.
            with tempfile.NamedTemporaryFile(
                dir=self.cache_dir, prefix=f".{key}-", suffix=".tmp", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                np.savez_compressed(temporary, **arrays)	# 먼저 임시 파일을 완성하여 불완전한 결과가 노출되지 않게 한다.
            os.replace(temporary_path, self.cache_dir / f"{key}.npz")	# 완성된 파일을 원자적으로 교체한다.
            paths = sorted(self._entry_paths(), key=lambda path: (path.stat().st_mtime_ns, path.name))	# 저장 시각과 이름으로 제거 순서를 일정하게 정한다.
            for path in paths[: max(0, len(paths) - self.max_disk_entries)]:	# 제한을 넘는 가장 오래된 항목만 제거한다.
                path.unlink()
        except OSError:
            pass
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)	# 저장 성공·실패와 무관하게 남은 임시 파일을 정리한다.
                except OSError:
                    pass

    def clear(self) -> None:
        """메모리와 이 관리 객체의 해시 이름 NPZ 파일만 지우고 다른 파일은 보존한다.

        자동 캐시 쓰기와 달리 명시적 정리는 파일시스템 오류를 호출자에게 알린다.
        재귀 삭제는 하지 않는다.
        """
        self._memory.clear()	# 명시적 정리 요청에서는 메모리 캐시부터 비운다.
        for path in self._entry_paths():	# 이 관리 객체가 사용하는 해시 이름 파일만 대상으로 삼는다.
            path.unlink()
