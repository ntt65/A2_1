# Step 1 공통 기반 사용법

수학 계산은 NumPy, 시각화는 Matplotlib을 사용한다. 관련 작업과 검증 기록은
[이슈 #4](https://github.com/ntt65/A2_1/issues/4)와 [Step 1 작업 기록](worklogs/step-1.md)에 있다.

## 수학·미분·시드

| API | 입력 → 출력 | 주요 계약 |
|---|---|---|
| `src.utils.math.sigmoid(x)` | 실수 배열 `(...)` → float64 `(...)` | 두 부호 분기로 overflow 방지, 스칼라는 shape `()` |
| `src.utils.math.softmax(x, axis=-1)` | 실수 배열 `(...)` → float64 `(...)` | 지정 축의 최댓값을 뺀 뒤 정규화, 비어 있는 배열·스칼라·잘못된 축 거부 |
| `src.utils.numerical.central_difference(f, x, h=1e-5)` | 스칼라 함수·x·h → float | `(f(x+h)-f(x-h))/(2h)`, 유한한 스칼라 및 양의 h, 구별 가능한 샘플점 필요 |
| `src.utils.random.set_seed(seed=42)` | 정수 `[0, 2**32-1]` → None | 독립 실행 시작점에서 호출, import 시 RNG 상태 변경 없음 |

수학 API는 복소수·bool·문자열·object 배열·비유한 입력을 거부한다. Sigmoid는 빈 배열의 shape를 유지한다.

```python
import numpy as np
from src.utils.math import sigmoid, softmax
from src.utils.numerical import central_difference
from src.utils.random import set_seed

set_seed()
assert float(sigmoid(0.0)) == 0.5
probabilities = softmax(np.array([[1000.0, 1000.0], [1e308, -1e308]]), axis=1)
assert np.all(np.abs(probabilities.sum(axis=1) - 1.0) <= 1e-6)
assert abs(central_difference(lambda x: x*x, 3.0) - 6.0) <= 1e-4
```

## 공통 플롯

`src.utils.plotting.contour_grid(function, x_range, y_range, points=101)`은
`X, Y, Z`를 각각 `(points, points)` float64 배열로 반환한다.
`function(X,Y)`는 같은 shape의 유한한 실수 배열을 반환해야 한다.

`contour_base(function, x_range, y_range, points=101, levels=20, title=...)`는
축 이름과 동일 배율을 설정한 `(Figure, Axes)`를 반환한다. 저장·표시·종료는 호출자가 맡는다.
이 함수는 미션별 gradient나 최적화 경로를 추가하지 않는 공통 기반이다.

## 메모리·디스크 캐시

`src.utils.cache_manager`에서 `make_cache_key`와 `CacheManager`를 가져온다.

```python
import numpy as np
from src.utils.cache_manager import CacheManager, make_cache_key

# 최소 작동 예제다. 실제 사용에서는 SVD·격자 등 비용이 큰 반복 계산에 적용한다.
inputs = np.array([1.0, 2.0])
key = make_cache_key("example", "v1", inputs, parameters={"seed": 42})
cache = CacheManager(".cache", max_memory_entries=32, max_disk_entries=128)
result = cache.get_or_compute(key, lambda: inputs * 2)
assert np.array_equal(result, [2.0, 4.0])
```

- 키는 namespace·계산 버전·전체 입력·배열 shape/dtype·매개변수를 SHA-256으로 구분한다. 난수가 관여하면 seed도 매개변수에 포함한다.
- 일반 NumPy ndarray(bool/숫자 dtype), JSON 스칼라, built-in list/tuple/문자열 키 dict를 지원한다. object/structured 배열과 ndarray 서브클래스는 거부한다.
- 반환값은 깊은 복사다. 호출자가 수정해도 캐시가 바뀌지 않는다. JSON float는 유한해야 한다.
- 메모리는 최근 사용 순서(LRU), 디스크는 마지막 저장 시각 순서로 오래된 항목을 제거한다. 한도는 **항목 수**이며 바이트 제한은 아니다.
- 한도를 0으로 지정하면 해당 저장 계층을 사용하지 않는다. `enabled=False`는 읽기·쓰기 없이 매번 계산한다.
- 디스크는 안전한 NPZ+JSON 형식이며 `allow_pickle=False`로 읽는다. 저장은 임시 파일을 완성한 뒤 원자적으로 교체한다.
- 누락·손상·형식 버전 불일치는 재계산한다. 저장 실패가 계산 자체를 막지 않으며 콜백 예외는 그대로 전달한다.
- `clear()`는 메모리와 64자리 해시 이름의 캐시 NPZ만 지운다. 같은 폴더의 다른 파일은 보존하며 명시적 정리 실패는 호출자에게 알린다.
- 한 관리자 인스턴스는 단일 소유 스레드에서 사용한다. 프로세스 간 공유·동시 정리·중복 계산 방지는 외부 조정이 필요하다.

## 코드 레지스트리

`.agent/registry.json`은 `schema_version`, `forward`, `reverse`를 포함한다.
파일 탐색은 먼저 해당 개념이나 파일 항목을 확인하고 실제 대상 파일만 읽는다.

- `forward[path]`: 원본 파일 SHA-256, 함수·클래스·메서드의 이름/줄/개념, import 의존성, 관련 테스트.
- `reverse[concept]`: 파일 경로·심볼·현재 줄·의존성·테스트 위치 목록. 같은 개념의 여러 위치를 지원한다.
- 노트북은 셀 ID·제목을 포함하고 코드 줄은 셀 내부 기준이다. 해시는 노트북 출력까지 포함한 원본 파일 바이트를 사용한다.
- 생성 대상은 `src/`, `tests/`, `scripts/`의 Python 파일과 `notebooks/`의 노트북이다. 숨겨진/캐시 폴더·심볼릭 링크를 따라가지 않는다.
- 소스를 실행하지 않고 AST를 읽는다. 노트북의 코드 셀은 일반 Python 문법이어야 하며 IPython magic은 현재 지원하지 않는다.

개념과 관련 테스트는 다음처럼 선언한다. 실제 존재하는 심볼·파일만 지정한다.

```python
# 구현 모듈
__concepts__ = {"Softmax": ["softmax"]}

# 테스트 모듈
__covers__ = ["src/utils/math.py"]
```

노트북은 모든 셀에 안정적인 `id`를 부여한다. `cell.metadata.concepts`는 개념 문자열 목록,
`cell.metadata.title`은 위치 설명이며 코드 셀은 `__concepts__`도 사용할 수 있다.

저장소 루트에서 다음을 실행한다(Windows는 `.venv/Scripts/python.exe`, Linux는 `.venv/bin/python` 사용).

```text
python scripts/update_registry.py --update
python scripts/update_registry.py --check
python -m unittest discover -s tests -v
```

`--update`는 초기 생성·갱신·복구를 위해 위 경로만 제한적으로 탐색한다. 결과에는 시각을 넣지 않아 재생성 결과가 결정적이다.
`--check`는 누락·편집·삭제·잘못된 역색인을 검사하고 불일치 시 0이 아닌 종료 코드를 반환한다.
코드나 노트북을 수정하면 갱신한 JSON도 함께 커밋한다. 일반 코드 탐색과 달리 정합성 검사에는 이 제한된 전체 대상 확인이 필요하다.
