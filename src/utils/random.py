"""모듈을 불러올 때 난수 상태를 바꾸지 않고 실험 시작점에서만 시드를 설정한다."""

import numpy as np

__concepts__ = {"Reproducible_Seed": ["set_seed"]}


def set_seed(seed: int = 42) -> None:
    """실험 시작점에서 NumPy의 기존 전역 난수 생성기에 시드를 한 번 설정한다.

    seed는 [0, 2**32 - 1] 범위의 정수 스칼라이며 배열을 반환하지 않는다.
    불리언과 정수가 아닌 값은 TypeError, 범위를 벗어난 값은 ValueError를
    발생시킨다. 이 모듈을 불러오는 것만으로 난수 상태가 변하지 않는다.
    """
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):	# 불리언이 정수의 하위 자료형이어도 시드로는 받아들이지 않는다.
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= 2**32 - 1:	# NumPy 전역 난수 생성기가 허용하는 시드 범위를 보장한다.
        raise ValueError("seed must lie between 0 and 2**32 - 1")
    np.random.seed(int(seed))	# 명시적으로 호출한 실험 시작점에서만 재현 가능한 난수 상태를 만든다.
