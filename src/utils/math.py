"""입력 배열의 shape를 유지하는 활성화 함수를 NumPy로 계산한다."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

__concepts__ = {"Sigmoid": ["sigmoid"], "Softmax": ["softmax"]}


def _as_finite_real_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """임의 shape의 실수 입력을 유한한 float64 배열로 변환한다.

    불리언을 포함한 비실수 자료형은 TypeError, float64 변환 뒤 비유한 값은
    ValueError를 발생시킨다. 가능한 경우 복사 없이 변환하므로 반환 배열은
    입력과 메모리를 공유할 수 있으며, 이 함수 자체는 입력을 수정하지 않는다.
    """
    array = np.asarray(value)	# 원래 자료형을 먼저 확인하여 불리언·복소수를 암묵적으로 받아들이지 않는다.
    if array.dtype.kind not in "iuf":	# 정수와 실수만 수학 함수의 입력으로 허용한다.
        raise TypeError(f"{name} must contain real numbers, excluding booleans")
    with np.errstate(over="ignore", invalid="ignore"):
        array = array.astype(np.float64, copy=False)	# 계산 정밀도를 float64로 맞추되 불필요한 복사는 피한다.
    if not np.all(np.isfinite(array)):	# 형 변환 중 생긴 무한대까지 입력 단계에서 차단한다.
        raise ValueError(f"{name} must contain only finite float64 values")
    return array


def sigmoid(x: ArrayLike) -> NDArray[np.float64]:
    """sigmoid(x)를 입력과 같은 shape의 float64 배열로 반환한다.

    스칼라 입력의 shape는 ()이다. x >= 0에서는 1 / (1 + exp(-x)),
    x < 0에서는 exp(x) / (1 + exp(x))를 사용하여 지수의 오버플로를 피한다.
    비실수 입력은 TypeError, 비유한 입력은 ValueError를 발생시킨다.
    빈 배열도 원래 shape를 유지한다.
    """
    values = _as_finite_real_array(x, "x")
    result = np.empty_like(values)
    nonnegative = values >= 0	# 입력 부호에 따라 안전한 지수식을 선택한다.
    with np.errstate(under="ignore"):
        result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))	# 음수 지수만 계산하므로 큰 양의 입력에서도 넘치지 않는다.
        exponential = np.exp(values[~nonnegative])	# 큰 음의 입력은 원래 지수를 사용하여 넘침을 피한다.
        result[~nonnegative] = exponential / (1.0 + exponential)	# 음수 구간의 같은 sigmoid 값을 안정적인 비율로 계산한다.
    return result


def softmax(x: ArrayLike, axis: int = -1) -> NDArray[np.float64]:
    """지정한 축에서 exp(x)를 정규화하여 같은 shape의 float64 배열을 반환한다.

    지수 계산 전에 해당 축의 최댓값을 빼서 수치 범위를 안정화한다.
    입력은 최소 1차원인 비어 있지 않은 유한 실수 배열이어야 한다.
    잘못된 숫자 자료형이나 축의 자료형은 TypeError, 범위를 벗어난 축과
    빈 배열은 ValueError를 발생시킨다. 스칼라는 정규화할 축이 없어 거부한다.
    """
    values = _as_finite_real_array(x, "x")
    if values.ndim == 0 or values.size == 0:	# 정규화 축이나 원소가 없는 입력을 명시적으로 거부한다.
        raise ValueError("softmax requires a nonempty array with an axis")
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise TypeError("axis must be an integer")
    if not -values.ndim <= axis < values.ndim:	# 음수 축을 허용하되 실제 차원 범위를 벗어나지 않게 한다.
        raise ValueError("axis is outside the input dimensions")
    # 부호가 반대인 유한 극값의 차는 -inf가 될 수 있다.
    # 이때 exp(-inf)=0은 정규화 확률에 대한 올바른 극한 기여도다.
    with np.errstate(over="ignore", under="ignore"):
        shifted = values - np.max(values, axis=axis, keepdims=True)	# 축 길이를 유지한 최댓값을 빼서 각 지수를 1 이하로 제한한다.
        weights = np.exp(shifted)	# 평행 이동해도 softmax 비율은 같으므로 안정적으로 가중치를 구한다.
    return weights / np.sum(weights, axis=axis, keepdims=True)	# 같은 축의 합으로 나누어 입력 shape와 축별 확률합을 보존한다.
