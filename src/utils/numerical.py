"""유한한 스칼라 입력 조건을 명시하고 중심차분으로 수치 미분한다."""

from collections.abc import Callable

import numpy as np

from src.utils.math import _as_finite_real_array

__concepts__ = {"Central_Difference": ["central_difference"]}


def _finite_scalar(value: object, name: str) -> float:
    """유한 실수 스칼라를 float로 반환하고 shape가 ()가 아니면 거부한다.

    비실수 입력은 TypeError, 비유한 값이나 스칼라가 아닌 shape는
    ValueError를 발생시킨다.
    """
    array = _as_finite_real_array(value, name)
    if array.shape != ():	# 길이 1 배열도 스칼라와 구별하여 입력 계약을 지킨다.
        raise ValueError(f"{name} must be a scalar with shape ()")
    return float(array)


def central_difference(
    function: Callable[[float], float], x: float, h: float = 1e-5
) -> float:
    """f'(x)를 (f(x+h)-f(x-h))/(2h)로 근사하여 스칼라 float를 반환한다.

    x, h와 두 함수 반환값은 shape ()의 유한 실수 스칼라여야 한다.
    h는 양수이고 float64에서 x+h와 x-h가 각각 x와 구별되어야 한다.
    비실수 값은 TypeError, 잘못된 shape·간격·표본점 또는 표현할 수 없는
    차이·결과는 ValueError를 발생시킨다. 전달된 함수가 발생시킨 예외는
    감추지 않고 호출자에게 그대로 전달한다.
    """
    if not callable(function):	# 표본점 평가 전에 호출 가능한 목적함수인지 확인한다.
        raise TypeError("function must be callable")
    point = _finite_scalar(x, "x")
    step = _finite_scalar(h, "h")
    if step <= 0:	# 중심차분의 분모가 0이 되거나 간격의 방향이 뒤집히지 않게 한다.
        raise ValueError("h must be positive")
    upper, lower = point + step, point - step	# 기준점 양쪽의 대칭 표본점을 만든다.
    denominator = 2.0 * step	# 두 표본점 사이의 전체 간격을 분모로 사용한다.
    if not all(np.isfinite(value) for value in (upper, lower, denominator)):	# 유한 입력끼리의 덧셈·곱셈에서도 생길 수 있는 넘침을 확인한다.
        raise ValueError("sampling points and 2*h must be finite")
    if upper == point or lower == point:	# 반올림으로 표본점이 기준점에 겹치면 유효한 차분으로 인정하지 않는다.
        raise ValueError("h is too small to distinguish x+h and x-h from x")
    value_upper = _finite_scalar(function(upper), "function(x+h)")	# 오른쪽 표본의 반환값도 유한한 스칼라인지 검증한다.
    value_lower = _finite_scalar(function(lower), "function(x-h)")	# 왼쪽 표본에도 같은 반환 계약을 적용한다.
    derivative = (value_upper - value_lower) / denominator	# 대칭 차분의 변화량을 표본 간격으로 나눈다.
    if not np.isfinite(derivative):	# 뺄셈이나 나눗셈에서 표현 범위를 벗어난 결과를 거부한다.
        raise ValueError("the central difference is not representable as a finite float64")
    return derivative
