"""이산 Shannon 정보량을 NumPy로 계산하며 단위는 nat이다.

확률은 검증만 하고 잘라내거나 다시 정규화하지 않는다. 클래스 축은
비어 있으면 안 된다. 0 log(0)=0과 p>0,q=0 => +inf 규약으로 확률 지지집합의
불일치를 보존하며, 임의의 epsilon으로 대체하지 않는다.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.utils.math import _as_finite_real_array

__concepts__ = {
    "Entropy": ["entropy"],
    "Cross_Entropy": ["cross_entropy"],
    "KL_Divergence": ["kl_divergence"],
    "Probability_Simplex_Validation": ["_distribution"],
}


def _distribution(value: ArrayLike, axis: int, name: str) -> tuple[NDArray[np.float64], int]:
    """실수 확률 배열을 검증하여 float64 값과 음수가 아닌 축 인덱스를 반환한다.

    각 클래스 벡터의 성분은 [0,1] 범위이며 합은 1과 절대오차 1e-12 이내여야
    한다(rtol=0). 스칼라 입력이나 빈 클래스 축은 ValueError를 발생시키지만
    빈 배치 차원은 허용한다. 잘못된 숫자·축 자료형은 TypeError를 발생시킨다.
    배열을 수정·클리핑·정규화하거나 난수로 초기화하지 않는다.
    """
    values = _as_finite_real_array(value, name)
    if values.ndim == 0:	# 클래스 축이 없는 스칼라를 확률분포로 해석하지 않는다.
        raise ValueError(f"{name} must have a class axis, not be scalar")
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise TypeError("axis must be an integer, excluding booleans")
    if not -values.ndim <= axis < values.ndim:
        raise ValueError("axis is outside the input dimensions")
    axis = int(axis) % values.ndim	# 음수 축을 동등한 음수가 아닌 인덱스로 통일한다.
    if values.shape[axis] == 0:	# 배치는 비어도 되지만 각 분포의 클래스 축은 비면 안 된다.
        raise ValueError("the class axis must be nonempty")
    if np.any(values < 0) or np.any(values > 1):	# 잘못된 확률을 클리핑으로 숨기지 않고 거부한다.
        raise ValueError(f"{name} probabilities must lie in [0,1]")
    if not np.all(np.abs(np.sum(values, axis=axis) - 1.0) <= 1e-12):	# 상대오차 없이 각 분포의 합이 1인지 검증한다.
        raise ValueError(f"{name} must sum to 1 on the class axis (atol=1e-12, rtol=0)")
    return values, axis


def _pair(p: ArrayLike, q: ArrayLike, axis: int) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """shape가 정확히 같은 정답·모델 분포를 검증하며 모든 브로드캐스팅을 거부한다."""
    target, normalized_axis = _distribution(p, axis, "p")
    model, _ = _distribution(q, axis, "q")
    if target.shape != model.shape:	# 브로드캐스팅으로 다른 분포가 암묵적으로 비교되는 것을 막는다.
        raise ValueError("p and q must have exactly the same shape; broadcasting is not allowed")
    return target, model, normalized_axis


def _sum_classes(terms: NDArray[np.float64], axis: int) -> float | NDArray[np.float64]:
    """검증된 클래스 축의 정보량 항을 합하고 결과 shape가 ()이면 float로 반환한다."""
    result = np.sum(terms, axis=axis)
    return float(result) if np.ndim(result) == 0 else result	# 단일 분포와 배치 분포의 반환 계약을 구분한다.


def entropy(p: ArrayLike, axis: int = -1) -> float | NDArray[np.float64]:
    """지정한 클래스 축을 줄여 H(p)=-sum(p*ln(p))를 nat 단위로 반환한다.

    p는 최소 1차원이고 성분이 [0,1] 범위의 유한 실수 확률이어야 한다.
    클래스 합은 atol=1e-12, rtol=0 안에서 1이어야 한다. 결과 shape는
    p.shape에서 axis를 제거한 형태이며, 단일 벡터는 Python float로 반환한다.
    빈 배치 축은 허용한다. 0*ln(0)=0 규약은 클리핑 없이 마스크로 구현한다.
    잘못된 자료형은 TypeError, 값·shape·축 오류는 ValueError를 발생시킨다.
    입력과 전역 난수 상태를 바꾸지 않으며 float64로 계산한다.
    """
    values, axis = _distribution(p, axis, "p")
    logarithms = np.zeros_like(values)
    np.log(values, out=logarithms, where=values > 0)	# 양의 성분에만 로그를 계산하여 0*ln(0)=0 규약을 구현한다.
    with np.errstate(under="ignore"):
        terms = -values * logarithms	# 각 확률의 자기 정보량 기여도를 계산한다.
    return _sum_classes(terms, axis)	# 선택한 클래스 축만 합하여 남은 배치 shape를 보존한다.


def cross_entropy(p: ArrayLike, q: ArrayLike, axis: int = -1) -> float | NDArray[np.float64]:
    """정확한 확률 경계 규약으로 CE(p,q)=-sum(p*ln(q))를 nat 단위로 반환한다.

    p와 q는 정확히 같은 shape의 유한 실수 배열이어야 한다. 성분은 [0,1]이고
    클래스 축은 비어 있지 않으며, 클래스 합은 atol=1e-12, rtol=0 안에서 1이다.
    암묵적 브로드캐스팅·클리핑·정규화는 하지 않는다. p=0인 항은 q=0이어도
    0이며 p>0,q=0인 항은 +inf를 더한다. 출력은 axis를 제거한 shape이며,
    단일 벡터는 Python float, 배치는 float64 ndarray로 반환한다.
    잘못된 자료형은 TypeError, 값·shape·축 오류는 ValueError를 발생시킨다.
    호출자의 배열과 전역 난수 상태를 보존하며 자연로그를 사용하므로 단위는
    bit가 아닌 nat이다.
    """
    target, model, axis = _pair(p, q, axis)
    positive_support = (target > 0) & (model > 0)	# 양의 확률이 겹치는 영역만 유한한 로그 계산에 사용한다.
    logarithms = np.zeros_like(model)
    np.log(model, out=logarithms, where=positive_support)	# 정답 확률이 0인 항에서 불필요한 log(0)을 평가하지 않는다.
    with np.errstate(under="ignore"):
        terms = -target * logarithms	# 모델의 음의 로그 확률을 정답 분포로 가중한다.
    terms[(target > 0) & (model == 0)] = np.inf	# 정답에 가능한 사건을 모델이 불가능하다고 하면 무한 정보량으로 남긴다.
    return _sum_classes(terms, axis)	# 선택한 클래스 축만 합하여 남은 배치 shape를 보존한다.


def kl_divergence(p: ArrayLike, q: ArrayLike, axis: int = -1) -> float | NDArray[np.float64]:
    """클래스 축을 줄여 KL(p||q)=sum(p*(ln(p)-ln(q)))를 nat 단위로 반환한다.

    아주 작은 양수 q에서 p/q가 넘치지 않도록 로그의 차를 사용한다. 입력은
    cross_entropy와 같은 정확한 shape, [0,1] 범위, 유한성, 정규화·축 계약을
    따른다. p=0인 항은 0이며 p>0,q=0인 항은 +inf다. 최종 합의 작은 음수
    반올림 오차까지 포함하여 epsilon이나 클리핑으로 값을 바꾸지 않는다.
    출력은 벡터 입력에서 Python float, 배치 입력에서 클래스 축을 제거한
    float64 ndarray다. 잘못된 입력은 TypeError 또는 ValueError로 알린다.
    입력과 난수 상태를 보존하며, CE와 KL이 유한한 경우 CE=H+KL은 반올림
    오차 범위 안에서 성립한다.
    """
    target, model, axis = _pair(p, q, axis)
    positive_support = (target > 0) & (model > 0)	# 양의 확률이 겹치는 영역만 유한한 로그 계산에 사용한다.
    target_log, model_log = np.zeros_like(target), np.zeros_like(model)
    np.log(target, out=target_log, where=positive_support)	# 유효한 지지집합에서 정답 확률의 로그를 계산한다.
    np.log(model, out=model_log, where=positive_support)	# 같은 지지집합에서 모델 확률의 로그를 계산한다.
    with np.errstate(under="ignore"):
        terms = target * (target_log - model_log)	# 비율 대신 로그 차를 사용해 작은 분모의 오버플로를 피한다.
    terms[(target > 0) & (model == 0)] = np.inf	# 정답에 가능한 사건을 모델이 불가능하다고 하면 무한 정보량으로 남긴다.
    return _sum_classes(terms, axis)	# 선택한 클래스 축만 합하여 남은 배치 shape를 보존한다.
