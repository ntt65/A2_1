"""편향을 보정한 Adam을 NumPy로 구현하고 공통 최적화 실행기의 계약을 따른다."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.optimizer import VanillaGD, _NonfiniteResult, _finite_result, _step_inputs
from src.utils.numerical import _finite_scalar

__concepts__ = {
    "Adam_Update": ["Adam"],
    "Optimizer_State_Rollback": ["Adam._snapshot_state", "Adam._restore_state"],
}


class Adam(VanillaGD):
    """유한한 벡터 (n,)에 원 논문의 알고리즘 1에 따른 편향 보정 Adam을 적용한다.

    m_t=beta1*m_(t-1)+(1-beta1)*g_t,
    v_t=beta2*v_(t-1)+(1-beta2)*g_t**2,
    m_hat=m_t/(1-beta1**t), v_hat=v_t/(1-beta2**t),
    theta_next=theta-lr*m_hat/(sqrt(v_hat)+epsilon).

    epsilon은 제곱근 바깥에 더한다. lr과 epsilon은 유한한 양의 스칼라이고
    beta1과 beta2는 [0,1) 범위의 유한 스칼라다. 모든 입력은 float64로 계산한다.
    theta와 gradient는 서로 같은 비어 있지 않은 (n,) 벡터여야 하며,
    reset() 전까지 처음 정해진 상태의 shape를 유지해야 한다. m/v 속성은
    복사본을 반환하고 t는 수락한 step() 횟수를 센다. 실패한 갱신은 상태에
    반영하지 않는다.
    불리언을 포함한 잘못된 숫자 자료형은 TypeError, 잘못된 shape·범위 또는
    비유한 입력은 ValueError를 발생시킨다. 표현할 수 없는 중간 계산은
    FloatingPointError를 발생시킨다. 가중 항을 포함한 모멘트 계산에서 정밀도를
    잃는 언더플로(0 또는 부정확한 비정규 수로의 반올림)도 거부한다. 기울기
    제곱이 소실되어 유한하지만 잘못된 정규화 갱신을 만드는 일을 막기 위해서다.
    이 정책은 극단적인 기울기와 모멘트 감쇠를 의도적으로 제한하며, 별도 척도나
    임의 정밀도의 모멘트 표현을 제공하지 않는다. 정확히 표현 가능한 비정규
    모멘트는 허용한다. 공통 실행기는 이러한 실패와 후보점의 콜백 거부 시
    이전 상태를 복원한다.
    """

    def __init__(self, lr: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        """스칼라 하이퍼파라미터를 검증하고 m/v=None, t=0으로 초기화한다."""
        super().__init__(lr)	# 공통 학습률 검증을 적용한다.
        self.beta1 = _finite_scalar(beta1, "beta1")
        self.beta2 = _finite_scalar(beta2, "beta2")
        self.epsilon = _finite_scalar(epsilon, "epsilon")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:	# 누적 계수와 편향 보정 분모가 유효한 범위인지 확인한다.
            raise ValueError("beta1 and beta2 must satisfy 0 <= beta < 1")
        if self.epsilon <= 0:	# 정규화 분모의 안정화 상수는 양수여야 한다.
            raise ValueError("epsilon must be positive")
        self._m = None	# 첫 갱신에서 입력 shape에 맞는 0 모멘트를 만들도록 표시한다.
        self._v = None	# 이전 2차 모멘트와 shape 정보를 초기화한다.
        self._t = 0	# 편향 보정의 시간 인덱스도 초기 상태로 맞춘다.

    @property
    def m(self) -> NDArray[np.float64] | None:
        """분리된 1차 모멘트 벡터 (n,)를 반환하며 초기화 전에는 None을 반환한다."""
        return None if self._m is None else self._m.copy()	# 호출자가 내부 모멘트를 직접 바꾸지 못하도록 복사한다.

    @property
    def v(self) -> NDArray[np.float64] | None:
        """분리된 2차 모멘트 벡터 (n,)를 반환하며 초기화 전에는 None을 반환한다."""
        return None if self._v is None else self._v.copy()	# 2차 모멘트 역시 내부 상태와 메모리를 분리한다.

    @property
    def t(self) -> int:
        """reset 이후 수락한 갱신 횟수를 반환하며 초기값은 0이다."""
        return self._t

    def reset(self) -> None:
        """모멘트와 갱신 횟수를 지워 다음 갱신에서 새로운 shape를 정할 수 있게 한다."""
        self._m = None	# 첫 갱신에서 입력 shape에 맞는 0 모멘트를 만들도록 표시한다.
        self._v = None	# 이전 2차 모멘트와 shape 정보를 초기화한다.
        self._t = 0	# 편향 보정의 시간 인덱스도 초기 상태로 맞춘다.

    def _snapshot_state(self) -> tuple:
        """후보점 거부 시 복원할 독립적인 (m(n,), v(n,), t) 상태를 반환한다."""
        return self.m, self.v, self._t	# 복사본 속성을 사용하여 이후 갱신과 분리된 복원 지점을 만든다.

    def _restore_state(self, state: tuple) -> None:
        """이 Adam 객체가 만든 상태를 복원하면서 두 모멘트 배열을 복사한다."""
        first, second, updates = state	# 같은 시점의 두 모멘트와 횟수를 함께 복원한다.
        self._m = None if first is None else first.copy()	# 스냅샷을 공유하지 않도록 1차 모멘트를 다시 복사한다.
        self._v = None if second is None else second.copy()	# 2차 모멘트도 별도 배열로 복원한다.
        self._t = updates	# 마지막 수락 시점에 저장한 갱신 횟수로 되돌린다.

    def step(self, theta: ArrayLike, gradient: ArrayLike) -> NDArray[np.float64]:
        """유한한 (n,) 갱신값을 반환하고 모멘트 언더플로를 거부하며 m/v/t를 함께 반영한다."""
        point, derivative = _step_inputs(theta, gradient)	# 입력 벡터의 유한성과 동일한 shape를 공통 규칙으로 검증한다.
        if self._m is not None and self._m.shape != point.shape:	# 서로 다른 차원의 누적 모멘트가 섞이지 않도록 한다.
            raise ValueError("reset Adam before changing the parameter shape")
        previous_m = np.zeros_like(point) if self._m is None else self._m	# 첫 단계에서만 입력과 같은 shape의 0 모멘트를 사용한다.
        previous_v = np.zeros_like(point) if self._v is None else self._v	# 2차 모멘트도 같은 초기화 규칙을 따른다.
        updates = self._t + 1	# 아직 상태를 바꾸지 않고 후보 갱신의 시간 인덱스를 계산한다.
        try:
            with np.errstate(over="ignore", invalid="ignore", under="raise"):
                moment = self.beta1 * previous_m + (1.0 - self.beta1) * derivative	# 기울기의 지수 이동평균으로 방향 정보를 누적한다.
        except FloatingPointError as error:
            raise _NonfiniteResult("Adam first moment underflow loses float64 precision") from error
        try:
            with np.errstate(over="ignore", invalid="ignore", under="raise"):
                variance = self.beta2 * previous_v + (1.0 - self.beta2) * derivative**2	# 기울기 제곱의 이동평균으로 좌표별 크기 정보를 누적한다.
        except FloatingPointError as error:
            raise _NonfiniteResult("Adam second moment underflow loses float64 precision") from error
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            corrected_m = moment / (1.0 - self.beta1**updates)	# 0 초기화로 작아진 1차 모멘트의 편향을 보정한다.
            corrected_v = variance / (1.0 - self.beta2**updates)	# 2차 모멘트에도 같은 시간 기준의 편향 보정을 적용한다.
            candidate = point - self.lr * (corrected_m / (np.sqrt(corrected_v) + self.epsilon))	# 제곱근 바깥에 epsilon을 더하여 좌표별 갱신 크기를 정규화한다.
        _finite_result(moment, "Adam first moment")	# 새 상태에 반영하기 전에 중간 모멘트의 유한성을 확인한다.
        _finite_result(variance, "Adam second moment")	# 제곱과 누적에서 발생한 오버플로를 최종 후보와 별도로 검출한다.
        _finite_result(corrected_m, "Adam corrected first moment")	# 편향 보정 과정의 넘침도 검사한다.
        _finite_result(corrected_v, "Adam corrected second moment")	# 정규화에 쓰이는 보정 모멘트가 유한해야 한다.
        _finite_result(candidate, "Adam update")	# 후보점까지 검증한 뒤에만 세 상태 값을 갱신한다.
        self._m = moment.copy()	# 검증이 끝난 모멘트만 독립 배열로 상태에 반영한다.
        self._v = variance.copy()	# 1차 모멘트와 동일한 단계의 2차 모멘트를 함께 저장한다.
        self._t = updates	# 실제로 수락된 단계에 대해서만 횟수를 증가시킨다.
        return candidate
