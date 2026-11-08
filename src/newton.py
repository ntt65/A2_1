"""유한 실수 벡터와 양의 정부호 Hessian에 대한 Newton 갱신을 구현한다.

역행렬을 만들지 않고 선형 시스템을 푼다. 고정 감쇠를 지원하지만
선 탐색이나 전역 수렴 보장은 제공하지 않는다.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.optimizer import VanillaGD, _finite_result, _output_array, _step_inputs, _vector

__concepts__ = {
    "Newton_Method": ["NewtonMethod"],
    "Quadratic_Hessian": ["circle_hessian", "elliptic_hessian"],
}


class NewtonMethod(VanillaGD):
    """H(theta) direction = gradient를 풀어 theta - lr*direction을 반환한다.

    theta와 gradient는 유한 실수 벡터 (n,)다. hessian은 분리된 점 복사본을
    받아 유한 실수 행렬 (n,n)을 반환해야 한다. 행렬은 atol=rtol=1e-12 안에서
    대칭이어야 한다. 정확히 대칭인 행렬은 평균 내지 않는다. 허용된 비대칭
    성분 쌍은 오버플로를 피한 중간값을 양쪽 삼각부에 동일하게 넣은 뒤
    Cholesky 분해와 np.linalg.solve에 전달한다. 이는 허용된 비대칭만 제거하며
    음의 곡률이나 특이성을 고치지 않는다. 정확한 중간값을 표현할 수 없으면
    float64 반올림의 영향을 받는다.
    H 전체가 비정규 수 범위이면 분해와 풀이 전에 H와 gradient를 max(abs(H))로
    나누어 비정규 수의 소거 연산을 피한다. 척도를 바꾼 gradient를 표현할 수
    없으면 유한 방향처럼 받아들이지 않고 거부한다. lr은 (0,1]의 스칼라이며
    lr=1이 완전한 Newton 갱신이다. 이 클래스는 상태가 없고 run_optimization과
    함께 사용할 수 있다. 고정 감쇠는 임의 목적함수나 시작점의 수렴을 보장하지 않는다.

    잘못된 입력·콜백 자료형은 TypeError를 발생시킨다. 잘못된 shape·감쇠율,
    비대칭, 특이하거나 양의 정부호가 아닌 Hessian은 ValueError를 발생시킨다.
    비유한 Hessian 출력·분해 결과·방향·갱신값은 경로 실행기가 처리하는 공통
    FloatingPointError 하위 예외를 발생시킨다. 다른 콜백 예외는 그대로
    전달하며 입력은 수정하지 않는다.
    """

    def __init__(self, hessian: Callable[[NDArray[np.float64]], ArrayLike], lr: float = 1.0):
        """호출 가능한 H:(n,)->(n,n)과 유한 감쇠율 0<lr<=1을 저장하고 나머지는 거부한다."""
        if not callable(hessian):	# Hessian을 평가할 수 없는 객체는 초기화 단계에서 거부한다.
            raise TypeError("hessian must be callable")
        super().__init__(lr)	# 양수·유한 학습률 조건을 공통 구현으로 검증한다.
        if self.lr > 1:	# 현재 구현의 감쇠율 계약을 완전 Newton 갱신 이하로 제한한다.
            raise ValueError("Newton lr must satisfy 0 < lr <= 1")
        self.hessian = hessian

    def step(self, theta: ArrayLike, gradient: ArrayLike) -> NDArray[np.float64]:
        """H(theta)가 대칭 양의 정부호인지 검증한 뒤 새로운 float64 (n,) 갱신값을 반환한다.

        콜백에는 theta의 복사본을 전달한다. 자료형·shape·곡률 오류는
        TypeError 또는 ValueError, 비유한 계산값은 FloatingPointError를
        발생시킨다. Hessian의 역행렬이나 고유값 풀이 함수는 호출하지 않는다.
        """
        point, derivative = _step_inputs(theta, gradient)	# 점과 기울기가 같은 차원의 유한 벡터인지 확인한다.
        dimension = point.size
        matrix = _output_array(self.hessian(point.copy()), (dimension, dimension), "hessian output")	# 콜백의 입력 변형을 격리하고 정방 Hessian의 자료형·shape를 검증한다.
        with np.errstate(over="ignore", invalid="ignore"):
            symmetric = np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-12)	# 명시한 절대·상대 허용치로 대칭 계약을 확인한다.
        if not symmetric:
            raise ValueError("hessian must be symmetric within atol=rtol=1e-12")
        if not np.array_equal(matrix, matrix.T):	# 정확 대칭인 극소 행렬은 불필요한 평균 계산으로 바꾸지 않는다.
            # 정확한 대칭, 특히 비정규 수인 대각 성분은 그대로 보존한다.
            # 같은 부호의 차와 반대 부호의 합은 오버플로를 일으키지 않는다.
            # 하나의 중간값을 양쪽에 넣어 삼각부의 대칭을 일치시킨다.
            row, column = np.triu_indices(dimension, k=1)	# 대각을 제외한 각 대칭 성분 쌍을 한 번씩만 처리한다.
            upper, lower = matrix[row, column], matrix[column, row]
            same_sign = np.signbit(upper) == np.signbit(lower)	# 성분 부호에 따라 안전한 중간값 계산식을 고른다.
            midpoint = np.empty_like(upper)
            midpoint[same_sign] = upper[same_sign] + 0.5 * (lower[same_sign] - upper[same_sign])	# 같은 부호에서는 차를 절반으로 나누어 합의 넘침을 피한다.
            midpoint[~same_sign] = 0.5 * (upper[~same_sign] + lower[~same_sign])	# 반대 부호에서는 안전한 합을 사용해 중간값을 계산한다.
            matrix[row, column] = midpoint	# 검증을 통과한 비대칭 성분을 하나의 중간값으로 맞춘다.
            matrix[column, row] = midpoint	# 반대 삼각부에도 같은 값을 넣어 정확한 대칭을 만든다.
        _finite_result(matrix, "Newton symmetric Hessian")
        system, right_hand_side = matrix, derivative
        scale = float(np.max(np.abs(matrix)))	# 행렬 전체가 비정규 수 범위인지 판단할 기준 척도를 구한다.
        if 0 < scale < np.finfo(np.float64).tiny:	# 비정규 수만으로 이루어진 비영 Hessian에서만 척도를 바꾼다.
            # LAPACK 소거 연산 전에 동치인 정규화 시스템으로 바꾼다.
            # 영행렬 Hessian은 이 분기를 건너뛰고 양의 정부호 검사에서 거부된다.
            with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
                system = matrix / scale	# 원래 방향 해를 유지하는 동치 선형 시스템을 만든다.
                right_hand_side = derivative / scale	# 좌변과 우변에 같은 척도를 적용한다.
            _finite_result(right_hand_side, "Newton scaled gradient")	# 정규화 우변의 오버플로를 잘못된 유한 해로 숨기지 않는다.
        try:
            with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
                factor = np.linalg.cholesky(system)	# 대칭 행렬의 양의 정부호 조건을 분해로 검증한다.
        except np.linalg.LinAlgError as error:
            raise ValueError("hessian must be positive definite and nonsingular") from error
        _finite_result(factor, "Newton Cholesky factor")
        try:
            with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
                direction = np.linalg.solve(system, right_hand_side)	# 역행렬을 만들지 않고 H*d=g의 방향을 직접 푼다.
        except np.linalg.LinAlgError as error:
            raise ValueError("hessian linear system is numerically singular") from error
        _finite_result(direction, "Newton direction")
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            candidate = point - self.lr * direction	# 고정 감쇠율을 적용한 후보점을 계산한다.
        return _finite_result(candidate, "Newton update")	# 최종 갱신값이 유한할 때만 호출자에게 반환한다.


def circle_hessian(point: ArrayLike) -> NDArray[np.float64]:
    """유한한 점 (2,)에서 f(x,y)=x**2+y**2의 Hessian 2*I를 shape (2,2)로 반환한다.

    입력을 검증하고 보존하며, 호출할 때마다 독립적인 float64 행렬을 반환한다.
    공통 벡터 계약에 따라 비실수 입력은 TypeError, 비유한 값이나 잘못된
    shape는 ValueError를 발생시킨다.
    """
    value = _vector(point, "point")
    if value.shape != (2,):	# 이 목적함수의 Hessian은 2차원 점에 대해서만 정의한다.
        raise ValueError("circle Hessian point must have shape (2,)")
    return np.diag([2.0, 2.0])	# 원형 이차함수의 두 방향 곡률 2를 대각에 배치한다.


def elliptic_hessian(point: ArrayLike) -> NDArray[np.float64]:
    """f(x,y)=x**2+10*y**2의 Hessian diag(2,20)을 shape (2,2)로 반환한다.

    point는 보존되는 유한 실수 벡터 (2,)여야 한다. 반환하는 float64 행렬은
    호출자가 독립적으로 소유한다. 비실수 입력은 TypeError, 비유한 값이나
    잘못된 shape는 ValueError를 발생시킨다.
    """
    value = _vector(point, "point")
    if value.shape != (2,):	# 이 목적함수의 Hessian은 2차원 점에 대해서만 정의한다.
        raise ValueError("elliptic Hessian point must have shape (2,)")
    return np.diag([2.0, 20.0])	# 타원형 이차함수의 서로 다른 곡률 2와 20을 반영한다.
