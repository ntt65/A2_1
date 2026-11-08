"""수치 미분과 f(x, y) = x**2 + y**2의 기하적 성질을 계산하고 시각화한다."""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.utils.math import _as_finite_real_array
from src.utils.numerical import central_difference

__concepts__ = {
    "Numerical_Derivative": ["numerical_derivative"],
    "Numerical_Gradient": ["numerical_gradient"],
    "Quadratic_Surface": ["quadratic"],
    "Quadratic_Gradient": ["quadratic_gradient"],
    "Gradient_Visualization": ["gradient_figure"],
    "Derivative_Error_Study": ["derivative_error_study"],
    "Derivative_Error_Visualization": ["derivative_error_figure"],
}


def numerical_derivative(
    function: Callable[[float], float], x: float, h: float = 1e-5
) -> float:
    """스칼라 함수의 x에서의 중심차분을 스칼라로 반환한다.

    (f(x+h)-f(x-h))/(2h) 계산과 입력 검증은 공통 유틸리티에 위임한다.
    x, h와 함수 출력의 shape는 ()이며 h는 양수·유한해야 하고, float64에서
    두 표본점 모두 x와 구별되어야 한다. 자료형 오류는 TypeError, shape·간격
    오류나 비유한 결과는 ValueError를 발생시킨다. 사용자 함수의 예외는
    그대로 전달한다. 통상적인 O(h**2) 절단오차 추정을 적용하려면 함수가
    결정적이고 x 근방에서 3차까지 연속 미분 가능해야 한다. 계산 결과가
    유한하다는 사실만으로 미분 가능성이 증명되지는 않는다.
    """
    return central_difference(function, x, h)	# 동일한 차분식과 검증 기준을 공통 구현에서 재사용한다.


def numerical_gradient(
    function: Callable[[NDArray[np.float64]], float],
    point: ArrayLike,
    h: float = 1e-5,
) -> NDArray[np.float64]:
    """비어 있지 않은 점 (n,)에서 스칼라 함수의 기울기를 근사한다.

    좌표 i마다 t -> function(point의 i번째 좌표를 t로 바꾼 점)에 공통
    중심차분을 적용한다. 반환값은 shape (n,)의 float64 배열이다. 각 함수
    평가에는 독립 복사본을 전달하므로 표본을 변경해도 입력이나 다른 표본에
    영향을 주지 않는다. 목적함수는 점 근방에서 결정적이고 충분히 매끄러워야
    하며, 유한한 계산값만으로 미분 가능성을 증명하지 않는다. 함수 출력은
    유한 실수 스칼라여야 한다. 양의 유한 스칼라 h는 모든 좌표에서 양쪽
    표본점을 구별할 수 있어야 한다. 숫자 자료형 오류는 TypeError,
    shape·표본점·간격·출력 오류는 ValueError를 발생시킨다. 사용자 예외는
    그대로 전달하고 난수 상태를 소비하거나 초기화하지 않는다.
    """
    if not callable(function):
        raise TypeError("function must be callable")
    values = _as_finite_real_array(point, "point")
    if values.ndim != 1 or values.size == 0:	# 기울기의 좌표축을 명확히 하도록 비어 있지 않은 벡터만 받는다.
        raise ValueError("point must be a nonempty one-dimensional array")
    values = values.copy()	# 사용자 함수가 표본을 바꾸더라도 호출자의 입력은 보존한다.
    gradient = np.empty_like(values)	# 입력과 같은 shape·dtype으로 좌표별 기울기를 담는다.
    for index in range(values.size):	# 한 번에 한 좌표만 변화시켜 편미분을 계산한다.

        def along_coordinate(coordinate: float) -> float:
            """다른 표본과 격리된 shape (n,)의 점에서 스칼라 목적함수를 평가한다."""
            sample = values.copy()	# 매 평가마다 독립 표본을 만들어 콜백의 변경을 격리한다.
            sample[index] = coordinate	# 현재 편미분 대상 좌표만 차분 표본값으로 바꾼다.
            return function(sample)

        gradient[index] = central_difference(along_coordinate, float(values[index]), h)	# 각 좌표의 편미분에도 동일한 중심차분 검증을 적용한다.
    return gradient


def quadratic(x: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
    """유한 실수 x와 y의 브로드캐스트 shape로 x**2 + y**2를 반환한다.

    스칼라는 shape ()의 float64 배열을 만들고 배열은 NumPy의 일반적인
    브로드캐스팅을 따른다. 입력은 수정하지 않는다. 비실수 입력은 TypeError,
    호환되지 않는 shape·비유한 입력·결과 오버플로는 ValueError를 발생시킨다.
    일반적인 float64 연산처럼 작은 제곱은 0으로 반올림될 수 있으며
    임의 정밀도 계산을 제공하지 않는다.
    """
    x_values = _as_finite_real_array(x, "x")
    y_values = _as_finite_real_array(y, "y")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = np.asarray(x_values**2 + y_values**2, dtype=np.float64)	# 브로드캐스팅한 두 좌표의 제곱합을 float64로 계산한다.
    if not np.all(np.isfinite(result)):	# 입력이 유한해도 제곱·덧셈 결과가 넘치면 거부한다.
        raise ValueError("quadratic result must be representable as finite float64")
    return result


def quadratic_gradient(points: ArrayLike) -> NDArray[np.float64]:
    """shape (..., 2)의 점에 대한 해석적 기울기 (2x, 2y)를 반환한다.

    단일 점 (2,) 또는 마지막 좌표축 길이가 2인 비어 있지 않은 배치를 받고,
    같은 shape의 새로운 float64 배열을 반환한다. 원점에서는 방향이 없는
    영벡터를 반환한다. 비실수 입력은 TypeError, 빈 배열·잘못된 shape,
    비유한 입력·결과 오버플로는 ValueError를 발생시킨다. 입력은 수정하지 않는다.
    """
    values = _as_finite_real_array(points, "points")
    if values.ndim == 0 or values.shape[-1] != 2 or values.size == 0:	# 마지막 축을 2차원 좌표로 해석할 수 있는지 확인한다.
        raise ValueError("points must be nonempty with shape (..., 2)")
    with np.errstate(over="ignore", invalid="ignore"):
        result = 2.0 * values	# x²+y²를 각 좌표로 미분한 (2x,2y)를 일괄 계산한다.
    if not np.all(np.isfinite(result)):	# 입력이 유한해도 제곱·덧셈 결과가 넘치면 거부한다.
        raise ValueError("quadratic gradient must be representable as finite float64")
    return result




def _gradient_samples() -> np.ndarray:
    """반지름 2인 원 위의 여덟 점과 원점을 shape (9,2)로 반환한다.

    여덟 각도는 0, pi/4, ..., 7*pi/4다. 매번 새로운 float64 배열을
    반환하므로 호출자가 값을 수정해도 이후 그림에 영향을 주지 않는다.
    """
    angles = np.arange(8, dtype=np.float64) * (np.pi / 4.0)	# 겹치는 끝점을 제외한 여덟 균등 방향을 만든다.
    circle = 2.0 * np.column_stack((np.cos(angles), np.sin(angles)))	# f=4 등고선 위에 반지름 2의 표본을 배치한다.
    return np.vstack((circle, np.zeros((1, 2))))	# 방향이 없는 영기울기 사례도 함께 검증하도록 원점을 추가한다.


def gradient_figure():
    """원의 등고선에 수직인 기울기를 보여 주는 Figure와 Axes를 반환한다.

    고정 목적함수는 x**2+y**2이며 두 좌표의 범위는 [-3.5,3.5]다.
    f=4 위의 여덟 표본에 바깥쪽 기울기와 반시계 방향 접선 화살표를 그린다.
    화살표 길이는 표시용으로 각각 좌표 단위 0.7과 0.55로 정규화하며,
    수학적인 크기가 같다는 뜻이 아니다. 원점의 기울기는 0이므로 방향
    화살표 없이 표시한다. 파일을 저장하지 않으며 그림 정리는 호출자가 담당한다.
    """
    from src.utils.plotting import contour_base
    import matplotlib.pyplot as plt

    figure, axes = contour_base(
        quadratic, (-3.5, 3.5), (-3.5, 3.5), points=141, levels=12,
        title="Gradient is normal to a level curve: f(x,y) = x² + y²",
    )
    try:
        figure.set_size_inches(8.4, 7.2)
        figure.set_layout_engine("constrained")
        samples = _gradient_samples()
        points = samples[:-1]	# 원점은 방향 정규화에서 제외하여 0으로 나누지 않는다.
        gradients = quadratic_gradient(points)
        gradient_directions = gradients / np.linalg.norm(gradients, axis=1, keepdims=True)	# 표시할 화살표는 방향만 남기도록 정규화한다.
        tangents = np.column_stack((-points[:, 1], points[:, 0]))	# 반지름을 90도 회전시켜 등고선의 접선 방향을 만든다.
        tangent_directions = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)	# 접선도 표시용 단위 방향으로 바꾼다.
        angles = np.linspace(0.0, 2.0 * np.pi, 361)
        axes.plot(2.0 * np.cos(angles), 2.0 * np.sin(angles), color="#424242",
                  linewidth=1.5, linestyle="--", label="Sample level curve: f = 4")
        axes.scatter(points[:, 0], points[:, 1], color="#303030", s=18, zorder=4)
        axes.quiver(
            points[:, 0], points[:, 1], gradient_directions[:, 0], gradient_directions[:, 1],
            angles="xy", scale_units="xy", scale=1.0 / 0.7, color="#cf4b16",
            width=0.006, zorder=5, label="Gradient direction (2x, 2y)",
        )
        axes.quiver(
            points[:, 0], points[:, 1], tangent_directions[:, 0], tangent_directions[:, 1],
            angles="xy", scale_units="xy", scale=1.0 / 0.55, color="#2466b0",
            width=0.005, zorder=5, label="Tangent direction (-y, x)",
        )
        axes.scatter([0.0], [0.0], marker="x", color="#202020", s=55, zorder=6)	# 원점에는 방향 화살표 대신 별도 표식을 사용한다.
        axes.annotate("Origin: zero gradient\n(no direction)", xy=(0.0, 0.0),
                      xytext=(0.15, -0.55), fontsize=9,
                      bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"})
        axes.set(xlim=(-3.5, 3.5), ylim=(-3.5, 3.5))
        axes.legend(loc="upper right", fontsize=8, framealpha=0.95)
        figure.supxlabel("Arrow lengths are normalized for display; axes use equal scales.", fontsize=9)
        return figure, axes
    except Exception:
        plt.close(figure)	# 그림 생성 도중 실패하면 자원을 회수한다.
        raise


def derivative_error_study() -> dict[str, np.ndarray]:
    """x=3에서 61개 간격의 중심차분 추정값과 오차를 반환한다.

    딕셔너리의 각 값은 shape (61,)의 float64 배열이다. h는 1e-15부터 1까지다.
    quadratic_derivative와 quadratic_error는 x**2를 정확한 미분값 6과,
    sine_derivative와 sine_error는 sin(x)를 cos(3)과 비교한다. 모든 계산은
    numerical_derivative를 재사용한다. 오차는 정확히 0인 부동소수점 결과까지
    포함한 원래의 절대오차다. h가 작아질수록 정확도가 단조롭게 좋아지지는 않는다.
    """
    steps = np.logspace(-15.0, 0.0, 61)	# 반올림 지배 구간부터 큰 간격까지 로그 간격으로 탐색한다.
    quadratic_estimates = np.array([
        numerical_derivative(lambda x: x * x, 3.0, float(h)) for h in steps
    ])
    sine_estimates = np.array([
        numerical_derivative(np.sin, 3.0, float(h)) for h in steps
    ])
    return {
        "h": steps,
        "quadratic_derivative": quadratic_estimates,
        "quadratic_error": np.abs(quadratic_estimates - 6.0),	# 계산 함수와 독립적인 해석 미분값 6을 기준으로 삼는다.
        "sine_derivative": sine_estimates,
        "sine_error": np.abs(sine_estimates - np.cos(3.0)),	# sin의 해석 미분값과 비교하여 일반적인 절단오차를 관찰한다.
    }


def derivative_error_figure():
    """x=3에서 간격에 따른 미분 오차를 비교하는 Figure와 Axes를 반환한다.

    가로축은 로그 축이다. 절대오차 세로축은 [0,1e-15]의 선형 구간이 있는
    symlog를 사용하여 정확한 0 오차를 그대로 보이게 한다. 측정값을 임의의
    양수 하한으로 대체하지 않는다. x**2는 정확한 산술에서 중심차분 절단오차가
    없다. sin(x)는 주도 절단오차 항 abs(cos(3))*h**2/6을 보여 준다.
    파일은 저장하지 않으며 그림 정리는 호출자가 담당한다.
    """
    import matplotlib.pyplot as plt

    study = derivative_error_study()	# 그림에 넣을 오차를 실제 공통 실험 함수에서 계산한다.
    figure, axes = plt.subplots(figsize=(9.0, 5.5), layout="constrained")
    try:
        steps = study["h"]
        axes.plot(steps, study["quadratic_error"], marker="o", markersize=3,
                  color="#cf4b16", label="x²: measured error (exact derivative 6)")
        axes.plot(steps, study["sine_error"], marker="s", markersize=3,
                  color="#2466b0", label="sin(x): measured error (exact derivative cos(3))")
        axes.plot(steps, abs(np.cos(3.0)) * steps**2 / 6.0,
                  linestyle="--", color="#555555", label="sin(x): leading O(h²) truncation term")
        axes.set_xscale("log")	# 여러 자릿수에 걸친 간격을 비교할 수 있게 한다.
        axes.set_yscale("symlog", linthresh=1e-15)	# 정확한 0 오차를 유지하면서 작은 양의 오차도 보여 준다.
        maximum = max(float(np.max(study["quadratic_error"])), float(np.max(study["sine_error"])))	# 두 함수의 실제 오차 범위를 모두 포함하도록 세로축 상한을 정한다.
        axes.set(ylim=(0.0, maximum * 2.0), xlabel="Central-difference step h",
                 ylabel="Absolute derivative error",
                 title="Step-size error at x = 3\nQuadratic central difference has no truncation error in exact arithmetic")
        axes.grid(True, which="major", alpha=0.25)
        axes.legend(fontsize=8, loc="upper center")
        figure.supxlabel("Y scale: logarithmic above 1e-15, linear near zero; zero errors are retained.", fontsize=9)
        return figure, axes
    except Exception:
        plt.close(figure)	# 그림 생성 도중 실패하면 자원을 회수한다.
        raise
