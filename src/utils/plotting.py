"""특정 미션의 목적함수에 종속되지 않는 등고선 격자와 그림을 제공한다."""

from collections.abc import Callable, Sequence

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.utils.math import _as_finite_real_array

__concepts__ = {"Contour_Grid": ["contour_grid"], "Contour_Plot": ["contour_base"]}


def _interval(bounds: Sequence[float], name: str) -> tuple[float, float]:
    """shape (2,)의 유한하고 엄격히 증가하는 구간을 검증한다.

    비실수 입력은 TypeError, 비유한 값·잘못된 shape·역순 또는 동일한
    경계값은 ValueError를 발생시킨다.
    """
    interval = _as_finite_real_array(bounds, name)
    if interval.shape != (2,) or interval[0] >= interval[1]:	# 두 경계가 올바른 순서인지 확인하여 뒤집힌 격자를 막는다.
        raise ValueError(f"{name} must contain two strictly increasing bounds")
    return float(interval[0]), float(interval[1])


def contour_grid(
    function: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    x_range: Sequence[float] = (-5.0, 5.0),
    y_range: Sequence[float] = (-5.0, 5.0),
    points: int = 101,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """각각 shape (points, points)인 float64 배열 X, Y, f(X,Y)를 반환한다.

    x_range와 y_range는 유한하고 증가하는 두 경계값이며 points는 2 이상의
    정수다. function은 두 격자 배열을 받아 정확히 같은 shape의 유한 실수
    배열을 반환해야 한다. 스칼라의 브로드캐스팅은 자동으로 추정하지 않는다.
    잘못된 인수 자료형은 TypeError, 잘못된 구간·크기·출력 shape 또는
    비유한 격자·함수 반환값은 ValueError를 발생시킨다.
    """
    if not callable(function):
        raise TypeError("function must be callable")
    x_limits, y_limits = _interval(x_range, "x_range"), _interval(y_range, "y_range")	# 두 좌표축의 범위를 독립적으로 검증한다.
    if isinstance(points, (bool, np.bool_)) or not isinstance(points, (int, np.integer)):
        raise TypeError("points must be an integer")
    if points < 2:	# 등고선 격자는 축마다 최소 두 표본이 필요하다.
        raise ValueError("points must be at least 2")
    with np.errstate(over="ignore", invalid="ignore"):
        x_values = np.linspace(*x_limits, points)	# x축 경계를 포함하는 균일 표본을 만든다.
        y_values = np.linspace(*y_limits, points)	# y축도 같은 표본 수로 구성한다.
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):	# 너무 넓은 범위에서 발생한 격자 생성 오버플로를 검출한다.
        raise ValueError("range is too wide to construct a finite grid")
    x_grid, y_grid = np.meshgrid(x_values, y_values, indexing="xy")	# 행은 y, 열은 x에 대응하는 좌표 격자를 만든다.
    values = _as_finite_real_array(function(x_grid, y_grid), "function output")	# 함수 평가 결과도 유한한 float64로 검증한다.
    if values.shape != x_grid.shape:	# 잘못된 함수 출력을 브로드캐스팅으로 숨기지 않는다.
        raise ValueError("function output must have shape (points, points)")
    return x_grid, y_grid, values


def contour_base(
    function: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    x_range: Sequence[float] = (-5.0, 5.0),
    y_range: Sequence[float] = (-5.0, 5.0),
    points: int = 101,
    levels: int = 20,
    title: str = "Function contours",
) -> tuple[Figure, Axes]:
    """축 설명이 있는 등고선을 그리고 재사용할 Figure와 Axes를 반환한다.

    격자 인수의 조건은 contour_grid와 같고 levels는 양의 정수다.
    파일을 저장하거나 창을 표시하지 않으며 그림 정리는 호출자가 담당한다.
    x/y 축 이름과 동일한 축척을 사용하여 추가 도형과 기하적으로 비교한다.
    잘못된 격자 인수나 levels는 TypeError 또는 ValueError를 발생시킨다.
    """
    if isinstance(levels, (bool, np.bool_)) or not isinstance(levels, (int, np.integer)):
        raise TypeError("levels must be an integer")
    if levels < 1:	# 등고선 수준 개수는 양수로 제한한다.
        raise ValueError("levels must be positive")
    x_grid, y_grid, values = contour_grid(function, x_range, y_range, points)	# 공통 격자 생성과 입력 검증을 재사용한다.
    figure, axes = plt.subplots()
    try:
        contours = axes.contour(x_grid, y_grid, values, levels=levels)
        axes.clabel(contours, inline=True, fontsize=8)
        axes.set(xlabel="x", ylabel="y", title=title)
        axes.set_aspect("equal", adjustable="box")	# 좌표축의 단위 길이를 같게 하여 기하적 방향을 왜곡하지 않는다.
    except Exception:
        plt.close(figure)	# 그리기 실패 시 이미 만든 그림 자원을 회수한다.
        raise
    return figure, axes
