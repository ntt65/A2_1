"""NumPy 최적화 기법과 유한하고 시점이 맞는 경로 기록을 남기는 공통 실행기를 제공한다."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.utils.math import _as_finite_real_array
from src.utils.numerical import _finite_scalar

__concepts__ = {
    "Vanilla_GD": ["VanillaGD"],
    "Momentum_Update": ["Momentum"],
    "Optimization_Trajectory": ["OptimizationResult", "run_optimization"],
    "Optimizer_State_Rollback": ["VanillaGD._snapshot_state", "VanillaGD._restore_state",
                                 "Momentum._snapshot_state", "Momentum._restore_state"],
    "Elliptic_Objective": ["elliptic", "elliptic_gradient"],
    "Optimizer_Comparison_Visualization": ["optimizer_comparison_figure"],
    "Learning_Rate_Stability_Visualization": ["learning_rate_figure"],
}


class _NonfiniteResult(FloatingPointError):
    """표현 범위를 벗어난 내부 계산이나 콜백 출력을 구분하는 예외다."""


def _vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """유한 실수의 비어 있지 않은 벡터 (n,)를 검증하고 자료형·shape 오류를 알린다."""
    array = _as_finite_real_array(value, name)
    if array.ndim != 1 or array.size == 0:	# 매개변수와 기울기를 비어 있지 않은 1차원 좌표 벡터로 제한한다.
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    return array


def _step_inputs(theta: ArrayLike, gradient: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """shape (n,)이 서로 같은 유한 theta·gradient 벡터를 반환한다."""
    point = _vector(theta, "theta")
    derivative = _vector(gradient, "gradient")
    if point.shape != derivative.shape:	# 브로드캐스팅으로 잘못된 갱신이 만들어지는 것을 막는다.
        raise ValueError("theta and gradient must have identical shapes")
    return point, derivative


def _finite_result(value: np.ndarray, name: str) -> np.ndarray:
    """유한한 계산값을 반환하거나 내부 FloatingPointError 하위 예외를 발생시킨다."""
    if not np.all(np.isfinite(value)):	# 표현 불가능한 계산값을 입력 계약 오류와 별도로 구분한다.
        raise _NonfiniteResult(f"{name} is not representable as finite float64")
    return value


class VanillaGD:
    """비어 있지 않은 벡터 (n,)에 상태 없는 theta_next = theta - lr*gradient를 적용한다.

    lr은 양의 유한 스칼라여야 한다. 입력은 보존하고 매번 새로운 float64
    배열을 반환한다. 잘못된 입력은 TypeError 또는 ValueError, 표현할 수 없는
    계산은 FloatingPointError를 발생시킨다.
    """

    def __init__(self, lr: float):
        """양의 유한 스칼라 학습률을 저장하고 그 밖의 값은 거부한다."""
        self.lr = _finite_scalar(lr, "lr")
        if self.lr <= 0:	# 기울기의 반대 방향으로 이동하는 양의 학습률만 허용한다.
            raise ValueError("lr must be positive")

    def reset(self) -> None:
        """상태 없는 기법의 초기화 메서드이며 입력 상태 없이 None을 반환한다."""

    def _snapshot_state(self) -> None:
        """실행기가 후보점을 갱신하기 전에 사용할 무상태 스냅샷을 반환한다."""
        return None

    def _restore_state(self, state: None) -> None:
        """무상태 기법의 스냅샷을 복원하며 실제로 변경할 상태는 없다."""

    def step(self, theta: ArrayLike, gradient: ArrayLike) -> NDArray[np.float64]:
        """두 입력을 수정하지 않고 shape (n,)의 theta-lr*gradient를 반환한다."""
        point, derivative = _step_inputs(theta, gradient)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            candidate = point - self.lr * derivative	# 현재 기울기의 반대 방향으로 학습률만큼 이동한다.
        return _finite_result(candidate, "GD update")	# 유한한 후보점만 다음 최적화 단계로 전달한다.


class Momentum(VanillaGD):
    """v_next=beta*v+gradient, theta_next=theta-lr*v_next, v0=0인 Momentum을 적용한다.

    lr은 양의 유한 스칼라이고 beta는 [0,1)의 유한 스칼라다. 입력 shape는
    (n,)이며 reset() 전까지 이미 정해진 속도 벡터의 shape와 같아야 한다.
    실패한 갱신은 속도에 반영하지 않는다. velocity는 변경 가능한 내부 상태가
    아닌 복사본을 반환한다. 이 구현의 규약에는 (1-beta) 계수가 없다.
    """

    def __init__(self, lr: float, beta: float = 0.9):
        """스칼라 하이퍼파라미터를 검증하고 속도를 None으로 초기화한다."""
        super().__init__(lr)
        self.beta = _finite_scalar(beta, "beta")
        if not 0 <= self.beta < 1:	# 이전 속도의 가중치가 감쇠하는 범위를 유지한다.
            raise ValueError("beta must satisfy 0 <= beta < 1")
        self._velocity = None	# 다음 갱신에서 입력 차원에 맞게 속도를 초기화하도록 표시한다.

    @property
    def velocity(self) -> NDArray[np.float64] | None:
        """속도 (n,)의 복사본을 반환하고 첫 갱신 전 또는 초기화 직후에는 None을 반환한다."""
        return None if self._velocity is None else self._velocity.copy()	# 반환 배열의 수정이 내부 속도에 영향을 주지 않게 한다.

    def reset(self) -> None:
        """속도를 지워 다음 갱신에서 입력에 맞는 영벡터로 시작한다."""
        self._velocity = None	# 다음 갱신에서 입력 차원에 맞게 속도를 초기화하도록 표시한다.

    def _snapshot_state(self) -> NDArray[np.float64] | None:
        """독립된 속도 (n,) 스냅샷을 반환하고 초기화 전에는 None을 반환한다."""
        return self.velocity	# 복사본 속성으로 이후 갱신과 분리된 복원 지점을 만든다.

    def _restore_state(self, state: NDArray[np.float64] | None) -> None:
        """이 기법이 만든 스냅샷을 독립 복사본으로 복원한다."""
        self._velocity = None if state is None else state.copy()	# 스냅샷과 내부 속도의 배열 공유를 피한다.

    def step(self, theta: ArrayLike, gradient: ArrayLike) -> NDArray[np.float64]:
        """유한한 Momentum 갱신값을 반환하고 검증된 속도를 한 번에 상태에 반영한다."""
        point, derivative = _step_inputs(theta, gradient)
        if self._velocity is not None and self._velocity.shape != point.shape:	# 초기화 없이 서로 다른 차원의 속도를 재사용하지 못하게 한다.
            raise ValueError("reset momentum before changing the parameter shape")
        previous = np.zeros_like(point) if self._velocity is None else self._velocity	# 첫 단계에서는 입력과 같은 shape의 영벡터로 시작한다.
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            velocity = self.beta * previous + derivative	# 현재 기울기에 감쇠된 이전 속도를 더한다.
            candidate = point - self.lr * velocity	# 누적 방향으로 이동할 후보점을 계산한다.
        _finite_result(velocity, "momentum velocity")	# 상태에 반영하기 전에 누적 속도의 유한성을 검증한다.
        _finite_result(candidate, "momentum update")	# 속도가 유한해도 최종 이동에서 넘칠 수 있으므로 별도로 검사한다.
        self._velocity = velocity.copy()	# 모든 검증이 끝난 단계의 속도만 상태에 반영한다.
        return candidate


@dataclass(frozen=True)
class OptimizationResult:
    """수락된 경로 path(k+1,n), losses/gradient_norms(k+1)과 갱신 횟수 k를 담는다.

    status는 budget_exhausted, gradient_tolerance 또는 nonfinite다.
    결과 객체가 배열을 소유하며 모든 값은 유한하다. 필드를 고정해도 내부
    NumPy 배열까지 불변이 되는 것은 아니다. 거부한 후보점은 모든 기록에서
    제외하므로 iterations는 len(path)-1과 같다.
    """

    path: NDArray[np.float64]
    losses: NDArray[np.float64]
    gradient_norms: NDArray[np.float64]
    iterations: int
    status: str


def _output_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    """콜백의 자료형·shape를 검증하여 비유한 출력과 입력 계약 위반을 구분한다."""
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numbers, excluding booleans")
    if array.shape != shape:	# 콜백이 반환한 좌표 수나 손실의 스칼라 계약을 정확히 확인한다.
        raise ValueError(f"{name} must have shape {shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        array = array.astype(np.float64, copy=True)	# 콜백 배열을 독립적인 계산 상태로 보관한다.
    return _finite_result(array, name)


def _stable_norm(vector: np.ndarray) -> float:
    """유한 벡터의 유클리드 노름을 척도 조정으로 계산하고 결과 오버플로를 거부한다."""
    scale = float(np.max(np.abs(vector)))	# 큰 값의 제곱에서 생기는 넘침을 피할 기준 척도를 구한다.
    if scale == 0:	# 영벡터는 정규화 분모를 만들지 않고 노름 0으로 처리한다.
        return 0.0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = scale * float(np.sqrt(np.sum((vector / scale)**2)))	# 척도를 줄인 벡터의 노름을 계산한 뒤 원래 단위로 복원한다.
    if not np.isfinite(result):
        raise _NonfiniteResult("gradient norm exceeds float64 range")
    return result


def _evaluate(objective: Callable, gradient: Callable, point: np.ndarray) -> tuple[float, np.ndarray, float]:
    """독립된 점 복사본을 평가하여 스칼라 손실, gradient(n,), 노름을 반환한다."""
    loss = _output_array(objective(point.copy()), (), "objective output")	# 손실 콜백의 변형을 격리하고 스칼라 반환을 검증한다.
    derivative = _output_array(gradient(point.copy()), point.shape, "gradient output")	# 기울기도 별도 점 복사본으로 평가하여 상태 오염을 막는다.
    return float(loss), derivative, _stable_norm(derivative)


def run_optimization(
    objective: Callable[[NDArray[np.float64]], float],
    gradient: Callable[[NDArray[np.float64]], ArrayLike],
    initial: ArrayLike,
    optimizer: VanillaGD,
    steps: int = 100,
    gradient_tolerance: float | None = None,
) -> OptimizationResult:
    """스칼라 목적함수와 점 (n,)에 대해 지정한 갱신 예산으로 최적화를 실행한다.

    매 실행마다 최적화기 상태를 초기화한다. 기본 steps=100은 수락된 갱신을
    최대 100회 수행한다는 뜻이며 초기 지표도 기록한다. 선택적인 음수 아닌
    값 gradient_tolerance는 초기점까지 포함하여 기울기 노름에 따른 조기
    종료를 허용한다. objective는 스칼라, gradient는 (n,)을 반환해야 한다.
    각 콜백은 독립 복사본을 받는다. 초기 비유한 지표는 ValueError를
    발생시킨다. 이후의 비유한 출력이나 내부 계산은 status nonfinite로 종료하고
    마지막으로 수락된 최적화기 상태를 복원한다. 잘못된 자료형·shape와 사용자
    콜백 예외도 상태를 먼저 복원한 뒤 그대로 전달한다. optimizer는 VanillaGD
    또는 그 스냅샷·복원 계약을 구현한 하위 클래스(Momentum, Adam, Newton)여야
    한다. 입력 initial은 보존한다.
    """
    if not callable(objective) or not callable(gradient):
        raise TypeError("objective and gradient must be callable")
    if not isinstance(optimizer, VanillaGD):	# 공통 갱신·상태 복원 계약을 지원하는 최적화기만 사용한다.
        raise TypeError("optimizer must be VanillaGD or a compatible subclass")
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)):
        raise TypeError("steps must be an integer")
    if steps < 0:	# 갱신 예산 0은 허용하지만 음수 예산은 거부한다.
        raise ValueError("steps must be nonnegative")
    tolerance = None if gradient_tolerance is None else _finite_scalar(gradient_tolerance, "gradient_tolerance")
    if tolerance is not None and tolerance < 0:	# 기울기 노름의 종료 기준은 음수가 될 수 없다.
        raise ValueError("gradient_tolerance must be nonnegative")
    point = _vector(initial, "initial").copy()	# 실행 중 후보점 교체가 사용자의 초기점에 영향을 주지 않게 한다.
    optimizer.reset()	# 서로 독립인 실험 사이에 누적 상태를 이어받지 않는다.
    try:
        loss, derivative, norm = _evaluate(objective, gradient, point)
    except _NonfiniteResult as error:
        raise ValueError("initial objective, gradient, and norm must be finite") from error
    path, losses, norms = [point.copy()], [loss], [norm]	# 초기점을 포함해 모든 기록의 시점을 일치시킨다.
    status = "budget_exhausted"	# 다른 종료 원인이 없으면 갱신 예산 소진으로 기록한다.
    for _ in range(int(steps)):
        if tolerance is not None and norm <= tolerance:	# 새 갱신 전에 초기점 또는 현재점의 종료 조건을 확인한다.
            status = "gradient_tolerance"
            break
        previous_state = optimizer._snapshot_state()	# 후보점이나 콜백이 실패하면 되돌릴 상태를 먼저 보존한다.
        try:
            candidate = optimizer.step(point, derivative)	# 현재점의 기울기로 하나의 후보 갱신을 만든다.
            next_loss, next_derivative, next_norm = _evaluate(objective, gradient, candidate)	# 후보점의 모든 지표까지 유효해야 갱신을 수락한다.
        except _NonfiniteResult:
            optimizer._restore_state(previous_state)	# 거부한 후보점의 모멘트·속도·횟수가 상태에 남지 않게 한다.
            status = "nonfinite"	# 계산 범위 초과는 유한한 기존 기록을 보존한 채 종료한다.
            break
        except Exception:
            optimizer._restore_state(previous_state)	# 거부한 후보점의 모멘트·속도·횟수가 상태에 남지 않게 한다.
            raise
        point, loss, derivative, norm = candidate, next_loss, next_derivative, next_norm	# 검증을 통과한 점과 같은 시점의 지표를 한꺼번에 채택한다.
        path.append(point.copy())	# 수락한 점만 독립 복사본으로 경로에 남긴다.
        losses.append(loss)	# 경로와 손실 기록의 길이를 맞춘다.
        norms.append(norm)	# 동일 시점의 기울기 노름도 함께 기록한다.
    if status == "budget_exhausted" and tolerance is not None and norm <= tolerance:	# 마지막 허용 갱신에서 달성한 종료 조건도 놓치지 않는다.
        status = "gradient_tolerance"
    return OptimizationResult(np.array(path), np.array(losses), np.array(norms), len(path)-1, status)	# 초기점을 제외한 수락 갱신 수와 정렬된 기록을 반환한다.


def elliptic(theta: ArrayLike) -> float:
    """유한 실수 점 (2,)의 x**2+10*y**2를 반환하며 넘침은 FloatingPointError로 알린다."""
    point = _vector(theta, "theta")
    if point.shape != (2,):	# 고정 타원형 목적함수는 두 좌표만 받는다.
        raise ValueError("elliptic theta must have shape (2,)")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = np.asarray(point[0]**2 + 10.0*point[1]**2)	# y 방향의 곡률이 더 큰 이차 목적함수를 계산한다.
    return float(_finite_result(result, "elliptic objective"))


def elliptic_gradient(theta: ArrayLike) -> NDArray[np.float64]:
    """유한 실수 점 (2,)의 (2*x,20*y)를 반환하며 넘침은 FloatingPointError로 알린다."""
    point = _vector(theta, "theta")
    if point.shape != (2,):	# 고정 타원형 목적함수는 두 좌표만 받는다.
        raise ValueError("elliptic theta must have shape (2,)")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = point * np.array([2.0, 20.0])	# 각 좌표의 해석 미분 계수로 기울기를 구한다.
    return _finite_result(result, "elliptic gradient")



def optimizer_comparison_figure(runs: dict, objective, title: str):
    """2차원 경로, 손실, 기울기 노름을 비교하는 Figure와 Axes(3,)를 반환한다.

    runs는 표시 이름을 유한한 path(k+1,2), losses(k+1,),
    gradient_norms(k+1,) 기록에 대응시킨다. objective는 두 격자 배열을 받아
    같은 shape를 반환해야 한다. 실험 호출자가 검증된 경로 기록을 제공한다.
    경로의 두 좌표축은 같은 축척을 사용하며 symlog는 정확한 0 손실·노름을
    보존한다. 파일은 저장하지 않고 호출자가 Figure를 닫는다. 잘못된 기록은
    예외를 발생시킨다.
    """
    import matplotlib.pyplot as plt
    from src.utils.plotting import contour_grid

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    try:
        extent = max(1.0, max(float(np.max(np.abs(run["path"]))) for run in runs.values())) * 1.15
        x, y, z = contour_grid(objective, (-extent, extent), (-extent, extent), points=101)
        contours = axes[0].contour(x, y, z, levels=12, cmap="Greys", alpha=0.55)
        axes[0].clabel(contours, inline=True, fontsize=7)
        for index, (label, run) in enumerate(runs.items()):
            color = ("#2466b0", "#cf4b16", "#29864a")[index % 3]
            path = np.asarray(run["path"])
            updates = np.arange(len(path))	# 초기점 0부터 수락된 갱신 횟수까지 같은 가로축을 사용한다.
            axes[0].plot(path[:, 0], path[:, 1], "o--", markersize=2.5,
                         linewidth=1.2, color=color, label=label)
            axes[0].scatter(*path[0], marker="s", color=color, s=45, zorder=4)
            axes[0].scatter(*path[-1], marker="x", color=color, s=50, zorder=5)
            axes[1].plot(updates, run["losses"], color=color, label=label)
            axes[2].plot(updates, run["gradient_norms"], color=color, label=label)
        axes[0].set(xlabel="x", ylabel="y", title="Paths (square: start, x: end)",
                    xlim=(-extent, extent), ylim=(-extent, extent))
        axes[0].set_aspect("equal", adjustable="box")	# 경로의 각도와 타원 모양이 좌표축 비율로 왜곡되지 않게 한다.
        for axis, ylabel, threshold in zip(axes[1:], ("Objective value", "Gradient norm"), (1e-18, 1e-12)):
            axis.set(xlabel="Completed updates", ylabel=ylabel, title=ylabel)
            axis.set_yscale("symlog", linthresh=threshold)	# 손실·노름의 정확한 0도 제거하지 않고 표시한다.
            axis.set_ylim(bottom=0)	# 음수가 아닌 지표의 의미에 맞게 표시 범위를 정한다.
            axis.grid(True, alpha=0.25)
        for axis in axes:
            axis.legend(fontsize=8)
        figure.suptitle(title, fontsize=12)
        figure.supxlabel("Same initial point and update budget; symlog axes retain zero values.", fontsize=9)
        return figure, axes
    except Exception:
        plt.close(figure)	# 시각화 실패 시 생성된 그림 자원을 회수한다.
        raise


def learning_rate_figure(runs: dict):
    """원형 목적함수에서 GD 학습률 0.5, 1.0, 1.1을 비교하는 Figure와 Axes(3,)를 반환한다.

    runs는 이 숫자 학습률을 검증된 경로 기록에 대응시킨다. 발산 경로가
    다른 사례를 가리지 않도록 각 패널은 독립된 범위에서 x/y 축척을 같게 한다.
    초기·최종 좌표를 포함하며 어떤 값도 클리핑하지 않는다. 파일을 저장하지
    않고 호출자가 그림 정리를 담당한다. 잘못된 기록은 예외를 발생시킨다.
    """
    import matplotlib.pyplot as plt
    from src.calculus import quadratic
    from src.utils.plotting import contour_grid

    figure, axes = plt.subplots(1, 3, figsize=(14.4, 4.8), layout="constrained")
    try:
        descriptions = {0.5: "One-step convergence", 1.0: "Undamped sign oscillation", 1.1: "Growing oscillation"}
        for axis, rate in zip(axes, (0.5, 1.0, 1.1)):
            run = runs[rate]
            path = np.asarray(run["path"])
            extent = max(1.0, float(np.max(np.abs(path)))) * 1.15	# 각 학습률의 전체 경로가 보이도록 독립된 좌표 범위를 정한다.
            x, y, z = contour_grid(quadratic, (-extent, extent), (-extent, extent), points=81)
            axis.contour(x, y, z, levels=10, cmap="Greys", alpha=0.5)
            axis.plot(path[:, 0], path[:, 1], "o--", color="#2466b0", markersize=4, linewidth=1,
                      label="GD path")
            axis.scatter(*path[0], marker="s", color="#cf4b16", s=55, zorder=5, label="Start")
            axis.scatter(*path[-1], marker="x", color="#202020", s=70, zorder=6, label="End")
            axis.set(xlabel="x", ylabel="y", xlim=(-extent, extent), ylim=(-extent, extent),
                     title=f"lr={rate:g}: {descriptions[rate]}\nFinal radius={run['final_norm']:.4g}")
            axis.set_aspect("equal", adjustable="box")	# 각 패널 안에서 x와 y의 단위 길이를 같게 한다.
            axis.legend(fontsize=8, loc="upper left")
        figure.suptitle("Circle GD: theta[t+1] = (1 - 2 lr) theta[t]; start=(5,5), 20 updates", fontsize=12)
        figure.supxlabel("Each panel has its own coordinate range; lr=0.5 does not diverge.", fontsize=9)
        return figure, axes
    except Exception:
        plt.close(figure)	# 시각화 실패 시 생성된 그림 자원을 회수한다.
        raise
