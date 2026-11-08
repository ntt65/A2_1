"""NumPy 선형 변환, 독립적인 면적 검증과 행렬 분해를 구현한다."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.utils.math import _as_finite_real_array
from src.utils.numerical import _finite_scalar

from dataclasses import dataclass
from matplotlib.axes import Axes
from src.utils.cache_manager import CacheManager, make_cache_key

__concepts__ = {
    "Power_Iteration": ["PowerIterationResult", "power_iteration"],
    "SVD_Decomposition": ["svd_decompose"],
    "SVD_Reconstruction": ["reconstruct_svd"],
    "SVD_Compression": ["compress_image"],
    "SVD_Comparison": ["svd_comparison_figure"],
    "Rotation_Matrix": ["rotation_matrix"],
    "Scaling_Matrix": ["scaling_matrix"],
    "Shear_Matrix": ["shear_matrix"],
    "Linear_Transform": ["unit_circle", "transform_points", "transformation_figure"],
    "Determinant_Area": ["polygon_area", "determinant_area_check"],
}


def _transform_matrix(matrix: ArrayLike) -> NDArray[np.float64]:
    """유한 실수의 2x2 변환 행렬을 검증하며 잘못된 shape는 ValueError로 알린다."""
    result = _as_finite_real_array(matrix, "matrix")
    if result.shape != (2, 2):	# 평면의 선형 변환에 맞는 두 행과 두 열을 요구한다.
        raise ValueError("matrix must have shape (2, 2)")
    return result


def _plane_points(points: ArrayLike, minimum: int = 1) -> NDArray[np.float64]:
    """n >= minimum인 shape (2, n)의 유한 실수 열벡터 점들을 검증한다."""
    result = _as_finite_real_array(points, "points")
    if result.ndim != 2 or result.shape[0] != 2 or result.shape[1] < minimum:	# 각 열을 점으로 해석할 수 있는 shape와 최소 점 수를 확인한다.
        raise ValueError(f"points must have shape (2, n) with n >= {minimum}")
    return result


def rotation_matrix(theta: float) -> NDArray[np.float64]:
    """라디안 각도 theta의 R(theta)=[[cos,-sin],[sin,cos]]를 shape (2,2)로 반환한다."""
    angle = _finite_scalar(theta, "theta")	# 각도도 유한 실수 스칼라라는 공통 계약으로 검증한다.
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)	# 열벡터에 적용하는 반시계 회전 행렬을 구성한다.


def scaling_matrix(sx: float, sy: float) -> NDArray[np.float64]:
    """shape (2,2)의 diag(sx,sy)를 반환하며 유한한 음수·0 배율도 허용한다."""
    return np.diag([_finite_scalar(sx, "sx"), _finite_scalar(sy, "sy")])	# 두 축의 배율을 독립적으로 적용하며 음수와 0도 보존한다.


def shear_matrix(k: float) -> NDArray[np.float64]:
    """유한 스칼라 k의 수평 전단 [[1,k],[0,1]]을 shape (2,2)로 반환한다."""
    return np.array([[1.0, _finite_scalar(k, "k")], [0.0, 1.0]])	# x 좌표에 k*y만 더하는 수평 전단을 구성한다.


def unit_circle(n: int = 400) -> NDArray[np.float64]:
    """끝점을 중복하지 않고 순서대로 나열한 단위원의 n개 꼭짓점을 shape (2,n)로 반환한다.

    n은 3 이상의 정수여야 한다. 꼭짓점을 이은 다각형은 원의 근사이며,
    변환 전후 다각형 면적의 비를 비교하면 이 표본화 오차가 상쇄된다.
    """
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise TypeError("n must be an integer")
    if n < 3:	# 면적을 정의할 다각형에는 최소 세 꼭짓점이 필요하다.
        raise ValueError("n must be at least 3")
    angles = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)	# 같은 끝점을 두 번 넣지 않고 원 둘레를 균등 표본화한다.
    return np.stack((np.cos(angles), np.sin(angles)))	# 각 점이 열벡터가 되도록 좌표를 shape (2,n)로 쌓는다.


def transform_points(matrix: ArrayLike, points: ArrayLike) -> NDArray[np.float64]:
    """A(2,2) @ P(2,n)을 적용하여 유한한 float64 열벡터 점 (2,n)을 반환한다.

    잘못된 입력은 TypeError 또는 ValueError를 발생시킨다. 결과를 표현할 수
    없으면 비유한 좌표를 반환하는 대신 ValueError를 발생시킨다.
    """
    transform, vertices = _transform_matrix(matrix), _plane_points(points)	# 행렬과 점의 계약을 각각 검증한 뒤 곱한다.
    with np.errstate(over="ignore", invalid="ignore"):
        result = transform @ vertices	# 모든 열벡터에 동일한 선형 변환을 한 번에 적용한다.
    if not np.all(np.isfinite(result)):	# 유한 입력의 행렬곱에서도 생길 수 있는 넘침을 차단한다.
        raise ValueError("transformed coordinates are not finite")
    return result


def polygon_area(points: ArrayLike) -> float:
    """순서가 있는 단순 다각형 (2,n)의 음수 아닌 면적을 신발끈 공식으로 구한다.

    n>=3이며 꼭짓점은 자기 교차 없이 경계 순서로 주어져야 한다. 시계·반시계
    방향과 닫는 꼭짓점의 중복을 모두 허용한다. 계산은 변환 행렬의 행렬식과
    독립적이다. 퇴화한 다각형은 0을 반환하고 표현할 수 없는 면적은
    ValueError를 발생시킨다.
    """
    vertices = _plane_points(points, minimum=3)
    # 원점에서 먼 다각형은 평행 이동으로 뺄셈의 상쇄 오차를 줄인다.
    with np.errstate(over="ignore", invalid="ignore"):
        centered = vertices - vertices[:, :1]	# 면적이 평행 이동에 불변임을 이용해 좌표의 크기를 줄인다.
        x, y = centered
        area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))	# 인접 꼭짓점의 교차곱을 합한 신발끈 공식으로 면적을 구한다.
    if not np.isfinite(area):	# 면적 계산이 float64 범위를 벗어나면 검증값으로 사용하지 않는다.
        raise ValueError("polygon area is not representable as a finite float64")
    return float(area)


def _determinant_2x2(matrix: NDArray[np.float64]) -> float:
    """척도를 조정한 (2,2) 행렬에서 ad-bc를 계산하고 float64 범위 손실을 거부한다.

    직접 교차곱은 동일한 두 열에서 정확한 0을 보존한다. 이 경우 LU 기반
    행렬식은 반올림으로 잘못된 비영 값을 남길 수 있다. 척도 조정이나 곱셈이
    비영 성분을 없애거나, 비영 행렬식을 원래 척도로 복원할 때 언더플로·
    오버플로가 발생하면 ValueError를 발생시킨다.
    """
    scale = np.max(np.abs(matrix))	# 교차곱 전에 행렬 성분의 크기를 줄일 기준을 구한다.
    if scale == 0:	# 영행렬은 척도로 나누지 않고 정확한 행렬식 0을 반환한다.
        return 0.0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        normalized = matrix / scale	# 곱셈의 오버플로 위험을 줄이도록 모든 성분에 같은 척도를 적용한다.
        if np.any((matrix != 0) & (normalized == 0)):	# 척도 조정 중 비영 성분이 사라지면 다른 행렬로 계산하지 않는다.
            raise ValueError("matrix scaling loses nonzero entries in float64")
        first = normalized[0, 0] * normalized[1, 1]	# 2x2 행렬식의 첫 교차곱 ad를 계산한다.
        second = normalized[0, 1] * normalized[1, 0]	# 두 번째 교차곱 bc를 독립적으로 계산한다.
        if ((normalized[0, 0] != 0 and normalized[1, 1] != 0 and first == 0)
                or (normalized[0, 1] != 0 and normalized[1, 0] != 0 and second == 0)):
            raise ValueError("determinant cross product underflows in float64")
        scaled = first - second	# 동일한 열의 교차곱을 직접 빼서 정확한 0을 보존한다.
        determinant = (scaled * scale) * scale	# 행렬식이 척도의 제곱만큼 변하는 관계로 원래 단위를 복원한다.
    if not np.isfinite(determinant) or (scaled != 0 and determinant == 0):	# 원래 척도로 복원할 때의 넘침과 비영 값 소실을 모두 거부한다.
        raise ValueError("determinant is not representable in float64")
    return float(determinant)


def determinant_area_check(
    matrix: ArrayLike,
    points: ArrayLike | None = None,
    relative_tolerance: float = 0.01,
    absolute_tolerance: float = 1e-10,
) -> dict[str, float | str | bool]:
    """독립적으로 구한 면적비를 abs(det(A))와 비교하여 오차 지표를 반환한다.

    A의 shape는 (2,2)이고 선택적인 단순 다각형 점들의 shape는 (2,n)이다.
    행렬식이 0이 아니면 상대오차, 0이면 면적비의 절대오차를 사용한다.
    두 허용치는 유한하고 음수가 아니어야 한다. 변환 전 면적이 0이거나
    행렬식·면적비를 표현할 수 없으면 ValueError를 발생시킨다. 2x2 행렬식은
    다각형 면적과 독립적으로 척도를 조정한 ad-bc로 계산한다.
    """
    transform = _transform_matrix(matrix)
    relative_tolerance = _finite_scalar(relative_tolerance, "relative_tolerance")
    absolute_tolerance = _finite_scalar(absolute_tolerance, "absolute_tolerance")
    if relative_tolerance < 0 or absolute_tolerance < 0:	# 오차 허용치는 음수가 될 수 없다.
        raise ValueError("area tolerances must be nonnegative")
    vertices = unit_circle() if points is None else _plane_points(points, minimum=3)	# 사용자 다각형이 없으면 같은 표본의 단위원을 기준으로 사용한다.
    before = polygon_area(vertices)	# 분모가 될 변환 전 면적을 좌표로 직접 계산한다.
    if before == 0:	# 면적비를 정의할 수 없는 원본 다각형은 거부한다.
        raise ValueError("source polygon must have positive area")
    determinant = _determinant_2x2(transform)	# 면적 계산과 독립적인 ad-bc를 비교 기준으로 구한다.
    after = polygon_area(transform_points(transform, vertices))	# 변환된 실제 좌표에서 면적을 다시 측정한다.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ratio = after / before	# 동일한 꼭짓점 표본의 변환 전후 면적비를 구한다.
        expected = abs(determinant)	# 방향 반전을 제외한 면적 크기 변화는 행렬식의 절댓값이다.
        absolute_error = abs(ratio - expected)
        error = absolute_error / expected if expected != 0 else absolute_error	# 특이 행렬에서는 0을 분모로 쓰지 않고 절대오차를 적용한다.
    if not all(np.isfinite(value) for value in (determinant, ratio, error)):
        raise ValueError("determinant or area-ratio error is not finite")
    kind = "relative" if expected != 0 else "absolute"
    tolerance = relative_tolerance if expected != 0 else absolute_tolerance	# 오차 종류에 맞는 허용치를 선택한다.
    return {
        "determinant": determinant, "area_before": before, "area_after": after,
        "area_ratio": ratio, "expected_ratio": expected,
        "error": error, "error_kind": kind, "tolerance": tolerance,
        "passed": bool(error <= tolerance),	# 실측 오차를 선택된 허용치와 직접 비교한다.
    }


def transformation_figure(n: int = 400) -> tuple[Figure, np.ndarray]:
    """회전·스케일링·전단을 비교하는 Figure와 세 Axes를 반환한다.

    각 패널에서 원래 점과 변환한 원의 점 (2,n)을 겹쳐 표시한다. 화살표는
    같은 pi/4 방향을 추적하므로 원 모양이 같아도 회전을 확인할 수 있다.
    창을 띄우거나 파일을 저장하지 않으며 그림 정리는 호출자가 담당한다.
    """
    original = unit_circle(n)
    reference = np.array([[np.sqrt(0.5)], [np.sqrt(0.5)]])	# 원의 윤곽이 겹쳐도 회전을 볼 수 있게 pi/4 방향을 추적한다.
    examples = (
        ("Rotation: 45 degrees", rotation_matrix(np.pi / 4)),
        ("Scaling: (2, 0.5)", scaling_matrix(2, 0.5)),
        ("Horizontal shear: k=1", shear_matrix(1)),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), layout="constrained")
    try:
        for axis, (title, matrix) in zip(axes, examples):
            transformed = transform_points(matrix, original)
            for label, vertices, color, style in (
                ("Original", original, "#777777", "--"),
                ("Transformed", transformed, "#1565C0", "-"),
            ):
                closed = np.column_stack((vertices, vertices[:, 0]))	# 계산용 표본은 유지하고 그릴 때만 경계선을 닫는다.
                axis.plot(closed[0], closed[1], style, color=color, label=label)
            for endpoint, color in (
                (reference[:, 0], "#777777"),
                (transform_points(matrix, reference)[:, 0], "#C45100"),
            ):
                axis.annotate("", xy=endpoint, xytext=(0, 0),
                              arrowprops={"arrowstyle": "->", "color": color, "lw": 2})
                axis.plot(*endpoint, "o", color=color, markersize=4)
            result = determinant_area_check(matrix, original)
            axis.set(title=f"{title}\ndet={result['determinant']:.2f}, area ratio={result['area_ratio']:.2f}",
                     xlabel="x", ylabel="y")
            axis.axhline(0, color="#dddddd", linewidth=0.7)
            axis.axvline(0, color="#dddddd", linewidth=0.7)
            axis.grid(alpha=0.2)
            axis.set_aspect("equal", adjustable="box")	# 축척 차이가 변환의 기하적 의미를 왜곡하지 않도록 한다.
            axis.legend(loc="upper right", fontsize=8)
    except Exception:
        plt.close(figure)	# 그리기 중 오류가 나면 생성한 그림 자원을 회수한다.
        raise
    return figure, axes


@dataclass(frozen=True)
class PowerIterationResult:
    """실수 고유쌍의 근삿값과 명시적인 종료 정보를 담는다.

    eigenvector는 호출자가 소유한 단위 노름의 float64 배열 (n,)이며,
    eigenvalue와 residual은 스칼라다. iterations는 행렬·벡터 곱의 정규화
    갱신 횟수를 세므로 초기 고유쌍이 정확하면 갱신은 0회다. converged는
    상대 고유쌍 잔차가 요청한 기준을 만족했다는 뜻이며, 고유값의 절댓값이
    최대라는 증명은 아니다. reason은 residual_tolerance, zero_matrix,
    nullspace_breakdown 또는 max_iterations다. 필드는 고정되어 있지만
    독립적으로 소유한 벡터의 원소는 수정할 수 있다.
    """

    eigenvalue: float
    eigenvector: NDArray[np.float64]
    iterations: int
    residual: float
    converged: bool
    reason: str


def _power_normalize(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """유한한 비영 벡터 (n,)를 원래 크기 그대로 제곱하지 않고 정규화한다."""
    peak = float(np.max(np.abs(vector)))	# 원래 벡터를 제곱하기 전에 최대 성분으로 척도를 조정한다.
    if peak == 0.0:	# 영벡터는 나눗셈 전에 정규화 또는 잔차 규약에 맞게 처리한다.
        raise ValueError("initial vector must not be zero")
    with np.errstate(under="ignore"):
        scaled = vector / peak	# 극대·극소 성분의 제곱에서 정밀도를 잃을 위험을 줄인다.
    return scaled / np.linalg.norm(scaled)	# 안전한 척도에서 단위 노름 벡터를 만든다.


def _power_residual_norm(vector: NDArray[np.float64]) -> float:
    """아주 작은 성분의 제곱이 언더플로를 일으키지 않도록 잔차 벡터의 크기를 구한다."""
    peak = float(np.max(np.abs(vector)))	# 원래 벡터를 제곱하기 전에 최대 성분으로 척도를 조정한다.
    if peak == 0.0:	# 영벡터는 나눗셈 전에 정규화 또는 잔차 규약에 맞게 처리한다.
        return 0.0
    with np.errstate(under="ignore"):
        scaled = vector / peak	# 극대·극소 성분의 제곱에서 정밀도를 잃을 위험을 줄인다.
    return float(peak * np.linalg.norm(scaled))	# 잔차 노름을 원래 척도로 되돌린다.


def _power_eigenvalue(scaled_value: float, matrix_scale: float) -> float:
    """스칼라 고유값의 원래 척도를 복원하고 float64 범위 오류를 거부한다."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        value = float(np.float64(scaled_value) * np.float64(matrix_scale))	# 정규화 행렬에서 얻은 고유값을 원래 행렬의 단위로 복원한다.
    if not np.isfinite(value) or (scaled_value != 0.0 and value == 0.0):	# 고유값 복원에서의 오버플로와 비영 값 소실을 모두 검출한다.
        raise ValueError("the estimated eigenvalue is not representable in float64")
    return value


def power_iteration(
    A: ArrayLike,
    initial: ArrayLike | None = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> PowerIterationResult:
    """정규화한 행렬·벡터 곱을 반복하여 실수 고유쌍을 계산한다.

    A는 n >= 1인 shape (n,n)의 유한 실수 정방 배열이다. initial은 0이 아닌
    유한 실수 벡터 (n,)이며 생략하면 NumPy 전역 난수 생성기로 정규분포 벡터를
    뽑는다. 실험 시작점에서 공통 set_seed(42)를 호출해야 하며 이 함수 자체는
    시드를 초기화하지 않는다. 입력은 수정하지 않는다.

    갱신식은 v <- A@v / ||A@v||2이고 lambda는 Rayleigh 몫이다.
    ||A@v-lambda*v||2/(||A||F*||v||2) <= tol이거나 max_iter회 갱신하면 종료한다.
    행렬의 척도는 상대 잔차에서 상쇄되며 노름의 오버플로를 방지한다. 잔차
    노름도 절댓값 최댓값으로 척도를 조정하므로 극소 성분의 제곱 때문에
    정확히 수렴했다고 잘못 판정하지 않는다. tol은 양의 유한 스칼라이고
    max_iter는 양의 정수여야 한다. 자료형·shape·값 오류는 TypeError 또는
    ValueError를 발생시킨다. 척도 조정이 행렬의 비영 성분을 없애거나 고유값이
    float64 범위를 벗어나면 문제를 조용히 바꾸는 대신 ValueError로 거부한다.

    영행렬은 정확한 0 고유쌍을 반환한다. 비영 행렬에서 반복 벡터가 사라지면
    converged=False와 nullspace_breakdown을 반환한다. 음의 지배 고유값에서는
    v의 부호가 뒤집힐 수 있으나 잔차 기준으로 수렴을 판단한다. 절댓값이 같은
    고유값이나 복소 지배 성분은 반복 예산을 소진할 수 있다. 지배 고유쌍에
    도달하려면 초기 벡터에 적절한 성분도 있어야 한다. 절댓값이 최대가 아닌
    정확한 고유벡터도 즉시 잔차 기준을 만족할 수 있다. 고유값 풀이 함수를
    호출하지 않으며 일반적인 수렴 보장을 주장하지 않는다.
    """
    matrix = _as_finite_real_array(A, "A")
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:	# 비어 있지 않은 정방행렬에 대해서만 반복 고유쌍을 정의한다.
        raise ValueError("A must be a nonempty square matrix of shape (n,n)")
    tolerance = _as_finite_real_array(tol, "tol")
    if tolerance.shape != () or float(tolerance) <= 0.0:	# 잔차 종료 기준을 양의 유한 스칼라로 제한한다.
        raise ValueError("tol must be a positive finite scalar")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)):
        raise TypeError("max_iter must be an integer")
    if max_iter < 1:	# 반복 예산은 양의 정수여야 한다.
        raise ValueError("max_iter must be positive")
    vector = _as_finite_real_array(
        np.random.normal(size=matrix.shape[0]) if initial is None else initial,	# 초기 벡터가 없는 경우에만 실험 시작점의 난수 상태를 사용한다.
        "initial",
    )
    if vector.shape != (matrix.shape[0],):	# 초기 벡터가 행렬 차원과 정확히 맞는지 확인한다.
        raise ValueError("initial must have shape (n,)")
    vector = _power_normalize(vector)	# 초기 방향을 단위 노름으로 맞춘다.
    matrix_scale = float(np.max(np.abs(matrix)))	# 행렬곱과 잔차 노름을 안전한 척도에서 계산할 기준을 구한다.
    if matrix_scale == 0.0:
        return PowerIterationResult(0.0, vector.copy(), 0, 0.0, True, "zero_matrix")	# 영행렬은 갱신 없이 정확한 0 고유쌍으로 종료한다.
    with np.errstate(under="ignore"):
        scaled_matrix = matrix / matrix_scale	# 스칼라 배율은 고유벡터와 상대 잔차를 바꾸지 않는다.
    if np.any((matrix != 0.0) & (scaled_matrix == 0.0)):	# 행렬의 비영 성분을 잃는 척도 조정은 문제를 바꾸므로 거부한다.
        raise ValueError("A's dynamic range cannot be preserved when scaled in float64")
    matrix_norm = float(np.linalg.norm(scaled_matrix, ord="fro"))	# 상대 잔차 분모에 사용할 Frobenius 노름을 한 번 구한다.

    for iterations in range(int(max_iter) + 1):	# 초기점과 마지막 허용 갱신점까지 잔차를 평가한다.
        product = scaled_matrix @ vector	# 역행렬이나 고유값 풀이 없이 행렬·벡터 곱으로 방향을 강화한다.
        if not np.any(product):	# 비영 행렬에서 방향이 영공간에 들어가면 수렴과 구분하여 종료한다.
            return PowerIterationResult(
                0.0, vector.copy(), iterations, 0.0, False, "nullspace_breakdown"
            )
        vector_norm = float(np.linalg.norm(vector))
        scaled_value = float(np.dot(vector, product) / np.dot(vector, vector))	# 현재 방향의 Rayleigh 몫으로 고유값을 추정한다.
        residual = float(
            _power_residual_norm(product - scaled_value * vector) / (matrix_norm * vector_norm)	# 척도에 독립적인 고유쌍 상대 잔차를 계산한다.
        )
        eigenvalue = _power_eigenvalue(scaled_value, matrix_scale)	# 결과 고유값만 원래 행렬 척도로 복원한다.
        if residual <= float(tolerance):	# 잔차 기준을 만족했을 때만 수렴 상태를 부여한다.
            return PowerIterationResult(
                eigenvalue, vector.copy(), iterations, residual, True, "residual_tolerance"
            )
        if iterations == max_iter:	# 허용한 갱신 수를 넘겨 추가 반복하지 않는다.
            return PowerIterationResult(
                eigenvalue, vector.copy(), iterations, residual, False, "max_iterations"
            )
        vector = _power_normalize(product)	# 다음 반복을 위해 곱의 방향만 단위 벡터로 남긴다.

    raise RuntimeError("unreachable iteration state")


def _svd_image(image: ArrayLike) -> NDArray[np.float64]:
    """shape (m, n)의 비어 있지 않은 유한 실수 흑백 배열을 검증한다.

    픽셀 값이 [0,1] 안에 있을 필요는 없다. 비실수 입력은 TypeError,
    비유한 값·빈 차원·다른 shape는 ValueError를 발생시킨다.
    """
    values = _as_finite_real_array(image, "image")
    if values.ndim != 2 or 0 in values.shape:	# 흑백 영상은 두 차원이 모두 비어 있지 않아야 한다.
        raise ValueError("image must have nonempty shape (m, n)")
    return values


def _svd_rank(k: int, available: int) -> int:
    """[1, available] 범위의 정수 rank를 검증하고 불리언과 범위 밖 값의 클리핑을 거부한다."""
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")
    if not 1 <= k <= available:	# 요청 rank를 임의로 잘라내지 않고 유효 범위를 직접 검증한다.
        raise ValueError(f"k must lie between 1 and {available}")
    return int(k)


def _svd_factors(
    factors: tuple[ArrayLike, ArrayLike, ArrayLike],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """1 <= r <= min(m,n)인 유한 실수 U(m,r), s(r,), Vt(r,n)을 검증한다.

    특잇값은 음수가 아니고 내림차순이어야 한다. 직교성은 다시 계산하지
    않으므로 호출자는 필요하면 일부 성분만 남긴 올바른 SVD 인자를 제공해야
    한다. 컨테이너·자료형 오류는 TypeError, shape·값 오류는 ValueError를
    발생시킨다.
    """
    if not isinstance(factors, (tuple, list)) or len(factors) != 3:	# U·특잇값·Vt의 세 인자가 함께 주어져야 한다.
        raise TypeError("factors must contain exactly (U, s, Vt)")
    u, singular_values, vt = (
        _as_finite_real_array(value, name)
        for value, name in zip(factors, ("U", "s", "Vt"))
    )
    if u.ndim != 2 or singular_values.ndim != 1 or vt.ndim != 2:	# 행렬·벡터·행렬이라는 SVD 인자의 차원 계약을 확인한다.
        raise ValueError("factor shapes must be U(m,r), s(r,), Vt(r,n)")
    rank = singular_values.size
    if (
        rank == 0 or u.shape[1] != rank or vt.shape[0] != rank
        or rank > min(u.shape[0], vt.shape[1])
    ):
        raise ValueError("factor dimensions are incompatible")
    if np.any(singular_values < 0) or np.any(singular_values[1:] > singular_values[:-1]):	# 큰 특잇값부터 k개를 고르는 전제인 비음수 내림차순을 검증한다.
        raise ValueError("singular values must be nonnegative and descending")
    return u, singular_values, vt


def svd_decompose(
    image: ArrayLike, cache: CacheManager | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """흑백 영상의 축약 SVD U(m,r), s(r,), Vt(r,n), r=min(m,n)을 반환한다.

    미션에서 허용한 NumPy SVD를 사용한다. 공통 캐시가 주어지면 서로 다른
    요청 rank 사이에서 전체 분해를 재사용한다. 키는 float64 영상, 알고리즘
    버전과 NumPy 버전을 포함한다. 입력을 수정하지 않으며 캐시 배열은 독립
    복사본이다. 비유한 값이나 표현 불가능한 인자는 ValueError를 발생시키고,
    분해가 수렴하지 않을 때 NumPy LinAlgError는 그대로 전달한다.
    """
    values = _svd_image(image)
    if cache is not None and not isinstance(cache, CacheManager):	# 동일한 복사·손상 처리 계약을 가진 공통 캐시만 허용한다.
        raise TypeError("cache must be CacheManager or None")

    def compute() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """전체 축약 분해를 한 번 계산하고 인자의 shape·값을 검증한다."""
        with np.errstate(over="ignore", invalid="ignore"):
            return _svd_factors(np.linalg.svd(values, full_matrices=False))	# 축약 SVD를 계산한 뒤 인자의 shape와 유한성을 다시 확인한다.

    if cache is None:	# 캐시를 생략하면 동일한 계산을 직접 수행한다.
        return compute()
    key = make_cache_key(
        "SVD_Decomposition", "v1", values,
        parameters={"full_matrices": False, "numpy_version": np.__version__},	# 분해 옵션과 NumPy 버전이 다른 결과를 구분한다.
    )
    return _svd_factors(cache.get_or_compute(key, compute))	# 캐시 적중 결과에도 인자 검증을 동일하게 적용한다.


def reconstruct_svd(
    factors: tuple[ArrayLike, ArrayLike, ArrayLike], k: int,
) -> NDArray[np.float64]:
    """A_k = U[:, :k] diag(s[:k]) Vt[:k, :]를 shape (m, n)로 반환한다.

    k는 [1, len(s)] 범위의 정수여야 한다. diag(s) 배열을 만드는 대신 열별
    배율을 곱한다. 복원값은 클리핑하지 않으며 오차는 표시 전의 수치 근사값을
    기준으로 측정해야 한다. 인자·rank 오류는 TypeError 또는 ValueError,
    비유한 복원값은 ValueError를 발생시킨다.
    """
    u, singular_values, vt = _svd_factors(factors)
    rank = _svd_rank(k, singular_values.size)
    with np.errstate(over="ignore", invalid="ignore"):
        reconstructed = (u[:, :rank] * singular_values[:rank]) @ vt[:rank, :]	# 대각행렬 할당 없이 U의 열에 특잇값을 곱해 rank-k 근사를 복원한다.
    if not np.all(np.isfinite(reconstructed)):	# 복원 곱셈에서 발생한 비유한 값을 측정 결과로 사용하지 않는다.
        raise ValueError("reconstruction is not representable as finite float64")
    return reconstructed


def compress_image(
    image: ArrayLike, k: int, cache: CacheManager | None = None,
) -> NDArray[np.float64]:
    """클리핑하지 않은 rank-k 근사 영상을 원본의 shape (m,n)로 반환한다.

    압축 파일이 아닌 복원 픽셀을 반환한다. SVD 인자를 저장하는 데 필요한
    숫자는 k*(m+n+1)개이므로 원본 m*n개보다 많을 수 있다. 비용이 큰 SVD를
    계산하거나 캐시하기 전에 rank를 검증한다.
    """
    values = _svd_image(image)
    rank = _svd_rank(k, min(values.shape))	# 비용이 큰 분해를 시작하기 전에 요청 rank를 검증한다.
    return reconstruct_svd(svd_decompose(values, cache), rank)	# 전체 분해를 재사용하고 선택한 rank의 복원만 수행한다.


def _svd_relative_error(
    image: NDArray[np.float64], reconstructed: NDArray[np.float64],
) -> float:
    """상대 Frobenius 오차를 반환하며 영영상에서는 절대오차를 사용한다.

    큰 유한 픽셀 값의 오버플로를 피하려고 뺄셈·노름 계산 전에 두 배열의
    척도를 조정한다. 두 입력은 비어 있지 않은 같은 shape (m,n)이다.
    """
    scale = float(np.max(np.abs(image)))
    if scale == 0:	# 영입력은 상대오차 분모가 없으므로 별도 규약으로 처리한다.
        return float(np.linalg.norm(reconstructed, ord="fro"))	# 영영상 기준에서는 복원값의 절대 Frobenius 오차를 사용한다.
    scaled = image / scale	# 뺄셈과 노름 전에 공통 척도를 적용해 큰 픽셀 값의 넘침을 피한다.
    return float(np.linalg.norm(scaled - reconstructed / scale, ord="fro") / np.linalg.norm(scaled, ord="fro"))


def svd_comparison_figure(
    image: ArrayLike, ranks: tuple[int, ...] | list[int],
    cache: CacheManager | None = None,
) -> tuple[Figure, NDArray]:
    """원본과 rank별 근사를 비교하고 Figure와 1차원 Axes 배열을 반환한다.

    모든 패널은 원본의 같은 회색조 범위를 공유하며, 렌더러 오버플로를 피하도록
    표시 전에 공통 유한 크기로 척도를 조정한다. 범위 밖 값은 표시할 때만
    클리핑한다. 제목은 rank, 클리핑 전 Frobenius 상대오차(영영상은 절대오차),
    저장 원소 수/원본 원소 수를 보여 준다. 비율이 1보다 크면 저장량을 줄이지
    못한다는 뜻이다. 파일을 저장하거나 창을 띄우지 않고 호출자가 그림을
    정리한다. 잘못된 rank는 그림을 만들기 전에 거부한다.
    """
    values = _svd_image(image)
    if not isinstance(ranks, (tuple, list)) or not ranks:
        raise TypeError("ranks must be a nonempty list or tuple of integers")
    validated = [_svd_rank(rank, min(values.shape)) for rank in ranks]	# 그림이나 분해를 만들기 전에 모든 요청 rank를 검증한다.
    factors = svd_decompose(values, cache)	# 여러 rank 비교에서 전체 SVD는 한 번만 계산하거나 읽는다.
    approximations = [reconstruct_svd(factors, rank) for rank in validated]	# 동일한 분해와 입력으로 rank 효과만 비교한다.
    errors = [_svd_relative_error(values, approximation) for approximation in approximations]	# 표시용 클리핑 전에 원래 수치 복원 오차를 측정한다.
    low, high = float(np.min(values)), float(np.max(values))
    display_scale = max(abs(low), abs(high)) or 1.0	# 영영상의 0 나눗셈을 피하면서 공통 표시 척도를 정한다.
    display_low, display_high = low / display_scale, high / display_scale
    figure, grid = plt.subplots(
        1, len(validated) + 1, figsize=(3.4*(len(validated)+1), 4.2),
        squeeze=False, layout="constrained",
    )
    axes = grid[0]
    try:
        axes[0].imshow(values / display_scale, cmap="gray", vmin=display_low, vmax=display_high)
        axes[0].set_title(f"Original {values.shape[0]} x {values.shape[1]}")
        error_name = "abs. error" if np.all(values == 0) else "rel. error"
        for axis, rank, approximation, error in zip(axes[1:], validated, approximations, errors):
            storage_ratio = rank * (values.shape[0] + values.shape[1] + 1) / values.size	# 세 SVD 인자의 원소 수를 원본 픽셀 수와 비교한다.
            displayed = np.clip(approximation, low, high) / display_scale	# 수치 측정은 보존하고 그림에 보이는 값만 원본 회색조 범위로 제한한다.
            axis.imshow(displayed, cmap="gray", vmin=display_low, vmax=display_high)
            axis.set_title(f"k={rank} | {error_name}={error:.3g}\nstored/original={storage_ratio:.2f}x")
        for axis in axes:
            axis.set_axis_off()
    except Exception:
        plt.close(figure)	# 그리기 중 오류가 나면 생성한 그림 자원을 회수한다.
        raise
    return figure, axes
