"""NumPy linear transforms, independent area checks, and matrix decompositions."""

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
    """Validate a finite real 2x2 transform; invalid shapes raise ValueError."""
    result = _as_finite_real_array(matrix, "matrix")
    if result.shape != (2, 2):
        raise ValueError("matrix must have shape (2, 2)")
    return result


def _plane_points(points: ArrayLike, minimum: int = 1) -> NDArray[np.float64]:
    """Validate finite real column points of shape (2, n), with n >= minimum."""
    result = _as_finite_real_array(points, "points")
    if result.ndim != 2 or result.shape[0] != 2 or result.shape[1] < minimum:
        raise ValueError(f"points must have shape (2, n) with n >= {minimum}")
    return result


def rotation_matrix(theta: float) -> NDArray[np.float64]:
    """Return R(theta)=[[cos,-sin],[sin,cos]], shape (2,2); theta is radians."""
    angle = _finite_scalar(theta, "theta")
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)


def scaling_matrix(sx: float, sy: float) -> NDArray[np.float64]:
    """Return diag(sx,sy), shape (2,2); finite negative/zero scales are allowed."""
    return np.diag([_finite_scalar(sx, "sx"), _finite_scalar(sy, "sy")])


def shear_matrix(k: float) -> NDArray[np.float64]:
    """Return horizontal shear [[1,k],[0,1]], shape (2,2), for finite scalar k."""
    return np.array([[1.0, _finite_scalar(k, "k")], [0.0, 1.0]])


def unit_circle(n: int = 400) -> NDArray[np.float64]:
    """Return n ordered unit-circle vertices, shape (2,n), without a repeat endpoint.

    n must be an integer >=3. The enclosed polygon approximates the circle;
    comparing transformed/original polygon areas cancels this sampling error.
    """
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise TypeError("n must be an integer")
    if n < 3:
        raise ValueError("n must be at least 3")
    angles = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    return np.stack((np.cos(angles), np.sin(angles)))


def transform_points(matrix: ArrayLike, points: ArrayLike) -> NDArray[np.float64]:
    """Apply A(2,2) @ P(2,n), returning finite float64 column points (2,n).

    Invalid inputs raise TypeError/ValueError. An unrepresentable result raises
    ValueError instead of returning nonfinite coordinates.
    """
    transform, vertices = _transform_matrix(matrix), _plane_points(points)
    with np.errstate(over="ignore", invalid="ignore"):
        result = transform @ vertices
    if not np.all(np.isfinite(result)):
        raise ValueError("transformed coordinates are not finite")
    return result


def polygon_area(points: ArrayLike) -> float:
    """Return a simple ordered polygon's nonnegative shoelace area from (2,n).

    Vertices must follow the boundary without self-intersections, n>=3. Either
    winding direction and a repeated closing vertex are accepted. This
    computation is independent of the transform's determinant. Degenerate
    polygons return zero; unrepresentable areas raise ValueError.
    """
    vertices = _plane_points(points, minimum=3)
    # Translation reduces cancellation for polygons far from the origin.
    with np.errstate(over="ignore", invalid="ignore"):
        centered = vertices - vertices[:, :1]
        x, y = centered
        area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))
    if not np.isfinite(area):
        raise ValueError("polygon area is not representable as a finite float64")
    return float(area)


def _determinant_2x2(matrix: NDArray[np.float64]) -> float:
    """Evaluate ad-bc on a scaled (2,2) matrix and reject float64 range loss.

    Direct cross products preserve an exact zero for identical columns, where
    LU-based determinant rounding can leave a spurious nonzero value. Scaling
    or products that erase nonzero values, and a nonzero determinant that
    underflows/overflows on rescaling, raise ValueError.
    """
    scale = np.max(np.abs(matrix))
    if scale == 0:
        return 0.0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        normalized = matrix / scale
        if np.any((matrix != 0) & (normalized == 0)):
            raise ValueError("matrix scaling loses nonzero entries in float64")
        first = normalized[0, 0] * normalized[1, 1]
        second = normalized[0, 1] * normalized[1, 0]
        if ((normalized[0, 0] != 0 and normalized[1, 1] != 0 and first == 0)
                or (normalized[0, 1] != 0 and normalized[1, 0] != 0 and second == 0)):
            raise ValueError("determinant cross product underflows in float64")
        scaled = first - second
        determinant = (scaled * scale) * scale
    if not np.isfinite(determinant) or (scaled != 0 and determinant == 0):
        raise ValueError("determinant is not representable in float64")
    return float(determinant)


def determinant_area_check(
    matrix: ArrayLike,
    points: ArrayLike | None = None,
    relative_tolerance: float = 0.01,
    absolute_tolerance: float = 1e-10,
) -> dict[str, float | str | bool]:
    """Compare independent area ratio with abs(det(A)) and return error metrics.

    A has shape (2,2); optional simple-polygon points have shape (2,n).
    Nonzero determinants use relative error; a zero determinant uses absolute
    area-ratio error. Both tolerances must be finite and nonnegative. A zero
    source area or unrepresentable determinant/ratio raises ValueError. The
    2x2 determinant is evaluated as scaled ad-bc independently of polygon area.
    """
    transform = _transform_matrix(matrix)
    relative_tolerance = _finite_scalar(relative_tolerance, "relative_tolerance")
    absolute_tolerance = _finite_scalar(absolute_tolerance, "absolute_tolerance")
    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("area tolerances must be nonnegative")
    vertices = unit_circle() if points is None else _plane_points(points, minimum=3)
    before = polygon_area(vertices)
    if before == 0:
        raise ValueError("source polygon must have positive area")
    determinant = _determinant_2x2(transform)
    after = polygon_area(transform_points(transform, vertices))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ratio = after / before
        expected = abs(determinant)
        absolute_error = abs(ratio - expected)
        error = absolute_error / expected if expected != 0 else absolute_error
    if not all(np.isfinite(value) for value in (determinant, ratio, error)):
        raise ValueError("determinant or area-ratio error is not finite")
    kind = "relative" if expected != 0 else "absolute"
    tolerance = relative_tolerance if expected != 0 else absolute_tolerance
    return {
        "determinant": determinant, "area_before": before, "area_after": after,
        "area_ratio": ratio, "expected_ratio": expected,
        "error": error, "error_kind": kind, "tolerance": tolerance,
        "passed": bool(error <= tolerance),
    }


def transformation_figure(n: int = 400) -> tuple[Figure, np.ndarray]:
    """Return a Figure and three Axes comparing rotation, scaling and shear.

    Each panel overlays original/transformed circle points (2,n). Arrows mark
    the same pi/4 direction, making rotation visible even for a circular shape.
    No window is shown or file saved; the caller owns Figure cleanup.
    """
    original = unit_circle(n)
    reference = np.array([[np.sqrt(0.5)], [np.sqrt(0.5)]])
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
                closed = np.column_stack((vertices, vertices[:, 0]))
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
            axis.set_aspect("equal", adjustable="box")
            axis.legend(loc="upper right", fontsize=8)
    except Exception:
        plt.close(figure)
        raise
    return figure, axes


@dataclass(frozen=True)
class PowerIterationResult:
    """Approximate real eigenpair and explicit termination information.

    eigenvector is a caller-owned float64 array of shape (n,) with unit norm;
    eigenvalue and residual are scalars. iterations counts matrix-vector
    normalization updates, so an exact initial eigenpair needs zero updates.
    converged means the relative eigenpair residual met the requested bound,
    not a proof that the eigenvalue has largest magnitude. reason is one of
    residual_tolerance, zero_matrix, nullspace_breakdown, or max_iterations.
    Fields are frozen, while the independently owned vector remains mutable.
    """

    eigenvalue: float
    eigenvector: NDArray[np.float64]
    iterations: int
    residual: float
    converged: bool
    reason: str


def _power_normalize(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize a finite nonzero (n,) vector without squaring its raw scale."""
    peak = float(np.max(np.abs(vector)))
    if peak == 0.0:
        raise ValueError("initial vector must not be zero")
    with np.errstate(under="ignore"):
        scaled = vector / peak
    return scaled / np.linalg.norm(scaled)


def _power_residual_norm(vector: NDArray[np.float64]) -> float:
    """Measure a residual vector without underflow from squaring tiny entries."""
    peak = float(np.max(np.abs(vector)))
    if peak == 0.0:
        return 0.0
    with np.errstate(under="ignore"):
        scaled = vector / peak
    return float(peak * np.linalg.norm(scaled))


def _power_eigenvalue(scaled_value: float, matrix_scale: float) -> float:
    """Restore a scalar eigenvalue's scale, rejecting float64 range failures."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        value = float(np.float64(scaled_value) * np.float64(matrix_scale))
    if not np.isfinite(value) or (scaled_value != 0.0 and value == 0.0):
        raise ValueError("the estimated eigenvalue is not representable in float64")
    return value


def power_iteration(
    A: ArrayLike,
    initial: ArrayLike | None = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> PowerIterationResult:
    """Compute a real eigenpair by repeated normalized matrix-vector products.

    A is a finite real square array of shape (n,n), n >= 1. initial is a
    nonzero finite real vector (n,); when omitted, NumPy's global RNG draws a
    normal vector. Call the shared set_seed(42) at the experiment entry point;
    this function never resets a seed. Inputs are not modified.

    The update is v <- A@v / ||A@v||2 and lambda is the Rayleigh quotient.
    Stop when ||A@v-lambda*v||2/(||A||F*||v||2) <= tol or after max_iter
    updates. Matrix scaling cancels from this relative residual and avoids
    overflowing norms. Residual norms also use max-absolute-value scaling so
    squaring tiny residual entries does not falsely report exact convergence.
    tol must be positive finite scalar; max_iter must be
    a positive integer. Invalid types/shapes/values raise TypeError/ValueError.
    Scaling that erases nonzero matrix entries, or an eigenvalue outside
    float64 range, raises ValueError instead of silently changing the problem.

    A zero matrix gives an exact zero eigenpair. A vanishing iterate for a
    nonzero matrix gives nullspace_breakdown with converged=False. Negative
    dominant eigenvalues may flip v's sign; residual convergence handles this.
    Equal-magnitude or complex dominant modes may exhaust the iteration budget.
    Dominance also requires a suitable initial component: a subdominant exact
    eigenvector can meet the residual criterion immediately. No eigensolver is
    called and no general convergence guarantee is claimed.
    """
    matrix = _as_finite_real_array(A, "A")
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A must be a nonempty square matrix of shape (n,n)")
    tolerance = _as_finite_real_array(tol, "tol")
    if tolerance.shape != () or float(tolerance) <= 0.0:
        raise ValueError("tol must be a positive finite scalar")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)):
        raise TypeError("max_iter must be an integer")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    vector = _as_finite_real_array(
        np.random.normal(size=matrix.shape[0]) if initial is None else initial,
        "initial",
    )
    if vector.shape != (matrix.shape[0],):
        raise ValueError("initial must have shape (n,)")
    vector = _power_normalize(vector)
    matrix_scale = float(np.max(np.abs(matrix)))
    if matrix_scale == 0.0:
        return PowerIterationResult(0.0, vector.copy(), 0, 0.0, True, "zero_matrix")
    with np.errstate(under="ignore"):
        scaled_matrix = matrix / matrix_scale
    if np.any((matrix != 0.0) & (scaled_matrix == 0.0)):
        raise ValueError("A's dynamic range cannot be preserved when scaled in float64")
    matrix_norm = float(np.linalg.norm(scaled_matrix, ord="fro"))

    for iterations in range(int(max_iter) + 1):
        product = scaled_matrix @ vector
        if not np.any(product):
            return PowerIterationResult(
                0.0, vector.copy(), iterations, 0.0, False, "nullspace_breakdown"
            )
        vector_norm = float(np.linalg.norm(vector))
        scaled_value = float(np.dot(vector, product) / np.dot(vector, vector))
        residual = float(
            _power_residual_norm(product - scaled_value * vector) / (matrix_norm * vector_norm)
        )
        eigenvalue = _power_eigenvalue(scaled_value, matrix_scale)
        if residual <= float(tolerance):
            return PowerIterationResult(
                eigenvalue, vector.copy(), iterations, residual, True, "residual_tolerance"
            )
        if iterations == max_iter:
            return PowerIterationResult(
                eigenvalue, vector.copy(), iterations, residual, False, "max_iterations"
            )
        vector = _power_normalize(product)

    raise RuntimeError("unreachable iteration state")


def _svd_image(image: ArrayLike) -> NDArray[np.float64]:
    """Validate a nonempty finite real grayscale array of shape (m, n).

    Pixel values need not lie in [0, 1]. Non-real inputs raise TypeError;
    nonfinite values, empty dimensions, and other shapes raise ValueError.
    """
    values = _as_finite_real_array(image, "image")
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("image must have nonempty shape (m, n)")
    return values


def _svd_rank(k: int, available: int) -> int:
    """Validate an integer rank in [1, available], rejecting bool and clipping."""
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")
    if not 1 <= k <= available:
        raise ValueError(f"k must lie between 1 and {available}")
    return int(k)


def _svd_factors(
    factors: tuple[ArrayLike, ArrayLike, ArrayLike],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Validate real finite U(m,r), s(r,), Vt(r,n), with 1 <= r <= min(m,n).

    Singular values must be nonnegative and descending. Orthogonality is not
    recomputed: callers should provide SVD factors, optionally truncated.
    Invalid container/types raise TypeError; invalid shapes/values raise ValueError.
    """
    if not isinstance(factors, (tuple, list)) or len(factors) != 3:
        raise TypeError("factors must contain exactly (U, s, Vt)")
    u, singular_values, vt = (
        _as_finite_real_array(value, name)
        for value, name in zip(factors, ("U", "s", "Vt"))
    )
    if u.ndim != 2 or singular_values.ndim != 1 or vt.ndim != 2:
        raise ValueError("factor shapes must be U(m,r), s(r,), Vt(r,n)")
    rank = singular_values.size
    if (
        rank == 0 or u.shape[1] != rank or vt.shape[0] != rank
        or rank > min(u.shape[0], vt.shape[1])
    ):
        raise ValueError("factor dimensions are incompatible")
    if np.any(singular_values < 0) or np.any(singular_values[1:] > singular_values[:-1]):
        raise ValueError("singular values must be nonnegative and descending")
    return u, singular_values, vt


def svd_decompose(
    image: ArrayLike, cache: CacheManager | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return economy SVD U(m,r), s(r,), Vt(r,n), r=min(m,n), for a grayscale image.

    Uses NumPy's SVD as allowed by the mission. When provided, the shared cache
    reuses the complete factorization across different requested ranks. The key
    includes the float64 image, algorithm version, and NumPy version. Input is
    never modified; cached arrays are owned copies. Nonfinite/unrepresentable
    factors raise ValueError, and NumPy LinAlgError propagates on nonconvergence.
    """
    values = _svd_image(image)
    if cache is not None and not isinstance(cache, CacheManager):
        raise TypeError("cache must be CacheManager or None")

    def compute() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Compute and validate the complete economy factorization once."""
        with np.errstate(over="ignore", invalid="ignore"):
            return _svd_factors(np.linalg.svd(values, full_matrices=False))

    if cache is None:
        return compute()
    key = make_cache_key(
        "SVD_Decomposition", "v1", values,
        parameters={"full_matrices": False, "numpy_version": np.__version__},
    )
    return _svd_factors(cache.get_or_compute(key, compute))


def reconstruct_svd(
    factors: tuple[ArrayLike, ArrayLike, ArrayLike], k: int,
) -> NDArray[np.float64]:
    """Return A_k = U[:, :k] diag(s[:k]) Vt[:k, :] with shape (m, n).

    k must be an integer in [1, len(s)]. Multiplication uses column scaling in
    place of allocating diag(s). Reconstruction is not clipped: errors must be
    measured on the numerical approximation before display. Invalid factors or
    rank raise TypeError/ValueError; nonfinite reconstruction raises ValueError.
    """
    u, singular_values, vt = _svd_factors(factors)
    rank = _svd_rank(k, singular_values.size)
    with np.errstate(over="ignore", invalid="ignore"):
        reconstructed = (u[:, :rank] * singular_values[:rank]) @ vt[:rank, :]
    if not np.all(np.isfinite(reconstructed)):
        raise ValueError("reconstruction is not representable as finite float64")
    return reconstructed


def compress_image(
    image: ArrayLike, k: int, cache: CacheManager | None = None,
) -> NDArray[np.float64]:
    """Return the unclipped rank-k approximation with the image's (m,n) shape.

    This returns reconstructed pixels, not a compressed file. The stored SVD
    factors require k*(m+n+1) numbers, which can exceed the original m*n numbers.
    Validate rank before performing or caching a potentially expensive SVD.
    """
    values = _svd_image(image)
    rank = _svd_rank(k, min(values.shape))
    return reconstruct_svd(svd_decompose(values, cache), rank)


def _svd_relative_error(
    image: NDArray[np.float64], reconstructed: NDArray[np.float64],
) -> float:
    """Return relative Frobenius error, using absolute error for a zero image.

    Scale both arrays before subtraction/norm evaluation to avoid overflow for
    large finite pixel values. Both inputs have the same nonempty (m,n) shape.
    """
    scale = float(np.max(np.abs(image)))
    if scale == 0:
        return float(np.linalg.norm(reconstructed, ord="fro"))
    scaled = image / scale
    return float(np.linalg.norm(scaled - reconstructed / scale, ord="fro") / np.linalg.norm(scaled, ord="fro"))


def svd_comparison_figure(
    image: ArrayLike, ranks: tuple[int, ...] | list[int],
    cache: CacheManager | None = None,
) -> tuple[Figure, NDArray]:
    """Compare original and rank approximations; return Figure and 1D Axes array.

    All panels share the original grayscale mapping, scaled by a common finite
    magnitude before display to prevent renderer overflow. Out-of-range values
    are clipped only for display. Titles show rank, unclipped Frobenius relative
    error (absolute for a zero image), and stored/original element count ratio.
    A ratio above 1 means no storage reduction. No file is saved or window shown;
    the caller owns figure cleanup. Invalid ranks fail before creating a figure.
    """
    values = _svd_image(image)
    if not isinstance(ranks, (tuple, list)) or not ranks:
        raise TypeError("ranks must be a nonempty list or tuple of integers")
    validated = [_svd_rank(rank, min(values.shape)) for rank in ranks]
    factors = svd_decompose(values, cache)
    approximations = [reconstruct_svd(factors, rank) for rank in validated]
    errors = [_svd_relative_error(values, approximation) for approximation in approximations]
    low, high = float(np.min(values)), float(np.max(values))
    display_scale = max(abs(low), abs(high)) or 1.0
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
            storage_ratio = rank * (values.shape[0] + values.shape[1] + 1) / values.size
            displayed = np.clip(approximation, low, high) / display_scale
            axis.imshow(displayed, cmap="gray", vmin=display_low, vmax=display_high)
            axis.set_title(f"k={rank} | {error_name}={error:.3g}\nstored/original={storage_ratio:.2f}x")
        for axis in axes:
            axis.set_axis_off()
    except Exception:
        plt.close(figure)
        raise
    return figure, axes
