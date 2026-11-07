"""Linear algebra utilities for AI math mission."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)


def unit_circle(num_points: int = 100) -> np.ndarray:
    """Return 2xN points on a unit circle."""
    theta = np.linspace(0, 2 * np.pi, num_points)
    return np.vstack((np.cos(theta), np.sin(theta)))


def rotation_matrix(theta_rad: float) -> np.ndarray:
    """Return a 2D rotation matrix."""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s], [s, c]], dtype=float)


def scaling_matrix(scale_x: float, scale_y: float) -> np.ndarray:
    """Return a 2D scaling matrix."""
    return np.array([[scale_x, 0.0], [0.0, scale_y]], dtype=float)


def shear_matrix(k: float) -> np.ndarray:
    """Return a 2D shear matrix."""
    return np.array([[1.0, k], [0.0, 1.0]], dtype=float)


def apply_transformation(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a linear transformation matrix to 2xN points."""
    return matrix @ points


def polygon_area(points: np.ndarray) -> float:
    """Compute polygon area from 2xN points using shoelace formula."""
    x, y = points[0], points[1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def determinant_area_check(matrix: np.ndarray, num_points: int = 400) -> dict:
    """Compare determinant magnitude with transformed area ratio."""
    original = unit_circle(num_points)
    transformed = apply_transformation(original, matrix)

    area_before = polygon_area(original)
    area_after = polygon_area(transformed)
    area_ratio = area_after / area_before
    det_value = float(np.linalg.det(matrix))
    det_abs = abs(det_value)
    rel_error = abs(area_ratio - det_abs) / (det_abs + 1e-12)

    return {
        "determinant": det_value,
        "det_abs": det_abs,
        "area_ratio": area_ratio,
        "relative_error": rel_error,
        "within_1_percent": rel_error <= 0.01,
    }


def visualize_transformation(
    matrix: np.ndarray,
    name: str,
    output_path: str | Path,
    num_points: int = 200,
) -> Path:
    """Visualize unit-circle transform and save to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original = unit_circle(num_points)
    transformed = apply_transformation(original, matrix)

    plt.figure(figsize=(6, 6))
    plt.plot(original[0], original[1], label="Original Unit Circle", color="tab:blue")
    plt.plot(transformed[0], transformed[1], label=f"Transformed ({name})", color="tab:red")
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.axvline(0, color="gray", linewidth=0.8)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(f"Matrix Transformation: {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def power_iteration(
    matrix: np.ndarray,
    num_iters: int = 1000,
    tol: float = 1e-6,
    random_state: int = 42,
) -> Tuple[float, np.ndarray, int]:
    """Compute dominant eigenpair via Power Iteration."""
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")

    rng = np.random.default_rng(random_state)
    vector = rng.normal(size=matrix.shape[0])
    vector = vector / np.linalg.norm(vector)

    prev_value = 0.0
    for iteration in range(1, num_iters + 1):
        next_vector = matrix @ vector
        next_norm = np.linalg.norm(next_vector)
        if next_norm == 0:
            raise ValueError("Power iteration failed: zero vector encountered")
        vector = next_vector / next_norm
        eigenvalue = float(vector.T @ matrix @ vector)

        if abs(eigenvalue - prev_value) < tol:
            return eigenvalue, vector, iteration
        prev_value = eigenvalue

    return eigenvalue, vector, num_iters


def compare_power_iteration(matrix: np.ndarray) -> dict:
    """Compare power-iteration result against np.linalg.eig validation."""
    eigval_pi, eigvec_pi, iters = power_iteration(matrix)

    eigvals, eigvecs = np.linalg.eig(matrix)
    dominant_idx = int(np.argmax(np.abs(eigvals)))
    eigval_np = float(np.real(eigvals[dominant_idx]))
    eigvec_np = np.real(eigvecs[:, dominant_idx])

    value_error_pct = abs(eigval_pi - eigval_np) / (abs(eigval_np) + 1e-12) * 100

    return {
        "power_iteration": {"eigenvalue": eigval_pi, "eigenvector": eigvec_pi, "iterations": iters},
        "numpy": {"eigenvalue": eigval_np, "eigenvector": eigvec_np},
        "error_percent": value_error_pct,
        "within_5_percent": value_error_pct <= 5,
    }


def compress_image_svd(image: np.ndarray, k: int) -> np.ndarray:
    """Compress and reconstruct grayscale image with top-k singular values."""
    if image.ndim != 2:
        raise ValueError("image must be 2D grayscale")
    if k <= 0:
        raise ValueError("k must be positive")

    u, s, vt = np.linalg.svd(image, full_matrices=False)
    k = min(k, len(s))
    reconstructed = (u[:, :k] * s[:k]) @ vt[:k, :]
    return np.clip(reconstructed, 0.0, 1.0)


def create_sample_image(size: int = 64) -> np.ndarray:
    """Create deterministic synthetic grayscale image for SVD demo."""
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    image = 0.6 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy) + 0.4 * xx + 0.2 * yy
    image = (image - image.min()) / (image.max() - image.min())
    return image


def compare_svd_compression(
    image: np.ndarray,
    ks: Iterable[int] = (10, 50, 100),
    output_path: str | Path = "outputs/svd_comparison.png",
) -> Path:
    """Save one figure comparing original image and SVD reconstructions."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ks = list(ks)
    fig, axes = plt.subplots(1, len(ks) + 1, figsize=(4 * (len(ks) + 1), 4))

    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for ax, k in zip(axes[1:], ks):
        recon = compress_image_svd(image, k)
        ax.imshow(recon, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"k={k}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
