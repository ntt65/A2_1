"""Reproduce the Step 2 figures and measured metrics with NumPy-generated data."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.linear_algebra import (
    determinant_area_check, reconstruct_svd, rotation_matrix,
    scaling_matrix, shear_matrix, svd_comparison_figure, svd_decompose,
    transformation_figure,
)
from src.utils.cache_manager import CacheManager
from src.utils.random import set_seed
from scripts.verify_power_iteration import verify_mission_eigenpair, write_report

__concepts__ = {
    "Synthetic_Grayscale": ["generate_grayscale_image"],
    "Linear_Algebra_Experiment": ["run_experiment", "main"],
}


def generate_grayscale_image(size: int) -> np.ndarray:
    """Create a textured synthetic scene of shape (size,size), with pixels in [0,1].

    size is an integer >=8. A landscape, disk, curved texture and Gaussian
    texture give meaningful higher-rank detail. The global RNG is consumed;
    call set_seed(42) once at the experiment boundary for repeatability.
    """
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)):
        raise TypeError("size must be an integer")
    if size < 8:
        raise ValueError("size must be at least 8")
    coordinates = np.linspace(-1.0, 1.0, int(size))
    x, y = np.meshgrid(coordinates, coordinates)
    background = 0.22 + 0.25 * (1.0 - y)
    boundary = 0.20 + 0.18*np.sin(5*x) + 0.10*np.sin(13*x)
    scene = np.where(y > boundary, 0.18 + 0.10*(x+1), background)
    disk = (x + 0.45)**2 + (y + 0.40)**2 <= 0.16**2
    scene = np.where(disk, 0.92, scene)
    texture = 0.07*np.sin(40*x*y) + 0.05*np.sin(25*x + 8*y*y)
    texture += np.random.normal(0.0, 0.06, size=(int(size), int(size)))
    return np.clip(scene + texture, 0.0, 1.0)


def run_experiment(output_dir: Path, use_cache: bool = True) -> dict:
    """Write three PNGs, measured JSON, and explicit eig comparison JSON/Markdown.

    Uses seed 42, fixed transform cases, and the specified symmetric matrix.
    SVD compares a 64x64 main experiment and a separate 128x128 supplement.
    Required numerical assertions run before the metrics report is written.
    Output directory creation/file errors propagate; no external data is read.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    cache = CacheManager(".cache/linear_algebra", enabled=use_cache)
    transforms = {}
    for name, matrix in (
        ("rotation", rotation_matrix(np.pi/4)),
        ("scaling", scaling_matrix(2, 0.5)),
        ("shear", shear_matrix(1)),
        ("reflection", scaling_matrix(-2, 3)),
        ("singular", scaling_matrix(2, 0)),
    ):
        result = determinant_area_check(matrix)
        assert result["passed"], (name, result)
        transforms[name] = result
    figure, _ = transformation_figure()
    try:
        figure.savefig(output_dir / "linear_transforms.png", dpi=140)
    finally:
        plt.close(figure)

    verification = verify_mission_eigenpair()
    eigenpair = verification["power_iteration"]
    write_report(verification, output_dir)

    images = []
    for size, ranks, filename in (
        (64, (10, 50, 64), "svd_64.png"),
        (128, (10, 50, 100), "svd_128.png"),
    ):
        original = generate_grayscale_image(size)
        factors = svd_decompose(original, cache)
        norm = float(np.linalg.norm(original, ord="fro"))
        full_error = float(np.linalg.norm(original-reconstruct_svd(factors, size), ord="fro") / norm)
        assert full_error <= 1e-10
        rows = []
        for rank in ranks:
            reconstruction = reconstruct_svd(factors, rank)
            error = float(np.linalg.norm(original-reconstruction, ord="fro") / norm)
            ratio = rank * (2*size+1) / original.size
            rows.append({"k": rank, "relative_error": error, "stored_over_original": ratio})
        assert np.all(np.diff([row["relative_error"] for row in rows]) <= 1e-12)
        numerical_rank = int(np.count_nonzero(factors[1] > factors[1][0]*size*np.finfo(float).eps))
        assert numerical_rank == size, "the comparison data should contain full-rank detail"
        figure, _ = svd_comparison_figure(original, list(ranks), cache)
        try:
            figure.savefig(output_dir / filename, dpi=140)
        finally:
            plt.close(figure)
        images.append({
            "shape": [size, size], "filename": filename, "numerical_rank": numerical_rank,
            "full_rank_relative_error": full_error, "comparisons": rows,
        })
    report = {
        "seed": 42, "numpy_version": np.__version__, "dtype": "float64",
        "transforms": transforms, "power_iteration": eigenpair, "svd": images,
        "eig_comparison_artifacts": ["power_iteration_eig_comparison.json", "power_iteration_eig_comparison.md"],
        "notes": [
            "64x64 main experiment uses valid ranks 10,50,64.",
            "128x128 supplemental experiment permits required ranks 10,50,100.",
            "Stored/original compares numeric element counts, not encoded file sizes.",
            "Error uses unclipped reconstructions; a storage ratio above one is not a saving.",
            "A small eigenpair residual alone does not prove dominance for arbitrary initial vectors.",
        ],
    }
    (output_dir / "linear_algebra_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
        encoding="utf-8", newline="\n",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the reproducible experiment from CLI and print its measured summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--no-cache", action="store_true", help="recompute without cache reads or writes")
    arguments = parser.parse_args(argv)
    report = run_experiment(arguments.output_dir, not arguments.no_cache)
    eigenpair = report["power_iteration"]
    print(f"Power Iteration: lambda={eigenpair['eigenvalue']:.12f}, "
          f"relative error={eigenpair['relative_error']:.3e}, residual={eigenpair['relative_residual']:.3e}")
    comparison = eigenpair["numpy_eig_comparison"]
    print(f"np.linalg.eig: lambda={comparison['numpy_eigenvalue']:.15f}; "
          f"relative error={comparison['relative_eigenvalue_error']:.6e} <= 0.05 (5%): PASS")
    print(f"Figures and metrics saved in {arguments.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
