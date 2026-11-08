"""Submit explicit NumPy eig comparison evidence, separate from the algorithm.

This is verification code only: src.linear_algebra.power_iteration still uses
manual matrix-vector iteration and does not call a NumPy eigensolver.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from src.linear_algebra import power_iteration
from src.utils.random import set_seed

__concepts__ = {
    "Power_Iteration_Eig_Verification": ["verify_mission_eigenpair", "write_report", "main"],
}


def verify_mission_eigenpair() -> dict:
    """Compare the fixed real (2,2) mission matrix with verification-only eig.

    Uses explicit initial (2,) vector [1,1], float64, and the existing manual
    implementation. NumPy's reference is selected by absolute eigenvalue;
    eigenvector signs are aligned before comparison. The independent closed
    form cross-checks both calculations. Raises AssertionError on a failed
    accuracy, convergence, unit-vector, or reference check; returns JSON-safe
    evidence on success. No RNG is consumed by this fixed-input function.
    """
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    initial = np.array([1.0, 1.0], dtype=np.float64)
    manual = power_iteration(matrix, initial=initial)
    values, vectors = np.linalg.eig(matrix)  # Reference verification only.
    index = int(np.argmax(np.abs(values)))
    reference = float(values[index])
    vector_raw = vectors[:, index] / np.linalg.norm(vectors[:, index])
    vector_aligned = vector_raw.copy()
    if np.dot(manual.eigenvector, vector_aligned) < 0:
        vector_aligned = -vector_aligned
    analytic = float((7.0 + np.sqrt(5.0)) / 2.0)
    relative_error = float(abs(manual.eigenvalue - reference) / abs(reference))
    vector_error = float(np.linalg.norm(manual.eigenvector - vector_aligned))
    residual = float(np.linalg.norm(matrix @ manual.eigenvector - manual.eigenvalue * manual.eigenvector)
                     / (np.linalg.norm(matrix, ord="fro") * np.linalg.norm(manual.eigenvector)))
    comparison = {
        "reference_method": "numpy.linalg.eig",
        "numpy_eigenvalue": reference,
        "absolute_eigenvalue_error": float(abs(manual.eigenvalue - reference)),
        "relative_eigenvalue_error": relative_error,
        "relative_eigenvalue_tolerance": 0.05,
        "relative_eigenvalue_error_percent": 100.0 * relative_error,
        "aligned_eigenvector_l2_error": vector_error,
        "eigenvector_tolerance": 1e-8,
        "independent_relative_residual": residual,
        "relative_residual_tolerance": 1e-6,
        "numpy_vs_analytic_absolute_error": float(abs(reference - analytic)),
    }
    passed = bool(manual.converged and relative_error <= 0.05 and vector_error <= 1e-8
                  and residual <= 1e-6 and abs(reference - analytic) <= 1e-14
                  and abs(manual.eigenvalue - analytic) / abs(analytic) <= 0.05
                  and abs(np.linalg.norm(manual.eigenvector) - 1.0) <= 1e-14)
    comparison["passed"] = passed
    assert passed, f"Power Iteration / np.linalg.eig verification failed: {comparison}"
    return {
        "seed": 42, "dtype": "float64", "numpy_version": np.__version__,
        "purpose": "verification only; np.linalg.eig is not used by the manual algorithm",
        "initialization": "explicit vector [1,1]; no random draw",
        "power_iteration": {
            "matrix": matrix.tolist(), "initial": initial.tolist(),
            "eigenvalue": manual.eigenvalue, "eigenvector": manual.eigenvector.tolist(),
            "reference_formula": "(7 + sqrt(5)) / 2", "reference": analytic,
            "relative_error": float(abs(manual.eigenvalue - analytic) / abs(analytic)),
            "relative_residual": manual.residual, "iterations": manual.iterations,
            "converged": manual.converged, "reason": manual.reason,
            "numpy_eig_comparison": comparison,
        },
        "numpy_eig": {
            "eigenvalues": values.tolist(), "selected_index": index,
            "selection": "argmax(abs(eigenvalues))", "eigenvalue": reference,
            "eigenvector_raw": vector_raw.tolist(),
            "eigenvector_sign_aligned": vector_aligned.tolist(),
        },
        "passed": passed,
    }


def comparison_summary(report: dict) -> str:
    """Format the verified report's two eigenpairs and measured error for CLI output."""
    manual, reference = report["power_iteration"], report["numpy_eig"]
    comparison = manual["numpy_eig_comparison"]
    return (
        f"Power Iteration: lambda={manual['eigenvalue']:.15f}, vector={manual['eigenvector']}\n"
        f"np.linalg.eig: lambda={reference['eigenvalue']:.15f}, "
        f"vector(sign aligned)={reference['eigenvector_sign_aligned']}\n"
        f"Relative eigenvalue error={comparison['relative_eigenvalue_error']:.6e} "
        f"<= {comparison['relative_eigenvalue_tolerance']:.2f} (5%): PASS; "
        f"iterations={manual['iterations']}, residual={manual['relative_residual']:.6e}"
    )


def write_report(report: dict, output_dir: Path) -> None:
    """Write successful JSON and Markdown evidence to output_dir; I/O errors propagate."""
    assert report["passed"] is True, "Refusing to publish a failed report as PASS"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "power_iteration_eig_comparison"
    path.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n",
    )
    manual = report["power_iteration"]
    comparison = manual["numpy_eig_comparison"]
    text = (
        "# Executed Power Iteration versus np.linalg.eig verification\n\n"
        "Reproduce from the repository root: `python -m scripts.verify_power_iteration`.\n"
        "Use the Python 3.11 project environment described in README.md.\n\n"
        f"NumPy {report['numpy_version']}; float64; seed 42; matrix {manual['matrix']}; "
        f"explicit initial vector {manual['initial']}.\n\n"
        "```text\n" + comparison_summary(report) + "\n```\n\n"
        "Relative eigenvalue error = abs(lambda_manual - lambda_numpy) / abs(lambda_numpy).\n\n"
        f"Independent reference: (7 + sqrt(5)) / 2 = {manual['reference']:.15f}.\n"
        f"NumPy versus independent reference absolute error: {comparison['numpy_vs_analytic_absolute_error']:.6e}.\n"
        f"Sign-aligned eigenvector L2 error: {comparison['aligned_eigenvector_l2_error']:.6e} "
        f"<= {comparison['eigenvector_tolerance']:.1e}.\n\n"
        "The reference eigenvalue is selected by argmax(abs(eigenvalues)). Eigenvectors have an "
        "arbitrary sign, so the NumPy vector is sign-aligned before comparison. "
        "np.linalg.eig is called only in this verification script and comparison tests; "
        "the algorithm in src/linear_algebra.py remains manual.\n\n"
        "This fixed symmetric matrix has a separated dominant eigenvalue. A small residual "
        "alone does not prove dominance for arbitrary matrices or initial directions.\n"
    )
    path.with_suffix(".md").write_text(text, encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    """Run the fixed comparison, publish evidence, and return 0 only after assertions pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    arguments = parser.parse_args(argv)
    set_seed(42)
    report = verify_mission_eigenpair()
    write_report(report, arguments.output_dir)
    print(comparison_summary(report))
    print(f"Comparison JSON and Markdown saved in {arguments.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
