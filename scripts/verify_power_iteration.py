"""알고리즘과 분리된 명시적인 NumPy eig 비교 근거를 생성한다.

이 파일은 검증 전용이다. src.linear_algebra.power_iteration은 여전히
행렬·벡터 곱을 직접 반복하며 NumPy 고유값 해법을 호출하지 않는다.
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
    """고정된 실수 미션 행렬 (2,2)을 검증 전용 eig와 비교한다.

    명시적인 초기 벡터 [1,1]의 shape (2,), float64와 기존 직접 구현을 사용한다.
    NumPy 기준은 고유값의 절댓값으로 선택하고 고유벡터 부호를 맞춘 뒤 비교한다.
    독립적인 닫힌 해로 두 계산을 함께 교차 검증한다. 정확도·수렴·단위벡터·
    기준값 검사가 실패하면 AssertionError를 발생시키며 성공하면 JSON으로
    표현 가능한 근거를 반환한다. 이 고정 입력 함수는 난수를 소비하지 않는다.
    """
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)	# 비교 대상인 고정 대칭 행렬을 float64로 생성한다.
    initial = np.array([1.0, 1.0], dtype=np.float64)
    manual = power_iteration(matrix, initial=initial)	# 수작업 행렬·벡터 반복 알고리즘의 결과를 얻는다.
    values, vectors = np.linalg.eig(matrix)	# 비교 검증에서만 NumPy 고유값 해법을 호출한다.
    index = int(np.argmax(np.abs(values)))	# 대수적으로 가장 큰 값이 아닌 절댓값 최대 고유값을 선택한다.
    reference = float(values[index])
    vector_raw = vectors[:, index] / np.linalg.norm(vectors[:, index])	# 기준 고유벡터를 단위 길이로 맞춘다.
    vector_aligned = vector_raw.copy()
    if np.dot(manual.eigenvector, vector_aligned) < 0:	# v와 -v가 같은 고유방향인 점을 고려한다.
        vector_aligned = -vector_aligned
    analytic = float((7.0 + np.sqrt(5.0)) / 2.0)	# 두 구현과 독립적인 해석 고유값을 계산한다.
    relative_error = float(abs(manual.eigenvalue - reference) / abs(reference))	# NumPy 기준값으로 고유값 상대오차를 측정한다.
    vector_error = float(np.linalg.norm(manual.eigenvector - vector_aligned))
    residual = float(np.linalg.norm(matrix @ manual.eigenvector - manual.eigenvalue * manual.eigenvector)	# 원래 행렬·고유벡터·고유값에서 상대 잔차를 다시 계산한다.
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
    passed = bool(manual.converged and relative_error <= 0.05 and vector_error <= 1e-8	# 수렴·고유값·벡터·잔차·독립 해석 기준을 함께 판정한다.
                  and residual <= 1e-6 and abs(reference - analytic) <= 1e-14
                  and abs(manual.eigenvalue - analytic) / abs(analytic) <= 0.05
                  and abs(np.linalg.norm(manual.eigenvector) - 1.0) <= 1e-14)
    comparison["passed"] = passed
    assert passed, f"Power Iteration / np.linalg.eig verification failed: {comparison}"	# 어느 기준이든 실패하면 성공 근거를 생성하지 않는다.
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
    """검증 보고서의 두 고유쌍과 실측 오차를 명령행 출력용으로 구성한다."""
    manual, reference = report["power_iteration"], report["numpy_eig"]
    comparison = manual["numpy_eig_comparison"]
    return (	# 기존 요약 문자열과 숫자 표시 형식을 그대로 유지한다.
        f"Power Iteration: lambda={manual['eigenvalue']:.15f}, vector={manual['eigenvector']}\n"
        f"np.linalg.eig: lambda={reference['eigenvalue']:.15f}, "
        f"vector(sign aligned)={reference['eigenvector_sign_aligned']}\n"
        f"Relative eigenvalue error={comparison['relative_eigenvalue_error']:.6e} "
        f"<= {comparison['relative_eigenvalue_tolerance']:.2f} (5%): PASS; "
        f"iterations={manual['iterations']}, residual={manual['relative_residual']:.6e}"
    )


def write_report(report: dict, output_dir: Path) -> None:
    """성공한 JSON/Markdown 근거를 output_dir에 저장하며 입출력 오류는 전달한다."""
    assert report["passed"] is True, "Refusing to publish a failed report as PASS"	# 실패한 기록에 PASS 표기를 붙여 저장하지 못하게 한다.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)	# 산출물을 저장할 디렉터리를 준비한다.
    path = output_dir / "power_iteration_eig_comparison"
    path.with_suffix(".json").write_text(	# 검증한 고유쌍과 조건을 JSON으로 저장한다.
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n",
    )
    manual = report["power_iteration"]
    comparison = manual["numpy_eig_comparison"]
    text = (	# 같은 측정값에서 기존 Markdown 보고서를 구성한다.
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
    path.with_suffix(".md").write_text(text, encoding="utf-8", newline="\n")	# 본문 문자열을 바꾸지 않고 UTF-8 Markdown으로 저장한다.


def main(argv: list[str] | None = None) -> int:
    """고정 비교를 실행하고 근거를 저장하며 모든 assert가 통과해야 0을 반환한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    arguments = parser.parse_args(argv)	# 호출자가 제공한 명령행 인자를 해석한다.
    set_seed(42)	# 독립 실험의 시작점에서 난수 시드를 한 번 고정한다.
    report = verify_mission_eigenpair()	# 입력과 기준이 고정된 고유쌍 비교를 실행한다.
    write_report(report, arguments.output_dir)
    print(comparison_summary(report))	# 검증한 비교 요약을 원래 형식으로 출력한다.
    print(f"Comparison JSON and Markdown saved in {arguments.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())	# 명령행 실행 결과를 프로세스 종료 코드로 전달한다.
