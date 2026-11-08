"""NumPy로 Step 3의 미분 오차와 gradient 기하를 재현한다."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")	# 파일 저장용 비대화형 백엔드를 pyplot보다 먼저 선택한다.
import matplotlib.pyplot as plt
import numpy as np

from src.calculus import (
    derivative_error_figure, derivative_error_study, gradient_figure,
    numerical_derivative, numerical_gradient, quadratic, quadratic_gradient,
)
from src.utils.random import set_seed

__concepts__ = {"Calculus_Experiment": ["run_experiment", "main"]}


def run_experiment(output_dir: Path) -> dict:
    """스칼라·벡터 오차를 assert로 검증하고 PNG 두 개와 JSON을 저장한다.

    float64, seed 42, x=3/h=1e-5, 반경 2인 점 여덟 개와 원점을 사용한다.
    수치 gradient의 shape는 (9,2)다. 기하 관계는 원점 이외의 점에서만
    검사하며 영벡터에는 방향을 부여하지 않는다.
    파일 생성 오류와 수치 assert 실패는 호출자에게 전달한다.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)	# 산출물을 저장할 디렉터리를 준비한다.
    set_seed(42)	# 독립 실험의 시작점에서 난수 시드를 한 번 고정한다.
    derivative = numerical_derivative(lambda x: x*x, 3.0, h=1e-5)	# 고정한 x와 h로 x²의 중심차분을 계산한다.
    derivative_error = abs(derivative - 6.0)	# 구현과 독립적인 해석 미분값 6과 비교한다.
    assert derivative_error <= 1e-4	# 고정 입력의 미분 정확도를 수용 기준으로 검사한다.

    angles = np.arange(8) * np.pi / 4
    points = np.vstack((2*np.column_stack((np.cos(angles), np.sin(angles))), [0, 0]))	# 반경 2인 여덟 점에 원점을 더해 (9,2) 검증 입력을 만든다.
    analytic = quadratic_gradient(points)
    numeric = np.array([	# 각 점에서 공통 수치 gradient 함수를 호출한다.
        numerical_gradient(lambda p: quadratic(p[0], p[1]), point, h=1e-5)
        for point in points
    ])
    np.testing.assert_allclose(analytic, 2*points, atol=0, rtol=0)	# 해석 gradient도 독립 식 2*points와 정확히 일치하는지 검사한다.
    gradient_error = float(np.max(np.abs(numeric - analytic)))	# 전체 좌표 성분 중 가장 큰 절대오차를 기록한다.
    assert gradient_error <= 1e-8
    tangent = np.column_stack((-points[:-1, 1], points[:-1, 0]))	# 방향이 없는 원점은 접선·수직성 검사에서 제외한다.
    normal_unit = analytic[:-1] / np.linalg.norm(analytic[:-1], axis=1, keepdims=True)	# gradient의 크기를 제거해 방향만 비교한다.
    tangent_unit = tangent / np.linalg.norm(tangent, axis=1, keepdims=True)
    normalized_dot = np.sum(normal_unit * tangent_unit, axis=1)	# 각 표본의 정규화된 법선·접선 내적을 구한다.
    orthogonality_error = float(np.max(np.abs(normalized_dot)))
    assert orthogonality_error <= 1e-12	# 정규화 내적의 최대 절댓값으로 수직성을 확인한다.
    np.testing.assert_array_equal(analytic[-1], [0, 0])	# 원점의 해석 gradient가 영벡터인지 확인한다.
    np.testing.assert_array_equal(numeric[-1], [0, 0])	# 원점의 수치 gradient도 영벡터인지 확인한다.

    sweep = derivative_error_study()	# 여러 간격의 표본 오차를 별도로 측정한다.
    report = {
        "seed": 42, "numpy_version": np.__version__, "dtype": "float64",
        "derivative": {
            "function": "x**2", "x": 3.0, "h": 1e-5,
            "numerical": derivative, "analytic": 6.0,
            "absolute_error": derivative_error, "tolerance": 1e-4,
        },
        "gradient": {
            "function": "x**2+y**2", "points": points.tolist(),
            "analytic": analytic.tolist(), "numerical": numeric.tolist(),
            "maximum_absolute_error": gradient_error, "tolerance": 1e-8,
            "nonzero_tangents": tangent.tolist(),
            "normalized_dot_products": normalized_dot.tolist(),
            "maximum_normalized_dot": orthogonality_error,
            "orthogonality_tolerance": 1e-12,
            "origin_gradient": analytic[-1].tolist(), "origin_direction": None,
        },
        "step_size_study": {key: np.asarray(value).tolist() for key, value in sweep.items()},
        "notes": [
            "Quadratic central differences have zero truncation error in exact arithmetic.",
            "Sine central differences have O(h**2) truncation error; small h suffers roundoff.",
            "No monotonic error decrease or universal best h is asserted.",
            "The origin gradient is zero and has no direction.",
            "Arrow lengths are normalized for display; metric gradients are unnormalized.",
        ],
    }
    for filename, make_figure in (
        ("gradient_geometry.png", gradient_figure),
        ("derivative_error.png", derivative_error_figure),
    ):
        figure, _ = make_figure()
        try:
            figure.savefig(output_dir / filename, dpi=140)	# 축·범례가 포함된 그림을 지정 출력 경로에 저장한다.
        finally:
            plt.close(figure)	# 저장 성공 여부와 관계없이 그림 자원을 해제한다.
    (output_dir / "calculus_metrics.json").write_text(	# 그림과 대응하는 원시 검증 수치를 JSON으로 저장한다.
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
        encoding="utf-8", newline="\n",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """명령행에서 Step 3 실험을 실행하며 --output-dir로 쓰기 가능한 폴더를 선택한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    arguments = parser.parse_args(argv)	# 호출자가 제공한 명령행 인자를 해석한다.
    report = run_experiment(arguments.output_dir)	# 명령행에서 선택한 폴더에 독립 실험을 실행한다.
    print(f"Derivative error: {report['derivative']['absolute_error']:.3e}; "
          f"gradient error: {report['gradient']['maximum_absolute_error']:.3e}; "
          f"orthogonality: {report['gradient']['maximum_normalized_dot']:.3e}")
    print(f"Figures and metrics saved in {arguments.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())	# 명령행 실행 결과를 프로세스 종료 코드로 전달한다.
