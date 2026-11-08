"""NumPy로 생성한 데이터로 Step 2의 그림과 실측 지표를 재현한다."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")	# 파일 저장용 비대화형 백엔드를 pyplot보다 먼저 선택한다.
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
    """픽셀 범위 [0,1], shape (size,size)의 질감 있는 합성 장면을 만든다.

    size는 8 이상의 정수다. 풍경, 원판, 곡선 질감과 Gaussian 잡음을 넣어
    높은 rank에서도 의미 있는 세부가 남게 한다. 전역 난수 상태를 소비하므로
    재현하려면 실험 시작점에서 set_seed(42)를 한 번 호출한다.
    """
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)):	# 불리언을 정수 크기로 받아들이지 않는다.
        raise TypeError("size must be an integer")
    if size < 8:
        raise ValueError("size must be at least 8")
    coordinates = np.linspace(-1.0, 1.0, int(size))	# 합성 이미지의 가로·세로 공통 좌표를 만든다.
    x, y = np.meshgrid(coordinates, coordinates)	# 좌표 벡터를 이차원 격자로 확장한다.
    background = 0.22 + 0.25 * (1.0 - y)
    boundary = 0.20 + 0.18*np.sin(5*x) + 0.10*np.sin(13*x)
    scene = np.where(y > boundary, 0.18 + 0.10*(x+1), background)	# 산의 경계를 기준으로 전경과 배경의 밝기를 나눈다.
    disk = (x + 0.45)**2 + (y + 0.40)**2 <= 0.16**2	# 원판 내부에 해당하는 픽셀을 선택한다.
    scene = np.where(disk, 0.92, scene)
    texture = 0.07*np.sin(40*x*y) + 0.05*np.sin(25*x + 8*y*y)
    texture += np.random.normal(0.0, 0.06, size=(int(size), int(size)))	# 실험 시작 시 설정한 전역 난수 상태로 질감을 추가한다.
    return np.clip(scene + texture, 0.0, 1.0)	# 화소 값만 [0,1] 범위로 제한해 반환한다.


def run_experiment(output_dir: Path, use_cache: bool = True) -> dict:
    """PNG 세 개, 실측 JSON, 명시적인 eig 비교 JSON/Markdown을 저장한다.

    seed 42, 고정 변환 사례와 지정된 대칭 행렬을 사용한다.
    SVD는 64x64 본 실험과 별도의 128x128 보충 실험을 비교한다.
    지표 보고서를 쓰기 전에 필수 수치 assert 검증을 수행한다.
    출력 디렉터리 생성·파일 오류는 호출자에게 전달하며 외부 데이터는 읽지 않는다.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)	# 산출물을 저장할 디렉터리를 준비한다.
    set_seed(42)	# 독립 실험의 시작점에서 난수 시드를 한 번 고정한다.
    cache = CacheManager(".cache/linear_algebra", enabled=use_cache)
    transforms = {}
    for name, matrix in (
        ("rotation", rotation_matrix(np.pi/4)),
        ("scaling", scaling_matrix(2, 0.5)),
        ("shear", shear_matrix(1)),
        ("reflection", scaling_matrix(-2, 3)),
        ("singular", scaling_matrix(2, 0)),
    ):
        result = determinant_area_check(matrix)	# 변환 좌표의 면적비와 행렬식을 독립적으로 비교한다.
        assert result["passed"], (name, result)	# 허용오차를 벗어난 변환은 보고서 저장 전에 중단한다.
        transforms[name] = result
    figure, _ = transformation_figure()
    try:
        figure.savefig(output_dir / "linear_transforms.png", dpi=140)	# 축·범례가 포함된 그림을 지정 출력 경로에 저장한다.
    finally:
        plt.close(figure)	# 저장 성공 여부와 관계없이 그림 자원을 해제한다.

    verification = verify_mission_eigenpair()	# 알고리즘과 분리된 eig 비교 검증을 실행한다.
    eigenpair = verification["power_iteration"]
    write_report(verification, output_dir)	# 고유쌍 비교의 수치와 해석을 별도 근거 파일로 남긴다.

    images = []
    for size, ranks, filename in (
        (64, (10, 50, 64), "svd_64.png"),
        (128, (10, 50, 100), "svd_128.png"),
    ):
        original = generate_grayscale_image(size)	# 각 입력 크기에 맞는 서로 다른 합성 원본을 만든다.
        factors = svd_decompose(original, cache)	# 전체 입력과 조건이 반영된 캐시로 SVD 인자를 구한다.
        norm = float(np.linalg.norm(original, ord="fro"))
        full_error = float(np.linalg.norm(original-reconstruct_svd(factors, size), ord="fro") / norm)	# 완전 rank 복원의 Frobenius 상대오차를 계산한다.
        assert full_error <= 1e-10	# 완전 복원도 명시한 수치 허용치로 검사한다.
        rows = []
        for rank in ranks:
            reconstruction = reconstruct_svd(factors, rank)
            error = float(np.linalg.norm(original-reconstruction, ord="fro") / norm)	# 화면 표시 전 배열에서 rank별 상대 복원오차를 측정한다.
            ratio = rank * (2*size+1) / original.size	# 압축 파일 바이트가 아닌 수치 인자의 원소 수를 비교한다.
            rows.append({"k": rank, "relative_error": error, "stored_over_original": ratio})
        assert np.all(np.diff([row["relative_error"] for row in rows]) <= 1e-12)	# rank가 증가할 때 복원오차가 수치 허용치 이상 증가하지 않는지 확인한다.
        numerical_rank = int(np.count_nonzero(factors[1] > factors[1][0]*size*np.finfo(float).eps))	# 특잇값과 기계 정밀도로 입력의 수치 rank를 판정한다.
        assert numerical_rank == size, "the comparison data should contain full-rank detail"	# 높은 rank 비교에 필요한 세부가 입력에 실제로 있는지 검사한다.
        figure, _ = svd_comparison_figure(original, list(ranks), cache)
        try:
            figure.savefig(output_dir / filename, dpi=140)	# 축·범례가 포함된 그림을 지정 출력 경로에 저장한다.
        finally:
            plt.close(figure)	# 저장 성공 여부와 관계없이 그림 자원을 해제한다.
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
    (output_dir / "linear_algebra_metrics.json").write_text(	# 검증을 마친 수치·환경·조건을 JSON으로 저장한다.
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
        encoding="utf-8", newline="\n",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """명령행에서 재현 실험을 실행하고 실측 요약을 출력한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--no-cache", action="store_true", help="recompute without cache reads or writes")
    arguments = parser.parse_args(argv)	# 호출자가 제공한 명령행 인자를 해석한다.
    report = run_experiment(arguments.output_dir, not arguments.no_cache)	# 명령행의 캐시 사용 여부를 실험에 전달한다.
    eigenpair = report["power_iteration"]
    print(f"Power Iteration: lambda={eigenpair['eigenvalue']:.12f}, "
          f"relative error={eigenpair['relative_error']:.3e}, residual={eigenpair['relative_residual']:.3e}")
    comparison = eigenpair["numpy_eig_comparison"]
    print(f"np.linalg.eig: lambda={comparison['numpy_eigenvalue']:.15f}; "
          f"relative error={comparison['relative_eigenvalue_error']:.6e} <= 0.05 (5%): PASS")
    print(f"Figures and metrics saved in {arguments.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())	# 명령행 실행 결과를 프로세스 종료 코드로 전달한다.
