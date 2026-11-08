"""Step 4의 최적화 경로, 목적함수 이력과 안정성 사례를 재현한다."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")	# 파일 저장용 비대화형 백엔드를 pyplot보다 먼저 선택한다.
import matplotlib.pyplot as plt
import numpy as np

from src.calculus import quadratic, quadratic_gradient
from src.optimizer import (
    Momentum, VanillaGD, elliptic, elliptic_gradient, learning_rate_figure,
    optimizer_comparison_figure, run_optimization,
)
from src.utils.cache_manager import CacheManager, make_cache_key
from src.utils.random import set_seed

__concepts__ = {
    "Optimizer_Trajectory_Cache": ["cached_trajectory"],
    "Optimizer_Experiment": ["run_experiment", "main"],
}


def cached_trajectory(
    objective_name: str, method: str, lr: float, steps: int = 100,
    beta: float = 0.9, cache: CacheManager | None = None,
) -> dict:
    """초기점 [5,5]인 고정 이차원 실험 하나를 계산하거나 캐시에서 읽는다.

    objective_name은 'circle' 또는 'ellipse', method는 'GD' 또는 'Momentum'이다.
    기록에는 path(k+1,2), losses/norms(k+1), 설정과 종료 상태가 들어간다.
    캐시 키는 초기점 전체, 목적함수·알고리즘 버전, 방법, lr, beta, 예산,
    허용치 정책과 NumPy 버전을 포함한다. 난수 상태를 소비하지 않는다.
    잘못된 이름은 ValueError를 발생시키며 최적화 입력 오류는 그대로 전달한다.
    """
    if objective_name not in ("circle", "ellipse"):	# 지원하는 두 목적함수 이외의 이름을 거부한다.
        raise ValueError("objective_name must be circle or ellipse")
    if method not in ("GD", "Momentum"):
        raise ValueError("method must be GD or Momentum")
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)):	# 업데이트 횟수는 불리언이 아닌 정수여야 한다.
        raise TypeError("steps must be an integer")
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    steps = int(steps)
    initial = np.array([5.0, 5.0])	# 모든 기본 비교의 초기 좌표를 고정한다.
    optimizer = VanillaGD(lr) if method == "GD" else Momentum(lr, beta)	# 방법마다 새 최적화 상태를 생성한다.
    objective = (lambda p: quadratic(p[0], p[1])) if objective_name == "circle" else elliptic
    gradient = quadratic_gradient if objective_name == "circle" else elliptic_gradient	# 목적함수에 맞는 해석 gradient를 선택한다.
    parameters = {	# 경로에 영향을 주는 설정과 종료 정책을 캐시 키에 포함한다.
        "objective": objective_name, "method": method, "lr": float(lr),
        "beta": float(beta) if method == "Momentum" else None,
        "steps": steps, "gradient_tolerance": None, "numpy": np.__version__,
    }

    def compute() -> dict:
        """새 최적화 상태로 실행해 캐시가 지원하는 값과 원시 이력을 반환한다."""
        result = run_optimization(objective, gradient, initial, optimizer, steps=steps)	# 공통 실행기로 초기점과 모든 수락 갱신의 이력을 얻는다.
        radii = np.linalg.norm(result.path, axis=1)	# 초기점을 포함한 각 좌표의 원점 거리를 계산한다.
        hits = np.flatnonzero(radii <= 0.1)	# 반경 기준을 처음 만족한 기록을 찾는다.
        outside = np.flatnonzero(radii > 0.1)
        remaining_start = int(outside[-1] + 1) if outside.size else 0	# 마지막 이탈 뒤부터 관측 종료까지 유지하는 시작점을 구한다.
        return {
            **parameters, "initial": initial.copy(), "path": result.path,
            "losses": result.losses, "gradient_norms": result.gradient_norms,
            "iterations": result.iterations, "status": result.status,
            "final_norm": float(radii[-1]),
            "first_radius_0_1": int(hits[0]) if hits.size else None,
            "remaining_inside_radius_0_1_through_budget": remaining_start if remaining_start < len(radii) else None,
        }

    if cache is None:
        return compute()
    key = make_cache_key("optimizer_trajectory", "fixed-quadratics-v1", initial, parameters)	# 초기점 전체와 버전·설정으로 안정적인 캐시 키를 만든다.
    return cache.get_or_compute(key, compute)	# 캐시 적중과 새 계산에 동일한 반환 정책을 적용한다.


def _json_record(record: dict) -> dict:
    """평탄한 경로 기록의 ndarray를 값의 변경 없이 JSON 목록으로 변환한다."""
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in record.items()}	# 배열만 목록으로 바꾸고 스칼라·설정 값은 보존한다.


def run_experiment(output_dir: Path, use_cache: bool = True) -> dict:
    """독립 assert 검증 뒤 비교 PNG 세 개와 JSON 보고서를 저장한다.

    seed 42와 고정 초기점 [5,5]를 사용한다. 원형·타원형 비교는 각각
    100회 갱신하며 안정성 패널은 유한한 20회 갱신을 사용한다.
    캐시 사용 여부와 무관하게 동일한 수치 계약을 적용한다.
    파일 오류와 assert 실패는 전달하며 모든 그림은 저장 후 닫는다.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)	# 산출물을 저장할 디렉터리를 준비한다.
    set_seed(42)	# 독립 실험의 시작점에서 난수 시드를 한 번 고정한다.
    cache = CacheManager(".cache/optimizer", enabled=use_cache)
    comparisons = {}
    for name, rate, surface in (
        ("circle", 0.1, quadratic), ("ellipse", 0.05, lambda x, y: x*x + 10*y*y),
    ):
        runs = {method: cached_trajectory(name, method, rate, cache=cache) for method in ("GD", "Momentum")}	# 같은 목적함수와 학습률에서 두 방법을 비교한다.
        assert all(run["path"].shape == (101, 2) and run["iterations"] == 100 for run in runs.values())	# 초기점을 포함한 경로 길이와 실제 갱신 횟수를 확인한다.
        assert all(run["status"] == "budget_exhausted" for run in runs.values())
        if name == "circle":
            expected = np.full(2, 5.0 * 0.8**100)	# 원형 GD의 닫힌 해를 독립 기대값으로 만든다.
            np.testing.assert_allclose(runs["GD"]["path"][-1], expected, atol=1e-22, rtol=1e-12)	# 실측 최종 좌표를 닫힌 해와 비교한다.
            assert runs["GD"]["final_norm"] <= 0.1
        figure, _ = optimizer_comparison_figure(
            runs, surface, f"{name.title()}: lr={rate:g}, Momentum beta=0.9, start=(5,5), 100 updates",
        )
        try:
            figure.savefig(output_dir / f"optimizer_{name}.png", dpi=140)	# 축·범례가 포함된 그림을 지정 출력 경로에 저장한다.
        finally:
            plt.close(figure)	# 저장 성공 여부와 관계없이 그림 자원을 해제한다.
        comparisons[name] = {method: _json_record(run) for method, run in runs.items()}

    learning_rates = {rate: cached_trajectory("circle", "GD", rate, steps=20, cache=cache)
                      for rate in (0.5, 1.0, 1.1)}
    np.testing.assert_array_equal(learning_rates[0.5]["path"][1:], np.zeros((20, 2)))	# lr=0.5에서 한 번 갱신한 뒤 원점에 머무는지 검사한다.
    np.testing.assert_array_equal(learning_rates[1.0]["path"],	# lr=1에서 좌표 부호만 번갈아 바뀌는지 검사한다.
                                  np.tile([5.0, 5.0], (21, 1)) * (-1.0)**np.arange(21)[:, None])
    assert np.all(np.diff(learning_rates[1.1]["losses"]) > 0)	# lr=1.1에서 손실이 매번 증가하는지 확인한다.
    assert np.all(np.isfinite(learning_rates[1.1]["path"]))	# 발산 관측 구간의 계산값은 여전히 유한해야 한다.
    figure, _ = learning_rate_figure(learning_rates)
    try:
        figure.savefig(output_dir / "optimizer_learning_rates.png", dpi=140)	# 축·범례가 포함된 그림을 지정 출력 경로에 저장한다.
    finally:
        plt.close(figure)	# 저장 성공 여부와 관계없이 그림 자원을 해제한다.
    report = {
        "seed": 42, "numpy_version": np.__version__, "dtype": "float64",
        "comparisons": comparisons,
        "learning_rates": {str(rate): _json_record(run) for rate, run in learning_rates.items()},
        "notes": [
            "Momentum uses v_next=beta*v+gradient, theta_next=theta-lr*v_next, with v0=0.",
            "Fixed budgets include the initial point; optional early stopping is not used here.",
            "Circle GD convergence is 0<lr<1; lr=0.5 reaches zero in one update.",
            "Ellipse GD convergence is 0<lr<0.1; lr=0.05 eliminates y in one update.",
            "Momentum is not asserted to be faster at every step or for every setting.",
            "first_radius_0_1 is a first crossing, not a guarantee of remaining inside the radius.",
            "remaining_inside_radius_0_1_through_budget describes only the recorded finite horizon.",
        ],
    }
    (output_dir / "optimizer_metrics.json").write_text(	# 경로와 설정·종료 해석을 JSON으로 함께 보존한다.
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
        encoding="utf-8", newline="\n",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """출력 폴더와 캐시 우회 옵션을 받아 재현 가능한 최적화 실험을 실행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--no-cache", action="store_true")
    arguments = parser.parse_args(argv)	# 호출자가 제공한 명령행 인자를 해석한다.
    report = run_experiment(arguments.output_dir, not arguments.no_cache)	# 출력 위치와 캐시 우회 여부를 실제 실험에 전달한다.
    for name, runs in report["comparisons"].items():
        print(f"{name}: " + "; ".join(f"{method} radius={run['final_norm']:.6e}, loss={run['losses'][-1]:.6e}"
                                      for method, run in runs.items()))
    print(f"Figures and metrics saved in {arguments.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())	# 명령행 실행 결과를 프로세스 종료 코드로 전달한다.
