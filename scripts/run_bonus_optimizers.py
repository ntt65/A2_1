"""명시한 갱신 예산으로 미션의 Adam·Newton 비교를 재현한다."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")	# 파일 저장용 비대화형 백엔드를 pyplot보다 먼저 선택한다.
import matplotlib.pyplot as plt
import numpy as np

from scripts.run_optimizer_sensitivity import radius_summary
from src.adam import Adam
from src.calculus import quadratic, quadratic_gradient
from src.newton import NewtonMethod, circle_hessian, elliptic_hessian
from src.optimizer import (
    Momentum, VanillaGD, elliptic, elliptic_gradient,
    optimizer_comparison_figure, run_optimization,
)
from src.utils.cache_manager import CacheManager, make_cache_key
from src.utils.math import _as_finite_real_array
from src.utils.numerical import _finite_scalar
from src.utils.random import set_seed

__concepts__ = {
    "Bonus_Optimizer_Comparison": ["trajectory", "run_experiment", "main"],
    "Bonus_Optimizer_Cache": ["trajectory"],
}


def trajectory(objective_name: str, method: str, lr: float, steps: int = 200,
               initial=(5.0, 5.0), gradient_tolerance=None,
               cache: CacheManager | None = None) -> dict:
    """지정한 이차함수의 path(k+1,2)와 손실·norm 이력을 계산하거나 캐시에서 읽는다.

    목적함수 이름은 circle/ellipse, 방법은 GD/Momentum/Adam/Newton이다.
    모든 입력은 float64로 변환한다. initial은 정확히 (2,), steps는 음이 아닌
    정수이며 선택적 tolerance는 유한한 음이 아닌 스칼라다. 방법별 하이퍼파라미터는
    고정하고 기록한다. 캐시 식별에는 전체 수치 입력, dtype/shape, 버전, seed와
    종료 정책을 포함한다. 잘못된 입력의 예외는 호출자에게 전달한다.
    난수를 소비하지 않으며 캐시 반환 배열은 독립적인 소유권을 갖는다.
    """
    if objective_name not in ("circle", "ellipse"):
        raise ValueError("objective_name must be circle or ellipse")
    if method not in ("GD", "Momentum", "Adam", "Newton"):
        raise ValueError("method must be GD, Momentum, Adam or Newton")
    point = _as_finite_real_array(initial, "initial").copy()	# 호출자의 초기 배열을 변경하지 않도록 복사한다.
    if point.shape != (2,):	# 이 실험은 정확히 이차원 초기점만 허용한다.
        raise ValueError("initial must have shape (2,)")
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)):
        raise TypeError("steps must be an integer")
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    tolerance = None if gradient_tolerance is None else _finite_scalar(gradient_tolerance, "gradient_tolerance")	# 조기 종료 허용치가 있으면 유한 스칼라로 검증한다.
    if tolerance is not None and tolerance < 0:
        raise ValueError("gradient_tolerance must be nonnegative")
    hessian = circle_hessian if objective_name == "circle" else elliptic_hessian	# 목적함수에 맞는 Hessian 함수를 선택한다.
    if method == "GD":
        optimizer, settings = VanillaGD(lr), {}
    elif method == "Momentum":
        optimizer, settings = Momentum(lr, beta=0.9), {"beta": 0.9}
    elif method == "Adam":
        settings = {"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8}
        optimizer = Adam(lr=lr, **settings)	# 고정된 모멘트 계수와 epsilon으로 Adam 상태를 시작한다.
    else:
        optimizer, settings = NewtonMethod(hessian, lr=lr), {"hessian_policy": "symmetric_positive_definite"}	# 유한·대칭·양의 정부호 Hessian 계약을 가진 Newton을 생성한다.
    objective = (lambda p: quadratic(p[0], p[1])) if objective_name == "circle" else elliptic
    gradient = quadratic_gradient if objective_name == "circle" else elliptic_gradient
    parameters = {"objective": objective_name, "method": method, "lr": optimizer.lr,	# 캐시와 결과에 동일한 실행 설정을 기록한다.
                  "settings": settings, "steps_budget": int(steps),
                  "gradient_tolerance": tolerance, "seed": 42, "numpy_version": np.__version__}

    def compute() -> dict:
        """새 최적화 상태로 실행하여 유한한 이력과 반경 통계를 반환한다."""
        result = run_optimization(objective, gradient, point, optimizer, steps=int(steps),	# 선택한 종료 기준으로 공통 최적화 실행기를 호출한다.
                                  gradient_tolerance=tolerance)
        return {**parameters, "initial": point.copy(), "path": result.path,
                "losses": result.losses, "gradient_norms": result.gradient_norms,
                "iterations": result.iterations, "status": result.status,
                **radius_summary(np.linalg.norm(result.path, axis=1))}

    if cache is None:
        return compute()
    key = make_cache_key("bonus_optimizer", "adam-corrected-newton-spd-v1", point, parameters)	# 입력 좌표 전체와 계산 버전·설정으로 캐시를 구분한다.
    return cache.get_or_compute(key, compute)


def json_record(record: dict) -> dict:
    """평탄한 기록의 NumPy 배열을 목록으로 바꾸되 스칼라 메타데이터는 보존한다."""
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in record.items()}	# 원시 기록의 배열만 JSON 목록으로 변환한다.


def report_markdown(report: dict) -> str:
    """실측 속도·정확도를 표시하고 갱신 횟수와 전체 계산 비용을 구분한다."""
    lines = ["# Adam and Newton: executed comparison", "",	# 실측 보고서의 기존 Markdown 본문과 표 구조를 구성한다.
             "Reproduce: `python -m scripts.run_bonus_optimizers --no-cache`.", "",
             "float64; seed 42; start=(5,5). Adam: beta1=.9, beta2=.999, epsilon=1e-8; "
             "Momentum: beta=.9. Same rate within each three-method comparison; 200 updates.", "",
             "| Objective | Method | lr | Updates | Final radius | Final loss | First <=.1 | Stay through budget |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, methods in report["adam_comparisons"].items():
        for method, record in methods.items():
            lines.append(f"| {name} | {method} | {record['lr']} | {record['iterations']} | "
                         f"{record['final_radius']:.6e} | {record['losses'][-1]:.6e} | "
                         f"{record['first_radius_0_1']} | {record['remaining_inside_radius_0_1_through_budget']} |")
    lines += ["", "None means the criterion was not achieved; a first crossing can be followed by an exit. "
              "A finite-horizon stay is not an asymptotic guarantee. These fixed settings do not establish "
              "that one method is universally faster; equal rates have different meanings for normalized Adam updates.", "",
              "## Newton versus GD", "",
              "Ellipse x^2+10y^2, same initial point, maximum 200 updates, "
              "same stopping criterion ||gradient|| <= 1e-8. GD lr=.05; pure Newton lr=1.", "",
              "| Method | Accepted updates | Final gradient norm | Final loss | Stop |",
              "|---|---:|---:|---:|---|"]
    for method, record in report["newton_comparison"].items():
        lines.append(f"| {method} | {record['iterations']} | {record['gradient_norms'][-1]:.6e} | "
                     f"{record['losses'][-1]:.6e} | {record['status']} |")
    lines += ["", "Newton solves H*d=g with H=diag(2,20), hence it reaches this quadratic's minimizer "
              "in one update. It also requires Hessian construction and a dense linear solve per update "
              "(generally O(n^2) storage and O(n^3) solve work). Update counts are not runtime comparisons. "
              "The implementation requires a finite symmetric positive-definite Hessian and rejects singular/indefinite inputs. "
              "Fixed damping does not guarantee global convergence on arbitrary nonlinear objectives.", ""]
    for figure in report["figures"]:
        lines += [f"![{figure['filename']}]({figure['filename']})", ""]
    return "\n".join(lines)	# 본문 문자열을 변경하지 않고 줄바꿈으로 결합한다.


def run_experiment(output_dir: Path, use_cache: bool = True) -> dict:
    """독립적인 이차함수 검증 뒤 PNG 세 개와 JSON/Markdown을 생성한다.

    실험 시작점에서 seed를 한 번 설정하며 기존 기본 산출물을 덮어쓰지 않는다.
    고정 예산 비교의 경로는 (201,2)이고, 조기 종료하는 Newton/GD 경로는
    수락한 갱신 수에 초기점 하나를 더한 길이다. assert 실패와 파일 시스템
    오류는 전달하며 각 Figure는 저장 후 닫는다.
    """
    set_seed(42)	# 독립 실험의 시작점에서 난수 시드를 한 번 고정한다.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)	# 산출물을 저장할 디렉터리를 준비한다.
    cache = CacheManager(".cache/bonus_optimizers", enabled=use_cache)
    comparisons, figures = {}, []
    for name, rate in (("circle", 0.1), ("ellipse", 0.05)):
        runs = {method: trajectory(name, method, rate, cache=cache) for method in ("GD", "Momentum", "Adam")}	# 원형·타원형 각각 같은 조건에서 세 방법을 실행한다.
        for record in runs.values():
            assert record["path"].shape == (201, 2) and record["iterations"] == 200	# 200회 갱신과 초기점을 포함한 (201,2) 경로를 확인한다.
            assert record["status"] == "budget_exhausted"
            assert np.all(np.isfinite(record["path"]))
        factors = np.array([1-2*rate, 1-2*rate if name == "circle" else 1-20*rate])	# 각 좌표의 GD 갱신 배율을 독립 식으로 구한다.
        expected = np.full((201, 2), 5.0) * factors[None, :]**np.arange(201)[:, None]	# 전체 201개 좌표에 대한 닫힌 해를 구성한다.
        np.testing.assert_allclose(runs["GD"]["path"], expected, atol=1e-14, rtol=1e-12)	# 실제 GD 경로 전체를 독립 기대값과 비교한다.
        surface = quadratic if name == "circle" else (lambda x, y: x*x + 10*y*y)
        figure, _ = optimizer_comparison_figure(
            runs, surface, f"{name.title()}: GD / Momentum / Adam, lr={rate}, start=(5,5), 200 updates")
        filename = f"bonus_adam_{name}.png"
        try:
            figure.savefig(output_dir / filename, dpi=140)	# 축·범례가 포함된 그림을 지정 출력 경로에 저장한다.
        finally:
            plt.close(figure)	# 저장 성공 여부와 관계없이 그림 자원을 해제한다.
        figures.append({"filename": filename, "dpi": 140})
        comparisons[name] = {method: json_record(run) for method, run in runs.items()}
    newton = {method: trajectory("ellipse", method, rate, gradient_tolerance=1e-8, cache=cache)	# 두 방법에 같은 gradient 종료 허용치를 적용한다.
              for method, rate in (("GD", 0.05), ("Newton", 1.0))}
    np.testing.assert_allclose(newton["Newton"]["path"][-1], [0, 0], atol=1e-12, rtol=0)	# 순수 Newton이 이 이차함수의 원점에 도달했는지 검사한다.
    assert newton["Newton"]["iterations"] == 1	# 원점 도달에 수락된 갱신이 한 번인지 확인한다.
    assert all(record["status"] == "gradient_tolerance" for record in newton.values())	# GD와 Newton 모두 동일한 gradient 기준으로 종료해야 한다.
    figure, _ = optimizer_comparison_figure(
        newton, lambda x, y: x*x+10*y*y,
        "Ellipse: GD lr=.05 / Newton lr=1; start=(5,5); max 200; gradient tolerance 1e-8")
    try:
        figure.savefig(output_dir / "bonus_newton_ellipse.png", dpi=140)	# 축·범례가 포함된 그림을 지정 출력 경로에 저장한다.
    finally:
        plt.close(figure)	# 저장 성공 여부와 관계없이 그림 자원을 해제한다.
    figures.append({"filename": "bonus_newton_ellipse.png", "dpi": 140})
    report = {"seed": 42, "dtype": "float64", "numpy_version": np.__version__,
              "adam_comparisons": comparisons,
              "newton_comparison": {method: json_record(run) for method, run in newton.items()},
              "figures": figures, "all_assertions_passed": True}
    (output_dir / "bonus_optimizer_metrics.json").write_text(	# 모든 수치 검증 뒤 보너스 결과를 JSON으로 저장한다.
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    (output_dir / "bonus_optimizer_report.md").write_text(report_markdown(report), encoding="utf-8", newline="\n")	# 같은 실측 기록으로 읽기용 Markdown을 저장한다.
    return report


def main(argv: list[str] | None = None) -> int:
    """지정한 최적화 실험을 실행하며 검증이 성공하면 종료 코드 0을 반환한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)	# 호출자가 제공한 명령행 인자를 해석한다.
    report = run_experiment(args.output_dir, not args.no_cache)	# 명령행 옵션에 따라 보너스 실험을 실행한다.
    for name, methods in report["adam_comparisons"].items():
        print(name+": "+"; ".join(f"{method} radius={r['final_radius']:.6e}, stay={r['remaining_inside_radius_0_1_through_budget']}"
                                  for method, r in methods.items()))
    print("Newton/GD accepted updates: "+str({method: r["iterations"] for method, r in report["newton_comparison"].items()}))
    print(f"Bonus optimizer PNG, JSON and Markdown saved in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())	# 명령행 실행 결과를 프로세스 종료 코드로 전달한다.
