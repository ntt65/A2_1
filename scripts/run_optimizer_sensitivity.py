"""선언한 초기점 다섯 개에서 결정론적인 GD/Momentum 민감도 실험을 재현한다."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")	# 파일 저장용 비대화형 백엔드를 pyplot보다 먼저 선택한다.
import matplotlib.pyplot as plt
import numpy as np

from src.calculus import quadratic, quadratic_gradient
from src.optimizer import Momentum, VanillaGD, elliptic, elliptic_gradient, run_optimization
from src.utils.cache_manager import CacheManager, make_cache_key
from src.utils.math import _as_finite_real_array
from src.utils.numerical import _finite_scalar
from src.utils.plotting import contour_grid
from src.utils.random import set_seed

__concepts__ = {
    "Optimizer_Sensitivity": ["trajectory", "radius_summary", "aggregate", "run_experiment"],
    "Optimizer_Stability_Evidence": ["stability_evidence"],
}

INITIAL_POINTS = ((5.0, 5.0), (-5.0, 5.0), (1.0, -3.0), (-2.0, -1.0), (0.0, 0.0))


def radius_summary(radii, radius: float = 0.1) -> dict:
    """t=0을 포함한 유한 반경 배열 (k+1,)을 요약하며 잘못된 입력은 예외를 발생시킨다.

    first_radius_0_1은 기록상 최초 진입이다. remaining_inside는 기준보다 큰
    마지막 반경 직후부터 시작하되 그 뒤에 수락한 점이 남아 있어야 한다.
    어느 지표도 무한 시간 동안의 성질을 보장하지 않는다.
    진입·유지가 없으면 예산 값으로 대체하지 않고 None을 반환한다.
    """
    values = _as_finite_real_array(radii, "radii")	# 반경 입력을 유한한 실수 배열로 검증한다.
    threshold = _finite_scalar(radius, "radius")
    if values.ndim != 1 or not values.size or np.any(values < 0):
        raise ValueError("radii must be a nonempty nonnegative vector")
    if threshold <= 0:
        raise ValueError("radius must be positive")
    inside = np.flatnonzero(values <= threshold)	# 초기점도 포함하여 반경 기준의 최초 진입을 찾는다.
    outside = np.flatnonzero(values > threshold)
    stay = int(outside[-1] + 1) if outside.size else 0	# 마지막 이탈 뒤 남아 있는 기록부터 유지 여부를 판단한다.
    return {
        "radius_threshold": threshold,
        "first_radius_0_1": int(inside[0]) if inside.size else None,
        "remaining_inside_radius_0_1_through_budget": stay if stay < len(values) else None,
        "final_radius": float(values[-1]),
    }


def trajectory(objective_name: str, method: str, initial, lr: float,
               steps: int = 100, beta: float = 0.9, cache: CacheManager | None = None) -> dict:
    """길이가 맞는 손실·norm 배열과 캐시된 float64 path(k+1,2)를 반환한다.

    원형·타원형 목적함수에 기존 GD/Momentum 구현을 재사용한다.
    초기점 shape는 정확히 (2,), steps는 음이 아닌 정수이며 핵심 구현이
    lr과 beta를 검증한다. 캐시 식별에는 초기 좌표 전체, shape/dtype,
    알고리즘·목적함수 버전, 학습률, 예산, seed와 NumPy를 포함한다.
    잘못된 이름·자료형·shape는 예외를 발생시키며 난수 상태를 소비하지 않는다.
    """
    if objective_name not in ("circle", "ellipse") or method not in ("GD", "Momentum"):
        raise ValueError("use circle/ellipse and GD/Momentum")
    point = _as_finite_real_array(initial, "initial").copy()	# 초기 좌표를 복사하여 호출자의 배열을 보호한다.
    if point.shape != (2,):
        raise ValueError("initial must have shape (2,)")
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)):
        raise TypeError("steps must be an integer")
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    optimizer = VanillaGD(lr) if method == "GD" else Momentum(lr, beta)	# 각 경로에 독립적인 최적화 상태를 생성한다.
    objective = (lambda p: quadratic(p[0], p[1])) if objective_name == "circle" else elliptic
    gradient = quadratic_gradient if objective_name == "circle" else elliptic_gradient
    parameters = {
        "objective": objective_name, "method": method, "lr": optimizer.lr,
        "beta": optimizer.beta if method == "Momentum" else None,
        "steps": int(steps), "gradient_tolerance": None, "seed": 42,
        "numpy_version": np.__version__, "radius_threshold": 0.1,
    }

    def compute() -> dict:
        """새 유한 경로를 계산하고 유한한 관측 구간의 반경 요약을 반환한다."""
        result = run_optimization(objective, gradient, point, optimizer, steps=int(steps))	# 전체 갱신 예산의 경로와 목적함수 이력을 계산한다.
        radii = np.linalg.norm(result.path, axis=1)	# 초기점을 포함한 각 기록의 원점 거리를 구한다.
        return {**parameters, "initial": point.copy(), "path": result.path,
                "losses": result.losses, "gradient_norms": result.gradient_norms,
                "iterations": result.iterations, "status": result.status,
                **radius_summary(radii)}

    if cache is None:
        return compute()
    key = make_cache_key("optimizer_sensitivity", "quadratic-sensitivity-v1", point, parameters)	# 초기 좌표와 모든 실행 조건을 캐시 키에 포함한다.
    return cache.get_or_compute(key, compute)


def aggregate(records: list[dict]) -> dict:
    """도달하지 않은 값을 대체하지 않고 선언한 비어 있지 않은 기록 집합을 요약한다.

    진입·유지 횟수의 평균과 분산은 성공한 기록만 사용하며 성공 수와 분모를
    명시한다. 최종 반경 통계에는 모든 기록을 포함한다. 모집단 분산은
    ddof=0을 사용한다. 이 결정론적 집합은 무작위 모집단의 추정치가 아니다.
    잘못된 기록은 예외를 발생시킨다.
    """
    if not records:
        raise ValueError("records must be nonempty")
    summary = {"sample_count": len(records), "variance_ddof": 0}
    for field in ("first_radius_0_1", "remaining_inside_radius_0_1_through_budget"):
        values = [record[field] for record in records if record[field] is not None]	# 미도달 None을 마지막 횟수로 대체하지 않고 성공만 집계한다.
        summary[field] = {
            "success_count": len(values), "failure_count": len(records) - len(values),
            "mean_successful_updates": float(np.mean(values)) if values else None,
            "variance_successful_updates": float(np.var(values, ddof=0)) if values else None,
        }
    final = np.asarray([record["final_radius"] for record in records], dtype=np.float64)	# 최종 반경 통계의 분모에는 전체 표본을 포함한다.
    summary["final_radius_all_samples"] = {
        "mean": float(np.mean(final)), "variance": float(np.var(final, ddof=0)),
        "minimum": float(np.min(final)), "maximum": float(np.max(final)),
    }
    return summary


def stability_evidence() -> list[dict]:
    """닫힌 반복식으로 원형 GD의 lr=.5/1/1.1을 검증하고 스칼라 근거를 반환한다.

    theta0=(5,5)에서 theta[t]=(1-2lr)**t theta0이고 손실 배율은
    (1-2lr)**2다. 유한한 20회 갱신으로 관측 행동을 확인하며
    이론적 분류는 이 특정 이차함수의 반복식에서 도출한다.
    독립 검증에 실패하면 AssertionError를 발생시킨다.
    """
    evidence = []
    for rate, classification in ((0.5, "one_step_convergence"),
                                 (1.0, "constant_radius_sign_oscillation"),
                                 (1.1, "divergence_for_nonzero_initial")):
        record = trajectory("circle", "GD", [5.0, 5.0], rate, steps=20)	# 고정한 초기점에서 유한한 20회 갱신을 관측한다.
        multiplier = 1.0 - 2.0 * rate
        expected = np.full((21, 2), 5.0) * multiplier ** np.arange(21)[:, None]	# 실행기를 재사용하지 않은 닫힌 식으로 전체 기대 경로를 만든다.
        np.testing.assert_allclose(record["path"], expected, atol=1e-12, rtol=1e-13)	# 관측한 모든 좌표를 닫힌 해와 비교한다.
        assert np.all(np.isfinite(record["losses"]))
        nonzero = record["losses"][:-1] != 0	# 손실 배율 계산에서 0인 분모를 제외한다.
        observed = record["losses"][1:][nonzero] / record["losses"][:-1][nonzero]
        np.testing.assert_allclose(observed, multiplier**2, atol=1e-13, rtol=1e-13)	# 손실의 배율은 좌표 배율의 제곱과 일치해야 한다.
        evidence.append({
            "lr": rate, "steps": 20, "initial": [5.0, 5.0],
            "coordinate_multiplier": multiplier, "theoretical_loss_multiplier": multiplier**2,
            "observed_loss_ratio_min": float(np.min(observed)),
            "observed_loss_ratio_max": float(np.max(observed)),
            "observed_ratio_count": int(len(observed)), "classification": classification,
            "final_radius": record["final_radius"], "all_values_finite": True,
            "observed_strict_loss_increase": bool(np.all(np.diff(record["losses"]) > 0)),
        })
    return evidence


def sensitivity_figure(experiments: dict):
    """공간 축 비율을 맞춘 패널에 방법별 다섯 경로를 그려 Figure/Axes(2,2)를 반환한다.

    입력은 검증한 실험 기록이다. 좌표에는 단위가 없으며 방법·초기점 범례가
    경로를 가리지 않도록 패널을 분리한다. 호출자가 Figure를 저장하고 닫는다.
    잘못된 기록에서 발생하는 오류는 호출자에게 전달한다.
    """
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), layout="constrained")
    colors = ("#245c9e", "#d06a19", "#14866a", "#a2428c", "#272727")
    try:
        for row, (objective, rates) in enumerate(experiments.items()):
            surface = quadratic if objective == "circle" else (lambda x, y: x*x + 10*y*y)
            x, y, z = contour_grid(surface, (-6, 6), (-6, 6), points=101)	# 공통 등고선 격자를 사용해 목적함수의 곡률을 보여 준다.
            for column, (method, group) in enumerate(rates.items()):
                axis = axes[row, column]
                axis.contour(x, y, z, levels=12, cmap="Greys", alpha=0.4)
                for index, record in enumerate(group["runs"]):
                    path = np.asarray(record["path"])
                    initial = record["initial"]
                    axis.plot(path[:, 0], path[:, 1], "--", color=colors[index], linewidth=1.2,
                              label=f"s{index+1}: ({initial[0]:g}, {initial[1]:g})")
                    axis.scatter(*path[0], color=colors[index], s=35, marker="s", zorder=4)
                    axis.scatter(*path[-1], color=colors[index], s=40, marker="x", zorder=5)
                axis.set(xlabel="x (unitless)", ylabel="y (unitless)", xlim=(-6, 6), ylim=(-6, 6),
                         title=f"{objective.title()} / {method}, lr={group['lr']:g}")
                axis.set_aspect("equal", adjustable="box")	# 가로·세로의 좌표 길이가 동일하게 보이도록 축 비율을 맞춘다.
                axis.legend(fontsize=8, loc="lower right")
        figure.suptitle("Sensitivity to five fixed initial points: GD and Momentum", fontsize=14)
        figure.supxlabel("100 updates; Momentum beta=0.9; square=start, x=end; zero start included", fontsize=10)
        return figure, axes
    except Exception:
        plt.close(figure)	# 사용이 끝난 그림 자원을 해제한다.
        raise


def _json_record(record: dict) -> dict:
    """평탄한 기록 하나의 배열을 목록으로 변환하며 스칼라 측정값은 보존한다."""
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in record.items()}	# 배열만 JSON 목록으로 바꾸고 통계의 None은 보존한다.


def report_markdown(report: dict) -> str:
    """검증한 민감도·통계·반복식 근거를 재사용 가능한 Markdown으로 구성한다."""
    lines = ["# Optimizer sensitivity and stability evidence", "",	# 기존 보고서 본문과 표 형식을 변경 없이 구성한다.
             "Reproduce: `python -m scripts.run_optimizer_sensitivity --no-cache`.", "",
             "Five declared starts: (5,5), (-5,5), (1,-3), (-2,-1), (0,0). "
             "float64; seed 42; 100 updates; Momentum beta=0.9; radius threshold=0.1.", "",
             "This is a deterministic sensitivity set, not repeated random trials. "
             "The optimum start is deliberately included. Statistics use population variance (ddof=0). "
             "Missing hits remain null; first crossing does not imply staying inside. "
             "A stay is measured only through the recorded budget.", "",
             "| Objective | Method | Start | lr | Updates | Final radius | First <=0.1 | Stay through budget |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for name, methods in report["experiments"].items():
        for method, group in methods.items():
            for run in group["runs"]:
                lines.append(f"| {name} | {method} | {run['initial']} | {run['lr']:g} | {run['iterations']} "
                             f"| {run['final_radius']:.6e} | {run['first_radius_0_1']} "
                             f"| {run['remaining_inside_radius_0_1_through_budget']} |")
    lines += ["", "## Conditional success statistics", "",
              "Means/variances below include only starts meeting the corresponding criterion. "
              "The JSON additionally includes all-start final-radius statistics and every raw trajectory.", "",
              "| Objective | Method | Metric | Success / all | Mean updates | Variance updates² |",
              "|---|---|---|---:|---:|---:|"]
    for name, methods in report["experiments"].items():
        for method, group in methods.items():
            for field, label in (("first_radius_0_1", "first crossing"),
                                 ("remaining_inside_radius_0_1_through_budget", "stay through budget")):
                item = group["summary"][field]
                lines.append(f"| {name} | {method} | {label} | {item['success_count']} / 5 "
                             f"| {item['mean_successful_updates']} | {item['variance_successful_updates']} |")
    lines += ["", "## Circle GD stability", "",
              "For a nonzero start, theta[t]=(1-2lr)^t theta[0]; loss multiplies by (1-2lr)^2. "
              "Classification follows this exact quadratic recurrence, not an arbitrary divergence threshold.", "",
              "| lr | Coordinate factor | Loss factor | Observed ratio range | Classification |",
              "|---:|---:|---:|---|---|"]
    for item in report["stability"]:
        lines.append(f"| {item['lr']} | {item['coordinate_multiplier']:.6g} "
                     f"| {item['theoretical_loss_multiplier']:.6g} "
                     f"| [{item['observed_loss_ratio_min']:.6g}, {item['observed_loss_ratio_max']:.6g}] "
                     f"| {item['classification']} |")
    lines += ["", "Ratios with zero denominators are excluded; lr=0.5 has only the first ratio (0). "
              "All 20 observed updates are finite. Newton/Adam and claims of universal superiority are outside this report.",
              "", "![Fixed-start trajectories](optimizer_sensitivity.png)", ""]
    return "\n".join(lines)	# 완성한 보고서 행을 줄바꿈으로 연결한다.


def run_experiment(output_dir: Path, use_cache: bool = True) -> dict:
    """고정 실험의 PNG/JSON/Markdown을 저장하고 JSON으로 표현 가능한 근거를 반환한다.

    이 실행 시작점에서 seed 42를 정확히 한 번 설정한다. 캐시 비활성화는
    읽기와 쓰기를 모두 우회한다. 파일 시스템 오류와 독립 assert 실패는
    전달한다. 각 실행은 (101,2) 경로와 길이가 맞는 유한 이력을 가져야 한다.
    """
    set_seed(42)	# 독립 실험의 시작점에서 난수 시드를 한 번 고정한다.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)	# 산출물을 저장할 디렉터리를 준비한다.
    cache = CacheManager(".cache/optimizer_sensitivity", enabled=use_cache)
    experiments = {}
    for name, rate in (("circle", 0.1), ("ellipse", 0.05)):
        methods = {}
        for method in ("GD", "Momentum"):
            records = [trajectory(name, method, start, rate, cache=cache) for start in INITIAL_POINTS]	# 같은 조건에서 선언한 다섯 초기점을 각각 실행한다.
            for record in records:
                assert record["path"].shape == (101, 2) and record["iterations"] == 100	# 100회 갱신과 초기점 하나를 경로 길이로 구분한다.
                assert record["status"] == "budget_exhausted"
                if method == "GD":
                    factors = np.array([1-2*rate, 1-2*rate if name == "circle" else 1-20*rate])
                    expected = record["initial"] * factors[None, :] ** np.arange(101)[:, None]	# 각 초기점과 좌표별 배율로 독립 GD 기대 경로를 계산한다.
                    np.testing.assert_allclose(record["path"], expected, atol=1e-14, rtol=1e-12)	# 전체 경로의 정확도를 원래 허용치로 검사한다.
            methods[method] = {"lr": rate, "summary": aggregate(records),
                               "runs": [_json_record(record) for record in records]}
        experiments[name] = methods
    report = {"seed": 42, "dtype": "float64", "numpy_version": np.__version__,
              "initial_points": [list(p) for p in INITIAL_POINTS], "steps": 100,
              "experiments": experiments, "stability": stability_evidence(),
              "figure": {"filename": "optimizer_sensitivity.png", "format": "PNG",
                         "width_pixels": 1680, "height_pixels": 1400, "dpi": 140},
              "interpretation": "Deterministic fixed-start sensitivity, not random retry statistics; "
              "conditional crossing means exclude failures, and finite-horizon stays are not asymptotic guarantees."}
    figure, _ = sensitivity_figure(experiments)
    try:
        figure.savefig(output_dir / "optimizer_sensitivity.png", dpi=140)	# 축·범례가 포함된 그림을 지정 출력 경로에 저장한다.
    finally:
        plt.close(figure)	# 사용이 끝난 그림 자원을 해제한다.
    (output_dir / "optimizer_sensitivity.json").write_text(	# 분모·미도달·원시 경로를 포함한 결과를 JSON으로 남긴다.
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    (output_dir / "optimizer_sensitivity.md").write_text(report_markdown(report), encoding="utf-8", newline="\n")	# 동일한 검증 결과로 읽기용 보고서를 저장한다.
    return report


def main(argv: list[str] | None = None) -> int:
    """출력 경로와 선택적 캐시 우회 설정으로 고정 실험 명령을 실행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)	# 호출자가 제공한 명령행 인자를 해석한다.
    report = run_experiment(args.output_dir, use_cache=not args.no_cache)	# 캐시 우회 옵션을 실제 실험 실행에 전달한다.
    for objective, methods in report["experiments"].items():
        for method, group in methods.items():
            summary = group["summary"]["remaining_inside_radius_0_1_through_budget"]
            print(f"{objective} / {method}: stay <=0.1 in {summary['success_count']}/5 starts; "
                  f"conditional mean={summary['mean_successful_updates']} updates")
    print(f"Sensitivity PNG, JSON and Markdown saved in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())	# 명령행 실행 결과를 프로세스 종료 코드로 전달한다.
