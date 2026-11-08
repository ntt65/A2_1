"""검증된 최적화 측정값을 읽기 쉬운 Markdown 보고서로 렌더링한다."""

import numpy as np

__concepts__ = {"Optimizer_Readable_Report": ["render_optimizer"]}


def _array(value, shape: tuple, name: str) -> np.ndarray:
    """입력을 변경하지 않고 요구한 shape의 유한 실수 배열을 반환한다.

    잘못된 dtype·shape·비유한 측정값에는 ValueError를 발생시킨다.
    일관성 검사에는 float64를 사용하며 호출자의 배열에 쓰지 않는다.
    """
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or array.shape != shape:	# 실수 배열의 정확한 shape를 확인한다.
        raise ValueError(f"{name} must be a real array with shape {shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(array, dtype=np.float64)	# 비교 dtype을 float64로 맞춘다.
    if not np.all(np.isfinite(result)):	# 변환 과정에서 발생한 비유한 값도 거부한다.
        raise ValueError(f"{name} must contain finite float64 measurements")
    return result


def _validate_run(run: dict, objective: str, method: str, steps: int, lr: float) -> tuple:
    """고정 실행 하나를 검사하고 경로·손실·gradient norm·반경 배열을 반환한다.

    shape는 (steps+1,2)와 (steps+1,)다. 저장된 요약은 전체 이력과 일치해야 한다.
    c는 원형에서 1, 타원형에서 10이다. 손실은 x²+c*y²이고 gradient norm은
    해당 해석 미분의 노름과 같아야 한다.
    수치 일관성에는 rtol=1e-12, atol=0을 사용한다. 잘못되거나 모순된 근거는
    보고서를 반환하기 전에 ValueError를 발생시킨다.
    """
    if not isinstance(run, dict):
        raise ValueError("each optimizer run must be a dictionary")
    expected = {"objective": objective, "method": method, "lr": lr,	# 목적함수·방법·학습률·예산·종료 상태의 고정 조건을 구성한다.
                "beta": 0.9 if method == "Momentum" else None,
                "steps": steps, "iterations": steps, "gradient_tolerance": None,
                "status": "budget_exhausted"}
    for key, value in expected.items():
        if key not in run or isinstance(run[key], (bool, np.bool_)) or run[key] != value:	# 누락 필드와 수치로 위장한 불리언 설정을 거부한다.
            raise ValueError(f"{objective}/{method}: unexpected {key}")
    initial = _array(run.get("initial"), (2,), "initial")
    path = _array(run.get("path"), (steps+1, 2), "path")	# 초기점 하나를 포함한 전체 경로 shape를 확인한다.
    losses = _array(run.get("losses"), (steps+1,), "losses")
    gradient_norms = _array(run.get("gradient_norms"), (steps+1,), "gradient_norms")
    if not np.array_equal(initial, [5.0, 5.0]) or not np.array_equal(path[0], initial):	# 초기점과 실제 경로의 첫 점이 모두 (5,5)여야 한다.
        raise ValueError("the fixed experiment must start at (5,5)")
    coefficient = 1.0 if objective == "circle" else 10.0
    with np.errstate(over="ignore", invalid="ignore"):
        radii = np.linalg.norm(path, axis=1)	# 원시 경로에서 각 기록의 반경을 구한다.
        expected_losses = path[:, 0]**2 + coefficient*path[:, 1]**2	# 저장 손실을 신뢰하지 않고 좌표에서 목적함수를 재계산한다.
        expected_norms = np.linalg.norm(path*np.array([2.0, 2.0*coefficient]), axis=1)	# 해석 gradient에서 실제 norm을 다시 계산한다.
    if not np.all(np.isfinite(radii)) or not np.all(np.isfinite(expected_norms)):
        raise ValueError("path-derived measurements must be finite")
    for stored, computed, name in ((losses, expected_losses, "losses"),
                                    (gradient_norms, expected_norms, "gradient norms")):
        if not np.allclose(stored, computed, rtol=1e-12, atol=0):	# 저장된 이력과 경로로 재계산한 수치가 일치해야 한다.
            raise ValueError(f"{name} disagree with the recorded path")
    final = float(_array(run.get("final_norm"), (), "final_norm"))
    if not np.isclose(final, radii[-1], rtol=1e-12, atol=0):	# 최종 반경 요약과 경로의 마지막 좌표를 대조한다.
        raise ValueError("final_norm disagrees with the recorded path")
    inside = np.flatnonzero(radii <= 0.1)	# 반경 기준의 진입 여부를 전체 이력에서 다시 찾는다.
    outside = np.flatnonzero(radii > 0.1)
    remaining = int(outside[-1]+1) if outside.size else 0	# 마지막 이탈 이후의 관측 구간 유지 시작을 계산한다.
    crossings = {"first_radius_0_1": int(inside[0]) if inside.size else None,
                 "remaining_inside_radius_0_1_through_budget": remaining if remaining < len(path) else None}
    for key, expected_value in crossings.items():
        value = run.get(key)
        if key not in run or (value is not None and	# 진입·유지 인덱스는 정수 또는 None만 허용한다.
                              (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)))):
            raise ValueError(f"{key} must be an integer update index or null")
        if value != expected_value:	# 저장된 도달 요약과 실제 반경 이력이 같아야 한다.
            raise ValueError(f"{key} disagrees with the complete radius history")
    return path, losses, gradient_norms, radii


def _number(value) -> str:
    """측정 스칼라는 유효숫자 일곱 자리로, 누락된 도달 인덱스는 미도달로 표시한다."""
    return "미도달" if value is None else f"{float(value):.6e}"	# 미도달을 숫자로 대체하지 않고 따로 표시한다.


def _point(value) -> str:
    """좌표쌍 (2,) 하나의 값을 바꾸지 않고 표시한다."""
    return f"({_number(value[0])}, {_number(value[1])})"	# 좌표의 두 성분을 같은 정밀도로 표시한다.


def render_optimizer(report: dict) -> str:
    """입출력 없이 고정 run_optimizer JSON 사전으로 Markdown을 반환한다.

    seed 42/float64, 100회 GD/Momentum 실행 네 개, 원형 GD의 20회 안정성
    사례 세 개를 요구한다. 모든 경로와 요약을 검증하며 잘못되거나 누락되거나
    모순된 근거는 ValueError를 발생시킨다. 최적화를 재실행하거나 report,
    원본 JSON, PNG 산출물을 변경하지 않는다.
    """
    if not isinstance(report, dict) or report.get("seed") != 42 or report.get("dtype") != "float64":	# 고정 재현 조건에 대응하는 보고서인지 확인한다.
        raise ValueError("optimizer report must record seed 42 and dtype float64")
    try:
        runs = []
        for objective, rate in (("circle", 0.1), ("ellipse", 0.05)):
            for method in ("GD", "Momentum"):
                run = report["comparisons"][objective][method]
                arrays = _validate_run(run, objective, method, 100, rate)	# 본문 표를 만들기 전에 기본 네 실행을 전부 검증한다.
                runs.append((objective, method, run, arrays))
        stability = []
        for rate, label in ((0.5, "1회 만에 원점"), (1.0, "반경 일정·부호 진동"), (1.1, "반경·손실 증가")):
            run = report["learning_rates"][str(rate)]
            arrays = _validate_run(run, "circle", "GD", 20, rate)	# 20회 안정성 사례도 같은 원시 이력 검사를 적용한다.
            expected = np.full((21, 2), 5.0)*(1.0-2.0*rate)**np.arange(21)[:, None]	# 원형 GD 반복식으로 독립 기대 경로를 구성한다.
            if not np.allclose(arrays[0], expected, rtol=1e-12, atol=0):	# 학습률 설명이 실제 경로와 모순되면 보고서를 거부한다.
                raise ValueError("learning-rate path contradicts the circle GD recurrence")
            stability.append((rate, label, run, arrays))
    except (KeyError, TypeError) as error:
        raise ValueError("missing or malformed optimizer measurements") from error
    if runs[0][3][3][-1] > 0.1:	# 필수 원형 GD 반경 기준을 만족한 경우만 보고서를 반환한다.
        raise ValueError("fixed circle GD run fails the required final-radius bound 0.1")
    names = {"circle": "원형", "ellipse": "타원형"}
    lines = ["# 기본 최적화 실험 결과", "",
             "**미션 연결: #9~#12 — GD 수렴·학습률·Momentum·타원형 비교.** 최적화는 앞에서 구한 기울기를 "
             "매개변수의 이동으로 바꾸는 과정이다. 원형 함수는 모든 방향의 곡률이 같고 타원형 함수는 y 방향의 "
             "곡률이 x 방향보다 열 배 크므로 같은 학습률에도 방향별 이동이 달라진다. 경로·손실·기울기를 함께 "
             "관찰하며 함수의 곡률, 학습률, 이전 상태의 누적이 수렴 양상을 어떻게 바꾸는지 확인한다.", "",
             "원시 수치: [optimizer_metrics.json](optimizer_metrics.json). seed=42, dtype=float64; 초기점=(5,5).",
             "수치·그림 재생성: `python -m scripts.run_optimizer --no-cache`; 보고서 생성: `python -m scripts.run_submission_reports`.",
             "원형은 f=x²+y², 타원형은 f=x²+10y². GD: θ←θ−lr·g; Momentum: v←0.9v+g, θ←θ−lr·v, v₀=0.", "",
             "## GD·Momentum: 같은 초기점과 100회 업데이트", "",
             "각 경로 shape=(101,2), 손실·gradient norm shape=(101,). 초기점은 0회이며 표의 최종값은 100회다.",
             "반경은 ||θ||₂. 최초 진입은 반경≤0.1인 첫 시점, 유지 시작은 그 뒤 **100회까지** 계속 만족한 첫 시점이다. 무한히 유지한다는 보장은 아니다.", "",
             "| 함수·방법 | lr | 최종 좌표 | 최종 손실 | 최종 반경 | 최종 gradient norm | 최초 진입 | 유지 시작 |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    for objective, method, run, (path, losses, norms, radii) in runs:
        lines.append(f"| {names[objective]} {method} | {run['lr']:g} | {_point(path[-1])} "
                     f"| {_number(losses[-1])} | {_number(radii[-1])} | {_number(norms[-1])} "
                     f"| {run['first_radius_0_1']} | {run['remaining_inside_radius_0_1_through_budget']} |")
    gd = runs[0][3]
    lines += ["", f"원형 GD의 최종 반경 {_number(gd[3][-1])}은 기준 0.1 이내다. "
              "최초 진입이 빠르더라도 이후 진동으로 다시 벗어날 수 있으므로 유지 시작과 함께 비교한다. "
              "결과는 표시한 설정의 관측값이며 Momentum의 보편적인 우월성을 의미하지 않는다.", "",
              "Momentum은 이전 기울기의 영향을 누적해 한 방향의 이동을 가속할 수 있지만 최소점을 지나친 뒤에도 "
              "관성이 남아 진동할 수 있다. 위 원형 실험에서는 최초 진입은 Momentum이 빠르고, 기록된 100회까지 "
              "계속 반경 안에 머무는 시점은 GD가 빠르다. 타원형의 이 설정에서도 GD의 최초 진입이 더 빠르다. "
              "따라서 어느 수렴 지표를 비교하는지 먼저 정해야 결과를 올바르게 해석할 수 있다.", "",
              "### 대표 경로 수치", "",
              "| 함수·방법 | 업데이트 | 좌표 (x,y) | 손실 | 반경 |",
              "|---|---:|---|---:|---:|"]
    for objective, method, run, (path, losses, norms, radii) in runs:
        for step in (0, 1, 2, 5, 10, 20, 50, 100):	# 초기점과 선택한 갱신 시점의 원시 경로를 함께 표시한다.
            lines.append(f"| {names[objective]} {method} | {step} | {_point(path[step])} "
                         f"| {_number(losses[step])} | {_number(radii[step])} |")
    lines += ["", "## 학습률 0.5·1.0·1.1: 20회 관측", "",
              "원형 GD는 θₜ=(1−2lr)ᵗθ₀이고 손실 배율은 (1−2lr)²다. "
              "0<lr<1에서 수렴, lr=0.5는 1회 만에 원점, lr=1은 진동, lr>1은 비영 초기점에서 발산한다.",
              "각 경로 shape=(21,2), 손실·gradient norm shape=(21,). 유한한 20회 관측과 이론적 분류를 구분한다.", "",
              "| lr | 좌표 배율 | 손실 배율 | 최종 좌표 | 최종 손실 | 최종 반경 | 해석 |",
              "|---:|---:|---:|---|---:|---:|---|"]
    for rate, label, run, (path, losses, norms, radii) in stability:
        factor = 1.0-2.0*rate	# 학습률에 따른 좌표 배율을 본문에 명시한다.
        lines.append(f"| {rate:g} | {factor:.6g} | {factor*factor:.6g} | {_point(path[-1])} "
                     f"| {_number(losses[-1])} | {_number(radii[-1])} | {label} |")
    lines += ["", "| lr | 업데이트 | 좌표 (x,y) | 손실 | 반경 |", "|---:|---:|---|---:|---:|"]
    for rate, label, run, (path, losses, norms, radii) in stability:
        for step in (0, 1, 5, 10, 20):
            lines.append(f"| {rate:g} | {step} | {_point(path[step])} | {_number(losses[step])} | {_number(radii[step])} |")
    lines += ["", "기존 그림: [원형 경로·손실·gradient](optimizer_circle.png) · "
              "[타원형 경로·손실·gradient](optimizer_ellipse.png) · [학습률별 경로](optimizer_learning_rates.png).", "",
              "원형 비교 그림은 왼쪽에서 등고선 위 경로, 가운데에서 손실, 오른쪽에서 기울기 norm을 보여 준다. "
              "사각형은 시작점, x 표시는 마지막 점이며 점선으로 연결한 점은 각 업데이트다. 경로가 원점을 지나갔다는 "
              "사실만으로 안정적으로 수렴했다고 볼 수 없어 손실과 기울기 그래프를 함께 읽는다. 두 세로축은 "
              "0 근처를 선형으로 표시해 정확한 0도 보존한다.", "",
              "![원형 함수에서 GD·Momentum의 경로와 손실·기울기 비교](optimizer_circle.png)", "",
              "타원형 그림도 같은 세 패널로 두 방법을 비교한다. 지정한 lr=0.05에서는 GD의 y 성분이 첫 갱신에 "
              "0이 되어 이후 x축을 따라 이동한다. 이 조건에서 지속적인 지그재그를 기대하거나 그림을 근거로 "
              "Momentum이 언제나 더 부드럽다고 해석하지 않는다.", "",
              "![타원형 함수에서 GD·Momentum의 방향별 이동과 수렴 비교](optimizer_ellipse.png)", "",
              "학습률 그림은 한 번에 도착·크기가 일정한 진동·커지는 진동을 분리해 보여 준다. 각 패널이 자기 좌표 "
              "범위를 사용하므로 화면의 선 길이만 비교하지 말고 축 눈금과 최종 반경을 읽어야 한다. 유한한 20회 "
              "관측은 위 점화식의 수렴·진동·발산 분류와 함께 해석한다.", "",
              "![학습률 0.5·1.0·1.1에 따른 즉시 수렴·진동·발산](optimizer_learning_rates.png)", ""]
    return "\n".join(lines)	# 검증한 값과 기존 설명 문자열을 결합해 반환한다.
