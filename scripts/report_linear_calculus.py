"""실험을 다시 실행하지 않고 기존 선형대수·미적분 JSON 근거를 보고서로 렌더링한다."""

import numpy as np

from scripts.run_evaluation_evidence import step_size_summary

__concepts__ = {
    "Submission_Reports": ["render_linear_algebra", "render_calculus"],
    "Linear_Algebra_Experiment": ["render_linear_algebra"],
    "Calculus_Experiment": ["render_calculus"],
}


def _array(value, name: str, shape=None) -> np.ndarray:
    """유한 실수 근거를 float64로 읽으며 bool/complex와 예상과 다른 shape는 거부한다."""
    result = np.asarray(value)
    if result.dtype.kind not in "iuf" or (shape is not None and result.shape != shape):	# 실수 자료형과 명시한 shape를 먼저 확인한다.
        raise ValueError(f"{name}: require real values with shape {shape}")
    result = result.astype(np.float64)	# 근거 비교에 공통 float64를 사용한다.
    if not np.all(np.isfinite(result)):	# 비유한 값을 성공 근거로 받아들이지 않는다.
        raise ValueError(f"{name}: evidence must be finite")
    return result


def _scalar(value, name: str, nonnegative: bool = False) -> float:
    """필요하면 비음수 조건도 적용하여 유한 스칼라 측정값을 읽고 잘못된 근거에는 ValueError를 낸다."""
    result = float(_array(value, name, ()))	# shape () 검증을 거쳐 실제 스칼라만 읽는다.
    if nonnegative and result < 0:
        raise ValueError(f"{name}: require a nonnegative measurement")
    return result


def _agree(actual, expected, name: str, atol: float = 1e-14) -> None:
    """같은 shape와 명시적 허용치로 요약값·알려진 입력값의 불일치를 거부한다."""
    reference = _array(expected, name)
    observed = _array(actual, name, reference.shape)	# 기대값과 정확히 같은 shape를 요구해 브로드캐스팅 비교를 막는다.
    if not np.allclose(observed, reference, atol=atol, rtol=1e-12):	# 요약값과 원래 입력에서 도출한 값의 일관성을 검사한다.
        raise ValueError(f"{name}: evidence is inconsistent with its inputs")


def _fmt(value: float) -> str:
    """float를 다시 읽어도 같은 값이 되도록 자릿수를 보존하며 정수인 측정값의 .0만 생략한다."""
    number = float(value)
    return str(int(number)) if number.is_integer() else repr(number)	# 값을 보존하면서 보고서 표시만 구성한다.


def _verdict(passed: bool) -> str:
    """저장된 PASS 표기가 아니라 실제로 검사한 수치 조건에서 판정을 만든다."""
    return "PASS" if passed else "FAIL"	# 호출 시 계산한 조건으로 PASS/FAIL을 결정한다.


def _header(report: dict, title: str, stem: str) -> list[str]:
    """고정 seed/dtype과 재현 안내 머리말을 만들고 지원하지 않는 실험 메타데이터는 거부한다."""
    if isinstance(report["seed"], bool) or report["seed"] != 42 or report["dtype"] != "float64":	# 보고서가 지정된 재현 조건에 대응하는지 확인한다.
        raise ValueError("these reports require the seed-42 float64 experiment")
    return [
        f"# {title}", "",
        f"원본: [{stem}_metrics.json]({stem}_metrics.json); seed=42, dtype=float64.",
        "전체 재현: `python -m scripts.run_submission_reports`.",
        f"개별 실험: `python -m scripts.run_{stem}`. 아래 판정은 저장 수치와 고정 허용치를 대조한 결과다.", "",
    ]


def render_linear_algebra(report: dict) -> str:
    """입출력이나 난수 사용 없이 기본 run_linear_algebra 사전을 Markdown으로 렌더링한다.

    보고서는 변환 다섯 사례, (2,2) 고유값 문제, (64,64)/(128,128)의 SVD 근거를
    포함한다. 측정값을 고정 입력과 수용 기준에 대조하며 실패한 측정은 FAIL로
    표시한다. 누락 필드는 KeyError, 잘못되거나 모순된 근거는 ValueError를 낸다.
    eig/SVD/면적 실험을 다시 실행하지 않으며 입력 사전을 변경하지 않는다.
    """
    lines = _header(report, "선형대수 실행 결과", "linear_algebra")
    lines += [
        "**미션 연결: #1~#4 — 행렬 변환·면적·고유값·SVD.** "
        "신경망의 선형층은 입력 좌표를 섞고 늘리는 행렬 연산이다. 이 실험은 그 연산이 공간의 모양과 면적을 "
        "어떻게 바꾸는지 확인한 뒤, 변환을 대표하는 방향과 이미지 복원에 필요한 성분을 살펴본다. "
        "수치표는 계산의 정확도를, 그림은 변환과 근사의 기하학적 의미를 보여 준다.", "",
        "## 행렬 변환과 면적", "",
        "고정 입력은 `scripts/run_linear_algebra.py:run_experiment`의 설정이다(JSON에는 변환 행렬 미저장). "
        "c=cos(π/4), s=sin(π/4); 단위 원 400점 P:(2,400)에 A:(2,2) @ P를 적용한다.", "",
        "| 변환 | A | det(A) | 면적 전 → 후 | 면적비 | 오차 종류·값 | 허용치 | 판정 |",
        "|---|---|---:|---|---:|---|---:|---|",
    ]
    transforms = report["transforms"]
    cases = (
        ("rotation", "[[c,-s],[s,c]]", 1), ("scaling", "[[2,0],[0,0.5]]", 1),
        ("shear", "[[1,1],[0,1]]", 1), ("reflection", "[[-2,0],[0,3]]", -6),
        ("singular", "[[2,0],[0,0]]", 0),
    )
    if set(transforms) != {case[0] for case in cases}:	# 기본 다섯 변환 사례의 누락이나 추가를 거부한다.
        raise ValueError("the five fixed transform cases are required")
    for name, matrix, known_determinant in cases:
        data = transforms[name]
        determinant = _scalar(data["determinant"], "determinant")
        before = _scalar(data["area_before"], "area_before", True)
        after = _scalar(data["area_after"], "area_after", True)
        ratio = _scalar(data["area_ratio"], "area_ratio", True)
        error = _scalar(data["error"], "area error", True)
        if before == 0:
            raise ValueError("source polygon area must be positive")
        _agree(determinant, known_determinant, "fixed transform determinant")	# 고정 행렬의 독립 행렬식과 실측을 비교한다.
        _agree(data["expected_ratio"], abs(determinant), "expected area ratio")
        _agree(ratio, after / before, "area ratio")	# 원시 면적에서 계산한 비율과 저장 요약을 대조한다.
        expected_error = abs(ratio - abs(determinant))	# 먼저 면적비의 절대 차이를 계산한다.
        if known_determinant != 0:
            expected_error /= abs(determinant)	# 비특이 행렬에만 상대오차 분모를 적용한다.
        _agree(error, expected_error, "area error")
        kind = "relative" if known_determinant else "absolute"
        tolerance = .01 if known_determinant else 1e-10
        _agree(data["tolerance"], tolerance, "area tolerance", atol=0)	# 검증된 수용 기준의 허용치가 변경되지 않았는지 확인한다.
        if data["error_kind"] != kind:
            raise ValueError("area error_kind does not match singular/non-singular case")
        lines.append(f"| {name} | `{matrix}` | {_fmt(determinant)} | {_fmt(before)} → {_fmt(after)} "
                     f"| {_fmt(ratio)} | {kind}: {_fmt(error)} | {_fmt(tolerance)} "
                     f"| {_verdict(error <= tolerance)} |")
    lines += ["", "상대오차 = abs(면적비−abs(det))/abs(det); 특이 행렬은 면적비 절대오차를 사용한다.",
        "[변환 PNG](linear_transforms.png)", "",
        "세 패널의 회색 점선은 원본 단위 원, 파란 실선은 변환 결과다. 화살표는 같은 기준점의 이동을 보여 준다. "
        "원은 회전해도 윤곽이 겹치므로 회전 여부는 화살표를 함께 읽어야 한다. 스케일링은 가로를 늘리고 세로를 "
        "줄여 타원을 만들지만 배율의 곱이 1이어서 면적은 유지된다. 전단도 이 예제에서는 면적을 보존한다. "
        "표의 음의 행렬식은 방향 반전을 뜻하며 면적 크기는 절댓값과 비교한다.", "",
        "![단위 원의 회전·스케일링·전단과 기준 방향의 변화](linear_transforms.png)", "",
        "## Power Iteration과 eig", "",
        "Power Iteration은 변환을 반복할 때 상대적으로 가장 강하게 남는 방향을 찾는다. 매번 벡터를 "
        "정규화해 크기를 제어하고 방향의 변화를 관찰한다. 이 예제는 두 고유값의 절댓값이 분리되고 초기 벡터가 "
        "지배 방향의 성분을 가져 수렴한다. 고유값 오차는 기준값과의 일치, 잔차는 Av≈λv의 일치를 점검한다.", "",
        "반복식은 w=A v_t, v_(t+1)=w/||w||₂이고, 고유값은 Rayleigh quotient "
        "λ=(vᵀAv)/(vᵀv)로 추정한다. 고유기저에서 v₀=Σ c_i u_i이면 A^t v₀=Σ c_i λ_i^t u_i이므로 "
        "유일하게 절댓값이 큰 고유값과 그 방향의 0이 아닌 초기 성분이 있을 때 다른 성분의 상대 비중이 줄어든다. "
        "일반 행렬에서 이러한 조건 없이 지배값 수렴을 보장하지 않으며, 여기의 양의 대칭 행렬에서는 최대 절댓값과 최대 고유값이 같다. "
        "초기 영벡터는 정규화할 수 없어 거부한다. Av=0이면 길이로 나누지 않고 영 곱 종료를 보고한다.", ""]
    eigenpair = report["power_iteration"]
    comparison = eigenpair["numpy_eig_comparison"]
    _agree(eigenpair["matrix"], [[4, 1], [1, 3]], "eigenproblem matrix", atol=0)	# 저장된 행렬이 고정 고유값 문제와 같은지 확인한다.
    _agree(eigenpair["initial"], [1, 1], "initial eigenvector", atol=0)
    eigenvalue = _scalar(eigenpair["eigenvalue"], "eigenvalue")
    reference = _scalar(comparison["numpy_eigenvalue"], "numpy eigenvalue")
    independent = (7 + np.sqrt(5.0)) / 2	# 반복 구현이나 eig 결과와 독립적인 해석 기준을 사용한다.
    _agree(reference, independent, "independent eigenvalue")
    _agree(eigenpair["reference"], independent, "analytic eigenvalue")
    relative = abs(eigenvalue - reference) / abs(reference)
    _agree(comparison["relative_eigenvalue_error"], relative, "eig relative error")	# 보고된 고유값 오차가 원시 두 값의 차이와 일치해야 한다.
    _agree(eigenpair["relative_error"], abs(eigenvalue - independent) / independent,
           "analytic relative error")
    residual = _scalar(eigenpair["relative_residual"], "residual", True)
    vector = _array(eigenpair["eigenvector"], "eigenvector", (2,))	# 요약 잔차만 믿지 않고 실제 고유벡터를 검증한다.
    _agree(np.linalg.norm(vector), 1., "normalized eigenvector")	# 반복 결과 벡터의 단위 길이를 확인한다.
    matrix = _array(eigenpair["matrix"], "eigenproblem matrix", (2, 2))
    observed_residual = np.linalg.norm(matrix @ vector - eigenvalue * vector) / (	# 원시 고유쌍으로 상대 잔차를 다시 계산한다.
        np.linalg.norm(matrix, ord="fro") * np.linalg.norm(vector))
    _agree(residual, observed_residual, "eigenpair residual")	# 저장된 잔차가 실제 고유쌍과 모순되면 거부한다.
    _agree(comparison["independent_relative_residual"], residual, "independent residual")
    _agree(comparison["relative_eigenvalue_tolerance"], .05, "eig tolerance", atol=0)
    _agree(comparison["relative_residual_tolerance"], 1e-6, "residual tolerance", atol=0)
    iterations = _scalar(eigenpair["iterations"], "iterations", True)
    if iterations != int(iterations):
        raise ValueError("iterations must be an integer")
    lines += [
        "A=[[4,1],[1,3]], 초기 v=[1,1]; 기준 λ=(7+sqrt(5))/2. 수동 반복에는 eig를 호출하지 않는다.",
        f"정규화 고유벡터 v=[{_fmt(vector[0])}, {_fmt(vector[1])}], shape=(2,).",
        f"λ_power={_fmt(eigenvalue)}, λ_eig={_fmt(reference)}; 상대오차={_fmt(relative)} ≤ 0.05: "
        f"{_verdict(relative <= .05)}.",
        f"상대 잔차 ||Av−λv||₂/(||A||F·||v||₂)={_fmt(residual)} ≤ 1e-6: "
        f"{_verdict(residual <= 1e-6)}; 반복={int(iterations)}, converged={eigenpair['converged']}, "
        f"종료={eigenpair['reason']}.",
        "작은 잔차만으로 임의 입력의 지배 고유값 도달을 보장하지 않는다. "
        "[eig 상세 보고서](power_iteration_eig_comparison.md)", "", "## SVD 복원", "",
        "SVD는 입력과 출력의 직교 기저 사이에서 행렬을 특잇값 성분으로 나누고, 큰 성분부터 남겨 이미지를 "
        "근사한다. 정방행렬의 Av=λv를 찾는 고유값 문제와 구분되며 직사각형 행렬에도 적용할 수 있다. "
        "rank를 늘릴 때 복원 오차가 얼마나 줄고 저장할 원소가 얼마나 늘어나는지를 함께 비교한다.", "",
        "A_k=U[:,:k] diag(s[:k]) Vt[:k,:]; 오차=||A−A_k||F/||A||F (표시용 clipping 전). "
        "저장 비율=k(m+n+1)/(mn), 원소 수 기준이며 1보다 크면 저장량 절감이 아니다.", "",
        "| 원본 shape | k | 상대 Frobenius 오차 | 저장/원본 원소 비율 |",
        "|---|---:|---:|---:|",
    ]
    images = report["svd"]
    if len(images) != 2:	# 64와 128 입력을 혼합하거나 누락하지 못하게 한다.
        raise ValueError("both the 64 and 128 SVD experiments are required")
    full_rows = []
    for image, size, ranks in zip(images, (64, 128), ((10, 50, 64), (10, 50, 100))):
        _agree(image["shape"], [size, size], "SVD shape", atol=0)
        if image["filename"] != f"svd_{size}.png":
            raise ValueError("SVD filename does not match its shape")
        _agree(image["numerical_rank"], size, "SVD numerical rank", atol=0)
        rows = image["comparisons"]
        if [row["k"] for row in rows] != list(ranks):	# 각 입력 크기에서 선언한 유효 rank만 허용한다.
            raise ValueError("SVD comparisons do not contain the fixed valid ranks")
        errors = []
        for row in rows:
            error = _scalar(row["relative_error"], "SVD error", True)
            ratio = _scalar(row["stored_over_original"], "SVD storage ratio", True)
            _agree(ratio, row["k"] * (2 * size + 1) / size**2, "SVD storage ratio", atol=0)	# 저장 비율을 수치 인자의 원소 수 식으로 대조한다.
            errors.append(error)
            lines.append(f"| {size}×{size} | {row['k']} | {_fmt(error)} | {_fmt(ratio)} |")
        full_error = _scalar(image["full_rank_relative_error"], "full rank error", True)
        if ranks[-1] == size:
            _agree(full_error, errors[-1], "full-rank comparison error")	# 전체 rank 행과 별도 완전 복원 검사값이 같아야 한다.
        full_rows.append(f"{size}×{size}: 완전 rank 오차={_fmt(full_error)} ≤ 1e-10: "
                         f"{_verdict(full_error <= 1e-10)}; k 증가 시 오차 비증가(atol=1e-12): "
                         f"{_verdict(bool(np.all(np.diff(errors) <= 1e-12)))}.")
    lines += ["", *full_rows,
              "64×64 입력에서 rank 100은 불가능하므로 본 실험은 10/50/64, "
              "별도 128×128 실험은 10/50/100을 사용한다. 배열 크기이며 전체 PNG 크기가 아니다.",
              "[64×64 PNG](svd_64.png) · [128×128 PNG](svd_128.png)", "",
              "각 그림의 왼쪽은 원본, 오른쪽은 세 rank 복원이며 한 그림 안에서는 같은 밝기 기준을 사용한다. "
              "고정 시드로 만든 합성 풍경·원판·질감의 세부가 얼마나 복원되는지와 저장 비율을 함께 읽는다. "
              "가장 선명한 결과가 가장 효율적인 압축이라는 결론은 나오지 않는다. 두 크기는 서로 다른 입력 실험이므로 "
              "그림 사이의 차이를 rank 효과 하나로 설명하지 않는다.", "",
              "![64×64 합성 이미지의 원본과 rank 10·50·64 복원](svd_64.png)", "",
              "![128×128 합성 이미지의 원본과 rank 10·50·100 복원](svd_128.png)"]
    return "\n".join(lines) + "\n"	# 검증 후 기존 보고서 문자열을 줄바꿈으로 연결한다.


def render_calculus(report: dict) -> str:
    """입출력이나 실험 재실행 없이 기본 run_calculus 사전을 Markdown으로 렌더링한다.

    고정 x=3/h=1e-5 사례, (9,2) gradient 근거와 유한한 순서 있는 h 격자가 필요하다.
    저장된 배열과 오차 요약을 대조한다. 기준을 벗어난 측정은 FAIL로 표시하지만
    잘못되거나 모순된 사전은 ValueError, 누락 필드는 KeyError를 발생시킨다.
    입력과 난수 상태는 변경하지 않는다.
    """
    lines = _header(report, "미적분 실행 결과", "calculus")
    lines += [
        "**미션 연결: #5~#6 — 중심차분과 Gradient 기하.** "
        "미분은 입력을 조금 바꿀 때 함수가 얼마나 변하는지 알려 준다. 중심차분은 양쪽 함수값으로 이 변화율을 "
        "근사하므로 손으로 구한 미분과 이후 역전파를 독립적으로 확인하는 도구가 된다. 여기서는 근사의 오차를 먼저 "
        "측정하고, 여러 변수에서 기울기가 가리키는 방향을 등고선과 연결한다.", "",
    ]
    derivative = report["derivative"]
    if derivative["function"] != "x**2":
        raise ValueError("the fixed derivative function must be x**2")
    _agree([derivative["x"], derivative["h"], derivative["analytic"], derivative["tolerance"]],	# 고정 입력·해석 미분·허용치가 원래 조건과 일치하는지 확인한다.
           [3, 1e-5, 6, 1e-4], "fixed derivative conditions", atol=0)
    numerical = _scalar(derivative["numerical"], "numerical derivative")
    error = abs(numerical - 6)
    _agree(derivative["absolute_error"], error, "derivative error")	# 보고된 오차를 원시 수치 미분값으로 교차 검증한다.
    lines += [
        "## 중심차분과 h 탐색", "",
        "f(x)=x², x=3, h=1e-5; (f(x+h)−f(x−h))/(2h), 해석값 f′(3)=6.", "",
        "| 검사 | 측정 | 독립 기대값 | 절대오차 | 허용치 | 판정 |",
        "|---|---:|---:|---:|---:|---|",
        f"| 고정 중심차분 | {_fmt(numerical)} | 6 | {_fmt(error)} | 0.0001 | {_verdict(error <= 1e-4)} |", "",
    ]
    study = report["step_size_study"]
    steps = _array(study["h"], "h grid")
    if steps.ndim != 1 or steps.size == 0:
        raise ValueError("h grid must be a nonempty vector")
    quadratic_values = _array(study["quadratic_derivative"], "quadratic derivatives", steps.shape)
    errors = _array(study["quadratic_error"], "quadratic errors", steps.shape)
    _agree(errors, np.abs(quadratic_values - 6), "sampled quadratic errors")	# h별 원시 미분값에서 오차 배열을 재계산한다.
    summary = step_size_summary(steps, errors, tolerance=1e-4)	# 연속 구간을 보장하지 않는 표본 요약 함수를 재사용한다.
    count, passing = summary["sample_count"], summary["passing_count"]
    groups = "; ".join(
        f"{_fmt(group['first_h'])}~{_fmt(group['last_h'])} ({group['sample_count']}표본)"
        for group in summary["contiguous_sample_groups"]
    ) or "없음"
    minima = ", ".join(_fmt(value) for value in summary["minimizing_sampled_h"])
    sine_values = _array(study["sine_derivative"], "sine derivatives", steps.shape)
    sine_errors = _array(study["sine_error"], "sine errors", steps.shape)
    _agree(sine_errors, np.abs(sine_values - np.cos(3.0)), "sampled sine errors")	# sine의 독립 해석 미분 cos(3)과 오차를 비교한다.
    sine_summary = step_size_summary(steps, sine_errors, tolerance=1e-4)
    lines += [
        f"x² 탐색: h={_fmt(steps[0])}~{_fmt(steps[-1])}, {count}개 표본. 절대오차 ≤ 1e-4: "
        f"PASS {passing}/{count}, FAIL {count-passing}/{count} (고정 h 검증과 별개).",
        f"통과한 인접 표본 묶음: {groups}. 실수 연속 구간 전체의 통과를 보장하지 않는다.",
        f"최소 관측오차={_fmt(summary['minimum_observed_error'])}, 해당 표본 h={minima}; "
        f"전체 최대오차={_fmt(np.max(errors))}.",
        f"sin(x), x=3 비교: 최소 관측오차={_fmt(sine_summary['minimum_observed_error'])}, "
        f"전체 최대오차={_fmt(np.max(sine_errors))}.",
        "x²의 중심차분은 정확 연산에서 절단오차가 0이다. sin(x)는 O(h²) 절단오차가 있고 "
        "작은 h에서는 반올림 오차가 커질 수 있다. 최소오차를 보인 표본 h를 보편적인 최적 간격으로 일반화하지 않는다.",
        "[h별 오차 PNG](derivative_error.png)", "",
        "그림의 가로축은 간격 h, 세로축은 해석값 대비 절대오차다. 두 실선은 x²와 sin(x)의 실제 오차이고 "
        "점선은 sin(x)의 2차 절단오차 주도항이다. 큰 h의 근사 오차와 작은 h의 뺄셈·반올림 영향을 구분해 읽는다. "
        "x²는 중심차분의 절단오차가 사라지는 특별한 예제이므로 일반 함수의 권장 h를 정하는 근거로 사용하지 않는다. "
        "세로축은 원점 부근을 선형으로 표시해 오차 0도 그대로 보존한다.", "",
        "![중심차분 간격에 따른 x²·sin(x) 오차와 sin(x)의 절단오차 주도항](derivative_error.png)", "",
        "## Gradient와 등고선 접선", "",
    ]
    gradient = report["gradient"]
    if gradient["function"] != "x**2+y**2":
        raise ValueError("the gradient function must be x**2+y**2")
    points = _array(gradient["points"], "gradient points", (9, 2))
    analytic = _array(gradient["analytic"], "analytic gradients", (9, 2))
    numeric = _array(gradient["numerical"], "numerical gradients", (9, 2))
    tangents = _array(gradient["nonzero_tangents"], "tangents", (8, 2))
    _agree(analytic, 2 * points, "analytic gradient", atol=0)	# 수치 함수와 독립적인 2*points 식으로 해석 gradient를 확인한다.
    _agree(points[-1], [0, 0], "origin", atol=0)
    _agree(np.linalg.norm(points[:-1], axis=1), np.full(8, 2.), "radius-two sample points")
    _agree(tangents, np.column_stack((-points[:-1, 1], points[:-1, 0])), "tangents", atol=0)
    gradient_error = float(np.max(np.abs(numeric - analytic)))	# 전체 성분의 최대 수치 미분 오차를 다시 구한다.
    _agree(gradient["maximum_absolute_error"], gradient_error, "maximum gradient error")
    _agree(gradient["tolerance"], 1e-8, "gradient tolerance", atol=0)
    dots = np.sum(analytic[:-1] * tangents, axis=1) / (	# 원점을 제외한 gradient와 접선의 정규화 내적을 계산한다.
        np.linalg.norm(analytic[:-1], axis=1) * np.linalg.norm(tangents, axis=1))
    _agree(gradient["normalized_dot_products"], dots, "normalized dot products")
    orthogonality = _scalar(gradient["maximum_normalized_dot"], "maximum normalized dot", True)
    _agree(orthogonality, np.max(np.abs(dots)), "maximum normalized dot")	# 저장된 최대 내적과 원시 내적 배열을 대조한다.
    _agree(gradient["orthogonality_tolerance"], 1e-12, "orthogonality tolerance", atol=0)
    _agree(gradient["origin_gradient"], analytic[-1], "origin gradient", atol=0)
    if gradient["origin_direction"] is not None:	# 영벡터인 원점에 임의 방향을 부여한 기록은 거부한다.
        raise ValueError("the origin must not be assigned a gradient direction")
    origin_ok = bool(np.all(numeric[-1] == 0))
    lines += [
        "f(x,y)=x²+y², ∇f=(2x,2y), 접선=(-y,x). 반경 2의 8점과 원점; "
        "측정 gradient shape=(9,2), 중심차분 h=1e-5.", "",
        "| 검사 | 측정 오차 | 허용치 | 판정 |", "|---|---:|---:|---|",
        f"| 수치/해석 gradient 최대 절대오차 | {_fmt(gradient_error)} | 1e-8 | {_verdict(gradient_error <= 1e-8)} |",
        f"| 원점 제외 정규화 내적 최대 절댓값 | {_fmt(orthogonality)} | 1e-12 | {_verdict(orthogonality <= 1e-12)} |",
        f"| 원점 수치 gradient [0,0] 일치 | {_fmt(np.max(np.abs(numeric[-1])))} | 0 | {_verdict(origin_ok)} |", "",
        "원점의 해석 gradient=[0,0], 방향 없음(null); 수직성 검사에서 제외한다. "
        "그림의 화살표 길이 정규화와 위 수치의 원래 크기는 구분한다.",
        "[Gradient/등고선 PNG](gradient_geometry.png)", "",
        "등고선을 따라 움직이면 함수값이 변하지 않아 그 방향의 변화율은 0이다. 반경 2의 여덟 점에서 "
        "주황색 화살표는 가장 빠르게 증가하는 기울기 방향, 파란색은 등고선의 접선 방향을 나타낸다. "
        "둘이 수직이므로 경사하강법은 주황색 화살표의 반대쪽으로 이동한다. 화살표는 방향을 읽기 쉽도록 "
        "길이를 각각 일정하게 표시했으며 실제 기울기 크기를 비교하는 그림은 아니다. 표의 내적 검증은 "
        "눈으로 본 수직 관계를 수치로 확인한다.", "",
        "![원형 등고선에 수직인 기울기와 접선 방향 및 원점의 영기울기](gradient_geometry.png)",
    ]
    return "\n".join(lines) + "\n"	# 검증을 마친 수치와 원래 설명으로 Markdown을 반환한다.
