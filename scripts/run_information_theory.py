"""독립 기대값과 Bernoulli 곡선으로 정보 이론 보너스를 재현한다."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")	# 파일 저장용 비대화형 백엔드를 pyplot보다 먼저 선택한다.
import matplotlib.pyplot as plt
import numpy as np

from src.information_theory import cross_entropy, entropy, kl_divergence
from src.utils.random import set_seed

__concepts__ = {
    "Information_Theory_Experiment": ["example_report", "run_experiment", "main"],
    "Information_Theory_Visualization": ["information_figure"],
}


def _json_number(value: float) -> float | str:
    """유한 스칼라는 수치로, +inf는 설명된 문자열 '+inf'로 저장하며 NaN/-inf는 거부한다."""
    value = float(value)
    if np.isposinf(value):	# 수학적으로 허용한 양의 무한대만 명시적 문자열로 인코딩한다.
        return "+inf"
    if not np.isfinite(value):	# NaN이나 음의 무한대를 성공한 근거로 저장하지 않는다.
        raise ValueError("only finite values or positive infinity belong in the report")
    return value


def example_report() -> list[dict]:
    """독립 상수를 검증하고 nats 단위의 스칼라 예제 기록을 반환한다.

    각 p/q는 정규화된 벡터 (n,)이며 broadcasting이나 clipping을 적용하지 않는다.
    유리수 사례의 기준값은 구현과 독립적으로 고정했다. 오차에는 atol=1e-14
    (subnormal 사례는 2e-13), rtol=0을 사용한다.
    경계의 무한대는 명시적으로 직렬화하며 assert 실패는 호출자에게 전달한다.
    """
    cases = [	# 구현과 독립적으로 정한 입력·정답 상수를 검증 사례로 사용한다.
        ("uniform_four", [.25, .25, .25, .25], [.25, .25, .25, .25],
         (1.3862943611198906, 1.3862943611198906, 0.)),
        ("point_mass", [0., 1.], [0., 1.], (0., 0., 0.)),
        ("rational_forward", [.25, .75], [.5, .5],
         (.5623351446188083, .6931471805599453, .13081203594113697)),
        ("rational_reverse", [.5, .5], [.25, .75],
         (.6931471805599453, .8369882167858358, .14384103622589045)),
        ("support_mismatch", [1., 0.], [0., 1.], (0., np.inf, np.inf)),
        ("smallest_positive_q", [1., 0.], [np.nextafter(0., 1.), 1.],
         (0., 744.4400719213812, 744.4400719213812)),
    ]
    records = []
    for label, p, q, expected in cases:
        actual = [entropy(p), cross_entropy(p, q), kl_divergence(p, q)]	# 같은 두 분포로 엔트로피·교차 엔트로피·KL을 실제 계산한다.
        tolerance = 2e-13 if label == "smallest_positive_q" else 1e-14	# 최소 양의 subnormal 사례에는 별도 명시한 절대 허용치를 적용한다.
        errors = []
        for measured, reference in zip(actual, expected):
            if np.isposinf(reference):
                assert np.isposinf(measured)	# 무한대 기대값을 임의의 유한 값으로 바꾸지 않는다.
                errors.append(None)
            else:
                error = abs(measured - reference)
                assert error <= tolerance	# 유한 사례의 절대오차를 독립 기준으로 검사한다.
                errors.append(float(error))
        if all(np.isfinite(value) for value in actual):
            assert abs(actual[1] - actual[0] - actual[2]) <= 2e-13	# 유한한 지지집합 사례에서 CE=H+KL 항등식을 확인한다.
            assert actual[2] >= -1e-14	# 이 독립 예제 범위에서 KL의 수치 비음수성을 검사한다.
        records.append({"name": label, "p": p, "q": q, "shape": [len(p)],
                        "entropy_nats": _json_number(actual[0]),
                        "cross_entropy_nats": _json_number(actual[1]),
                        "kl_nats": _json_number(actual[2]),
                        "independent_expected_H_CE_KL": [_json_number(value) for value in expected],
                        "absolute_errors_H_CE_KL": errors, "atol": tolerance, "rtol": 0., "passed": True})
    return records


def information_figure(parameters: np.ndarray, binary_entropy: np.ndarray,
                       model_parameters: np.ndarray, ce: np.ndarray, divergence: np.ndarray,
                       target_parameter: float, target_entropy: float):
    """nats 단위의 엔트로피와 유한한 Bernoulli CE/KL 곡선을 담은 Figure/Axes(2,)를 반환한다.

    입력 곡선은 shape (n,)이다. 호출자는 검증된 확률과 측정값을 제공한다.
    경계의 +inf는 그리는 선분에서 제외하고 설명으로 명시하며, 값을 잘라내거나
    유한한 값으로 표시하지 않는다. 확률 축은 단위가 없고 정보량 축은 자연로그의
    nats 단위다. 반환한 그림은 호출자가 닫아야 한다.
    """
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), layout="constrained")
    try:
        axes[0].plot(parameters, binary_entropy, color="#2466b0", linewidth=2,
                     label="H(Bernoulli(p))")
        axes[0].scatter([0., .5, 1.], [0., np.log(2.), 0.], color="#c45100", zorder=4,
                        label="Endpoints and maximum")
        axes[0].axhline(np.log(2.), color="#777777", linestyle=":", label="ln(2) nats")
        axes[0].set(xlabel="Outcome-1 probability p (unitless)", ylabel="Entropy (nats)",
                    title="Binary entropy: maximum at p=0.5", xlim=(0, 1), ylim=(-.025, .76))
        finite = np.isfinite(ce) & np.isfinite(divergence)	# 무한 경계 두 점을 유한 곡선으로 잘못 이어 그리지 않는다.
        axes[1].plot(model_parameters[finite], ce[finite], color="#2466b0", linewidth=2,
                     label="CE(P,Q)")
        axes[1].plot(model_parameters[finite], divergence[finite], color="#c45100", linewidth=2,
                     label="KL(P || Q)")
        axes[1].axhline(target_entropy, color="#29864a", linestyle="--", label="H(P); CE = H + KL")
        axes[1].axvline(target_parameter, color="#777777", linestyle=":", label="q=p=0.3")
        axes[1].scatter([target_parameter, target_parameter], [target_entropy, 0.],
                        color=["#2466b0", "#c45100"], zorder=4)
        axes[1].text(.5, .97, "q=0 or 1: CE and KL are +infinity\nEndpoint infinities are not drawn",	# 그리지 않은 무한대 경계의 의미를 그림 안에 명시한다.
                     transform=axes[1].transAxes, ha="center", va="top", fontsize=9,
                     bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
        axes[1].set(xlabel="Model outcome-1 probability q (unitless)", ylabel="Information (nats)",
                    title="Fixed P=Bernoulli(0.3), varying Q=Bernoulli(q)", xlim=(0, 1))
        for axis in axes:
            axis.grid(alpha=.25)
            axis.legend(fontsize=8, loc="lower center" if axis is axes[0] else "center")
        figure.suptitle("Entropy, cross-entropy and KL divergence — natural logarithms", fontsize=13)
        figure.supxlabel("501 grid values; float64; exact zero-probability conventions; no clipping or renormalization", fontsize=9)
        return figure, axes
    except Exception:
        plt.close(figure)	# 사용이 끝난 그림 자원을 해제한다.
        raise


def report_markdown(report: dict) -> str:
    """예제·수식·shape·확률 지지집합 규칙·실행한 검증을 Markdown으로 구성한다."""
    lines = ["# Information theory bonus: executed evidence", "",	# 기존 보고서 본문·수식·표 문자열을 그대로 구성한다.
             "Reproduce: `python -m scripts.run_information_theory`.", "",
             "NumPy float64; seed 42 set once at the experiment entry; natural logs, units nats.", "",
             "## Formulas and shape contract", "",
             "- `H(p) = -sum_i p_i ln(p_i)`.",
             "- `CE(p,q) = -sum_i p_i ln(q_i)`.",
             "- `KL(p||q) = sum_i p_i (ln(p_i)-ln(q_i))`; subtract logs instead of forming a ratio.", "",
             "The selected class axis is removed. Input `(n,)` returns a scalar; `(batch,n)` on axis=-1 returns `(batch,)`.",
             "Every probability is finite and in [0,1], and each class sum must equal 1 within atol=1e-12, rtol=0.",
             "p/q shapes must match exactly. No broadcasting, clipping, renormalization, or function-level reseeding occurs.",
             "A zero target term is zero, including 0*ln(0). Positive p with q=0 yields +inf.",
             "The JSON string `\"+inf\"` encodes this mathematical infinity; it is not a finite cap. NaN is never serialized.", "",
             "## Independent examples", "",
             "| Example | H(p), nats | CE(p,q), nats | KL(p||q), nats | Result |", "|---|---:|---:|---:|---|"]
    for row in report["examples"]:
        lines.append(f"| {row['name']} | {row['entropy_nats']} | {row['cross_entropy_nats']} | {row['kl_nats']} | PASS |")
    checks = report["curve_checks"]
    lines += ["", "Rational reference: p=(1/4,3/4), q=(1/2,1/2). Reversing KL changes its value.",
              "For the smallest positive float64 q, -ln(q)=1074 ln(2); CE/KL remain finite without overflow.", "",
              "## Bernoulli curves", "",
              "The first curve uses 501 values p in [0,1]. The second fixes P=(0.7,0.3) and uses the same q grid for Q=(1-q,q).",
              f"- Maximum binary entropy: {checks['maximum_entropy_nats']:.15f} nats at p={checks['maximum_entropy_parameter']}.",
              f"- Minimum finite KL: {checks['minimum_finite_kl_nats']:.6e} at q={checks['minimum_kl_parameter']}.",
              f"- Maximum finite-support CE-(H+KL) absolute error: {checks['maximum_identity_error']:.6e} <= 1e-12.",
              f"- Maximum H(p)-H(1-p) absolute error: {checks['maximum_entropy_symmetry_error']:.6e} <= 1e-14.",
              "- q=0 and q=1 produce +inf for both CE and KL; finite line segments exclude these endpoints.", "",
              "![Information theory curves](information_theory.png)", "",
              "CE = H + KL follows by separating sum p ln(p) and -sum p ln(q). Gibbs' inequality gives KL>=0 for normalized distributions, with equality when p=q on support.",
              "Thus, for fixed P, minimizing cross-entropy over Q also minimizes KL. KL is not symmetric and is not a distance metric.",
              "The 1e-12 input sum tolerance permits floating-point roundoff, not arbitrary unnormalized weights. No universal continuous-distribution or sample-estimation claims are made.", ""]
    return "\n".join(lines)	# 작성한 보고서 행을 줄바꿈으로 연결한다.


def run_experiment(output_dir: Path) -> dict:
    """독립 예제와 501점 곡선을 assert로 검증한 뒤 PNG/JSON/Markdown을 저장한다.

    입력·출력 확률의 shape는 (501,2), 정보량 곡선은 (501,)이다.
    모델 경계의 무한대는 JSON에서 '+inf'로 표현한다.
    계산 비용이 작으므로 의도적으로 캐시하지 않는다. 파일 오류와 수치 assert
    실패는 호출자에게 전달하며 생성한 Figure는 사용 후 닫는다.
    """
    set_seed(42)	# 독립 실험의 시작점에서 난수 시드를 한 번 고정한다.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)	# 산출물을 저장할 디렉터리를 준비한다.
    examples = example_report()	# 고정 예제 검증이 먼저 통과해야 곡선 실험을 진행한다.
    parameters = np.linspace(0., 1., 501, dtype=np.float64)
    binary_probabilities = np.column_stack((1. - parameters, parameters))	# Bernoulli의 두 클래스 확률을 (501,2)로 구성한다.
    binary_entropy = entropy(binary_probabilities)
    target_parameter = .3
    target = np.array([1. - target_parameter, target_parameter])
    targets = np.tile(target, (len(parameters), 1))	# 브로드캐스팅에 의존하지 않고 목표 분포를 동일 shape로 반복한다.
    model_probabilities = binary_probabilities.copy()
    ce = cross_entropy(targets, model_probabilities)	# 고정 목표 분포와 변화하는 모델 분포의 교차 엔트로피를 구한다.
    divergence = kl_divergence(targets, model_probabilities)	# 같은 입력으로 KL을 계산한다.
    target_entropy = entropy(target)
    finite = np.isfinite(ce) & np.isfinite(divergence)
    identity_error = float(np.max(np.abs(ce[finite] - target_entropy - divergence[finite])))	# 경계 무한대를 제외한 유한 표본에서 항등식 오차를 구한다.
    symmetry_error = float(np.max(np.abs(binary_entropy - binary_entropy[::-1])))	# Bernoulli 엔트로피의 p와 1-p 대칭성을 검사한다.
    assert binary_entropy.shape == ce.shape == divergence.shape == (501,)	# 클래스 축을 제거한 정보량 곡선 shape를 확인한다.
    assert identity_error <= 1e-12 and symmetry_error <= 1e-14
    assert np.all(divergence[finite] >= -1e-14)
    assert np.all(np.isposinf(ce[[0, -1]])) and np.all(np.isposinf(divergence[[0, -1]]))	# 양 끝 모델 확률에서 무한대 경계 정책이 유지되어야 한다.
    assert abs(float(np.max(binary_entropy)) - .6931471805599453) <= 1e-14	# 엔트로피 최대값을 독립 상수 ln(2)와 비교한다.
    assert abs(target_entropy - .6108643020548935) <= 1e-14
    minimum_index = int(np.argmin(divergence))
    assert abs(parameters[minimum_index] - target_parameter) <= 1e-15	# KL 최소 표본이 목표 확률과 일치하는지 검사한다.
    assert abs(divergence[minimum_index]) <= 1e-14
    report = {
        "seed": 42, "dtype": "float64", "numpy_version": np.__version__, "units": "nats",
        "normalization_atol": 1e-12, "normalization_rtol": 0., "examples": examples,
        "binary_entropy": {"parameter": parameters.tolist(), "probabilities": binary_probabilities.tolist(),
                           "entropy_nats": binary_entropy.tolist()},
        "fixed_target": {"parameter": target_parameter, "probabilities": target.tolist(), "entropy_nats": target_entropy},
        "varying_model": {"parameter": parameters.tolist(), "probabilities": model_probabilities.tolist(),
                          "cross_entropy_nats": [_json_number(value) for value in ce],
                          "kl_nats": [_json_number(value) for value in divergence]},
        "curve_checks": {"maximum_entropy_nats": float(np.max(binary_entropy)),
                         "maximum_entropy_parameter": float(parameters[np.argmax(binary_entropy)]),
                         "minimum_finite_kl_nats": float(divergence[minimum_index]),
                         "minimum_kl_parameter": float(parameters[minimum_index]),
                         "maximum_identity_error": identity_error, "identity_atol": 1e-12,
                         "maximum_entropy_symmetry_error": symmetry_error, "symmetry_atol": 1e-14,
                         "finite_model_count": int(np.count_nonzero(finite)), "infinite_boundary_count": 2,
                         "passed": True},
        "infinity_encoding": "The string '+inf' denotes mathematical positive infinity when p>0 and q=0; no clipping is performed.",
        "figure": {"filename": "information_theory.png", "width_px": 1792, "height_px": 686,
                   "nominal_dpi": 140, "format": "PNG"},
    }
    figure, _ = information_figure(parameters, binary_entropy, parameters, ce, divergence,
                                   target_parameter, target_entropy)
    try:
        figure.savefig(output_dir / "information_theory.png", dpi=140)	# 축·범례가 포함된 그림을 지정 출력 경로에 저장한다.
    finally:
        plt.close(figure)	# 사용이 끝난 그림 자원을 해제한다.
    (output_dir / "information_theory.json").write_text(	# 비유한 수는 지정한 문자열 정책으로만 JSON에 기록한다.
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    (output_dir / "information_theory.md").write_text(report_markdown(report), encoding="utf-8", newline="\n")	# 동일 실측 자료로 읽기용 Markdown을 저장한다.
    return report


def main(argv: list[str] | None = None) -> int:
    """선택한 쓰기 가능 출력 디렉터리에서 정보 이론 실험을 실행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    arguments = parser.parse_args(argv)	# 호출자가 제공한 명령행 인자를 해석한다.
    report = run_experiment(arguments.output_dir)	# 명령행에서 선택한 출력 경로로 실험을 실행한다.
    print(f"Information theory: {len(report['examples'])} independent examples passed; "
          f"CE=H+KL maximum error={report['curve_checks']['maximum_identity_error']:.6e} nats")
    print(f"Entropy maximum=ln(2), KL minimum at q=0.3; boundary +inf retained. Saved in {arguments.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())	# 명령행 실행 결과를 프로세스 종료 코드로 전달한다.
