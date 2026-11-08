"""기본 실험 다섯 개를 다시 실행하고 검증된 결과를 읽기 쉬운 보고서로 저장한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import tempfile

import matplotlib
import numpy as np

from scripts.run_calculus import run_experiment as run_calculus
from scripts.run_linear_algebra import run_experiment as run_linear_algebra
from scripts.run_notebooks import execute_notebook
from scripts.run_optimizer import run_experiment as run_optimizer
from scripts.report_linear_calculus import render_calculus, render_linear_algebra
from scripts.report_notebooks import render_backprop, render_probability
from scripts.report_optimizer import render_optimizer


ROOT = Path(__file__).resolve().parents[1]
BEGIN = "<!-- BEGIN VERIFIED MISSION RESULTS -->"
END = "<!-- END VERIFIED MISSION RESULTS -->"
REPORT_NAMES = (
    "linear_algebra_report.md", "calculus_report.md", "backprop_report.md",
    "optimizer_report.md", "probability_report.md",
)
FINITE_DIFFERENCE_PATHS = (
    "calculus.derivative.numerical", "calculus.derivative.absolute_error",
    "calculus.gradient.numerical", "calculus.gradient.maximum_absolute_error",
    "calculus.step_size_study.quadratic_derivative", "calculus.step_size_study.quadratic_error",
    "calculus.step_size_study.sine_derivative", "calculus.step_size_study.sine_error",
    "backprop.numerical_gradient", "backprop.max_numerical_gradient_error",
    "probability.bce.numerical_sum_gradient", "probability.bce.max_gradient_error",
)
__concepts__ = {
    "Readable_Experiment_Reports": ["run_reports", "main"],
    "Report_Reproduction_Check": ["check_reproduction", "_check_approximation_accuracy"],
    "Readme_Result_Integration": ["updated_readme"],
}


def check_reproduction(actual: object, reference: object, location: str = "metrics",
                       approximation_paths: tuple[str, ...] = ()) -> None:
    """지표 구조와 유한 값의 일치를 atol=rtol=1e-12로 요구한다.

    입력은 같은 구조의 JSON 호환 값이어야 한다. NumPy 버전 표기는 달라도 된다.
    명시적 approximation_paths에는 새 실험과 렌더러가 독립 해석값으로 이미
    검증한 유한차분 측정값만 들어간다. 이 값도 shape·유한성은 검사하지만
    플랫폼별 상쇄 차이를 Windows의 마지막 자릿수와 비교하지 않는다.
    다른 값은 atol=rtol=1e-12를 유지한다. 매우 작은 h 입력은 atol=0을 사용해
    1e-15를 50% 바꾼 오류가 일반 절대 허용치에 가려지지 않게 한다.
    입력·h·해석값은 검사를 생략할 수 없으며 빈 튜플은 전체에 엄격 비교를 적용한다.
    첫 불일치에서 ValueError를 내고 NaN/inf를 근거로 허용하지 않는다.
    """
    if isinstance(reference, dict):
        if not isinstance(actual, dict) or actual.keys() != reference.keys():	# 매핑의 필드 구성이 원래 결과와 정확히 같아야 한다.
            raise ValueError(f"{location}: metric keys differ")
        for key in reference:
            if key != "numpy_version":
                check_reproduction(actual[key], reference[key], f"{location}.{key}", approximation_paths)	# NumPy 버전 표시 외의 모든 필드를 재귀적으로 검사한다.
    elif isinstance(reference, list):
        if not isinstance(actual, list) or len(actual) != len(reference):	# 중첩 목록 길이를 비교하여 shape 변화를 검출한다.
            raise ValueError(f"{location}: metric shape differs")
        for index, (left, right) in enumerate(zip(actual, reference)):
            check_reproduction(left, right, f"{location}[{index}]", approximation_paths)
    elif isinstance(reference, bool) or reference is None or isinstance(reference, str):
        if type(actual) is not type(reference) or actual != reference:	# 불리언·None·문자열 메타데이터는 자료형까지 동일해야 한다.
            raise ValueError(f"{location}: metric metadata differs")
    elif isinstance(reference, (int, float)):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise ValueError(f"{location}: expected a numeric measurement")
        if not math.isfinite(actual) or not math.isfinite(reference):	# 비유한 수치가 두 결과에 함께 있어도 성공으로 인정하지 않는다.
            raise ValueError(f"{location}: nonfinite measurement")
        if any(location == path or location.startswith(path + "[") for path in approximation_paths):	# 독립 해석 검증을 마친 명시적 유한차분 값만 플랫폼 끝자리 비교에서 제외한다.
            return
        is_step = location in ("calculus.derivative.h", "backprop.derivative_h") or location.startswith("calculus.step_size_study.h[")	# 매우 작은 h 입력에는 절대 허용치를 적용하지 않는다.
        if not math.isclose(actual, reference, rel_tol=1e-12, abs_tol=0. if is_step else 1e-12):	# 나머지 수치는 원래 절대·상대 허용치로 비교한다.
            raise ValueError(f"{location}: reproduced value {actual} differs from {reference}")
    else:
        raise ValueError(f"{location}: unsupported metric type")


def _rebase_links(text: str, source_dir: Path, destination_dir: Path) -> str:
    """URL·표시 문구·앵커를 보존하면서 Markdown 파일·그림 링크의 기준 경로를 바꾼다.

    경로 인자는 기준 디렉터리이며 text는 완전한 Markdown 문서다.
    Windows의 다른 드라이브를 출력으로 명시한 경우에만 파일 URI를 사용한다.
    파일을 읽거나 쓰지 않으며 잘못된 Markdown의 검토는 호출자가 담당한다.
    """
    def replace(match: re.Match) -> str:
        """기존 Markdown 링크 하나의 목적지만 상대 경로로 바꾸어 반환한다."""
        href = match.group(2)
        if "://" in href or href.startswith("#"):	# 외부 URL과 같은 문서 앵커는 변경하지 않는다.
            return match.group(0)
        path, separator, anchor = href.partition("#")
        target = (source_dir / path).resolve()
        try:
            relative = Path(os.path.relpath(target, destination_dir)).as_posix()	# 기존 파일의 실제 위치를 새 문서 기준 상대 경로로 표현한다.
        except ValueError:
            relative = target.as_uri()	# 다른 Windows 드라이브는 상대 경로가 불가능하므로 URI를 사용한다.
        return match.group(1) + relative + (separator + anchor if separator else "") + ")"
    return re.sub(r"(!?\[[^\]]*\]\()([^\s)]+)\)", replace, text)	# 링크 문구와 앵커는 보존하고 목적지 경로만 치환한다.


def _check_approximation_accuracy(name: str, report: dict) -> None:
    """미션 허용치를 바꾸지 않고 고정 미분·gradient 근거를 다시 검사한다.

    미적분은 스칼라 미분값 6과 (9,2) gradient 2*points를 기대한다.
    역전파는 매개변수 미분 아홉 개, BCE는 logit 미분 두 개를 검사한다.
    저장 기준과 독립적인 assert로 잘못된 유한차분을 거부한다.
    탐색용 h 격자는 정확한 유한차분 식 6과 cos(3)*sin(h)/h 및 이 고정
    초등함수에 보수적인 반올림 예산 8*eps*scale/h로 검사한다.
    모든 h에서 1e-4를 만족할 필요는 없다. 렌더러도 유한 배열과
    재계산한 오차 요약을 확인한다.
    """
    if name == "calculus":
        assert abs(report["derivative"]["numerical"] - 6.) <= 1e-4	# 고정 미분 입력은 원래 미션 허용치 1e-4로 검사한다.
        np.testing.assert_allclose(report["gradient"]["numerical"],	# gradient를 독립 해석값 2*points와 비교한다.
                                   2 * np.asarray(report["gradient"]["points"]), atol=1e-8, rtol=0)
        study = report["step_size_study"]
        h = np.asarray(study["h"], dtype=float)
        assert h.ndim == 1 and np.all(np.isfinite(h)) and np.all(h > 0)	# 간격 배열의 shape·유한성·양수 조건을 확인한다.
        eps = np.finfo(float).eps
        for key, exact, scale in (
            ("quadratic_derivative", np.full(h.shape, 6.), np.maximum(3., (3.+h)**2)),
            ("sine_derivative", np.cos(3.) * np.sin(h)/h, np.full(h.shape, 3.)),
        ):
            values = np.asarray(study[key], dtype=float)
            assert values.shape == h.shape and np.all(np.isfinite(values))
            budget = 8 * eps * scale / h + 8 * eps * np.abs(exact)	# 이 고정 함수들의 유한차분 상쇄에 대한 반올림 예산을 구한다.
            assert np.all(np.abs(values-exact) <= budget), f"{key}: finite-difference identity violated"	# h 탐색 실패 표본을 숨기지 않으면서 차분 항등식 자체를 검증한다.
    elif name == "backprop":
        np.testing.assert_allclose(report["numerical_gradient"], report["analytical_gradient"],	# 아홉 매개변수의 수치 gradient를 원래 기준으로 검사한다.
                                   atol=1e-6, rtol=0)
    elif name == "probability":
        bce = report["bce"]
        np.testing.assert_allclose(bce["numerical_sum_gradient"],	# BCE의 두 수치 미분을 해석값과 비교한다.
                                   np.asarray(bce["probabilities"]) - np.asarray(bce["y"]),
                                   atol=1e-8, rtol=0)


def updated_readme(original: str, reports: dict[str, str], destination_dir: Path = ROOT) -> str:
    """검증 이력 앞의 검증 결과 블록만 삽입하거나 교체한다.

    reports는 다섯 REPORT_NAMES를 outputs 기준 링크의 Markdown에 대응시킨다.
    destination_dir는 상대 링크의 기준인 README 소속 디렉터리다.
    블록 밖의 기존 본문과 이전 PNG 링크는 모두 보존한다.
    보고서 누락, 중복·역순·짝 없는 구분자, 최초 삽입 앵커 부재는
    파일 쓰기 전에 ValueError로 거부한다.
    """
    if set(reports) != set(REPORT_NAMES):	# 정해진 다섯 보고서가 모두 준비되었는지 확인한다.
        raise ValueError("exactly five core reports are required")
    counts = original.count(BEGIN), original.count(END)
    if counts not in ((0, 0), (1, 1)):	# 결과 구분자는 없거나 정확히 한 쌍이어야 한다.
        raise ValueError("README result markers must be absent or form one pair")
    sections = []
    anchors = ("mission-linear", "mission-calculus", "mission-backprop", "mission-optimizer", "mission-probability")
    for filename, anchor_id in zip(REPORT_NAMES, anchors):
        body = _rebase_links(reports[filename], ROOT / "outputs", destination_dir)	# README 위치에 맞춰 원본 근거 링크를 보정한다.
        body = re.sub(r"^(#{1,4}) ", r"\1## ", body, flags=re.MULTILINE)	# 주변 README의 상위 절 구조가 유지되도록 제목 수준을 낮춘다.
        sections.append(f'<a id="{anchor_id}"></a>\n\n' + body.strip())
    block = BEGIN + "\n\n## 기본 미션 실제 실행 결과\n\n" + (
        "아래 표·수식·판정은 기존 실험을 새로 실행해 생성했습니다. "
        "기존 PNG와 링크를 보존하며 그림을 만드는 원본 수치를 함께 기록합니다. "
        "재생성: `python -m scripts.run_submission_reports --update-readme`.\n\n"
    ) + "\n\n".join(sections) + "\n\n" + END
    if counts == (1, 1):
        start, finish = original.index(BEGIN), original.index(END)
        if finish < start:
            raise ValueError("README result markers are reversed")
        return original[:start] + block + original[finish + len(END):]	# 검증 결과 블록 바깥의 기존 본문은 그대로 둔다.
    anchor = "## 검증 현황"
    if original.count(anchor) != 1:	# 최초 삽입 위치가 하나로 결정되지 않으면 쓰기를 거부한다.
        raise ValueError("README requires one validation-history insertion anchor")
    position = original.index(anchor)
    return original[:position] + block + "\n\n" + original[position:]


def _environment_header(source: Path, raw: Path) -> str:
    """이번 실행 환경과 소스·측정 파일의 해시를 설명한다.

    source는 원래 스크립트·노트북이고 raw는 새로 실행한 지표 JSON이다.
    SHA-256은 전체 파일 바이트를 대상으로 하며 두 파일의 읽기 오류를 전달한다.
    동일한 인터프리터와 의존성에서 머리말은 결정론적이다.
    """
    return (	# 실제 실행 환경과 파일 전체 바이트의 해시를 머리말로 만든다.
        f"실행 환경: Python {platform.python_version()}, NumPy {np.__version__}, "
        f"Matplotlib {matplotlib.__version__}; float64, 독립 실험 시작 시 seed42.\n\n"
        f"실행 원본: `{source.relative_to(ROOT).as_posix()}`; "
        f"source SHA-256 `{hashlib.sha256(source.read_bytes()).hexdigest()}`.\n\n"
        f"이번 실행 JSON SHA-256 `{hashlib.sha256(raw.read_bytes()).hexdigest()}`. "
        "기존 제출 JSON과 구조·입력·직접 계산값 재현 비교 통과(atol=rtol=1e-12). "
        "중심차분의 플랫폼별 반올림 차이는 저장값의 끝자리 일치 대신 기존 해석값·오차 기준으로 검증했습니다. "
        "각 실험의 수학 수용 기준은 변경하지 않았습니다. "
        "PNG와 노트북 원본은 수정하지 않았습니다.\n\n"
    )


def run_reports(output_dir: Path, readme_path: Path | None = None) -> dict[str, str]:
    """다섯 기본 실험을 새로 실행하고 검증된 보고서 다섯 개를 저장한다.

    PNG/JSON은 ROOT/.cache 아래 임시 디렉터리에서만 다시 생성하며
    기존 outputs와 노트북 원본 바이트는 덮어쓰지 않는다.
    독립 수학 검증 뒤 check_reproduction으로 ROOT/outputs와 수치를 비교한다.
    유한차분은 플랫폼별 마지막 자릿수 일치 대신 원래 해석 수용 기준을 적용한다.
    파일명→표준 Markdown 매핑을 반환하며 링크는 outputs 기준이다.
    선택적 readme_path는 모든 실험이 통과한 뒤 지정된 결과 블록만 갱신한다.
    잘못된 구분자·수치 변화·커널 실패·입출력 오류는 전달한다.
    실험·검증 단계가 실패하면 보고서와 README를 저장하지 않는다.
    """
    destination = Path(output_dir).resolve()
    readme = None if readme_path is None else Path(readme_path).resolve()
    original = None if readme is None else readme.read_text(encoding="utf-8")
    if original is not None:
        updated_readme(original, dict.fromkeys(REPORT_NAMES, "validation placeholder"), readme.parent)	# 비싼 재실행 전에 README 구분자 구조부터 검증한다.
    (ROOT / ".cache").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="submission-replay-", dir=ROOT / ".cache") as folder:	# 기존 산출물에 영향 없이 임시 경로에서 실험을 재실행한다.
        staging = Path(folder)
        metrics = {	# 세 스크립트 실험을 새로 실행해 원시 지표를 수집한다.
            "linear_algebra": run_linear_algebra(staging, use_cache=False),
            "calculus": run_calculus(staging),
            "optimizer": run_optimizer(staging, use_cache=False),
        }
        notebooks = {}
        for name, filename in (("backprop", "backprop_derivation.ipynb"),
                               ("probability", "probability_loss.ipynb")):
            notebooks[name] = execute_notebook(ROOT / "notebooks" / filename, staging)	# 노트북도 새 커널에서 실행하되 원본 파일은 저장하지 않는다.
            metrics[name] = json.loads((staging / f"{name}_metrics.json").read_text(encoding="utf-8"))
        for name, data in metrics.items():
            _check_approximation_accuracy(name, data)	# 재현값 비교 전에 독립 수학 기준을 다시 검사한다.
        rendered = (	# 검증된 원시 지표와 실제 노트북 출력으로 본문을 렌더링한다.
            render_linear_algebra(metrics["linear_algebra"]),
            render_calculus(metrics["calculus"]),
            render_backprop(metrics["backprop"], notebooks["backprop"]),
            render_optimizer(metrics["optimizer"]),
            render_probability(metrics["probability"], notebooks["probability"]),
        )
        for name, data in metrics.items():
            reference = json.loads((ROOT / "outputs" / f"{name}_metrics.json").read_text(encoding="utf-8"))
            check_reproduction(data, reference, name, FINITE_DIFFERENCE_PATHS)	# 기존 기준 지표와 이번 실행의 구조·값을 대조한다.
        sources = (
            "scripts/run_linear_algebra.py", "scripts/run_calculus.py",
            "notebooks/backprop_derivation.ipynb", "scripts/run_optimizer.py",
            "notebooks/probability_loss.ipynb",
        )
        keys = ("linear_algebra", "calculus", "backprop", "optimizer", "probability")
        reports = {}
        for filename, body, source, key in zip(REPORT_NAMES, rendered, sources, keys):
            title, separator, rest = body.partition("\n")
            reports[filename] = title + "\n\n" + _environment_header(
                ROOT / source, staging / f"{key}_metrics.json",
            ) + rest.lstrip("\n") + ("" if body.endswith("\n") else "\n")
        replacement = None if original is None else updated_readme(original, reports, readme.parent)	# 모든 실험 통과 후 README 교체 본문을 준비한다.
    destination.mkdir(parents=True, exist_ok=True)
    for filename, report in reports.items():
        text = _rebase_links(report, ROOT / "outputs", destination)	# 선택한 저장 위치에서 읽을 수 있도록 링크 기준을 맞춘다.
        (destination / filename).write_text(text, encoding="utf-8", newline="\n")	# 최종 검증을 마친 보고서를 UTF-8로 저장한다.
    if readme is not None:
        readme.write_text(replacement, encoding="utf-8", newline="\n")	# 명시적으로 요청한 README 결과 블록만 갱신한다.
    return reports


def main(argv: list[str] | None = None) -> int:
    """보고서를 생성하며 선택적으로 검증된 전체 내용을 README에 추가한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--update-readme", action="store_true")
    arguments = parser.parse_args(argv)	# 호출자가 제공한 명령행 인자를 해석한다.
    reports = run_reports(arguments.output_dir, ROOT / "README.md" if arguments.update_readme else None)	# 출력 경로와 README 갱신 요청을 실행기에 전달한다.
    print(f"Published {len(reports)} verified Markdown reports; original PNG/JSON/notebooks unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())	# 명령행 실행 결과를 프로세스 종료 코드로 전달한다.
