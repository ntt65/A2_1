"""Reexecute the five core experiments and publish readable, checked results."""

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
    """Require matching metric structure and finite values within atol=rtol=1e-12.

    Inputs are JSON-compatible values with identical structures. NumPy labels
    may differ. Explicit approximation_paths contain only finite-difference
    measurements already checked against independent analytic values by the
    fresh experiment and renderer. Their shape/finiteness is still required,
    but platform-dependent cancellation is not compared to Windows last bits.
    Every other value retains atol=rtol=1e-12; inputs, h and analytic values
    cannot be skipped. An empty tuple requests strict comparison everywhere.
    Raise ValueError at the first mismatch, never accept NaN/inf as evidence.
    """
    if isinstance(reference, dict):
        if not isinstance(actual, dict) or actual.keys() != reference.keys():
            raise ValueError(f"{location}: metric keys differ")
        for key in reference:
            if key != "numpy_version":
                check_reproduction(actual[key], reference[key], f"{location}.{key}", approximation_paths)
    elif isinstance(reference, list):
        if not isinstance(actual, list) or len(actual) != len(reference):
            raise ValueError(f"{location}: metric shape differs")
        for index, (left, right) in enumerate(zip(actual, reference)):
            check_reproduction(left, right, f"{location}[{index}]", approximation_paths)
    elif isinstance(reference, bool) or reference is None or isinstance(reference, str):
        if type(actual) is not type(reference) or actual != reference:
            raise ValueError(f"{location}: metric metadata differs")
    elif isinstance(reference, (int, float)):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise ValueError(f"{location}: expected a numeric measurement")
        if not math.isfinite(actual) or not math.isfinite(reference):
            raise ValueError(f"{location}: nonfinite measurement")
        if any(location == path or location.startswith(path + "[") for path in approximation_paths):
            return
        if not math.isclose(actual, reference, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{location}: reproduced value {actual} differs from {reference}")
    else:
        raise ValueError(f"{location}: unsupported metric type")


def _rebase_links(text: str, source_dir: Path, destination_dir: Path) -> str:
    """Rebase Markdown file/image links while preserving URLs, labels and anchors.

    Paths are directory bases, text is a complete Markdown document. A file
    URI is used only for an explicitly chosen output on another Windows drive.
    No file is read or written; malformed Markdown is left to caller review.
    """
    def replace(match: re.Match) -> str:
        """Return one existing Markdown link with a relative destination path."""
        href = match.group(2)
        if "://" in href or href.startswith("#"):
            return match.group(0)
        path, separator, anchor = href.partition("#")
        target = (source_dir / path).resolve()
        try:
            relative = Path(os.path.relpath(target, destination_dir)).as_posix()
        except ValueError:
            relative = target.as_uri()
        return match.group(1) + relative + (separator + anchor if separator else "") + ")"
    return re.sub(r"(!?\[[^\]]*\]\()([^\s)]+)\)", replace, text)


def _check_approximation_accuracy(name: str, report: dict) -> None:
    """Recheck fixed derivative/gradient evidence with unchanged mission tolerances.

    Calculus expects scalar derivative 6 and (9,2) gradient 2*points;
    backprop has nine scalar parameter derivatives, BCE two logit derivatives.
    Assertions reject bad finite differences independently of baseline storage.
    The exploratory h sweep is checked against exact finite differences (6
    and cos(3)*sin(h)/h) with a conservative 8*eps*scale/h rounding budget for
    these fixed elementary functions. It need not meet 1e-4 at every h. The
    renderer also verifies finite arrays and recomputed error summaries.
    """
    if name == "calculus":
        assert abs(report["derivative"]["numerical"] - 6.) <= 1e-4
        np.testing.assert_allclose(report["gradient"]["numerical"],
                                   2 * np.asarray(report["gradient"]["points"]), atol=1e-8, rtol=0)
        study = report["step_size_study"]
        h = np.asarray(study["h"], dtype=float)
        assert h.ndim == 1 and np.all(np.isfinite(h)) and np.all(h > 0)
        eps = np.finfo(float).eps
        for key, exact, scale in (
            ("quadratic_derivative", np.full(h.shape, 6.), np.maximum(3., (3.+h)**2)),
            ("sine_derivative", np.cos(3.) * np.sin(h)/h, np.full(h.shape, 3.)),
        ):
            values = np.asarray(study[key], dtype=float)
            assert values.shape == h.shape and np.all(np.isfinite(values))
            budget = 8 * eps * scale / h + 8 * eps * np.abs(exact)
            assert np.all(np.abs(values-exact) <= budget), f"{key}: finite-difference identity violated"
    elif name == "backprop":
        np.testing.assert_allclose(report["numerical_gradient"], report["analytical_gradient"],
                                   atol=1e-6, rtol=0)
    elif name == "probability":
        bce = report["bce"]
        np.testing.assert_allclose(bce["numerical_sum_gradient"],
                                   np.asarray(bce["probabilities"]) - np.asarray(bce["y"]),
                                   atol=1e-8, rtol=0)


def updated_readme(original: str, reports: dict[str, str], destination_dir: Path = ROOT) -> str:
    """Insert/replace only the verified-results block before validation history.

    reports maps the five REPORT_NAMES to canonical outputs-relative Markdown.
    destination_dir is the README's containing directory for relative links.
    Existing text outside the block and all earlier PNG links are preserved.
    Reject missing reports, duplicate/reversed/unpaired markers or an absent
    initial insertion anchor with ValueError before any file write occurs.
    """
    if set(reports) != set(REPORT_NAMES):
        raise ValueError("exactly five core reports are required")
    counts = original.count(BEGIN), original.count(END)
    if counts not in ((0, 0), (1, 1)):
        raise ValueError("README result markers must be absent or form one pair")
    sections = []
    anchors = ("mission-linear", "mission-calculus", "mission-backprop", "mission-optimizer", "mission-probability")
    for filename, anchor_id in zip(REPORT_NAMES, anchors):
        body = _rebase_links(reports[filename], ROOT / "outputs", destination_dir)
        # Keep the surrounding README's main section hierarchy intact.
        body = re.sub(r"^(#{1,4}) ", r"\1## ", body, flags=re.MULTILINE)
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
        return original[:start] + block + original[finish + len(END):]
    anchor = "## 검증 현황"
    if original.count(anchor) != 1:
        raise ValueError("README requires one validation-history insertion anchor")
    position = original.index(anchor)
    return original[:position] + block + "\n\n" + original[position:]


def _environment_header(source: Path, raw: Path) -> str:
    """Describe this execution environment and its source/measurement hashes.

    source is the original script/notebook, raw is the newly executed metric
    JSON. SHA-256 covers their complete bytes; errors reading either propagate.
    The header is deterministic within the same interpreter and dependencies.
    """
    return (
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
    """Freshly execute all five core experiments and write five checked reports.

    PNG/JSON are regenerated only in a temporary directory under ROOT/.cache;
    existing outputs and notebook source bytes are never overwritten. Numeric
    report data are compared with ROOT/outputs using check_reproduction after
    independent math checks; finite differences use their original analytic
    acceptance criteria rather than platform-specific last-bit equality.
    Returns filename->canonical Markdown (outputs-relative links). Optional
    readme_path updates the bounded result block after every experiment passes.
    Invalid markers, drift, kernel failures and I/O errors propagate; no report
    or README is published if an experiment/validation step fails.
    """
    destination = Path(output_dir).resolve()
    readme = None if readme_path is None else Path(readme_path).resolve()
    original = None if readme is None else readme.read_text(encoding="utf-8")
    if original is not None:
        updated_readme(original, dict.fromkeys(REPORT_NAMES, "validation placeholder"), readme.parent)
    (ROOT / ".cache").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="submission-replay-", dir=ROOT / ".cache") as folder:
        staging = Path(folder)
        metrics = {
            "linear_algebra": run_linear_algebra(staging, use_cache=False),
            "calculus": run_calculus(staging),
            "optimizer": run_optimizer(staging, use_cache=False),
        }
        notebooks = {}
        for name, filename in (("backprop", "backprop_derivation.ipynb"),
                               ("probability", "probability_loss.ipynb")):
            notebooks[name] = execute_notebook(ROOT / "notebooks" / filename, staging)
            metrics[name] = json.loads((staging / f"{name}_metrics.json").read_text(encoding="utf-8"))
        for name, data in metrics.items():
            _check_approximation_accuracy(name, data)
        rendered = (
            render_linear_algebra(metrics["linear_algebra"]),
            render_calculus(metrics["calculus"]),
            render_backprop(metrics["backprop"], notebooks["backprop"]),
            render_optimizer(metrics["optimizer"]),
            render_probability(metrics["probability"], notebooks["probability"]),
        )
        for name, data in metrics.items():
            reference = json.loads((ROOT / "outputs" / f"{name}_metrics.json").read_text(encoding="utf-8"))
            check_reproduction(data, reference, name, FINITE_DIFFERENCE_PATHS)
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
        replacement = None if original is None else updated_readme(original, reports, readme.parent)
    destination.mkdir(parents=True, exist_ok=True)
    for filename, report in reports.items():
        text = _rebase_links(report, ROOT / "outputs", destination)
        (destination / filename).write_text(text, encoding="utf-8", newline="\n")
    if readme is not None:
        readme.write_text(replacement, encoding="utf-8", newline="\n")
    return reports


def main(argv: list[str] | None = None) -> int:
    """Generate reports, optionally adding their full verified content to README."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--update-readme", action="store_true")
    arguments = parser.parse_args(argv)
    reports = run_reports(arguments.output_dir, ROOT / "README.md" if arguments.update_readme else None)
    print(f"Published {len(reports)} verified Markdown reports; original PNG/JSON/notebooks unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
