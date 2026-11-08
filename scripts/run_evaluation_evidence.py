"""기존 그림의 메타데이터와 재현 가능한 수치 진단을 보고한다.

이 보충 실험은 원래 그림이나 지표를 덮어쓰지 않는다.
실제 측정값과 진단을 위해 의도적으로 주입한 실패를 구분한다.
"""

import argparse
import json
from pathlib import Path
import struct
import sys
import zlib

import matplotlib
matplotlib.use("Agg")	# 파일 저장용 비대화형 백엔드를 pyplot보다 먼저 선택한다.
import matplotlib.pyplot as plt
import numpy as np

from scripts.run_linear_algebra import generate_grayscale_image
from src.calculus import derivative_error_study
from src.linear_algebra import determinant_area_check, svd_comparison_figure
from src.utils.cache_manager import CacheManager
from src.utils.math import sigmoid, softmax
from src.utils.random import set_seed

__concepts__ = {
    "Evaluation_Evidence": ["run_experiment", "main"],
    "Figure_Metadata": ["png_metadata", "svd_panel_metadata"],
    "Numerical_Diagnostics": ["difference_report", "step_size_summary", "softmax_summary"],
}

FIGURES = (
    "linear_transforms.png", "svd_64.png", "svd_128.png",
    "gradient_geometry.png", "derivative_error.png",
    "optimizer_circle.png", "optimizer_ellipse.png", "optimizer_learning_rates.png",
    "probability_normal.png", "probability_bernoulli.png",
)


def png_metadata(path: Path) -> dict:
    """픽셀을 디코딩하지 않고 IHDR/pHYs에서 PNG 캔버스 크기와 물리적 DPI를 읽는다.

    스칼라 width/height, 형식, 원소 두 개인 DPI 목록을 반환하며 DPI가 없으면
    None을 사용한다. 청크 경계와 CRC를 검사한다. 잘못된 메타데이터는 ValueError,
    입출력 오류는 원래 예외로 전달한다. 여기서는 개별 패널의 기하를 추정하지 않는다.
    """
    data = Path(path).read_bytes()	# 원본 PNG를 읽기만 하며 픽셀이나 파일 내용을 변경하지 않는다.
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):	# PNG 서명부터 확인하여 다른 형식을 거부한다.
        raise ValueError("not a PNG file")
    offset, dimensions, dpi, ended = 8, None, None, False
    while offset + 12 <= len(data):
        length = struct.unpack(">I", data[offset:offset + 4])[0]	# 빅엔디언 청크 길이를 읽는다.
        kind = data[offset + 4:offset + 8]
        end = offset + 8 + length
        if end + 4 > len(data):	# 청크 내용과 CRC가 파일 경계 안에 있는지 확인한다.
            raise ValueError("truncated PNG chunk")
        payload = data[offset + 8:end]
        crc = struct.unpack(">I", data[end:end + 4])[0]
        if zlib.crc32(kind + payload) & 0xffffffff != crc:	# 청크 종류와 데이터의 CRC로 손상 여부를 검사한다.
            raise ValueError("PNG chunk CRC mismatch")
        if offset == 8 and kind != b"IHDR":
            raise ValueError("PNG must start with IHDR")
        if kind == b"IHDR":
            if dimensions is not None or length != 13:
                raise ValueError("invalid PNG IHDR")
            width, height = struct.unpack(">II", payload[:8])	# 전체 이미지의 가로·세로 픽셀 수를 읽는다.
            if width == 0 or height == 0:
                raise ValueError("PNG dimensions must be positive")
            dimensions = (width, height)
        elif kind == b"pHYs":
            if length != 9:
                raise ValueError("invalid PNG pHYs")
            x, y, unit = struct.unpack(">IIB", payload)
            if unit not in (0, 1):
                raise ValueError("invalid PNG physical unit")
            if unit == 1:
                dpi = [x * 0.0254, y * 0.0254]	# 미터당 픽셀 수를 인치당 픽셀 수로 환산한다.
        offset = end + 4
        if kind == b"IEND":
            if length != 0 or offset != len(data):
                raise ValueError("invalid PNG end")
            ended = True
            break
    if dimensions is None or not ended:	# 크기 정보와 정상 종료 청크가 모두 있어야 한다.
        raise ValueError("incomplete PNG metadata")
    return {"filename": Path(path).name, "format": "PNG", "width_px": dimensions[0],
            "height_px": dimensions[1], "dpi_xy": dpi,
            "measurement": "stored PNG canvas; not the input array or one panel"}


def svd_panel_metadata(cache: CacheManager | None = None) -> list[dict]:
    """seed를 고정한 두 SVD 그림을 140 DPI로 렌더링하여 이미지 Axes 영역을 측정한다.

    호출자는 실행 전에 seed 42를 한 번 설정해야 한다. 각 입력은 (size,size)이고
    원래 실험의 생성 순서인 64 다음 128을 따른다. 반환하는 소수 픽셀 bbox는
    제목을 제외하고 재생성한 이미지 패널의 영역이며 저장 PNG에서 복원한 정수
    자르기 영역이 아니다. 이미지를 파일로 저장하지 않는다.
    """
    records = []
    for size, ranks in ((64, [10, 50, 64]), (128, [10, 50, 100])):
        image = generate_grayscale_image(size)	# 원래 실험과 같은 생성 순서로 SVD 입력을 재생성한다.
        figure, axes = svd_comparison_figure(image, ranks, cache)
        try:
            figure.set_dpi(140)
            figure.canvas.draw()	# 실제 렌더링을 끝내 패널의 픽셀 좌표를 확정한다.
            renderer = figure.canvas.get_renderer()
            panels = []
            for label, axis in zip(["original"] + [f"rank {k}" for k in ranks], axes):
                bounds = axis.get_window_extent(renderer).bounds	# 전체 캔버스와 구분되는 이미지 Axes의 영역을 읽는다.
                panels.append({"label": label, "input_shape": [size, size],
                               "bbox_xywh_px": [float(value) for value in bounds]})
            records.append({"filename": f"svd_{size}.png", "input_shape": [size, size],
                            "role": "main" if size == 64 else "rank-100 supplement",
                            "figure_inches": figure.get_size_inches().tolist(),
                            "render_dpi": 140, "panels": panels,
                            "measurement": "recreated current-renderer image Axes; titles excluded"})
        finally:
            plt.close(figure)	# 사용이 끝난 그림 자원을 해제한다.
    return records


def difference_report(expected, observed, atol: float) -> dict:
    """같은 shape의 실수 배열에서 관측값−기대값의 부호 있는 오차를 반환한다.

    스칼라는 shape ()다. 입력은 비어 있지 않은 유한 실수 배열이어야 하며
    atol은 유한한 음이 아닌 값이어야 한다. broadcasting을 허용하지 않는다.
    자료형·shape·값 위반은 TypeError/ValueError를 발생시킨다.
    반환하는 passed 판정에는 명시적으로 rtol=0을 적용한다.
    """
    arrays = [np.asarray(value) for value in (expected, observed)]
    if any(value.dtype.kind not in "iuf" for value in arrays):
        raise TypeError("expected and observed must be real numeric values")
    if arrays[0].shape != arrays[1].shape or not arrays[0].size:	# broadcasting으로 다른 shape가 우연히 비교되지 않게 한다.
        raise ValueError("expected and observed must have the same nonempty shape")
    if (isinstance(atol, (bool, np.bool_)) or not np.isscalar(atol)
            or np.asarray(atol).dtype.kind not in "iuf"):
        raise TypeError("atol must be a real scalar")
    try:
        tolerance = float(atol)
    except (ValueError, TypeError) as error:
        raise TypeError("atol must be a real scalar") from error
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("atol must be finite and nonnegative")
    arrays = [value.astype(np.float64) for value in arrays]	# 공통 비교 dtype인 float64로 변환한다.
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("expected and observed must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        difference = arrays[1] - arrays[0]	# 관측값에서 독립 기대값을 빼 오차 부호를 보존한다.
    if not np.all(np.isfinite(difference)):	# 유한 입력끼리의 뺄셈에서도 범위 초과를 거부한다.
        raise ValueError("difference exceeds float64 range")
    return {"expected": arrays[0].tolist(), "observed": arrays[1].tolist(),
            "shape": list(arrays[0].shape), "signed_difference": difference.tolist(),
            "max_absolute_error": float(np.max(np.abs(difference))),
            "atol": tolerance, "rtol": 0.0,
            "passed": bool(np.all(np.abs(difference) <= tolerance))}


def step_size_summary(steps, errors, tolerance: float = 1e-4) -> dict:
    """표본 h/오차 격자 (n,)를 요약하되 연속 구간의 유효성을 단정하지 않는다.

    h는 유한한 양수로 엄격히 증가해야 하며 오차와 허용치는 유한하고 음이 아니어야 한다.
    통과한 모든 h, 연속된 표본 묶음, 최소오차를 갖는 모든 동률 표본을 반환한다.
    배열 shape·수치 조건 위반은 ValueError를 발생시킨다. 수치로 처리할 수 없는
    자료형은 NumPy의 변환·검사 과정에서 TypeError 또는 ValueError를 발생시킨다.
    """
    steps = np.asarray(steps, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    if (steps.ndim != 1 or errors.shape != steps.shape or steps.size == 0
            or not np.all(np.isfinite(steps)) or not np.all(np.isfinite(errors))
            or np.any(steps <= 0) or np.any(np.diff(steps) <= 0) or np.any(errors < 0)
            or not np.isscalar(tolerance) or not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError("require increasing positive h, nonnegative errors/tolerance")
    indices = np.flatnonzero(errors <= tolerance)	# 기준을 만족한 표본 인덱스만 선택한다.
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1) if indices.size else []	# 중간 실패 표본을 가로지르지 않도록 통과 표본을 묶는다.
    minimum = float(np.min(errors))
    return {"sample_count": int(steps.size), "tolerance": float(tolerance),
            "passing_count": int(indices.size), "passing_h": steps[indices].tolist(),
            "contiguous_sample_groups": [
                {"first_h": float(steps[g[0]]), "last_h": float(steps[g[-1]]), "sample_count": int(g.size)}
                for g in groups],
            "minimum_observed_error": minimum, "minimizing_sampled_h": steps[errors == minimum].tolist(),
            "scope": "only tested h values; gaps and intervening real h are not certified"}


def softmax_summary() -> list[dict]:
    """고정 float64 logit에 축 -1, 0, 1을 적용해 확률합 오차 행들을 반환한다.

    큰 이동량을 포함한 벡터 (3,), 행렬 (2,3), 텐서 (2,3,4)를 검사한다.
    확률은 모두 유한하고 음이 아니어야 하며 합에는 허용치 1e-6을 사용한다.
    기존의 shape 보존 구현이 회귀하면 assert 검증이 실패한다.
    """
    cases = [(np.array([1000.0, 1001.0, 1002.0]), -1),
             (np.array([[0., 1., 2.], [3., 4., 5.]]), 0),
             (np.array([[0., 1., 2.], [3., 4., 5.]]), 1),
             (np.arange(24, dtype=np.float64).reshape(2, 3, 4) - 1000, -1),
             (np.arange(24, dtype=np.float64).reshape(2, 3, 4) - 1000, 0),
             (np.arange(24, dtype=np.float64).reshape(2, 3, 4) - 1000, 1)]
    rows = []
    for logits, axis in cases:
        probabilities = softmax(logits, axis=axis)	# 각 사례의 지정 클래스 축으로 기존 Softmax를 호출한다.
        error = np.abs(np.sum(probabilities, axis=axis) - 1.0)	# 같은 축에서 확률합과 1의 차이를 측정한다.
        assert probabilities.shape == logits.shape	# 정규화 전후의 전체 shape가 같아야 한다.
        assert np.all(np.isfinite(probabilities)) and np.all(probabilities >= 0)
        assert np.max(error) <= 1e-6	# 확률합 최대오차를 수용 기준으로 검사한다.
        rows.append({"shape": list(logits.shape), "axis": axis, "input": logits.tolist(),
                     "sum_shape": list(error.shape), "sum_error_max": float(np.max(error)),
                     "sum_error_mean": float(np.mean(error)), "tolerance": 1e-6, "passed": True})
    return rows


def failure_diagnostics() -> dict:
    """실제 고정 예제를 계산한 뒤 잘못된 관측값을 명시적으로 주입한다.

    면적 사례는 정사각형 좌표 (2,4)와 diag(2,.5)를 사용한다. 역전파는 미션의
    W1(2,2), x(2,), W2(2,)와 독립적으로 고정한 gradient 기준 상수를 사용한다.
    교란 기록은 의도적인 진단 예제이며 제출 알고리즘에서 측정된 실패가 아니다.
    assert는 주입한 오답이 반드시 거부되는지 확인한다.
    """
    matrix = np.diag([2.0, 0.5])
    points = np.array([[0., 1., 1., 0.], [0., 0., 1., 1.]])
    area = determinant_area_check(matrix, points)	# 고정 면적 계산의 실제 성공 여부부터 확인한다.
    assert area["passed"] and abs(area["expected_ratio"] - 1.0) <= 1e-15
    injected_ratio = area["area_ratio"] + 0.02	# 진단용 오답임을 명시하여 면적비를 의도적으로 교란한다.
    area_failure = difference_report(area["expected_ratio"], injected_ratio, 0.01)
    assert not area_failure["passed"]	# 주입한 2% 면적 오차가 1% 기준에서 거부되어야 한다.
    x = np.array([1., 0.])
    w1 = np.array([[.1, .2], [.3, .4]])
    w2 = np.array([.5, .6])
    activation = sigmoid(w1 @ x)
    predicted = float(sigmoid(w2 @ activation))
    gradient = np.outer((predicted - 1) * w2 * activation * (1 - activation), x)	# 고정 신경망의 첫 층 gradient를 연쇄 법칙과 외적으로 구한다.
    reference = np.array([[-0.043978515808697015, 0.], [-0.05173350439092561, 0.]])	# 실측 결과를 재사용하지 않은 독립 gradient 상수다.
    genuine = difference_report(reference, gradient, 5e-5)
    perturbed = gradient.copy()
    perturbed[0, 0] += 0.001	# 진단용으로 한 gradient 성분에만 오차를 주입한다.
    failure = difference_report(reference, perturbed, 5e-5)
    assert genuine["passed"] and not failure["passed"]	# 원래 계산은 통과하고 주입된 오답은 거부되는지 확인한다.
    return {
        "area": {"matrix": matrix.tolist(), "points": points.tolist(), "genuine": area,
                 "injected_observation": area_failure,
                 "injection": "add 0.02 to area ratio; expected ratio=1 so abs error equals relative error",
                 "diagnostic": "REJECTED: measured ratio differs by 2%, exceeding 1% tolerance"},
        "backprop": {"x": x.tolist(), "W1": w1.tolist(), "b1": [0., 0.],
                     "W2": w2.tolist(), "b2": 0., "y_true": 1.,
                     "gradient_name": "dL/dW1", "genuine": genuine,
                     "injected_observation": failure, "injection": "add 0.001 to dL/dW1[0,0]",
                     "diagnostic": "REJECTED: absolute error exceeds 5e-5; signed difference identifies component"},
        "scope": "injected incorrect observations demonstrate diagnostics; genuine computations pass",
    }


def run_experiment(input_dir: Path, output_dir: Path, use_cache: bool = True) -> dict:
    """기존 PNG와 새로 계산한 검증으로 보충 JSON/Markdown을 저장한다.

    input_dir에는 원래 그림 열 개가 있어야 한다. output_dir에는 새로운
    evaluation_evidence 파일 두 개만 저장한다. 여기서 seed를 한 번 설정한다.
    누락 파일, 잘못된 메타데이터, 수치 검증 실패와 입출력 오류를 전달한다.
    기존 산출물은 읽기 전용으로 사용한다.
    """
    set_seed(42)	# 독립 실험의 시작점에서 난수 시드를 한 번 고정한다.
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    metadata = [png_metadata(input_dir / filename) for filename in FIGURES]	# 지정된 원본 그림 열 개의 형식·크기·DPI를 수집한다.
    cache = CacheManager(".cache/linear_algebra", enabled=use_cache)
    panels = svd_panel_metadata(cache)
    sweep = derivative_error_study()
    report = {
        "seed": 42, "dtype": "float64", "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__, "figure_metadata": metadata,
        "svd_panel_metadata": panels,
        "step_size": {"function": "x**2", "x": 3.0, "analytic_derivative": 6.0,
                      "h": sweep["h"].tolist(), "absolute_errors": sweep["quadratic_error"].tolist(),
                      **step_size_summary(sweep["h"], sweep["quadratic_error"]),
                      "interpretation": "central differences are exact for x**2 in exact arithmetic; sampled roundoff minima are not universal optimal h"},
        "softmax": softmax_summary(), "diagnostics": failure_diagnostics(),
        "notes": ["64x64 limits the main SVD input, not the entire Figure or rendered panels.",
                  "128x128 is the documented separate supplement required for rank 100.",
                  "Existing plots use nominal 140 DPI; preserve readable axes rather than shrinking every plot to 64px.",
                  "PNG pHYs stores integer pixels/metre, so recovered DPI can differ slightly from nominal 140.",
                  "Rendered panel geometry depends on renderer/fonts; it excludes titles and is not embedded PNG metadata."]}
    output_dir.mkdir(parents=True, exist_ok=True)	# 산출물을 저장할 디렉터리를 준비한다.
    (output_dir / "evaluation_evidence.json").write_text(	# 원시 수치와 명시적인 진단 조건을 새 JSON으로 기록한다.
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    lines = ["# Supplementary evaluation evidence", "",
             "Reproduce: `python -m scripts.run_evaluation_evidence --no-cache`.", "",
             "## Stored PNG metadata", "",
             "The 64x64 guide applies to the main SVD input, not entire plotted Figures.", "",
             "| PNG | Canvas pixels | Actual DPI (x/y) |", "|---|---:|---:|"]
    for row in metadata:
        dpi = row["dpi_xy"]
        lines.append(f"| {row['filename']} | {row['width_px']} x {row['height_px']} | {dpi} |")
    lines += ["", "## SVD input and rendered panels", "",
              "Image Axes dimensions measured after recreating the original seeded experiment at 140 DPI.",
              "These fractional pixel extents exclude titles and are not PNG crop metadata.", "",
              "| File | Input shape | Image panel | Width x height (pixels) |", "|---|---|---|---:|"]
    for row in panels:
        for panel in row["panels"]:
            _, _, width, height = panel["bbox_xywh_px"]
            lines.append(f"| {row['filename']} | {row['input_shape']} | {panel['label']} | {width:.3f} x {height:.3f} |")
    h_summary = report["step_size"]
    lines += ["", "## Sampled derivative h", "",
              f"x**2 at x=3, float64; {h_summary['passing_count']}/{h_summary['sample_count']} sampled h satisfy error <= 1e-4.",
              f"Passing consecutive sample groups: `{h_summary['contiguous_sample_groups']}`.",
              f"Minimum observed error: {h_summary['minimum_observed_error']:.6e}; sampled minimizers: `{h_summary['minimizing_sampled_h']}`.",
              "No continuous interval or universal optimal h is claimed. Quadratic central differences have zero truncation error in exact arithmetic.", "",
              "## Softmax axes", "", "| Input shape | Axis | Maximum sum error | Mean sum error |", "|---|---:|---:|---:|"]
    for row in report["softmax"]:
        lines.append(f"| {row['shape']} | {row['axis']} | {row['sum_error_max']:.6e} | {row['sum_error_mean']:.6e} |")
    lines += ["", "All six cases pass the 1e-6 tolerance; complete inputs are in the JSON.", "",
              "## Intentional failure diagnostics", "",
              "Both genuine fixed calculations pass. These are explicitly injected wrong observations:", "",
              "- Area: diag(2,0.5), unit square; adding 0.02 to ratio 1 yields 2% error > 1%: REJECTED.",
              "- Backprop: mission input, add 0.001 to dL/dW1[0,0]; max error > 5e-5: REJECTED.",
              f"- Signed gradient difference (observed minus independent reference): `{report['diagnostics']['backprop']['injected_observation']['signed_difference']}`.", "",
              "The JSON records full reproduction inputs, expected/observed values, shapes, differences and tolerances.", ""]
    (output_dir / "evaluation_evidence.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")	# 기존 보고서 문자열을 결합해 보충 Markdown만 저장한다.
    return report


def main(argv: list[str] | None = None) -> int:
    """메타데이터·진단 명령을 실행하며 선택적 입출력 디렉터리는 원본을 바꾸지 않는다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--no-cache", action="store_true")
    arguments = parser.parse_args(argv)	# 호출자가 제공한 명령행 인자를 해석한다.
    report = run_experiment(arguments.input_dir, arguments.output_dir, not arguments.no_cache)	# 입력·출력 경로와 캐시 옵션을 진단 실험에 전달한다.
    print(f"PNG metadata: {len(report['figure_metadata'])}; softmax axis cases: {len(report['softmax'])}; injected failures rejected: 2")
    print(f"Derivative h samples passing 1e-4: {report['step_size']['passing_count']}/61; evidence saved in {arguments.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())	# 명령행 실행 결과를 프로세스 종료 코드로 전달한다.
