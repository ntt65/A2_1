"""Render verified notebook measurements as self-contained Markdown reports.

These pure functions receive an executed notebook and its JSON measurements;
they do not run kernels, write files, or alter NumPy's random state.
"""

import numpy as np

__concepts__ = {
    "Notebook_Markdown_Reports": ["render_backprop", "render_probability"],
}

_BACKPROP_REFERENCE = {
    "z1": [0.1, 0.3],
    "a1": [0.5249791874789399861, 0.5744425168116589872],
    "z2": 0.60715510382646538534,
    "y_pred": 0.64729156999461775097,
    "L": 0.43495843685146062506,
    "dL_dy_pred": -1.5448988467566710784,
    "dL_dz2": -0.35270843000538224903,
    "dL_dW2": [-0.18516458500119814929, -0.20261071823298063973],
    "dL_da1": [-0.17635421500269112452, -0.21162505800322934942],
    "dL_dz1": [-0.04397851580869701354, -0.05173350439092561391],
    "dL_dW1": [[-0.04397851580869701354, 0.0], [-0.05173350439092561391, 0.0]],
    "dL_db1": [-0.04397851580869701354, -0.05173350439092561391],
    "dL_db2": -0.35270843000538224903,
}
_PARAMETER_ORDER = ["W1[0,0]", "W1[0,1]", "W1[1,0]", "W1[1,1]",
                    "b1[0]", "b1[1]", "W2[0]", "W2[1]", "b2"]


def _array(value, shape: tuple, name: str) -> np.ndarray:
    """Validate a finite real value of exact shape and return an owned float64 array.

    Scalars use shape (). Invalid types raise TypeError; shape/nonfinite
    values raise AssertionError so inconsistent evidence cannot claim PASS.
    """
    values = np.asarray(value)
    if values.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values")
    assert values.shape == shape, f"{name}: expected shape {shape}, got {values.shape}"
    values = values.astype(np.float64, copy=True)
    assert np.all(np.isfinite(values)), f"{name}: nonfinite measurement"
    return values


def _close(actual, expected, name: str, atol: float = 1e-14, rtol: float = 0) -> np.ndarray:
    """Assert finite actual values match independent expected values without broadcasting."""
    reference = np.asarray(expected, dtype=np.float64)
    values = _array(actual, reference.shape, name)
    np.testing.assert_allclose(values, reference, atol=atol, rtol=rtol, err_msg=name)
    return values


def _cell(notebook: dict, cell_id: str, cell_type: str) -> dict:
    """Find one stable notebook cell ID of the required type, raising precise ValueError."""
    matches = [cell for cell in notebook["cells"] if cell.get("id") == cell_id]
    if not matches:
        raise ValueError(f"missing required {cell_type} cell: {cell_id}")
    if len(matches) != 1:
        raise ValueError(f"duplicate notebook cell ID: {cell_id}")
    if matches[0].get("cell_type") != cell_type:
        raise ValueError(f"cell {cell_id} must have type {cell_type}")
    return matches[0]


def _markdown(notebook: dict, cell_id: str) -> str:
    """Return an original Markdown derivation cell unchanged apart from outer whitespace."""
    source = _cell(notebook, cell_id, "markdown")["source"]
    source = source if isinstance(source, str) else "".join(source)
    if not source.strip():
        raise ValueError(f"empty required markdown cell: {cell_id}")
    return source.strip()


def _execution(report: dict, notebook: dict, code_count: int, output_ids: tuple[str, ...]) -> None:
    """Require declared float64 seed-42 data and sequential, error-free executed code cells.

    Required evidence cells must contain nonempty text stream outputs. This
    checks submitted execution structure, not a new execution or cryptographic
    provenance. Missing/malformed cells raise ValueError; false claims assert.
    """
    assert report["seed"] == 42 and report["dtype"] == "float64"
    assert isinstance(report["numpy_version"], str) and report["numpy_version"]
    assert report["all_assertions_passed"] is True, "notebook assertions did not pass"
    evidence_cells = [_cell(notebook, cell_id, "code") for cell_id in output_ids]
    codes = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    assert len(codes) == code_count, "unexpected notebook code cell count"
    assert [cell.get("execution_count") for cell in codes] == list(range(1, code_count + 1)), "notebook is not fully executed in order"
    assert not any(output.get("output_type") == "error" for cell in codes for output in cell.get("outputs", [])), "notebook contains error output"
    for cell_id, cell in zip(output_ids, evidence_cells):
        assert any(output.get("output_type") == "stream" and output.get("text") for output in cell.get("outputs", [])), f"missing executed text output: {cell_id}"


def _format(value, decimals: int | None = None) -> str:
    """Format scalar/array measurements for a single Markdown table cell without mutation."""
    values = np.asarray(value, dtype=np.float64)
    if decimals is None:
        formatter = {"float_kind": lambda item: f"{item:.12g}"}
    else:
        formatter = {"float_kind": lambda item: f"{0.0 if round(float(item), decimals) == 0 else item:.{decimals}f}"}
    return np.array2string(values, separator=", ", max_line_width=10000,
                           formatter=formatter).replace("\n", " ")


def _stdout_sections(notebook: dict, cell_ids: tuple[str, ...]) -> list[str]:
    """Quote selected executed stdout streams verbatim inside safe Markdown fences.

    Call only after execution and numerical validation. Selected cells must
    contain stdout stream records, with string or serialized string-list text;
    stderr, errors, HTML/rich outputs and malformed records raise ValueError.
    Every selected stream is included in order, without rerunning or mutation.
    A fence longer than any matching backtick run keeps text literal.
    """
    lines = ["## 실제 실행 출력", "",
             "아래는 실행된 노트북의 필수 검증 셀에서 저장한 stdout 원문이다. 선택한 셀의 출력은 생략하지 "
             "않았으며 환경 소개·중간 확인·파일 저장 등 다른 셀의 출력은 제외했다. 위 표는 JSON 수치를 정리한 "
             "결과이고, 아래 블록은 재구성한 예시가 아닌 실제 출력이다.", ""]
    for cell_id in cell_ids:
        cell = _cell(notebook, cell_id, "code")
        chunks = []
        for output in cell.get("outputs", []):
            if output.get("output_type") != "stream" or output.get("name") != "stdout":
                raise ValueError(f"expected only stdout stream output: {cell_id}")
            raw = output.get("text")
            if isinstance(raw, str):
                chunks.append(raw)
            elif isinstance(raw, list) and all(isinstance(part, str) for part in raw):
                chunks.append("".join(raw))
            else:
                raise ValueError(f"malformed stdout text: {cell_id}")
        text = "".join(chunks)
        if not text.strip():
            raise ValueError(f"empty stdout text: {cell_id}")
        fence = "```"
        while fence in text:
            fence += "`"
        body = text if text.endswith("\n") else text + "\n"
        lines += [f"### `{cell_id}`", "", fence + "text\n" + body + fence, ""]
    return lines


def render_backprop(report: dict, notebook: dict) -> str:
    """Validate the fixed (2->2->1) BCE experiment and render its complete Markdown.

    report is the fresh backprop_metrics JSON mapping; notebook is the executed
    dict-compatible NotebookNode with stable IDs. Independent fixed constants,
    shapes, four-decimal displays, reported errors and nine gradient components
    are checked before rendering. Missing fields raise KeyError, missing cells
    raise ValueError, and inconsistent evidence raises AssertionError/TypeError.
    No inputs, files, execution state or RNG state are modified.
    """
    forward_text = _markdown(notebook, "backprop-forward-derivation")
    backward_text = _markdown(notebook, "backprop-chain-rule")
    _execution(report, notebook, 8, ("backprop-compare", "backprop-gradient-check"))
    parameters = {"x": [1., 0.], "W1": [[.1, .2], [.3, .4]], "b1": [0., 0.],
                  "W2": [.5, .6], "b2": 0., "y_true": 1.}
    for name, expected in parameters.items():
        _close(report["parameters"][name], expected, name, atol=0)
    actual = {**report["forward"], **report["backward"]}
    assert set(actual) == set(report["hand_reference"]) == set(_BACKPROP_REFERENCE)
    assert set(report["shapes"]) == set(report["hand_absolute_errors"]) == set(actual)
    rows, errors = [], []
    for name, reference in _BACKPROP_REFERENCE.items():
        hand = _close(report["hand_reference"][name], reference, f"hand {name}", atol=5e-15)
        value = _close(actual[name], hand, name, atol=5e-5)
        assert report["shapes"][name] == list(value.shape), f"wrong recorded shape: {name}"
        assert _format(value, 4) == _format(hand, 4), f"four-decimal mismatch: {name}"
        error = float(np.max(np.abs(value - hand)))
        _close(report["hand_absolute_errors"][name], error, f"reported error {name}", atol=0, rtol=1e-12)
        errors.append(error)
        rows.append(f"| `{name}` | `{value.shape}` | `{_format(hand)}` | `{_format(value)}` | `{_format(value, 4)}` | {error:.3e} |")
    _close(report["max_hand_absolute_error"], max(errors), "max hand error", atol=0, rtol=1e-12)
    assert report["four_decimal_match"] is True
    assert report["parameter_count"] == 9 and report["parameter_order"] == _PARAMETER_ORDER
    _close(report["derivative_h"], 1e-5, "derivative_h", atol=0)
    flattened = np.concatenate((np.asarray(actual["dL_dW1"]).ravel(), actual["dL_db1"],
                                actual["dL_dW2"], [actual["dL_db2"]]))
    analytic = _close(report["analytical_gradient"], flattened, "analytic gradient", atol=0)
    numeric = _close(report["numerical_gradient"], analytic, "numerical gradient", atol=1e-6)
    gradient_error = float(np.max(np.abs(numeric - analytic)))
    _close(report["max_numerical_gradient_error"], gradient_error, "max gradient error", atol=0, rtol=1e-12)
    lines = ["# 역전파: 실행된 손계산·NumPy 비교", "",
             "**미션 연결: #7~#8 — 연쇄 법칙 유도와 손계산 검증.** 이 작은 신경망은 자동 미분 없이 학습의 "
             "한 단계를 끝까지 추적하기 위한 예제다. 순전파는 입력에서 예측과 손실을 만들고, 역전파는 그 경로를 "
             "거슬러 각 가중치를 조금 바꿀 때 손실이 얼마나 달라지는지 구한다. shape 표기는 스칼라·벡터·행렬을 "
             "구분해 원소별 곱과 내적, 외적을 혼동하지 않도록 돕는다.", "",
             f"seed 42 · float64 · NumPy {report['numpy_version']}; 코드 8셀을 순서대로 실행했고 오류 출력은 0개다.",
             "재현: `python -m scripts.run_submission_reports`; 개별 실행: `python -m scripts.run_notebooks notebooks/backprop_derivation.ipynb`.",
             "원시 데이터: [backprop_metrics.json](backprop_metrics.json). 원본: [backprop_derivation.ipynb](../notebooks/backprop_derivation.ipynb).", "",
             "## 1. 고정 입력과 shape", "",
             "| 기호 | 값 | shape |", "|---|---|---|"]
    for name, value in parameters.items():
        lines.append(f"| `{name}` | `{_format(value)}` | `{np.shape(value)}` |")
    lines += ["", forward_text, "", backward_text, "",
              "출력 예측이 약 0.6473이고 정답은 1이므로 출력 logit에 대한 손실 기울기는 음수다. 이 지점에서 "
              "logit을 증가시키는 방향은 손실을 줄이며, 연쇄 법칙은 그 변화의 몫을 각 층의 매개변수에 전달한다. "
              "입력의 두 번째 성분이 0이라 첫 가중치 행렬의 두 번째 열은 이 표본의 예측에 영향을 주지 않고 "
              "그 기울기도 0이다. 이는 해당 표본의 특성이며 모든 데이터에서 그 가중치가 학습되지 않는다는 뜻은 아니다.", "",
              "## 4. 13개 체크포인트의 독립 기준·실측 비교", "",
              "표의 손계산·실측값은 12개 유효자리로 표시하고 오차는 원시 값에서 다시 계산했다. 손계산 허용치는 atol=5e-5, rtol=0이며 네 자리 표시도 각각 일치한다.", "",
              "| 체크포인트 | shape | 독립 손계산 | NumPy 실측 | 공통 네 자리 표시 | 최대 절대오차 |",
              "|---|---|---|---|---|---:|", *rows, "",
              "## 5. 독립 중심차분과 판정", "",
              "13개 손계산 비교는 어느 중간 단계에서 계산이 어긋나는지 찾게 해 준다. 별도의 중심차분 검사는 "
              "순전파 손실만으로 최종 기울기를 확인하므로 같은 잘못된 역전파 식을 손계산과 코드에 반복하는 "
              "실수를 줄인다. 중간값은 충분한 정밀도로 계산하고 네 자리 반올림은 마지막 표시에만 적용한다.", "",
              "9개 매개변수 순서: `" + ", ".join(_PARAMETER_ORDER) + "`.",
              "순전파 손실만 사용하는 `(L(theta+h e_i)-L(theta-h e_i))/(2h)`, h=1e-5, gradient shape `(9,)`, atol=1e-6, rtol=0으로 교차 검증했다.",
              f"손계산 최대 절대오차 **{max(errors):.16g} ≤ 5e-5**, 중심차분 최대오차 **{gradient_error:.16g} ≤ 1e-6**: **PASS**.",
              "원시 계산은 반올림하지 않는다. 이 결과는 고정 예제에 대한 것이며 임의 입력의 정확도를 보장하지 않는다.", ""]
    lines += _stdout_sections(notebook, ("backprop-compare", "backprop-gradient-check"))
    return "\n".join(lines)


def _validate_probability(report: dict) -> None:
    """Assert fixed distribution/loss measurements and recorded error fields are consistent.

    Distribution vectors have explicit shapes; independent constants validate
    the Gaussian, Bernoulli and categorical examples. Unsupported structures
    raise KeyError/TypeError; inaccurate or nonfinite evidence raises assertions.
    """
    assert len(report["normals"]) == len(report["bernoulli"]) == 2
    for row, mu, variance, peak in zip(report["normals"], (0., 2.), (1., .5),
                                     (0.3989422804014327, 0.5641895835477563)):
        _close(row["mu"], mu, "normal mu", atol=0)
        _close(row["variance"], variance, "normal variance", atol=0)
        _close(row["sigma"], np.sqrt(variance), "normal sigma", atol=1e-15)
        _close(row["peak"], peak, "normal peak", atol=1e-15)
        integral = float(_close(row["integral"], 1., "normal integral", atol=1e-4))
        _close(row["integral_absolute_error"], abs(integral - 1.), "normal mass error", atol=0, rtol=1e-12)
        assert row["grid_points"] == 10001 and float(_array(row["minimum_pdf"], (), "minimum PDF")) >= 0
    for row, probability in zip(report["bernoulli"], (.3, .7)):
        _close(row["p"], probability, "Bernoulli p", atol=0)
        _close(row["support"], [0., 1.], "Bernoulli support", atol=0)
        pmf = _close(row["pmf"], [1-probability, probability], "Bernoulli PMF", atol=1e-15)
        mass_error = abs(float(pmf.sum()) - 1.)
        assert mass_error <= 1e-12
        _close(row["mass_error"], mass_error, "Bernoulli mass error", atol=0, rtol=1e-12)
    soft = report["softmax"]
    _close(soft["logits"], [[1000., 1001., 1002.], [-1000., 0., 1000.]], "softmax logits", atol=0)
    probabilities = _close(soft["probabilities"], [[.09003057317038046, .24472847105479764, .6652409557748218], [0., 0., 1.]], "softmax probabilities", atol=1e-15)
    assert soft["axis"] == -1
    row_sums = probabilities.sum(axis=-1)
    _close(soft["row_sums"], row_sums, "softmax row sums", atol=0)
    error = float(np.max(np.abs(row_sums - 1.)))
    assert error <= 1e-6
    _close(soft["max_sum_error"], error, "softmax sum error", atol=0, rtol=1e-12)
    column_error = float(_array(soft["column_sum_error"], (), "softmax column error"))
    assert 0 <= column_error <= 1e-6
    gaussian = report["gaussian"]
    assert gaussian["n"] == 3
    for name, expected in (("variance", .5), ("sse", 1.5), ("mse", .5),
                           ("nll", 3.21709482877410026), ("independent_reference", 3.21709482877410026)):
        _close(gaussian[name], expected, "Gaussian " + name, atol=1e-14)
    _close(gaussian["mean_nll"], gaussian["nll"]/3, "Gaussian mean NLL", atol=1e-14)
    assert 0 <= float(_array(gaussian["identity_error"], (), "Gaussian identity error")) <= 1e-14
    bce = report["bce"]
    _close(bce["y"], [0., 1.], "BCE labels", atol=0)
    _close(bce["probabilities"], [.3, .7], "BCE probabilities", atol=0)
    _close(bce["logits"], [-.8472978603872036, .8472978603872034], "BCE logits", atol=1e-15)
    _close(bce["sum_loss"], .71334988787746476, "BCE sum", atol=1e-15)
    _close(bce["mean_loss"], bce["sum_loss"]/2, "BCE mean", atol=1e-15)
    _close(bce["logit_sum_loss"], bce["sum_loss"], "BCE logit loss", atol=1e-15)
    analytic = _close(bce["analytic_sum_gradient"], [.3, -.3], "BCE analytic gradient", atol=1e-15)
    numeric = _close(bce["numerical_sum_gradient"], analytic, "BCE numerical gradient", atol=1e-8)
    _close(bce["max_gradient_error"], float(np.max(np.abs(numeric-analytic))), "BCE gradient error", atol=0, rtol=1e-12)
    _close(bce["extreme_losses"], [0., 0., 1000., 1000.], "BCE extreme losses", atol=0)
    _close(bce["positive_tail_at_40"], 4.248354255291589e-18, "BCE positive tail", atol=0, rtol=1e-14)
    _close(bce["subtractive_formula_at_40"], 0., "BCE subtractive cancellation", atol=0)
    assert bce["boundary_policy_verified"] is True
    categorical = report["categorical"]
    _close(categorical["probabilities"], [.2, .3, .5], "categorical probabilities", atol=0)
    assert categorical["target"] == 2
    _close(categorical["probability_ce"], .69314718055994531, "categorical probability CE", atol=1e-15)
    _close(categorical["logit_ce"], categorical["probability_ce"], "categorical logit CE", atol=1e-15)
    _close(categorical["extreme_ce"], 2000., "categorical extreme CE", atol=0)
    assert categorical["boundary_policy_verified"] is True
    assert report["figures"] == ["probability_normal.png", "probability_bernoulli.png"]


def render_probability(report: dict, notebook: dict) -> str:
    """Validate and render distribution measurements and three likelihood derivations.

    report is probability_metrics JSON from the supplied executed notebook.
    Stable Markdown cells supply Gaussian/Bernoulli/categorical derivations;
    tables expose actual PDF/PMF/Softmax/loss values and boundary results.
    Missing cells raise ValueError and inconsistent measurements assert; the
    function performs no execution, file writes, input mutation or seed change.
    """
    derivations = [_markdown(notebook, cell_id) for cell_id in
                   ("gaussian-mle-derivation", "bernoulli-mle-derivation", "categorical-mle-derivation")]
    _execution(report, notebook, 9, ("normal-density-check", "bernoulli-check", "softmax-check",
                                    "gaussian-mle-check", "bce-check", "categorical-check"))
    _validate_probability(report)
    soft, gaussian, bce, categorical = (report[name] for name in ("softmax", "gaussian", "bce", "categorical"))
    lines = ["# 확률·손실: 실행된 분포와 MLE 유도", "",
             "**미션 연결: #13~#15 — 확률 분포·Softmax·MLE와 손실.** 손실 함수는 관측이 어떤 확률 모형에서 "
             "나왔다고 가정하는지와 연결된다. Gaussian 오차를 가정한 연속값 예측, Bernoulli를 가정한 이진 관측, "
             "categorical을 가정한 여러 클래스 중 하나의 관측을 구분하고, 데이터를 더 그럴듯하게 만드는 최대우도가 "
             "각각 어떤 손실의 최소화가 되는지 확인한다.", "",
             f"seed 42 · float64 · NumPy {report['numpy_version']}; 코드 9셀을 순서대로 실행했고 오류 출력은 0개다.",
             "재현: `python -m scripts.run_submission_reports`; 개별 실행: `python -m scripts.run_notebooks notebooks/probability_loss.ipynb`.",
             "원시 데이터: [probability_metrics.json](probability_metrics.json). 원본: [probability_loss.ipynb](../notebooks/probability_loss.ipynb).", "",
             "## 1. 정규 PDF와 Bernoulli PMF", "",
             "정규분포는 유한한 mu와 variance>0에서 N(mu,variance), PDF=exp(-(x-mu)²/(2*variance))/sqrt(2*pi*variance)다. 각 mu±8*sigma 구간의 10,001점에서 적분한다.",
             "Bernoulli는 0≤p≤1, k∈{0,1}에서 PMF(0)=1-p, PMF(1)=p이며 지지집합 밖은 0이다.", "",
             "| 분포 | 매개변수 | 실측값 | 오차·기준 |", "|---|---|---|---|"]
    for row in report["normals"]:
        lines.append(f"| Normal | mu={row['mu']:g}, variance={row['variance']:g}, sigma={row['sigma']:.12g} | integral={row['integral']:.16g}, peak={row['peak']:.12g} | {row['integral_absolute_error']:.3e} ≤ 1e-4 |")
    for row in report["bernoulli"]:
        lines.append(f"| Bernoulli | p={row['p']:g}, support=[0,1] | PMF=`{_format(row['pmf'])}` | sum error={row['mass_error']:.3e} ≤ 1e-12 |")
    lines += ["",
              "정규 PDF 그림에서 평균은 곡선의 중심을 옮기고 분산은 퍼짐을 바꾼다. 분산 0.5인 곡선은 더 좁고 "
              "높지만 전체 면적은 여전히 1이다. PDF 높이 자체는 한 점의 확률이 아니며 구간 확률은 면적으로 읽는다.", "",
              "![정규분포 PDF 비교](probability_normal.png)", "",
              "Bernoulli 그림의 막대 높이는 0과 1이라는 각 결과의 확률이다. 같은 p에 속한 두 막대의 높이를 더하면 "
              "1이 되며, p=0.3과 0.7에서는 두 결과의 확률이 서로 바뀐다. 연속 분포의 밀도와 이산 분포의 질량을 "
              "구분해 읽어야 한다.", "",
              "![Bernoulli PMF 비교](probability_bernoulli.png)", "",
              "## 2. Softmax의 실제 합 검증", "",
              "유한한 logits에서 선택 축의 최댓값 m을 빼고 p_j=exp(z_j-m)/sum_k exp(z_k-m)로 계산한다. 확률은 비음수이고 선택 축을 따라 합이 1이다.",
              f"logits=`{_format(soft['logits'])}`, shape `(2,3)`, axis=-1. 확률=`{_format(soft['probabilities'])}`.",
              f"행 합=`{soft['row_sums']}`, 최대 합 오차 **{soft['max_sum_error']:.16g} ≤ 1e-6**. axis=0 재검증의 열 합 오차도 **{soft['column_sum_error']:.16g} ≤ 1e-6**: PASS.", "",
              "Softmax는 여러 logit을 한 분포로 묶으므로 클래스 축을 어디로 선택했는지가 중요하다. 같은 최댓값을 "
              "빼는 계산은 확률을 바꾸지 않고 overflow를 피한다. 극단 logit의 작은 확률은 float64에서 0으로 표시될 "
              "수 있어, 아래 손실 검증은 확률의 로그를 우회 계산하는 대신 안정적인 logit 식도 함께 확인한다.", "",
              "## 3. 손실의 독립 수치 검증", "",
              "Gaussian NLL과 MSE는 아래 표에서 값이 다르지만 매개변수에 무관한 상수와 양의 배율 차이여서 같은 "
              "최소점을 갖는다. 이 연결에는 독립 관측과 고정 공통 분산 가정이 필요하다. 분산까지 학습하거나 관측마다 "
              "다른 분산을 사용하면 이 단순한 관계를 그대로 적용할 수 없다.", "",
              "| 예제 | 입력·조건 | 실측 결과 | 오차·기준 |", "|---|---|---|---|"]
    lines += [f"| Gaussian | observed=[0,1,2], predicted=[1,.5,1.5], n=3, variance=.5 | SSE={gaussian['sse']}, MSE={gaussian['mse']}, NLL={gaussian['nll']:.16g}, mean NLL={gaussian['mean_nll']:.16g} | density/NLL identity error={gaussian['identity_error']:.3e} ≤ 1e-14 |",
              f"| Bernoulli BCE | y=[0,1], p=[.3,.7] | sum={bce['sum_loss']:.16g}, mean={bce['mean_loss']:.16g}, logit sum={bce['logit_sum_loss']:.16g} | gradient max error={bce['max_gradient_error']:.3e} ≤ 1e-8 |",
              f"| Categorical CE | p=[.2,.3,.5], target index=2 (third class) | probability CE={categorical['probability_ce']:.16g}, logit CE={categorical['logit_ce']:.16g} | agreement ≤ 1e-15 |", "",
              *[text + "\n" for text in derivations],
              "## 7. 경계·수치 안정성의 실행 결과", "",
              f"BCE의 (y,p)=(0,0)/(1,1)은 0, (1,0)/(0,1)은 +inf임을 검증했다. 극단 logit [-1000,1000,-1000,1000]의 손실은 `{_format(bce['extreme_losses'])}`이며 모두 유한하다.",
              f"z=40,y=1의 안정식 손실은 {bce['positive_tail_at_40']:.16g}>0이고 subtractive logaddexp(0,z)-z는 {bce['subtractive_formula_at_40']:g}으로 상쇄된다.",
              f"Categorical에서 p=[0,0,1]의 정답 index=2는 0, index=0은 +inf다. logits=[-1000,0,1000], target=0의 CE는 {categorical['extreme_ce']:g}다. 불가능 사건을 clipping하지 않으며 JSON에는 경계 검증 여부를 기록한다.",
              "유한 logit에서 극소확률이 0으로 표시되는 표현 한계와, 확률 모형이 불가능 사건에 0을 직접 부여한 "
              "경우의 무한 손실은 구분한다.",
              "분포·합/평균·독립 손실값·수치 gradient·경계 assertion: **PASS**. 이는 명시한 고정 예제의 결과이며 모든 모델의 통계 가정을 보장하지 않는다.", ""]
    lines += _stdout_sections(notebook, ("normal-density-check", "bernoulli-check", "softmax-check",
                                       "gaussian-mle-check", "bce-check", "categorical-check"))
    return "\n".join(lines)
