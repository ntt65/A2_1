"""순수 노트북 보고서의 일관성과 실행 구조를 독립적으로 검증한다."""

import copy
import json
from pathlib import Path
import unittest

import nbformat
import numpy as np

from scripts.report_notebooks import render_backprop, render_probability

__covers__ = ["scripts/report_notebooks.py"]
ROOT = Path(__file__).resolve().parents[1]


def fixtures(name: str) -> tuple[dict, dict]:
    """실행하지 않고 저장된 실행 노트북 구조와 측정 JSON을 읽는다."""
    report = json.loads((ROOT / "outputs" / f"{name}_metrics.json").read_text(encoding="utf-8"))
    notebook_name = "backprop_derivation" if name == "backprop" else "probability_loss"
    notebook = nbformat.read(ROOT / "notebooks" / f"{notebook_name}.ipynb", as_version=4)
    return report, notebook


class NotebookReportTests(unittest.TestCase):
    """독립적으로 읽을 수 있는 근거, 누락 셀의 정확한 오류와 잘못된 PASS 거부를 확인한다."""

    def test_backprop_contains_fixed_values_all_shapes_and_derivations(self):
        """실제 실행한 셀 여덟 개에서 비교표 13행과 여섯 미분식이 보고서에 있는지 확인한다."""
        report, notebook = fixtures("backprop")
        text = render_backprop(report, notebook)
        for value in ("0.6072", "-1.5449", "-0.3527", "-0.0440", "-0.0517", "(2, 2)",
                      "backprop_metrics.json", "atol=5e-5", "atol=1e-6", "코드 8셀", "**PASS**"):
            assert value in text, value
        assert text.count("| `dL_") == 8	# 행렬·벡터별 미분 비교를 보여 주는 표 행이 빠지지 않는다.
        assert "\\delta_1x^T" in text and "\\delta_2 a_1" in text	# 역전파의 두 외적 유도식이 본문에 포함되어야 한다.
        assert "python -m scripts.run_submission_reports" in text
        assert "python -m scripts.run_notebooks notebooks/backprop_derivation.ipynb" in text
        assert "미션 연결: #7~#8" in text
        for cell_id in ("backprop-compare", "backprop-gradient-check"):
            cell = next(cell for cell in notebook["cells"] if cell["id"] == cell_id)
            raw = "".join(output["text"] for output in cell["outputs"])
            assert "```text\n" + raw + "```" in text	# 실행 셀의 원래 표준출력을 수정하지 않고 텍스트 블록으로 보존한다.
        assert "All 13 hand-reference checkpoints agree" in text
        assert "All 9 parameter gradients pass" in text	# 매개변수 아홉 개의 실제 기울기 검증 출력이 보고서에 있어야 한다.

    def test_probability_contains_distributions_likelihoods_and_boundaries(self):
        """실행 셀 아홉 개에서 PNG 링크 두 개와 독립 손실값 세 개를 확인한다."""
        report, notebook = fixtures("probability")
        text = render_probability(report, notebook)
        for value in ("3.2170948287741", "0.7133498878774649", "0.6931471805599453",
                      "variance=0.5", "axis=-1", "axis=0", "고정", "+inf", "코드 9셀",
                      "(probability_normal.png)", "(probability_bernoulli.png)", "**PASS**"):
            assert value in text, value
        assert "\\frac{n}{2\\sigma^2}\\operatorname{MSE}" in text	# Gaussian NLL과 MSE 사이의 비례계수를 본문에 남긴다.
        assert "L_{\\mathrm{mean}}=L_{\\mathrm{sum}}/n" in text	# 합 손실과 평균 손실의 1/n 관계를 명시한다.
        assert "\\mathcal L=\\prod_i\\prod_j p_{ij}^{y_{ij}}" in text	# 다중 클래스 우도의 곱 형태를 생략하지 않는다.
        assert "[0.9999999999999999, 1.0]" in text
        assert "미션 연결: #13~#15" in text
        for cell_id in ("normal-density-check", "bernoulli-check", "softmax-check",
                        "gaussian-mle-check", "bce-check", "categorical-check"):
            cell = next(cell for cell in notebook["cells"] if cell["id"] == cell_id)
            raw = "".join(output["text"] for output in cell["outputs"])
            assert "```text\n" + raw + "```" in text
        assert "다른 셀의 출력은 제외했다" in text	# 관련 없는 셀 출력까지 실행 증거로 섞지 않는 범위를 명시한다.

    def test_backprop_injected_mismatch_cannot_reuse_true_flags(self):
        """저장된 PASS가 모두 참이어도 바뀐 기울기나 잘못된 스칼라를 거부하는지 확인한다."""
        for field in ("gradient", "reference", "max_error", "numeric_gradient", "shape"):
            report, notebook = fixtures("backprop")
            if field == "gradient":
                report["backward"]["dL_dW1"][0][0] += .01
            elif field == "reference":
                report["forward"]["z2"] = report["hand_reference"]["z2"] = .7
            elif field == "max_error":
                report["max_hand_absolute_error"] = .01
            elif field == "numeric_gradient":
                report["numerical_gradient"][0] += .001
            else:
                report["shapes"]["z2"] = [1]
            with self.subTest(field=field), self.assertRaises(AssertionError):
                render_backprop(report, notebook)

    def test_probability_injected_mismatch_cannot_claim_pass(self):
        """PDF·PMF·Softmax·손실의 독립 기대값이 변조한 데이터를 거부하는지 확인한다."""
        for field in ("normal", "bernoulli", "softmax", "gaussian", "bce", "categorical", "boundary"):
            report, notebook = fixtures("probability")
            if field == "normal":
                report["normals"][0]["integral"] = .9
            elif field == "bernoulli":
                report["bernoulli"][0]["pmf"] = [.5, .5]
            elif field == "softmax":
                report["softmax"]["probabilities"][0] = [0., 0., 1.]
            elif field == "gaussian":
                report["gaussian"]["nll"] = report["gaussian"]["mse"]
            elif field == "bce":
                report["bce"]["sum_loss"] = .5
            elif field == "categorical":
                report["categorical"]["target"] = 1
            else:
                report["bce"]["boundary_policy_verified"] = False
            with self.subTest(field=field), self.assertRaises(AssertionError):
                render_probability(report, notebook)

    def test_missing_or_wrong_derivation_cells_raise_precise_errors(self):
        """필수 셀 ID가 빠지거나 설명 셀이 코드로 바뀐 경우를 조용히 무시하지 않는지 확인한다."""
        cases = [("backprop", render_backprop, "backprop-chain-rule"),
                 ("probability", render_probability, "gaussian-mle-derivation")]
        for name, renderer, cell_id in cases:
            report, notebook = fixtures(name)
            notebook["cells"] = [cell for cell in notebook["cells"] if cell["id"] != cell_id]
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "missing required markdown cell: " + cell_id):
                renderer(report, notebook)
            report, notebook = fixtures(name)
            next(cell for cell in notebook["cells"] if cell["id"] == cell_id)["cell_type"] = "code"
            with self.assertRaisesRegex(ValueError, "must have type markdown"):
                renderer(report, notebook)
        report, notebook = fixtures("backprop")
        notebook["cells"] = [cell for cell in notebook["cells"] if cell["id"] != "backprop-compare"]
        with self.assertRaisesRegex(ValueError, "missing required code cell: backprop-compare"):
            render_backprop(report, notebook)

    def test_unexecuted_or_error_outputs_are_rejected(self):
        """저장된 성공 플래그로 순차 실행 횟수와 실제 출력 구조를 대신하지 않는지 확인한다."""
        for name, renderer in (("backprop", render_backprop), ("probability", render_probability)):
            for failure in ("unexecuted", "error", "missing_output"):
                report, notebook = fixtures(name)
                code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
                if failure == "unexecuted":
                    code[0]["execution_count"] = None
                elif failure == "error":
                    code[-1]["outputs"] = [{"output_type": "error", "ename": "AssertionError", "evalue": "injected", "traceback": []}]
                else:
                    for cell in code:
                        cell["outputs"] = []
                with self.subTest(name=name, failure=failure), self.assertRaises(AssertionError):
                    renderer(report, notebook)
            selected_id = "backprop-compare" if name == "backprop" else "normal-density-check"
            for failure in ("rich_html", "stderr", "malformed_text"):
                report, notebook = fixtures(name)
                cell = next(cell for cell in notebook["cells"] if cell["id"] == selected_id)
                if failure == "rich_html":
                    cell["outputs"].append({"output_type": "display_data", "data": {"text/html": "<p>hidden</p>"}, "metadata": {}})
                elif failure == "stderr":
                    cell["outputs"][0]["name"] = "stderr"
                else:
                    cell["outputs"][0]["text"] = {"unexpected": "mapping"}
                with self.subTest(name=name, failure=failure), self.assertRaises(ValueError):
                    renderer(report, notebook)

    def test_renderers_preserve_inputs_and_random_state(self):
        """순수 렌더러가 호출자의 사전과 전역 난수 상태를 그대로 두는지 확인한다."""
        for name, renderer in (("backprop", render_backprop), ("probability", render_probability)):
            report, notebook = fixtures(name)
            before_report, before_notebook = copy.deepcopy(report), copy.deepcopy(notebook)
            random_state = np.random.get_state()
            first = renderer(report, notebook)
            assert first == renderer(report, notebook)
            assert report == before_report and notebook == before_notebook	# 렌더링 이후 측정값과 노트북 구조가 모두 원본과 같아야 한다.
            after_state = np.random.get_state()
            assert random_state[0] == after_state[0] and random_state[2:] == after_state[2:]
            np.testing.assert_array_equal(random_state[1], after_state[1])	# 순수 렌더러가 전역 난수 상태를 바꾸지 않는다.

    def test_plain_json_notebook_sources_are_supported(self):
        """nbformat의 문자열과 직렬화 JSON의 줄 목록이 같은 결과를 렌더하는지 확인한다."""
        for name, renderer in (("backprop", render_backprop), ("probability", render_probability)):
            report, notebook = fixtures(name)
            plain = json.loads(nbformat.writes(notebook))
            assert renderer(report, plain) == renderer(report, notebook)	# 직렬화 방식만 다른 노트북 소스는 동일하게 렌더한다.
