"""고정 이차함수의 독립 검증 자료로 순수 최적화 보고서 렌더러를 확인한다."""

import copy
import json
from pathlib import Path
import unittest

import numpy as np

from scripts.report_optimizer import render_optimizer

__covers__ = ["scripts/report_optimizer.py"]


def _run_fixture(objective: str, method: str, lr: float, steps: int) -> dict:
    """좌표별 닫힌형 해로 합성 측정 자료를 만든다.

    GD는 1-lr*h의 거듭제곱을 사용한다. Momentum은 실제 반복 구현과 독립적으로
    r²-(1+.9-lr*h)r+.9=0의 실수 삼각함수 해를 사용한다.
    경로 형상은 (steps+1,2)이며 고정 시작점 (5,5)를 포함한다.
    """
    hessian = np.array([2.0, 2.0 if objective == "circle" else 20.0])
    times = np.arange(steps+1)[:, None]
    if method == "GD":
        path = 5.0*(1.0-lr*hessian)**times
    else:
        radius = np.sqrt(0.9)
        angle = np.arccos((1.9-lr*hessian)/(2.0*radius))
        coefficient = ((1.0-lr*hessian)/radius-np.cos(angle))/np.sin(angle)
        path = 5.0*radius**times*(np.cos(times*angle)+coefficient*np.sin(times*angle))
    losses = np.sum(path**2*(hessian/2.0), axis=1)
    norms = np.linalg.norm(path*hessian, axis=1)
    radii = np.linalg.norm(path, axis=1)
    hits, outside = np.flatnonzero(radii <= 0.1), np.flatnonzero(radii > 0.1)
    stay = int(outside[-1]+1) if outside.size else 0
    return {"objective": objective, "method": method, "lr": lr,
            "beta": 0.9 if method == "Momentum" else None, "steps": steps,
            "iterations": steps, "gradient_tolerance": None, "status": "budget_exhausted",
            "initial": [5.0, 5.0], "path": path.tolist(), "losses": losses.tolist(),
            "gradient_norms": norms.tolist(), "final_norm": float(radii[-1]),
            "first_radius_0_1": int(hits[0]) if hits.size else None,
            "remaining_inside_radius_0_1_through_budget": stay if stay < len(path) else None}


def _fixture() -> dict:
    """독립적인 닫힌형 해만 사용하여 고정 보고서 레코드 일곱 개를 반환한다."""
    return {"seed": 42, "dtype": "float64", "comparisons": {
        name: {method: _run_fixture(name, method, rate, 100) for method in ("GD", "Momentum")}
        for name, rate in (("circle", 0.1), ("ellipse", 0.05))},
        "learning_rates": {str(rate): _run_fixture("circle", "GD", rate, 20) for rate in (0.5, 1.0, 1.1)}}


class OptimizerReportTests(unittest.TestCase):
    """읽을 수 있는 수치, 입력 보존과 모순된 근거의 거부를 확인한다."""

    def test_independent_fixture_reports_all_conditions_and_checkpoints(self):
        """알려진 닫힌형 해의 수치·학습 조건·체크포인트와 그림 링크가 보고서에 남는지 확인한다."""
        report = _fixture()
        original = copy.deepcopy(report)
        text = render_optimizer(report)
        assert report == original
        assert "1.440402e-09" in text and "2.074758e-18" in text	# 독립 닫힌형 경로에서 얻는 최종 반경과 손실이 본문에 있는지 확인한다.
        assert "2.710878e+02" in text and "7.348858e+04" in text
        assert "| 원형 Momentum | 0.1 |" in text
        assert "| 10 | 77 |" in text	# 첫 진입 10회와 예산 끝까지 유지 시작 77회를 구별한다.
        assert "| 타원형 GD | 0.05 |" in text and "| 38 | 38 |" in text
        assert "| 0.5 | 0 | 0 |" in text and "| 1 | -1 | 1 |" in text
        assert "| 1.1 | -1.2 | 1.44 |" in text
        assert "seed=42, dtype=float64" in text and "shape=(101,2)" in text and "shape=(21,2)" in text	# 시드·자료형·원시 경로 형상이 보고서에서 빠지지 않는다.
        assert "100회까지" in text and "보편적인 우월성" in text
        assert "미션 연결: #9~#12" in text
        for explanation in ("관성이 남아 진동", "왼쪽에서 등고선 위 경로", "각 패널이 자기 좌표"):
            assert explanation in text
        for label, image in (("원형 경로·손실·gradient", "optimizer_circle.png"),
                             ("타원형 경로·손실·gradient", "optimizer_ellipse.png"),
                             ("학습률별 경로", "optimizer_learning_rates.png")):
            assert f"[{label}]({image})" in text
            self.assertRegex(text, r"!\[[^\]]+\]\(" + image.replace(".", r"\.") + r"\)")
        for image in ("optimizer_circle.png", "optimizer_ellipse.png", "optimizer_learning_rates.png"):
            assert f"]({image})" in text
        for condition in ("원형 GD", "원형 Momentum", "타원형 GD", "타원형 Momentum"):
            for step in (0, 1, 2, 5, 10, 20, 50, 100):
                assert f"| {condition} | {step} | (" in text	# 모든 방법의 지정 체크포인트가 본문 표에 남아 있는지 확인한다.
        for rate in ("0.5", "1", "1.1"):
            for step in (0, 1, 5, 10, 20):
                assert f"| {rate} | {step} | (" in text

    def test_real_committed_measurements_render_without_mutation(self):
        """실제 생성기 형식을 받아들이면서 입력의 직렬화 내용과 저장 JSON 텍스트를 보존하는지 확인한다."""
        path = Path(__file__).resolve().parents[1]/"outputs"/"optimizer_metrics.json"
        source = path.read_text(encoding="utf-8")
        report = json.loads(source)
        before = json.dumps(report, sort_keys=True)
        text = render_optimizer(report)
        assert "1.018518e-09" in text and "2.016252e-02" in text
        assert "1.328070e-04" in text and "1.987508e-02" in text
        assert before == json.dumps(report, sort_keys=True)	# 순수 렌더링이 입력 측정 사전을 바꾸지 않는다.
        assert path.read_text(encoding="utf-8") == source	# 저장된 원본 JSON 텍스트도 변하지 않는다.

    def test_injected_summary_errors_are_rejected(self):
        """잘못된 최종 반경·첫 진입·유지 시작·레이블을 성공 보고서로 만들지 않는지 확인한다."""
        for field, value in (("final_norm", 0.0), ("first_radius_0_1", 0),
                             ("remaining_inside_radius_0_1_through_budget", 10),
                             ("steps", 99), ("iterations", 99), ("lr", 0.2),
                             ("beta", 0.8), ("status", "gradient_tolerance")):
            report = _fixture()
            report["comparisons"]["circle"]["Momentum"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                render_optimizer(report)

    def test_inconsistent_history_nonfinite_and_missing_data_are_rejected(self):
        """손실·노름·경로·형상·메타데이터가 바뀌면 그럴듯한 결과를 출력하지 않고 거부하는지 확인한다."""
        for field, value in (("losses", 99.0), ("gradient_norms", 99.0), ("losses", np.inf)):
            report = _fixture()
            report["comparisons"]["circle"]["GD"][field][1] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                render_optimizer(report)
        report = _fixture()
        report["learning_rates"]["1.1"]["path"] = [[5.0, 5.0]]
        with self.assertRaises(ValueError):
            render_optimizer(report)
        report = _fixture()
        del report["comparisons"]["ellipse"]["GD"]
        with self.assertRaises(ValueError):
            render_optimizer(report)
        for field, value in (("seed", 0), ("dtype", "float32")):
            report = _fixture()
            report[field] = value
            with self.assertRaises(ValueError):
                render_optimizer(report)
        report = _fixture()
        run = report["comparisons"]["circle"]["GD"]
        # 내부적으로 일관돼도 수렴하지 않은 경로를 반경 0.1 이내 성공으로 표시하면 안 된다.
        run["path"] = [[5.0, 5.0]]*101
        run["losses"] = [50.0]*101
        run["gradient_norms"] = [np.sqrt(200.0)]*101
        run["final_norm"] = np.sqrt(50.0)
        run["first_radius_0_1"] = None
        run["remaining_inside_radius_0_1_through_budget"] = None
        with self.assertRaisesRegex(ValueError, "final-radius bound"):
            render_optimizer(report)


if __name__ == "__main__":
    unittest.main()
