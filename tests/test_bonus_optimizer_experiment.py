"""제출한 최적화 보너스의 경로와 캐시를 독립적으로 검증한다."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.run_bonus_optimizers import main, trajectory
from src.optimizer import run_optimization
from src.utils.cache_manager import CacheManager

__covers__ = ["scripts/run_bonus_optimizers.py"]


class BonusExperimentTests(unittest.TestCase):
    """선언한 갱신 예산, 독립 최종점과 실제 성공 기준을 확인한다."""

    def test_independent_quadratic_and_newton_stop(self):
        """닫힌형 GD와 한 번에 도달하는 Newton이 서로 다른 수락 경로 길이를 갖는지 확인한다."""
        gd = trajectory("ellipse", "GD", .05, steps=3, initial=[2., -3.])
        np.testing.assert_allclose(gd["path"], [[2,-3],[1.8,0],[1.62,0],[1.458,0]], atol=1e-14, rtol=0)	# 타원형 GD의 처음 네 점을 독립 손계산과 비교한다.
        newton = trajectory("ellipse", "Newton", 1., initial=[2., -3.], gradient_tolerance=1e-8)
        assert newton["iterations"] == 1 and newton["status"] == "gradient_tolerance"
        assert newton["path"].shape == (2,2)
        np.testing.assert_allclose(newton["path"][-1], [0,0], atol=1e-12, rtol=0)	# Newton의 한 번 갱신이 알려진 최소점인 원점에 도달한다.

    def test_cache_reuse_configuration_and_copy(self):
        """초기점·예산·학습률·종료 정책이 각각 캐시 식별에 반영되는지 확인한다."""
        with tempfile.TemporaryDirectory() as directory:
            cache = CacheManager(directory)
            with patch("scripts.run_bonus_optimizers.run_optimization", wraps=run_optimization) as compute:
                first = trajectory("circle", "Adam", .1, steps=3, cache=cache)
                repeated = trajectory("circle", "Adam", .1, steps=3, cache=cache)
                assert compute.call_count == 1
                for kwargs in ({"initial":[1.,2.]},{"steps":4},{"gradient_tolerance":1e-8}):
                    parameters={"steps":3, **kwargs}
                    trajectory("circle", "Adam", .1, cache=cache, **parameters)
                trajectory("circle", "Adam", .05, steps=3, cache=cache)
                assert compute.call_count == 5
                uncached = trajectory("circle", "Adam", .1, steps=3, cache=CacheManager(directory, enabled=False))
                assert compute.call_count == 6
            for field in ("path","losses","gradient_norms"):
                np.testing.assert_array_equal(first[field], repeated[field])
                np.testing.assert_array_equal(first[field], uncached[field])	# 캐시 사용 여부와 무관하게 모든 원시 이력이 같다.
            first["path"][:] = 999
            restored = trajectory("circle", "Adam", .1, steps=3, cache=cache)
            np.testing.assert_array_equal(restored["path"], repeated["path"])

    def test_invalid_configuration(self):
        """잘못된 이름·형상·예산·임계값은 캐시에 접근하기 전에 거부되는지 확인한다."""
        for args in (("bad","Adam",.1),("circle","bad",.1)):
            with self.assertRaises(ValueError):
                trajectory(*args)
        for kwargs in ({"initial":[1.]},{"steps":-1},{"gradient_tolerance":-1}):
            with self.assertRaises(ValueError):
                trajectory("circle","GD",.1,**kwargs)
        with self.assertRaises(TypeError):
            trajectory("circle","GD",.1,steps=True)

    def test_cli_reports_all_three_bonus_optimizer_comparisons(self):
        """CLI가 원시 이력과 실제 PNG를 포함하되 Adam의 우월성을 강제하지 않는지 확인한다."""
        with tempfile.TemporaryDirectory() as directory:
            assert main(["--output-dir",directory,"--no-cache"]) == 0
            root = Path(directory)
            report=json.loads((root/"bonus_optimizer_metrics.json").read_text(encoding="utf-8"))
            assert report["all_assertions_passed"] is True
            for methods in report["adam_comparisons"].values():
                assert set(methods)=={"GD","Momentum","Adam"}
                for record in methods.values():
                    assert np.asarray(record["path"]).shape == (201,2)	# 200회 비교 경로는 초기점을 포함한 (201,2)여야 한다.
                    assert record["steps_budget"] == record["iterations"] == 200	# 계획한 예산과 실제 수락한 갱신 횟수가 모두 200인지 확인한다.
            assert report["newton_comparison"]["Newton"]["iterations"] == 1
            assert report["newton_comparison"]["GD"]["iterations"] > 1
            for figure in report["figures"]:
                assert (root/figure["filename"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            assert "Update counts are not runtime comparisons" in (root/"bonus_optimizer_report.md").read_text(encoding="utf-8")	# 반복 횟수 비교를 실행 시간 우월성으로 과장하지 않는 문구를 보존한다.
