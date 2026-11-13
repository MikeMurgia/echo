"""Run from the repo root:  python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision import (  # noqa: E402
    LocalResidualModel, MARGIN_GRID, ResidualModel, load_local_model, load_model, save_model, bootstrap_cost_difference, breakeven_ratio, decide, fit_margins, stage_for_rul)


def gaussian_model(n=20000, sd=10.0, seed=0):
    rng = np.random.default_rng(seed)
    residual = rng.normal(0, sd, n)
    stage = np.array(["mid"] * n)
    return ResidualModel.fit(residual, stage, min_group=30)


class DecisionTests(unittest.TestCase):
    def test_failure_probability_matches_the_gaussian_tail(self):
        # yhat = 40, H = 30: failure needs residual >= 10, i.e. one sd above zero -> 0.1587
        p = gaussian_model().failure_prob([40.0], ["mid"], 30.0)
        self.assertAlmostEqual(float(p[0]), 0.1587, delta=0.01)

    def test_probability_is_monotone_in_predicted_rul(self):
        p = gaussian_model().failure_prob([20.0, 30.0, 40.0, 60.0], ["mid"] * 4, 30.0)
        self.assertTrue(np.all(np.diff(p) < 0))

    def test_probability_is_never_exactly_zero_or_one(self):
        p = gaussian_model().failure_prob([-500.0, 5000.0], ["mid", "mid"], 30.0)
        self.assertTrue(np.all((p > 0) & (p < 1)))

    def test_breakeven_rule_flips_at_one_over_p(self):
        model = gaussian_model()
        p = float(model.failure_prob([40.0], ["mid"], 30.0)[0])
        below = decide(model, [40.0], ["mid"], 30.0, cost_ratio=0.9 / p)
        above = decide(model, [40.0], ["mid"], 30.0, cost_ratio=1.1 / p)
        self.assertFalse(below["echo_act"][0])
        self.assertTrue(above["echo_act"][0])
        self.assertFalse(above["point_act"][0])      # predicted RUL 40 > H = 30, point rule waits
        self.assertAlmostEqual(float(breakeven_ratio(p)), 1 / p)

    def test_small_stage_falls_back_to_pooled_residuals(self):
        rng = np.random.default_rng(1)
        residual = rng.normal(0, 5, 500)
        stage = np.array(["early"] * 480 + ["late"] * 20)
        model = ResidualModel.fit(residual, stage, min_group=30)
        self.assertTrue(model.fallback["late"])
        self.assertFalse(model.fallback["early"])

    def test_json_round_trip_keeps_probabilities(self):
        model = gaussian_model()
        again = ResidualModel.from_json(model.to_json())
        yhat = np.array([25.0, 35.0, 45.0])
        a = model.failure_prob(yhat, ["mid"] * 3, 30.0)
        b = again.failure_prob(yhat, ["mid"] * 3, 30.0)
        np.testing.assert_allclose(a, b, atol=0.01)

    def test_overestimating_stage_gives_higher_failure_probability(self):
        rng = np.random.default_rng(2)
        res = np.concatenate([rng.normal(0, 5, 2000), rng.normal(8, 5, 2000)])
        stage = np.array(["mid"] * 2000 + ["late"] * 2000)
        model = ResidualModel.fit(res, stage)
        p_mid = model.failure_prob([40.0], ["mid"], 30.0)[0]
        p_late = model.failure_prob([40.0], ["late"], 30.0)[0]
        self.assertGreater(p_late, p_mid)

    def test_unknown_stage_raises(self):
        with self.assertRaises(ValueError):
            gaussian_model().failure_prob([40.0], ["weird"], 30.0)

    def test_stage_function_matches_conformal_thresholds(self):
        np.testing.assert_array_equal(stage_for_rul([120, 100, 60, 30, 29.9]),
                                      ["early", "mid", "mid", "mid", "late"])


class BaselineTests(unittest.TestCase):
    def _cal(self, bias, n=6000, seed=3, horizon=30.0):
        rng = np.random.default_rng(seed)
        y = rng.uniform(5, 100, n)
        yhat = y + bias + rng.normal(0, 4, n)
        A = yhat[:, None] <= horizon + MARGIN_GRID[None, :]
        return A, y <= horizon, np.array(["mid"] * n)

    def test_margin_is_positive_when_rul_is_overestimated_and_costs_are_high(self):
        A, failure, stage = self._cal(bias=10.0)
        self.assertGreater(fit_margins(A, failure, stage, rho=50.0)["global"], 5.0)

    def test_margin_is_near_zero_for_an_unbiased_model_at_moderate_cost_ratio(self):
        A, failure, stage = self._cal(bias=0.0)
        self.assertLessEqual(abs(fit_margins(A, failure, stage, rho=2.0)["global"]), 3.0)

    def test_stage_without_windows_inherits_the_global_margin(self):
        A, failure, stage = self._cal(bias=5.0)
        m = fit_margins(A, failure, stage, rho=20.0)
        self.assertEqual(m["stage"]["late"], m["global"])

    def test_bootstrap_identical_costs_gives_zero_difference(self):
        rng = np.random.default_rng(0)
        units = np.repeat(np.arange(40), 5)
        cost = rng.random(200)
        diff, lo, hi, share = bootstrap_cost_difference(units, cost, cost, rng, 200)
        self.assertEqual((diff, lo, hi, share), (0.0, 0.0, 0.0, 0.0))

    def test_bootstrap_detects_a_clearly_cheaper_rule(self):
        rng = np.random.default_rng(1)
        units = np.repeat(np.arange(60), 4)
        base = rng.random(240)
        diff, lo, hi, share = bootstrap_cost_difference(units, base - 0.2, base, rng, 300)
        self.assertAlmostEqual(diff, -0.2, places=6)
        self.assertLess(hi, 0.0)
        self.assertEqual(share, 1.0)


class LocalModelTests(unittest.TestCase):
    def _heteroscedastic(self, n=60000, seed=5):
        rng = np.random.default_rng(seed)
        yhat = rng.uniform(30, 100, n)
        residual = rng.normal(0, 0.2 * yhat)          # errors grow with predicted RUL
        return yhat, residual, np.array(["mid"] * n)

    def test_local_probability_is_closer_to_the_truth_than_the_pooled_stage(self):
        from math import erf, sqrt
        yhat, residual, stage = self._heteroscedastic()
        stage_model = ResidualModel.fit(residual, stage)
        local = LocalResidualModel.fit(yhat, residual, 5.0, stage_model)
        truth = 0.5 * (1 - erf((10.0 / 8.0) / sqrt(2)))   # yhat = 40, sd = 8, need residual >= 10
        p_local = float(local.failure_prob([40.0], ["mid"], 30.0)[0])
        p_stage = float(stage_model.failure_prob([40.0], ["mid"], 30.0)[0])
        self.assertLess(abs(p_local - truth), abs(p_stage - truth))
        self.assertAlmostEqual(p_local, truth, delta=0.03)

    def test_sparse_neighbourhood_falls_back_to_stage_residuals(self):
        yhat, residual, stage = self._heteroscedastic(n=2000)
        stage_model = ResidualModel.fit(residual, stage)
        local = LocalResidualModel.fit(yhat, residual, 5.0, stage_model, min_local=50)
        far = np.array([500.0])                              # no calibration windows anywhere near
        np.testing.assert_allclose(local.failure_prob(far, ["mid"], 30.0), stage_model.failure_prob(far, ["mid"], 30.0))

    def test_local_probabilities_are_valid(self):
        yhat, residual, stage = self._heteroscedastic(n=5000)
        local = LocalResidualModel.fit(yhat, residual, 10.0, ResidualModel.fit(residual, stage))
        p = local.failure_prob(np.linspace(0, 150, 61), ["mid"] * 61, 30.0)
        self.assertTrue(np.all((p > 0) & (p < 1)))


class ExportTests(unittest.TestCase):
    def test_saved_model_reproduces_stage_and_local_probabilities(self):
        import tempfile
        rng = np.random.default_rng(9)
        yhat = rng.uniform(5, 120, 8000)
        residual = rng.normal(0, 3 + 0.12 * yhat)
        stage = np.where(yhat > 100, "early", np.where(yhat >= 30, "mid", "late"))
        stage_model = ResidualModel.fit(residual, stage)
        local = LocalResidualModel.fit(yhat, residual, 10.0, stage_model)
        query, query_stage = np.array([20.0, 35.0, 60.0, 110.0]), np.array(["late", "mid", "mid", "early"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            save_model(stage_model, path, local=local, bandwidths=[5, 10, 20])
            stage_back, local_back = load_model(path), load_local_model(path)
            wider = load_local_model(path, bandwidth=20.0)
        np.testing.assert_allclose(stage_back.failure_prob(query, query_stage, 30.0),
                                   stage_model.failure_prob(query, query_stage, 30.0), atol=0.01)
        np.testing.assert_allclose(local_back.failure_prob(query, query_stage, 30.0),
                                   local.failure_prob(query, query_stage, 30.0), atol=0.01)
        self.assertEqual(wider.bandwidth, 20.0)

    def test_file_without_a_local_model_loads_as_none(self):
        import tempfile
        rng = np.random.default_rng(10)
        model = ResidualModel.fit(rng.normal(0, 5, 500), np.array(["mid"] * 500))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            save_model(model, path)
            self.assertIsNone(load_local_model(path))

    def test_decide_uses_the_local_model_when_given(self):
        rng = np.random.default_rng(11)
        yhat = rng.uniform(30, 100, 40000)
        residual = rng.normal(0, 0.2 * yhat)
        stage_model = ResidualModel.fit(residual, np.array(["mid"] * len(yhat)))
        local = LocalResidualModel.fit(yhat, residual, 5.0, stage_model)
        a = decide(stage_model, [40.0], ["mid"], 30.0, 10.0)
        b = decide(stage_model, [40.0], ["mid"], 30.0, 10.0, local=local)
        self.assertNotEqual(float(a["p"][0]), float(b["p"][0]))


if __name__ == "__main__":
    unittest.main()