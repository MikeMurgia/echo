"""Run from the repo root:  python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision import (  # noqa: E402
    LocalResidualModel, ResidualModel, advice_shares, check_horizon, decide, decide_triage, dollar_summary,
    horizon_is_validated, p_reliable, ratio_from_costs, triage)


def gaussian_model(n=20000, sd=10.0, seed=0):
    rng = np.random.default_rng(seed)
    return ResidualModel.fit(rng.normal(0, sd, n), np.array(["mid"] * n), min_group=30)


class TriageTests(unittest.TestCase):
    def test_thresholds(self):
        # ratio 10, band 2: score = 10p, WAIT below 0.5, ACT above 2
        advice = triage([0.01, 0.049, 0.05, 0.1, 0.2, 0.21, 0.9], 10.0, band=2.0)
        self.assertEqual(advice.tolist(), ["WAIT", "WAIT", "INVESTIGATE", "INVESTIGATE", "INVESTIGATE", "ACT", "ACT"])

    def test_act_and_wait_agree_with_the_breakeven_rule(self):
        model = gaussian_model()
        yhat = np.linspace(0.0, 120.0, 241)
        for ratio in (2.0, 10.0, 50.0):
            out = decide_triage(model, yhat, ["mid"] * len(yhat), 30.0, ratio)
            self.assertTrue(np.all(out["echo_act"][out["advice"] == "ACT"]))
            self.assertFalse(np.any(out["echo_act"][out["advice"] == "WAIT"]))
            self.assertTrue(np.array_equal(out["flagged"], out["advice"] == "INVESTIGATE"))
            self.assertTrue(np.allclose(out["score"], out["p"] * ratio))

    def test_band_of_one_leaves_no_middle_zone(self):
        advice = triage([0.09, 0.11], 10.0, band=1.0)
        self.assertEqual(advice.tolist(), ["WAIT", "ACT"])

    def test_wider_band_flags_more_windows(self):
        model = gaussian_model()
        yhat = np.linspace(0.0, 120.0, 481)
        stage = ["mid"] * len(yhat)
        shares = [advice_shares(decide_triage(model, yhat, stage, 30.0, 20.0, band=b)["advice"])["INVESTIGATE"]
                  for b in (1.5, 2.0, 4.0)]
        self.assertTrue(shares[0] <= shares[1] <= shares[2])
        self.assertLess(shares[0], 1.0)  # the middle zone does not swallow everything

    def test_shares_sum_to_one(self):
        s = advice_shares(["WAIT", "ACT", "ACT", "INVESTIGATE"])
        self.assertAlmostEqual(sum(s.values()), 1.0)
        self.assertEqual(s["ACT"], 0.5)
        self.assertEqual(sum(advice_shares([]).values()), 0.0)

    def test_bad_inputs_raise(self):
        with self.assertRaises(ValueError):
            triage([0.5], 10.0, band=0.5)
        with self.assertRaises(ValueError):
            triage([0.5], 0.0)


class HorizonTests(unittest.TestCase):
    def test_range(self):
        self.assertEqual(check_horizon(5), 5.0)
        self.assertEqual(check_horizon(60), 60.0)
        for bad in (4.9, 60.1, float("nan"), -3):
            with self.assertRaises(ValueError):
                check_horizon(bad)

    def test_validated_horizons(self):
        self.assertTrue(all(horizon_is_validated(h) for h in (15, 30, 50, 30.0)))
        self.assertFalse(horizon_is_validated(20))

    def test_any_horizon_in_range_works_and_longer_horizons_raise_p(self):
        model = gaussian_model()
        ps = [float(decide_triage(model, [40.0], ["mid"], h, 10.0)["p"][0]) for h in (10, 20, 30, 45, 60)]
        self.assertTrue(np.all(np.diff(ps) > 0))  # a bigger window can only make failure more likely

    def test_decide_triage_rejects_out_of_range_horizon(self):
        with self.assertRaises(ValueError):
            decide_triage(gaussian_model(), [40.0], ["mid"], 200.0, 10.0)


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.residual = rng.normal(0, 5, 600)
        self.stage = np.array(["early"] * 400 + ["mid"] * 180 + ["late"] * 20)
        self.model = ResidualModel.fit(self.residual, self.stage, min_group=30)
        self.yhat = np.concatenate([rng.uniform(101, 125, 400), rng.uniform(31, 99, 180), rng.uniform(1, 29, 20)])

    def test_stage_fallback_is_flagged(self):
        flags = p_reliable(self.model, [110.0, 60.0, 10.0], ["early", "mid", "late"])
        self.assertEqual(flags.tolist(), [True, True, False])

    def test_local_support_rescues_or_not(self):
        local = LocalResidualModel.fit(self.yhat, self.residual, bandwidth=10.0, fallback=self.model, min_local=50)
        self.assertTrue(local.n_nearby([110.0])[0] >= 50)
        self.assertTrue(p_reliable(self.model, [110.0], ["early"], local)[0])
        self.assertLess(local.n_nearby([10.0])[0], 50)              # few calibration windows this low
        self.assertFalse(p_reliable(self.model, [10.0], ["late"], local)[0])  # and the stage fell back too
        self.assertTrue(p_reliable(self.model, [20.0], ["mid"], local)[0])    # stage model is fine -> still reliable

    def test_n_nearby_counts_the_window_edges(self):
        local = LocalResidualModel.fit([10.0, 12.0, 14.0, 30.0], [0.0] * 4, bandwidth=2.0, fallback=self.model, min_local=1)
        self.assertEqual(local.n_nearby([12.0, 40.0]).tolist(), [3, 0])


class DollarTests(unittest.TestCase):
    def test_ratio(self):
        self.assertEqual(ratio_from_costs(5_000, 100_000), 20.0)
        for bad in ((0, 10), (10, 0), (-1, 5)):
            with self.assertRaises(ValueError):
                ratio_from_costs(*bad)

    def test_summary_numbers(self):
        s = dollar_summary(0.10, 5_000, 100_000)
        self.assertAlmostEqual(s["wait_expected"], 10_000.0)
        self.assertEqual(s["act_cost"], 5_000.0)
        self.assertAlmostEqual(s["difference"], 5_000.0)
        self.assertEqual(s["cheaper"], "act")
        self.assertEqual(dollar_summary(0.02, 5_000, 100_000)["cheaper"], "wait")

    def test_dollars_agree_with_the_ratio_rule(self):
        model = gaussian_model()
        cs, cu = 4_000.0, 120_000.0
        for yhat in (20.0, 35.0, 45.0, 70.0):
            out = decide(model, [yhat], ["mid"], 30.0, ratio_from_costs(cs, cu))
            self.assertEqual(dollar_summary(out["p"][0], cs, cu)["cheaper"] == "act", bool(out["echo_act"][0]))


if __name__ == "__main__":
    unittest.main()