"""Run from the repo root:  python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.conformal_calibration import (  # noqa: E402
    aggregate_repeats, conformal_quantile, engine_bootstrap, final_quantiles, fit_quantiles,
    repeated_split_check, stage_for_rul, stratified_engine_split)


class ConformalTests(unittest.TestCase):
    def test_marginal_coverage_on_exchangeable_residuals(self):
        rng = np.random.default_rng(0)
        cal, test = np.abs(rng.laplace(0, 5, 5000)), np.abs(rng.laplace(0, 5, 50000))
        q = conformal_quantile(cal, 0.9)
        self.assertAlmostEqual(float((test <= q).mean()), 0.9, delta=0.015)

    def test_quantile_is_infinite_when_too_few_points(self):
        self.assertEqual(conformal_quantile(np.array([1.0, 2.0, 3.0]), 0.9), float("inf"))

    def test_stage_boundaries(self):
        st = stage_for_rul(np.array([150.0, 100.0, 99.9, 30.0, 29.9]), 100, 30)
        self.assertEqual(list(st), ["early", "mid", "mid", "mid", "late"])

    def test_small_stage_falls_back_to_global(self):
        rng = np.random.default_rng(1)
        res = np.abs(rng.normal(0, 4, 400))
        stage = np.array(["early"] * 300 + ["mid"] * 95 + ["late"] * 5)
        fq = fit_quantiles(res, stage, 0.9, min_group=30)
        self.assertTrue(fq["fallback"]["late"])
        self.assertFalse(fq["fallback"]["early"])
        self.assertEqual(fq["stage"]["late"], fq["global"])

    def test_engine_split_is_disjoint_complete_and_stratified(self):
        rng = np.random.default_rng(2)
        units = np.arange(1, 41)
        final = rng.integers(1, 200, 40).astype(float)
        cal, ev = stratified_engine_split(units, final, seed=3)
        self.assertEqual(cal & ev, set())
        self.assertEqual(cal | ev, set(int(u) for u in units))
        self.assertEqual((len(cal), len(ev)), (20, 20))
        # each consecutive pair (in final-RUL order) is split across the halves
        order = units[np.argsort(final, kind="stable")]
        for i in range(0, 40, 2):
            self.assertEqual(len({int(order[i]) in cal, int(order[i + 1]) in cal}), 2)

    def test_bootstrap_ci_brackets_the_point_estimate(self):
        rng = np.random.default_rng(4)
        unit = np.repeat(np.arange(30), 20)
        covered = rng.random(600) < 0.9
        lo, hi = engine_bootstrap(unit, covered, np.random.default_rng(5), 500)
        self.assertLessEqual(lo, covered.mean())
        self.assertGreaterEqual(hi, covered.mean())


def _fake_windows(n_units=60, per_unit=40, seed=0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        for c in range(per_unit):
            y = float(rng.uniform(1, 125))
            yhat = y + rng.normal(0, 5 + y / 10)
            rows.append((u, c, y, yhat))
    df = pd.DataFrame(rows, columns=["unit", "cycle", "y", "yhat"])
    df["abs_res"] = (df["yhat"] - df["y"]).abs()
    df["stage"] = stage_for_rul(df["yhat"].to_numpy(), 100.0, 30.0)
    df["y_raw"] = df["y"]
    return df


class RepeatTests(unittest.TestCase):
    class Args:
        repeats, seed, coverage, min_group = 4, 1, 0.9, 30
        sweep = [0.8, 0.9, 0.95]
        early_rul, late_rul, checkpoint = 100.0, 30.0, "x.pt"

    def test_repeats_cover_near_target_and_vary(self):
        win = _fake_windows()
        final = win.groupby("unit")["y_raw"].last()
        rows, q = repeated_split_check(win, final, self.Args)
        agg = aggregate_repeats(rows, 0.9)
        overall = agg[(agg.method == "mondrian") & (agg.slice == "overall")].iloc[0]
        self.assertEqual(int(overall.splits), 4)
        self.assertAlmostEqual(overall.coverage_mean, 0.9, delta=0.04)
        self.assertEqual(len(q), 4)
        self.assertGreater(q["q_late"].nunique(), 1)

    def test_final_quantiles_levels_are_ordered_and_use_all_engines(self):
        win = _fake_windows()
        meta = final_quantiles(win, self.Args, "FD002", "transformer")
        self.assertEqual(meta["calibration_engines_total"], 60)
        qs = [meta["levels"][k]["q_global"] for k in ("0.8", "0.9", "0.95")]
        self.assertLess(qs[0], qs[1])
        self.assertLess(qs[1], qs[2])


if __name__ == "__main__":
    unittest.main()