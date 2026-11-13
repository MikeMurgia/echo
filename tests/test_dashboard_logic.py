"""Run from the repo root:  python -m unittest discover tests"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard_logic import conformal_halfwidths, engine_frame, holds_up, load_bundle, most_urgent_unit  # noqa: E402
from src.decision import LocalResidualModel, ResidualModel, save_model, stage_for_rul  # noqa: E402


def fake_windows(n_units=40, seed=0):
    """Small stand-in for conformal_windows.csv: true RUL falls from 120 to 10, prediction noise sd 8."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, n_units + 1):
        for cycle in range(30, 141):
            y_raw = 150.0 - cycle
            y = min(y_raw, 125.0)
            yhat = max(y + rng.normal(0, 8), 0.0)
            rows.append((unit, cycle, 140, y_raw, y, yhat, yhat - y, abs(yhat - y), str(stage_for_rul(yhat)),
                         "cal" if unit % 2 == 0 else "eval"))
    df = pd.DataFrame(rows, columns=["unit", "cycle", "last_cycle", "y_raw", "y", "yhat", "residual", "abs_res",
                                     "stage", "split"])
    df["hw_mondrian"] = df["stage"].map(conformal_halfwidths(df[df["split"] == "cal"]))
    return df


class DashboardLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        folder = Path(cls.tmp.name) / "results" / "FD999"
        folder.mkdir(parents=True)
        cls.win = fake_windows()
        cls.win.to_csv(folder / "conformal_windows.csv", index=False)
        model = ResidualModel.fit(cls.win["residual"], cls.win["stage"])
        local = LocalResidualModel.fit(cls.win["yhat"], cls.win["residual"], 10.0, model)
        save_model(model, folder / "decision_model_final.json", local=local)
        cls.bundle = load_bundle(cls.tmp.name, "FD999")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_missing_results_give_none(self):
        self.assertIsNone(load_bundle(self.tmp.name, "FD123"))

    def test_missing_columns_raise(self):
        folder = Path(self.tmp.name) / "results" / "FD998"
        folder.mkdir(parents=True)
        self.win.drop(columns=["hw_mondrian"]).to_csv(folder / "conformal_windows.csv", index=False)
        (folder / "decision_model_final.json").write_text((Path(self.tmp.name) / "results" / "FD999" / "decision_model_final.json").read_text())
        with self.assertRaises(ValueError):
            load_bundle(self.tmp.name, "FD998")

    def test_halfwidths_cover_at_least_the_level(self):
        hw = conformal_halfwidths(self.win)
        covered = self.win["abs_res"] <= self.win["stage"].map(hw)
        for s in ("early", "mid", "late"):
            m = self.win["stage"] == s
            if m.sum() >= 30:
                self.assertGreaterEqual(covered[m].mean(), 0.9 - 1e-9)

    def test_engine_frame_shape_and_interval(self):
        f = engine_frame(self.bundle, 3, horizon=30, ratio=20)
        self.assertEqual(len(f), 111)
        self.assertTrue((f["cycle"].diff().dropna() > 0).all())
        self.assertTrue(((f["lo"] <= f["yhat"]) & (f["yhat"] <= f["hi"]) & (f["lo"] >= 0)).all())
        self.assertTrue(f["advice"].isin(["WAIT", "INVESTIGATE", "ACT"]).all())
        self.assertTrue(((f["p"] > 0) & (f["p"] < 1)).all())

    def test_advice_moves_from_wait_to_act_as_the_engine_ages(self):
        f = engine_frame(self.bundle, 3, horizon=30, ratio=20)
        self.assertEqual(f["advice"].iloc[0], "WAIT")
        self.assertEqual(f["advice"].iloc[-1], "ACT")

    def test_most_urgent_unit_has_the_lowest_latest_prediction(self):
        w = self.bundle["windows"]
        w2 = w.copy()
        w2.loc[(w2["unit"] == 7) & (w2["cycle"] == 140), "yhat"] = -1.0
        b = dict(self.bundle, windows=w2)
        self.assertEqual(most_urgent_unit(b), 7)

    def test_unknown_engine_and_bad_horizon(self):
        with self.assertRaises(KeyError):
            engine_frame(self.bundle, 9999, 30, 20)
        with self.assertRaises(ValueError):
            engine_frame(self.bundle, 3, 200, 20)

    def test_holds_up(self):
        h = holds_up(self.bundle, "mid", horizon=30, ratio=20)
        self.assertTrue(h["available"])
        self.assertGreater(h["coverage"], 0.8)
        self.assertLessEqual(h["cost_echo"], h["cost_point"] + 1e-9)
        self.assertLess(h["change_pct"], 0.0)
        self.assertTrue(0.0 <= h["investigate_share"] < 1.0)

    def test_higher_ratio_makes_waiting_dearer_for_the_point_rule(self):
        a = holds_up(self.bundle, "mid", 30, 5)["cost_point"]
        b = holds_up(self.bundle, "mid", 30, 50)["cost_point"]
        self.assertGreater(b, a)


if __name__ == "__main__":
    unittest.main()