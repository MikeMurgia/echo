"""Run from the repo root:  python -m unittest discover tests"""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard_logic import latest_table  # noqa: E402
from src.decision import LocalResidualModel, ResidualModel  # noqa: E402
from src.upload_logic import (  # noqa: E402
    MAX_BYTES, RAW_COLUMNS, find_checkpoint, normalize_engines, parse_upload, runtime_status, settings_warning,
    upload_bundle)

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


def fake_raw(units=(1, 2), n_cycles=40, seed=0, regimes=2):
    """Raw NASA-layout rows: 3 settings (one of `regimes` fixed combinations per row) and 21 noisy sensors."""
    rng = np.random.default_rng(seed)
    combos = np.array([[10.0 * (r + 1), 0.1 * (r + 1), 100.0] for r in range(regimes)])
    rows = []
    for u in units:
        for c in range(1, n_cycles + 1):
            rows.append([u, c, *combos[rng.integers(0, regimes)], *(rng.normal(500, 20, 21) + c)])
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def as_text(frame, sep=" ", trailing=""):
    return ("\n".join(sep.join(f"{v:g}" if i < 2 else f"{v:.6f}" for i, v in enumerate(r)) + trailing
                      for r in frame.to_numpy())).encode()


class ParseTests(unittest.TestCase):
    def test_valid_space_separated_with_trailing_spaces(self):
        frame, check = parse_upload(as_text(fake_raw(), trailing=" "))
        self.assertTrue(check.ok)
        self.assertEqual((check.engines, check.rows, check.windows), (2, 80, 2 * (40 - 30 + 1)))
        self.assertEqual(list(frame.columns), RAW_COLUMNS)
        self.assertEqual(frame["unit"].dtype.kind, "i")

    def test_comma_separated_and_header_row(self):
        text = (",".join(RAW_COLUMNS) + "\n").encode() + as_text(fake_raw(), sep=",")
        frame, check = parse_upload(text)
        self.assertTrue(check.ok)
        self.assertEqual(len(frame), 80)
        self.assertTrue(any("header" in w for w in check.warnings))

    def test_wrong_number_of_columns(self):
        frame, check = parse_upload(as_text(fake_raw().iloc[:, :22]))
        self.assertIsNone(frame)
        self.assertFalse(check.ok)
        self.assertIn("26", check.errors[0])

    def test_non_numeric_value(self):
        text = as_text(fake_raw()).replace(b"500", b"abc", 1)
        frame, check = parse_upload(text)
        self.assertIsNone(frame)
        self.assertFalse(check.ok)

    def test_cycles_must_increase(self):
        df = fake_raw()
        df.loc[5, "cycle"] = 2
        frame, check = parse_upload(as_text(df))
        self.assertIsNone(frame)
        self.assertIn("increase", check.errors[0])

    def test_short_engines_are_skipped_and_listed(self):
        df = pd.concat([fake_raw(units=(1,), n_cycles=40), fake_raw(units=(2,), n_cycles=10)], ignore_index=True)
        frame, check = parse_upload(as_text(df))
        self.assertTrue(check.ok)
        self.assertEqual(check.skipped_units, [2])
        self.assertEqual(sorted(frame["unit"].unique().tolist()), [1])

    def test_all_engines_too_short(self):
        frame, check = parse_upload(as_text(fake_raw(n_cycles=10)))
        self.assertIsNone(frame)
        self.assertFalse(check.ok)

    def test_empty_binary_and_oversized(self):
        for raw in (b"", b"\xff\xfe\x00\x01", b"x" * (MAX_BYTES + 1)):
            frame, check = parse_upload(raw)
            self.assertIsNone(frame)
            self.assertFalse(check.ok)


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn not installed")
class NormalizeTests(unittest.TestCase):
    def test_matches_the_training_pipeline(self):
        from scripts.normalize_data import normalize_subset
        train = fake_raw(units=(1, 2, 3, 4), n_cycles=60, seed=1)
        test = fake_raw(units=(5, 6), n_cycles=45, seed=2)
        keep = ["unit", "cycle", "op1", "op2", "op3", "s2", "s3", "s4", "s7"]
        _, test_ref, _, sensors, bundle = normalize_subset(train[keep], test[keep], "FDX", n_regimes=2)
        mine = normalize_engines(test, bundle)
        self.assertEqual(mine["regime"].tolist(), test_ref["regime"].tolist())
        self.assertTrue(np.allclose(mine[sensors].to_numpy(), test_ref[sensors].to_numpy(), atol=1e-9))

    def test_unlike_training_settings_are_flagged(self):
        from scripts.normalize_data import normalize_subset
        train = fake_raw(units=(1, 2, 3, 4), n_cycles=60, seed=1)
        keep = ["unit", "cycle", "op1", "op2", "op3", "s2", "s3"]
        _, _, _, _, bundle = normalize_subset(train[keep], train[keep], "FDX", n_regimes=2)
        self.assertIsNone(settings_warning(train, bundle))
        odd = fake_raw(units=(9,), n_cycles=40, seed=3)
        odd["op1"] = 80.0
        self.assertIn("unlike the training data", settings_warning(odd, bundle))


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class PredictTests(unittest.TestCase):
    def test_windows_end_at_every_cycle_from_the_window_size_and_match_the_model(self):
        import torch
        from src.models import build_model
        from src.upload_logic import predict_windows

        torch.manual_seed(0)
        model = build_model("transformer", input_size=3, d_model=16, num_layers=1, dropout=0.0)
        rng = np.random.default_rng(0)
        rows = [(u, c, 0, *rng.normal(size=3)) for u in (1, 2) for c in range(1, 41 if u == 1 else 36)]
        norm = pd.DataFrame(rows, columns=["unit", "cycle", "regime", "s2", "s3", "s4"])
        win = predict_windows(norm, model, window_size=30, feature_columns=["s2", "s3", "s4"])
        self.assertEqual(win.groupby("unit")["cycle"].min().tolist(), [30, 30])
        self.assertEqual(win.groupby("unit")["cycle"].max().tolist(), [40, 35])
        self.assertEqual(len(win), 11 + 6)
        self.assertTrue(win["y"].isna().all() and (win["split"] == "upload").all())
        first = norm[norm.unit == 1].iloc[:30][["s2", "s3", "s4"]].to_numpy(dtype=np.float32)
        model.eval()
        with torch.no_grad():
            direct = float(model(torch.from_numpy(first)[None]).reshape(-1)[0])
        self.assertAlmostEqual(float(win["yhat"].iloc[0]), direct, places=5)


class RuntimeStatusTests(unittest.TestCase):
    def test_checkpoint_lookup_handles_windows_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "results" / "FD002").mkdir(parents=True)
            (root / "results" / "checkpoints").mkdir(parents=True)
            self.assertIsNone(find_checkpoint(root, "FD002"))
            (root / "results" / "FD002" / "conformal_quantiles.json").write_text(
                '{"checkpoint": "results\\\\checkpoints\\\\transformer_FD002_seed1_best.pt"}')
            self.assertIsNone(find_checkpoint(root, "FD002"))  # named but the file is not there
            (root / "results" / "checkpoints" / "transformer_FD002_seed1_best.pt").write_bytes(b"x")
            self.assertEqual(find_checkpoint(root, "FD002").name, "transformer_FD002_seed1_best.pt")

    def test_missing_pieces_give_a_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok, message = runtime_status(tmp, "FD002")
            self.assertFalse(ok)
            self.assertTrue(message)


class LatestTableTests(unittest.TestCase):
    def test_one_row_per_engine_at_its_latest_cycle(self):
        rng = np.random.default_rng(0)
        resid = rng.normal(0, 8, 3000)
        yh = rng.uniform(0, 120, 3000)
        stage = np.where(yh > 100, "early", np.where(yh >= 30, "mid", "late"))
        model = ResidualModel.fit(resid, stage)
        local = LocalResidualModel.fit(yh, resid, 10.0, model)
        win = pd.DataFrame({"unit": [1, 1, 1, 2, 2], "cycle": [30, 31, 32, 30, 31], "yhat": [90.0, 50.0, 8.0, 100.0, 20.0],
                            "y": np.nan, "y_raw": np.nan})
        bundle = upload_bundle({"model": model, "local": local, "halfwidth": {"early": 20.0, "mid": 15.0, "late": 9.0}}, win)
        table = latest_table(bundle, horizon=30, ratio=20)
        self.assertEqual(table["engine"].tolist(), [1, 2])
        self.assertEqual(table["last_cycle"].tolist(), [32, 31])
        self.assertEqual(table["predicted_rul"].tolist(), [8.0, 20.0])
        self.assertEqual(table["interval_low"].tolist(), [0.0, 11.0])  # engine 2: yhat 20 is late stage, half-width 9
        self.assertTrue(set(table["advice"]) <= {"WAIT", "INVESTIGATE", "ACT"})
        self.assertEqual(table.loc[0, "advice"], "ACT")  # predicted RUL 8 with H = 30
        self.assertTrue(bundle["uploaded"])
        self.assertEqual(bundle["units"], [1, 2])


if __name__ == "__main__":
    unittest.main()