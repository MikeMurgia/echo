"""Score engines uploaded to the dashboard: parse, check, normalize like the NASA data, predict RUL.

Importing this module needs only numpy and pandas. torch, joblib and scikit-learn are imported inside the
functions that need them, so the dashboard still starts where they are not installed.

The file must have the raw NASA C-MAPSS layout: one row per engine and cycle, no header, 26 numeric columns
(engine id, cycle, three operating settings, 21 sensors), separated by spaces or commas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.decision import stage_for_rul

SETTING_COLS = ["op1", "op2", "op3"]
RAW_COLUMNS = ["unit", "cycle"] + SETTING_COLS + [f"s{i}" for i in range(1, 22)]
MAX_BYTES = 15 * 1024 * 1024
MAX_ENGINES = 200
MAX_WINDOWS = 40_000
SETTINGS_FAR = {"clustered": 1.0, "single": 5.0}  # standardized distance beyond which a row looks unlike training


@dataclass
class UploadCheck:
    """What the file checks found. `errors` block prediction; `warnings` do not."""

    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    skipped_units: list = field(default_factory=list)
    engines: int = 0
    rows: int = 0
    windows: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def _looks_numeric(tokens) -> bool:
    try:
        [float(t) for t in tokens]
        return True
    except ValueError:
        return False


def parse_upload(raw: bytes, window_size: int = 30) -> tuple[pd.DataFrame | None, UploadCheck]:
    """Read and check an uploaded file. Returns (frame or None, check). Never raises on bad content."""
    check = UploadCheck()
    if len(raw) > MAX_BYTES:
        check.errors.append(f"File is larger than {MAX_BYTES // (1024 * 1024)} MB.")
        return None, check
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        check.errors.append("File is not plain text (could not decode it as UTF-8).")
        return None, check
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        check.errors.append("File is empty.")
        return None, check
    import re
    rows = [re.split(r"[,\s]+", ln) for ln in lines]
    if not _looks_numeric(rows[0]):
        rows = rows[1:]
        check.warnings.append("The first row is not numeric, so it was treated as a header and ignored.")
    widths = {len(r) for r in rows}
    if widths != {len(RAW_COLUMNS)}:
        found = sorted(widths)
        check.errors.append(f"Expected {len(RAW_COLUMNS)} columns per row (engine id, cycle, 3 settings, 21 sensors), "
                            f"found {found}. Use the raw NASA layout with no header and no dropped sensors.")
        return None, check
    try:
        frame = pd.DataFrame(rows, columns=RAW_COLUMNS).astype(float)
    except ValueError:
        check.errors.append("Some values are not numbers.")
        return None, check
    if not np.isfinite(frame.to_numpy()).all():
        check.errors.append("Some values are missing or not finite.")
        return None, check
    if not ((frame["unit"] % 1 == 0).all() and (frame["cycle"] % 1 == 0).all() and (frame["cycle"] >= 1).all()):
        check.errors.append("Engine ids and cycles must be whole numbers, and cycles start at 1.")
        return None, check
    frame["unit"], frame["cycle"] = frame["unit"].astype(int), frame["cycle"].astype(int)
    bad = [int(u) for u, g in frame.groupby("unit") if not (g["cycle"].diff().dropna() > 0).all()]
    if bad:
        check.errors.append(f"Cycles must increase within each engine. Problem engines: {bad[:5]}"
                            + (" and more." if len(bad) > 5 else "."))
        return None, check
    counts = frame.groupby("unit")["cycle"].size()
    check.engines, check.rows = int(len(counts)), int(len(frame))
    if check.engines > MAX_ENGINES:
        check.errors.append(f"At most {MAX_ENGINES} engines per file (found {check.engines}).")
        return None, check
    short = counts[counts < window_size]
    if len(short):
        check.skipped_units = [int(u) for u in short.index]
        check.warnings.append(f"{len(short)} engine(s) have fewer than {window_size} cycles and are skipped: "
                              f"{check.skipped_units[:10]}" + (" and more." if len(short) > 10 else "."))
        frame = frame[~frame["unit"].isin(short.index)].reset_index(drop=True)
        counts = counts.drop(short.index)
    if frame.empty:
        check.errors.append(f"No engine has at least {window_size} cycles.")
        return None, check
    check.windows = int((counts - window_size + 1).sum())
    if check.windows > MAX_WINDOWS:
        check.errors.append(f"Too many windows to score here ({check.windows:,}; limit {MAX_WINDOWS:,}). Upload fewer engines.")
        return None, check
    return frame, check


def settings_warning(frame: pd.DataFrame, preproc: dict) -> str | None:
    """A warning if many rows have operating settings unlike anything seen in training (heuristic)."""
    z = preproc["setting_scaler"].transform(frame[preproc["setting_cols"]])
    clusterer = preproc.get("clusterer")
    if clusterer is not None:
        dist = clusterer.transform(z).min(axis=1)
        far = dist > SETTINGS_FAR["clustered"]
    else:
        far = np.abs(z).max(axis=1) > SETTINGS_FAR["single"]
    share = float(far.mean())
    if share > 0.01:
        return (f"{100 * share:.0f}% of rows have operating settings unlike the training data. "
                "Predictions for those rows may be unreliable.")
    return None


def normalize_engines(frame: pd.DataFrame, preproc: dict) -> pd.DataFrame:
    """Normalize raw rows exactly as scripts/normalize_data.py did for the test set (train-fitted regimes and scalers)."""
    sensors, settings = list(preproc["sensors"]), list(preproc["setting_cols"])
    missing = sorted((set(sensors) | set(settings)) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = frame[["unit", "cycle"] + settings + sensors].copy()
    out[sensors] = out[sensors].astype(float)
    z = preproc["setting_scaler"].transform(out[settings])
    clusterer = preproc.get("clusterer")
    regimes = clusterer.predict(z) if clusterer is not None else np.zeros(len(out), dtype=int)
    out["regime"] = regimes
    for regime, scaler in preproc["regime_scalers"].items():
        mask = regimes == regime
        if mask.any():
            out.loc[mask, sensors] = scaler.transform(frame.loc[mask, sensors])
    return out


def predict_windows(norm: pd.DataFrame, model, window_size: int, feature_columns, batch_size: int = 512,
                    stride: int = 1) -> pd.DataFrame:
    """RUL prediction for the window ending at every cycle (from cycle `window_size` on) of every engine.

    Returns a table shaped like results/<subset>/conformal_windows.csv, with no true RUL.
    """
    import torch
    from torch.utils.data import DataLoader
    from src.dataset import SlidingWindowDataset

    data = norm.copy()
    data["RUL"] = 0.0  # placeholder: the dataset class wants a label column; it is not used for prediction
    ds = SlidingWindowDataset(data, window_size=window_size, feature_columns=list(feature_columns), stride=stride)
    model.eval()
    preds = []
    with torch.inference_mode():
        for x, _ in DataLoader(ds, batch_size=batch_size, shuffle=False):
            preds.append(model(x).reshape(-1).cpu().numpy())
    yhat = np.concatenate(preds)
    meta = {int(u): (g["cycle"].to_numpy(), g["regime"].to_numpy()) for u, g in data.sort_values(["unit", "cycle"]).groupby("unit")}
    rows = []
    for engine_idx, start in ds.index:
        unit, values, _ = ds.engines[engine_idx]
        end = min(start + window_size, len(values))
        cycles, regimes = meta[int(unit)]
        rows.append((int(unit), int(cycles[end - 1]), len(values), int(regimes[end - 1])))
    win = pd.DataFrame(rows, columns=["unit", "cycle", "last_cycle", "regime"])
    nan = np.full(len(win), np.nan)
    win["y_raw"], win["y"], win["yhat"] = nan, nan, yhat
    win["residual"], win["abs_res"] = nan, nan
    win["stage"] = stage_for_rul(win["yhat"].to_numpy())
    win["split"], win["hw_mondrian"] = "upload", nan
    return win


@dataclass
class Runtime:
    """What scoring needs for one dataset: preprocessing, the calibrated model, and how it was found."""

    preproc: dict
    model: object
    window_size: int
    feature_columns: list
    checkpoint_name: str


def find_checkpoint(root, subset: str) -> Path | None:
    """The checkpoint the conformal step was calibrated with (from results/<subset>/conformal_quantiles.json)."""
    import json
    meta = Path(root) / "results" / subset / "conformal_quantiles.json"
    if not meta.exists():
        return None
    name = str(json.loads(meta.read_text()).get("checkpoint", "")).replace("\\", "/").split("/")[-1]
    path = Path(root) / "results" / "checkpoints" / name
    return path if name and path.exists() else None


def runtime_status(root, subset: str) -> tuple[bool, str]:
    """(ready, message): can this machine score uploads for `subset`?"""
    try:
        import torch  # noqa: F401
        import joblib  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError as err:
        return False, f"Uploading needs PyTorch, joblib and scikit-learn on this machine ({err.name} is missing)."
    if not (Path(root) / "data" / f"preproc_{subset}.joblib").exists():
        return False, f"Missing data/preproc_{subset}.joblib. Run scripts/normalize_data.py first."
    if find_checkpoint(root, subset) is None:
        return False, (f"Could not find the model checkpoint used for calibration. It is named in "
                       f"results/{subset}/conformal_quantiles.json and should be in results/checkpoints/.")
    return True, ""


def load_runtime(root, subset: str) -> Runtime:
    import joblib
    import torch
    from src.models import build_model

    preproc = joblib.load(Path(root) / "data" / f"preproc_{subset}.joblib")
    path = find_checkpoint(root, subset)
    if path is None:
        raise FileNotFoundError("Checkpoint not found")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    features = list(ckpt.get("feature_columns") or preproc["sensors"])
    if int(ckpt["input_size"]) != len(features):
        raise ValueError("Checkpoint input size does not match its feature list")
    model = build_model(ckpt.get("model", "transformer"), input_size=int(ckpt["input_size"]),
                        d_model=int(ckpt.get("d_model", 128)), num_layers=int(ckpt.get("num_layers", 3)),
                        dropout=float(ckpt.get("dropout", 0.2)))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return Runtime(preproc, model, int(ckpt["window_size"]), features, path.name)


def score_upload(frame: pd.DataFrame, runtime: Runtime) -> pd.DataFrame:
    """Normalize and predict for a parsed upload. Returns the windows table."""
    norm = normalize_engines(frame, runtime.preproc)
    return predict_windows(norm, runtime.model, runtime.window_size, runtime.feature_columns)


def upload_bundle(reference: dict, windows: pd.DataFrame) -> dict:
    """A dashboard bundle for uploaded engines that reuses the NASA bundle's calibration (models, half-widths)."""
    return dict(reference, windows=windows, units=sorted(int(u) for u in windows["unit"].unique()), uploaded=True)