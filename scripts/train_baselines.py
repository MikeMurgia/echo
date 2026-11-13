"""Random forest and XGBoost baselines on the same windows and engine split as the sequence models.

Each 30-cycle window becomes one feature vector: the last cycle's sensors, plus the
window mean, std and slope of every sensor. Writes
results/<subset>/<model>_seed<seed>_metrics.json in the same format as scripts/train.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import SlidingWindowDataset, read_normalized, sensor_columns, split_engine_ids  # noqa: E402


def rmse(y, pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(y)) ** 2)))


def nasa_score(y, pred) -> float:
    d = np.asarray(pred) - np.asarray(y)
    return float(np.sum(np.where(d < 0, np.expm1(-d / 13.0), np.expm1(d / 10.0))))


def window_features(dataset: SlidingWindowDataset) -> tuple[np.ndarray, np.ndarray]:
    """Flatten every window into [last, mean, std, slope] features."""
    rows, targets = [], []
    for i in range(len(dataset)):
        window, target = dataset[i]
        w = window.numpy()
        slope = (w[-1] - w[0]) / max(len(w) - 1, 1)
        rows.append(np.concatenate([w[-1], w.mean(axis=0), w.std(axis=0), slope]))
        targets.append(float(target))
    return np.asarray(rows, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def get_baselines(seed: int, fast: bool) -> dict:
    models = {"random_forest": RandomForestRegressor(
        n_estimators=60 if fast else 300, min_samples_leaf=5, n_jobs=-1, random_state=seed)}
    try:
        from xgboost import XGBRegressor
        models["xgboost"] = XGBRegressor(
            n_estimators=150 if fast else 600, learning_rate=0.05, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, n_jobs=-1, random_state=seed)
    except ImportError:
        print("xgboost not installed; running random forest only (pip install xgboost)")
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=("FD001", "FD002"), default="FD002")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42, help="use the same seed as the sequence model run you compare to")
    parser.add_argument("--fast", action="store_true", help="smaller models for a quick check")
    args = parser.parse_args()

    train_frame = read_normalized(args.data_dir, "train", args.subset)
    test_frame = read_normalized(args.data_dir, "test", args.subset)
    features = sensor_columns(train_frame)
    train_ids, val_ids = split_engine_ids(train_frame, args.validation_fraction, args.seed)
    common = dict(window_size=args.window_size, feature_columns=features, stride=args.stride)
    X_tr, y_tr = window_features(SlidingWindowDataset(train_frame[train_frame.unit.isin(train_ids)], **common))
    X_va, y_va = window_features(SlidingWindowDataset(train_frame[train_frame.unit.isin(val_ids)], **common))
    X_te, y_te = window_features(SlidingWindowDataset(test_frame, **common, final_window_only=True))
    y_te_raw = (test_frame.groupby("unit")["RUL_raw"].last().to_numpy(dtype=np.float32)
                if "RUL_raw" in test_frame.columns else None)
    print(f"{args.subset}: {X_tr.shape[0]} train, {X_va.shape[0]} validation, {X_te.shape[0]} test windows, "
          f"{X_tr.shape[1]} features")

    out_dir = args.results_dir / args.subset
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, model in get_baselines(args.seed, args.fast).items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        p_va, p_te = model.predict(X_va), model.predict(X_te)
        result = {
            "model": name, "tag": "default", "subset": args.subset, "seed": args.seed,
            "val_rmse": rmse(y_va, p_va), "val_nasa_score": nasa_score(y_va, p_va),
            "test_rmse_capped": rmse(y_te, p_te), "test_nasa_score_capped": nasa_score(y_te, p_te),
            "train_seconds": time.time() - t0, "window_size": args.window_size, "stride": args.stride,
        }
        if y_te_raw is not None and len(y_te_raw) == len(p_te):
            result.update({"test_rmse_raw": rmse(y_te_raw, p_te), "test_nasa_score_raw": nasa_score(y_te_raw, p_te)})
        (out_dir / f"{name}_seed{args.seed}_metrics.json").write_text(json.dumps(result, indent=2))
        print(f"{name}: val_RMSE={result['val_rmse']:.3f} test_RMSE(capped)={result['test_rmse_capped']:.3f}"
              + (f" test_RMSE(raw)={result['test_rmse_raw']:.3f}" if "test_rmse_raw" in result else "")
              + f" ({result['train_seconds']:.0f}s)")


if __name__ == "__main__":
    main()