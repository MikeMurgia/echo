"""Normalize C-MAPSS sensor readings within operating regimes.

Run from the repository root with::

    python scripts/normalize_data.py

For each subset, operating regimes and per-regime sensor means/standard
deviations are learned from training rows only. Test rows use those fitted
transformations. Outputs retain unit, cycle, settings, and RUL (when present),
and add a ``regime`` column for downstream window generation.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SUBSETS = ("FD001", "FD002")
DEFAULT_REGIMES = {"FD001": 1, "FD002": 6}
SETTING_COLS = ("op1", "op2", "op3")
META_COLS = ("unit", "cycle", "RUL", "RUL_raw", "regime")


def read_input(data_dir, split, subset):
    """Prefer RUL-labeled clean data, falling back to clean data."""
    candidates = (
        data_dir / f"rul_clean_{split}_{subset}.txt",
        data_dir / f"clean_{split}_{subset}.txt",
    )
    for path in candidates:
        if path.exists():
            return pd.read_csv(path, sep=r"\s+"), path
    raise FileNotFoundError(
        f"No cleaned {split} data for {subset}; expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def sensor_columns(df):
    return [column for column in df.columns if column.startswith("s") and column[1:].isdigit()]


def normalize_subset(train, test, subset, n_regimes=None):
    """Fit regime assignment and sensor standardization on train, apply to both."""
    if n_regimes is None:
        n_regimes = DEFAULT_REGIMES[subset]

    setting_cols = [column for column in SETTING_COLS if column in train.columns]
    if not setting_cols:
        raise ValueError(f"{subset} has no operating setting columns")
    sensors = sensor_columns(train)
    if not sensors:
        raise ValueError(f"{subset} has no sensor columns")

    missing = sorted((set(setting_cols) | set(sensors)) - set(test.columns))
    if missing:
        raise ValueError(f"{subset} test data is missing train columns: {missing}")
    if n_regimes < 1 or n_regimes > len(train):
        raise ValueError(f"Invalid regime count {n_regimes} for {subset} ({len(train)} rows)")

    # Scaling the settings before clustering prevents op1's larger numeric
    # range from dominating the distance calculation. Both steps see train only.
    setting_scaler = StandardScaler().fit(train[setting_cols])
    train_settings = setting_scaler.transform(train[setting_cols])
    test_settings = setting_scaler.transform(test[setting_cols])
    if n_regimes == 1:
        train_regimes = np.zeros(len(train), dtype=int)
        test_regimes = np.zeros(len(test), dtype=int)
    else:
        clusterer = KMeans(n_clusters=n_regimes, n_init=10, random_state=0)
        train_regimes = clusterer.fit_predict(train_settings)
        test_regimes = clusterer.predict(test_settings)

    clusterer_fitted = None if n_regimes == 1 else clusterer
    regime_scalers = {}
    train_out, test_out = train.copy(), test.copy()
    # Some sensor columns contain integer-valued raw readings; normalized
    # values are fractional, so make the output columns floating point first.
    train_out[sensors] = train_out[sensors].astype(float)
    test_out[sensors] = test_out[sensors].astype(float)
    train_out["regime"] = train_regimes
    test_out["regime"] = test_regimes

    # Fit a separate sensor standardizer using only training rows assigned to
    # each regime. Centering/scaling test data with these same values avoids
    # leakage and removes condition-specific baseline offsets.
    for regime in range(n_regimes):
        mask = train_regimes == regime
        if not mask.any():
            raise ValueError(f"{subset} regime {regime} has no training rows")
        scaler = StandardScaler().fit(train.loc[mask, sensors])
        regime_scalers[regime] = scaler
        train_out.loc[mask, sensors] = scaler.transform(train.loc[mask, sensors])
        test_mask = test_regimes == regime
        if test_mask.any():
            test_out.loc[test_mask, sensors] = scaler.transform(test.loc[test_mask, sensors])

    # Everything needed to preprocess a NEW engine the same way (used by the dashboard).
    bundle = {
        "subset": subset,
        "n_regimes": n_regimes,
        "setting_cols": setting_cols,
        "sensors": sensors,
        "setting_scaler": setting_scaler,
        "clusterer": clusterer_fitted,
        "regime_scalers": regime_scalers,
    }
    return train_out, test_out, setting_cols, sensors, bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--fd001-regimes", type=int, default=DEFAULT_REGIMES["FD001"],
        help="number of FD001 operating regimes (default: 1)",
    )
    parser.add_argument(
        "--fd002-regimes", type=int, default=DEFAULT_REGIMES["FD002"],
        help="number of FD002 operating regimes (default: 6)",
    )
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    regimes = {"FD001": args.fd001_regimes, "FD002": args.fd002_regimes}

    for subset in SUBSETS:
        try:
            train, train_path = read_input(data_dir, "train", subset)
            test, test_path = read_input(data_dir, "test", subset)
        except FileNotFoundError as err:
            print(f"{subset}: skipped ({err})")
            continue
        train_norm, test_norm, settings, sensors, bundle = normalize_subset(
            train, test, subset, regimes[subset]
        )
        train_out = data_dir / f"norm_{train_path.name}"
        test_out = data_dir / f"norm_{test_path.name}"
        joblib.dump(bundle, data_dir / f"preproc_{subset}.joblib")
        train_norm.to_csv(train_out, sep=" ", index=False, float_format="%.8g")
        test_norm.to_csv(test_out, sep=" ", index=False, float_format="%.8g")
        print(
            f"{subset}: {regimes[subset]} regime(s) from {settings}; "
            f"standardized {len(sensors)} sensors using train-only statistics.\n"
            f"  saved {train_out.name} ({len(train_norm)} rows)\n"
            f"  saved {test_out.name} ({len(test_norm)} rows)\n"
            f"  saved preproc_{subset}.joblib (fitted regimes + scalers)"
        )


if __name__ == "__main__":
    main()