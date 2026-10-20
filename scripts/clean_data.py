

#removes non-informative sensors : sensors with 0 value (1,5,6,10,16,18,19) for all

import pandas as pd
from pathlib import Path

INDEX_COLS = ["unit", "cycle"]
SETTING_COLS = ["op1", "op2", "op3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLUMNS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# Sensors that are always dropped
DROP_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]


def load_cmapss(path):
    """Load any CMAPSS train/test file (FD001-FD004)."""
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def clean_file(path):
    """Drop the fixed sensors and save clean_<filename> in the same directory."""
    path = Path(path)
    df = load_cmapss(path)

    df_clean = df.drop(columns=DROP_SENSORS)
    out_path = path.with_name(f"clean_{path.name}")
    df_clean.to_csv(out_path, sep=" ", index=False)   # space-separated, with header

    print(f"{path.name}: dropped {DROP_SENSORS}")
    print(f"  remaining sensors: {[c for c in SENSOR_COLS if c not in DROP_SENSORS]}")
    print(f"  saved: {out_path}")
    return out_path


if __name__ == "__main__":
    DATA_DIR = Path("data")      # <- change to your data directory
    for f in sorted(DATA_DIR.glob("train_FD00*.txt")) + sorted(DATA_DIR.glob("test_FD00*.txt")):
        clean_file(f)