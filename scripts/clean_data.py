"""Drop uninformative sensors and save clean_<filename> next to the raw file.

Which sensors are dropped depends on the subset:

* FD001 (one operating condition): s1, s5, s6, s10, s16, s18, s19 never vary.
* FD002 (six operating conditions): only s1, s5, s18, s19 have exactly zero
  spread inside every operating regime. s6, s10 and s16 vary slightly, so they
  are kept and normalized per regime (check them with feature analysis later).
"""
import re
from pathlib import Path

import pandas as pd

INDEX_COLS = ["unit", "cycle"]
SETTING_COLS = ["op1", "op2", "op3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLUMNS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

DROP_SENSORS = {
    "FD001": ["s1", "s5", "s6", "s10", "s16", "s18", "s19"],
    "FD002": ["s1", "s5", "s18", "s19"],
}


def load_cmapss(path):
    """Load any CMAPSS train/test file (FD001-FD004)."""
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def clean_file(path):
    """Drop the subset's constant sensors and save clean_<filename> in the same directory."""
    path = Path(path)
    subset = re.search(r"FD\d{3}", path.name).group()
    if subset not in DROP_SENSORS:
        raise ValueError(f"No sensor drop list defined for {subset}; add it to DROP_SENSORS")
    drop = DROP_SENSORS[subset]
    df = load_cmapss(path)

    df_clean = df.drop(columns=drop)
    out_path = path.with_name(f"clean_{path.name}")
    df_clean.to_csv(out_path, sep=" ", index=False)   # space-separated, with header

    print(f"{path.name}: dropped {drop}")
    print(f"  remaining sensors: {[c for c in SENSOR_COLS if c not in drop]}")
    print(f"  saved: {out_path}")
    return out_path


if __name__ == "__main__":
    DATA_DIR = Path("data")      # run from the repo root
    files = []
    for subset in DROP_SENSORS:
        files += [DATA_DIR / f"train_{subset}.txt", DATA_DIR / f"test_{subset}.txt"]
    for f in files:
        if f.exists():
            clean_file(f)
        else:
            print(f"skipped (not found): {f}")