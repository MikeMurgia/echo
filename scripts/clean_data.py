

#removes non-informative sensors : sensors with 0 value (1,5,10,16,18,19) for FD001

import pandas as pd
from pathlib import Path 

INDEX_COLS = ["unit", "cycle"]
SETTING_COLS = ["op1", "op2", "op3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLUMNS = INDEX_COLS + SETTING_COLS + SENSOR_COLS


def load_cmapss(path):
    """Load any CMAPSS train/test file (FD001-FD004)."""
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def add_condition_id(df, decimals=None):
    """Label each row with its operating condition by rounding the 3 settings.
    FD001/FD003 -> 1 condition, FD002/FD004 -> 6 conditions."""
    decimals = decimals or {"op1": 0, "op2": 1, "op3": 0}
    rounded = pd.DataFrame({c: df[c].round(d) for c, d in decimals.items()})
    return rounded.groupby(list(rounded.columns), sort=False).ngroup()


def find_non_informative(df, cols, by_condition=True, std_tol=1e-6, min_unique=3):
    """
    Returns a report DataFrame with a 'status' per column:
      drop   - constant (within every operating condition if by_condition=True)
      review - not constant, but very few unique values (e.g. s6 in FD001)
      keep   - informative
    """
    global_std = df[cols].std()
    if by_condition:
        cond = add_condition_id(df)
        # worst-case (largest) within-condition std for each column
        check_std = df.groupby(cond)[cols].std().max().fillna(0)
        n_cond = cond.nunique()
    else:
        check_std = global_std.fillna(0)
        n_cond = 1

    report = pd.DataFrame({
        "n_unique": df[cols].nunique(),
        "global_std": global_std,
        "max_within_cond_std": check_std,
    })
    report["status"] = "keep"
    report.loc[report["n_unique"] < min_unique, "status"] = "review"
    report.loc[report["max_within_cond_std"] <= std_tol, "status"] = "drop"
    report.attrs["n_conditions"] = n_cond
    return report


def remove_non_informative(df, cols, drop_review=False, **kwargs):
    report = find_non_informative(df, cols, **kwargs)
    to_drop = report.index[report["status"] == "drop"].tolist()
    if drop_review:
        to_drop += report.index[report["status"] == "review"].tolist()
    return df.drop(columns=to_drop), to_drop, report

def clean_file(path, drop_review=False):
    """Clean one CMAPSS file and save clean_<filename> in the same directory."""
    path = Path(path)
    df = load_cmapss(path)

    df, dropped_sensors, report = remove_non_informative(
        df, SENSOR_COLS, drop_review=drop_review
    )
    df, dropped_settings, _ = remove_non_informative(
        df, SETTING_COLS, by_condition=False
    )

    out_path = path.with_name(f"clean_{path.name}")
    df.to_csv(out_path, sep=" ", index=False)   # space-separated, with header

    print(f"{path.name}: conditions={report.attrs['n_conditions']}")
    print(f"  dropped sensors : {dropped_sensors}")
    print(f"  dropped settings: {dropped_settings}")
    print(f"  review manually : {report.index[report['status'] == 'review'].tolist()}")
    print(f"  saved           : {out_path}")
    return out_path


if __name__ == "__main__":
    DATA_DIR = Path("data")      # <- change to your data directory
    for f in sorted(DATA_DIR.glob("train_FD00*.txt")) + sorted(DATA_DIR.glob("test_FD00*.txt")):
        clean_file(f)

