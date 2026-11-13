import re
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")     # <- change to your data directory
RUL_CAP = 125               # try values between 120 and 130


def add_rul_train(df, cap=RUL_CAP):
    """Train engines run to failure: RUL = last cycle - current cycle, capped."""
    df = df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL_raw"] = max_cycle - df["cycle"]          # true RUL, uncapped (for reporting)
    df["RUL"] = df["RUL_raw"].clip(upper=cap)        # training target, capped
    return df


def add_rul_test(df, true_rul_last, cap=RUL_CAP):
    """Test engines stop early. true_rul_last = true RUL at each engine's last cycle
    (one value per engine, in unit order). RUL at earlier cycles = that value
    plus the number of cycles remaining until the engine's last recorded cycle."""
    df = df.copy()
    last_cycle = df.groupby("unit")["cycle"].max()          # indexed by unit, ascending
    assert len(true_rul_last) == len(last_cycle), "RUL file and test file have different engine counts"
    true_last = pd.Series(list(true_rul_last), index=last_cycle.index)
    df["RUL_raw"] = df["unit"].map(true_last) + df["unit"].map(last_cycle) - df["cycle"]
    df["RUL"] = df["RUL_raw"].clip(upper=cap)
    return df


def label_file(path, cap=RUL_CAP):
    """Label one clean_train_* or clean_test_* file and save rul_<filename> next to it."""
    path = Path(path)
    df = pd.read_csv(path, sep=" ")                          # clean files: space-separated, header
    dataset = re.search(r"FD\d{3}", path.name).group()

    if path.name.startswith("clean_train_"):
        df = add_rul_train(df, cap)
    else:                                                    # clean_test_*
        rul_path = path.with_name(f"RUL_{dataset}.txt")
        if not rul_path.exists():
            print(f"{path.name}: skipped, {rul_path.name} not found (needed for true test RUL)")
            return None
        true_rul_last = pd.read_csv(rul_path, header=None).iloc[:, 0]
        df = add_rul_test(df, true_rul_last, cap)

    out_path = path.with_name(f"rul_{path.name}")
    df.to_csv(out_path, sep=" ", index=False)
    print(f"{path.name}: {df['unit'].nunique()} engines, {len(df)} rows, "
          f"RUL (capped) {df['RUL'].min()}-{df['RUL'].max()}, RUL_raw {df['RUL_raw'].min()}-{df['RUL_raw'].max()} -> {out_path.name}")
    return out_path


if __name__ == "__main__":
    files = sorted(DATA_DIR.glob("clean_train_FD00*.txt")) + sorted(DATA_DIR.glob("clean_test_FD00*.txt"))
    for f in files:
        label_file(f)