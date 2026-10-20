import pandas as pd
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "train_FD002.txt"
df = pd.read_csv(DATA_FILE, sep=r"\s+", header=None)

print(df.shape)
print(df.head())