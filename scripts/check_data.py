import pandas as pd

df = pd.read_csv("../data/train_FD002.txt", sep=r"\s+", header=None)

print(df.shape)
print(df.head())