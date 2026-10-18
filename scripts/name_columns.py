import pandas as pd

OPS = ["op1", "op2", "op3"]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLS = ["unit", "cycle"] + OPS + SENSORS

df = pd.read_csv("../data/train_FD002.txt", sep=r"\s+", header=None)
df.columns = COLS

# last cycle of each engine, repeated on every row of that engine
last_cycle = df.groupby("unit")["cycle"].transform("max")
df["rul"] = last_cycle - df["cycle"]

print(df.shape)
print(df[["unit", "cycle", "rul"]].head(3))
print(df[["unit", "cycle", "rul"]].groupby("unit").tail(1).head(3))
print("engines:", df["unit"].nunique())
print("lifetime min/median/max:", df.groupby("unit")["cycle"].max().describe()[["min", "50%", "max"]].tolist())

import matplotlib.pyplot as plt

# 1. How many distinct operating-setting combinations are there?
combos = df[OPS].round(1).value_counts()
print(combos)

# 2. Plot the first two operating settings
plt.scatter(df["op1"], df["op2"], s=2)
plt.xlabel("op1")
plt.ylabel("op2")
plt.title("Operating settings, all rows")

# plt.show()  # Unhide to display the plot of operating settings





# Engine 1 only: raw sensor values over its life
e1 = df[df["unit"] == 1]

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
for ax, s in zip(axes, ["s3", "s7"]):
    ax.plot(e1["cycle"], e1[s], marker=".", linestyle="none", markersize=4)
    ax.set_ylabel(s)
axes[0].set_title("Engine 1, raw sensor values")
axes[1].set_xlabel("cycle")
# plt.show()  # Unhide to display the plot of raw sensor values





from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Give every row a regime id (0 to 5)
ops_scaled = StandardScaler().fit_transform(df[OPS])
df["regime"] = KMeans(n_clusters=6, n_init=10, random_state=0).fit_predict(ops_scaled)
print(df["regime"].value_counts().sort_index())

# 2. Which sensors never vary inside a regime?
within_sd = df.groupby("regime")[SENSORS].std().mean()
print(within_sd.round(4))
keep = [s for s in SENSORS if within_sd[s] > 1e-6]
print("dropping:", [s for s in SENSORS if s not in keep])

# 3. Normalise each kept sensor within its regime
g = df.groupby("regime")[keep]
z = (df[keep] - g.transform("mean")) / g.transform("std")

# 4. Same engine, same sensors, after normalising
e1 = df["unit"] == 1
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
for ax, s in zip(axes, ["s3", "s7"]):
    ax.plot(df.loc[e1, "cycle"], z.loc[e1, s], marker=".", linestyle="none", markersize=4)
    ax.set_ylabel(s + " (normalised)")
axes[0].set_title("Engine 1 after per-regime normalisation")
axes[1].set_xlabel("cycle")
plt.show()