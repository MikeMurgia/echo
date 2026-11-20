"""Plot the held-out FD001 engine 20 case study from conformal window results."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUBSET = "FD001"
UNIT = 20
START_CYCLE = 165
FAILURE_CYCLE = 181


def main() -> None:
    path = ROOT / "results" / SUBSET / "conformal_windows.csv"
    frame = pd.read_csv(path)
    data = frame[(frame["unit"] == UNIT) & (frame["split"] == "eval")
                 & (frame["cycle"] >= START_CYCLE)].sort_values("cycle")
    if data.empty:
        raise SystemExit(f"No held-out rows found for {SUBSET} engine {UNIT}")

    cycles = data["cycle"].to_numpy()
    truth = data["y_raw"].to_numpy()
    prediction = data["yhat"].to_numpy()
    halfwidth = data["hw_mondrian"].to_numpy()
    lower, upper = prediction - halfwidth, prediction + halfwidth
    failed = data["covered_mondrian"].astype(str).str.lower().eq("false").to_numpy()

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.fill_between(cycles, lower, upper, color="#4c78a8", alpha=0.18,
                    label="90% Mondrian interval")
    ax.plot(cycles, truth, color="#222222", linewidth=2.5, marker="o",
            markersize=4, label="True RUL")
    ax.plot(cycles, prediction, color="#e68613", linewidth=2.2, marker="s",
            markersize=3.5, label="ECHO prediction")
    if failed.any():
        ax.scatter(cycles[failed], truth[failed], color="#c43c39", marker="x",
                   s=75, linewidth=2.2, zorder=5, label="True RUL outside interval")
    ax.axvline(FAILURE_CYCLE, color="#c43c39", linestyle="--", linewidth=1.2,
               alpha=0.8)
    fail_row = data.loc[data["cycle"] == FAILURE_CYCLE]
    if not fail_row.empty:
        row = fail_row.iloc[0]
        ax.annotate(
            f"Cycle {FAILURE_CYCLE}: true {row['y_raw']:.0f}, "
            f"predicted {row['yhat']:.1f}",
            xy=(FAILURE_CYCLE, row["y_raw"]), xytext=(FAILURE_CYCLE - 13, row["y_raw"] - 8),
            arrowprops={"arrowstyle": "->", "color": "#c43c39"},
            color="#8e2d2b", fontsize=10,
        )
    ax.set_title("FD001 held-out engine 20: late-life prediction case")
    ax.set_xlabel("Observed cycle")
    ax.set_ylabel("Remaining useful life (cycles)")
    ax.set_xlim(START_CYCLE - 1, int(cycles.max()) + 1)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()

    output = ROOT / "results" / SUBSET / "case_study_engine20.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
