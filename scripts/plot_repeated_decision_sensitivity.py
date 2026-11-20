"""Plot repeated-split decision costs across horizons and failure-cost ratios.

Reads results/{FD001,FD002}/decision_repeats.csv and writes
results/decision_sensitivity_repeats.png. Points show means and error bars
show standard deviations across the 20 engine-level splits.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parents[1]
SUBSETS = ("FD001", "FD002")
HORIZONS = (15, 30, 50)
RHOS = (2, 5, 10, 20, 50)
METHODS = (
    ("cost_echo", "ECHO (stage-based)", "#e68613", "o"),
    ("cost_point", "Point rule", "#626b73", "s"),
    ("cost_interval", "Lower-bound interval rule", "#4c78a8", "^"),
)


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)

    for row, subset in enumerate(SUBSETS):
        data = pd.read_csv(ROOT / "results" / subset / "decision_repeats.csv")
        data = data[(data["scope"] == "all") & (data["p_source"] == "stage")]
        for col, horizon in enumerate(HORIZONS):
            ax = axes[row, col]
            d = data[(data["horizon"] == horizon) & data["rho"].isin(RHOS)].sort_values("rho")
            for mean_col, label, color, marker in METHODS:
                ax.errorbar(
                    d["rho"], d[f"{mean_col}_mean"], yerr=d[f"{mean_col}_std"],
                    label=label, color=color, marker=marker, linewidth=2,
                    markersize=5, capsize=3,
                )
            ax.set_xscale("log")
            ax.set_xticks(RHOS, labels=[str(rho) for rho in RHOS])
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_title(f"{subset} · H = {horizon} cycles")
            ax.grid(alpha=0.25, which="both")
            if col == 0:
                ax.set_ylabel("Mean cost per decision window")
            if row == 1:
                ax.set_xlabel("Failure-cost ratio, ρ = Cᵤ / Cₛ")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.015))
    fig.suptitle("Decision-cost sensitivity across 20 engine-level splits", y=1.07,
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    output = ROOT / "results" / "decision_sensitivity_repeats.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
