"""Build slide-ready cost sensitivity and pooled-vs-stage ablation figures from saved sweeps.

    python scripts/plot_slide_sensitivity.py

Writes cost_vs_rho.png in each subset results folder and a combined
results/ablation_pooled_vs_stage.png plus its underlying CSV.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUBSETS = ("FD001", "FD002")
HORIZONS = (15, 30, 50)


def plot_cost(subset: str, results_dir: Path) -> None:
    sweep = pd.read_csv(results_dir / subset / "decision_sweep.csv")
    data = sweep[(sweep.scope == "all") & (sweep.p_source == "stage")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), sharey=True)
    for ax, horizon in zip(axes, HORIZONS):
        d = data[data.horizon == horizon].sort_values("rho")
        ax.plot(d.rho, d.cost_echo, color="#f58518", lw=3.2, label="ECHO")
        ax.plot(d.rho, d.cost_point, color="#777777", lw=2.1, label="Point Rule")
        ax.plot(d.rho, d.cost_interval, color="#52677d", lw=2.1, label="Lower-Bound Rule")
        ax.set_xscale("log")
        ax.set_title(f"H = {horizon} cycles", fontsize=15, weight="bold")
        ax.set_xlabel("Cost ratio ρ = Cu / Cs", fontsize=14)
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("Expected cost per decision point", fontsize=14)
    axes[0].axvspan(5, 10, color="#f58518", alpha=0.10, zorder=0)
    row10 = data[(data.horizon == 15) & (data.rho == 10)]
    if not row10.empty:
        y = float(row10.cost_echo.iloc[0])
        axes[0].annotate("H=15 conservative-trigger crossover\n(ECHO vs Point, ρ≈5–10)",
                         xy=(10, y), xytext=(1.5, y * 1.65), fontsize=10,
                         arrowprops={"arrowstyle": "->", "color": "#555555"})
    axes[-1].legend(loc="upper left", fontsize=11, frameon=False)
    fig.suptitle(f"Maintenance cost sensitivity — {subset}", fontsize=17, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(results_dir / subset / "cost_vs_rho.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(results_dir: Path) -> None:
    records = []
    for subset in SUBSETS:
        repeats = pd.read_csv(results_dir / subset / "decision_repeats.csv")
        rows = repeats[(repeats.scope == "all") & (repeats.rho == 50)
                       & repeats.p_source.isin(["stage", "pooled"])]
        for r in rows.itertuples():
            records.append({"subset": subset, "horizon": int(r.horizon), "rho": float(r.rho),
                            "p_source": r.p_source, "splits": int(r.splits),
                            "cost_echo_mean": float(r.cost_echo_mean),
                            "cost_echo_std": float(r.cost_echo_std)})
    table = pd.DataFrame(records).sort_values(["subset", "horizon", "p_source"])
    table.to_csv(results_dir / "ablation_pooled_vs_stage.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    palette = {"stage": "#4c78a8", "pooled": "#9aa3ad"}
    for ax, subset in zip(axes, SUBSETS):
        d = table[table.subset == subset]
        x = np.arange(len(HORIZONS))
        width = 0.34
        for offset, source in ((-width / 2, "stage"), (width / 2, "pooled")):
            source_rows = [d[(d.horizon == h) & (d.p_source == source)].iloc[0] for h in HORIZONS]
            vals = [r.cost_echo_mean for r in source_rows]
            errors = [r.cost_echo_std for r in source_rows]
            ax.bar(x + offset, vals, width, yerr=errors, capsize=3, color=palette[source],
                   label=f"{source.title()} residuals")
        ax.set_xticks(x, [f"H={h}" for h in HORIZONS])
        ax.set_title(subset, fontsize=15, weight="bold")
        ax.set_xlabel("Planning horizon", fontsize=14)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean cost per decision point at ρ = 50 (error bars: SD)", fontsize=14)
    axes[-1].legend(frameon=False, fontsize=11)
    fig.suptitle("Residual source ablation — 20 engine splits", fontsize=17, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(results_dir / "ablation_pooled_vs_stage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results_dir = ROOT / "results"
    for subset in SUBSETS:
        plot_cost(subset, results_dir)
    plot_ablation(results_dir)
    print("Wrote results/FD001/cost_vs_rho.png, results/FD002/cost_vs_rho.png, ")
    print("results/ablation_pooled_vs_stage.png, and results/ablation_pooled_vs_stage.csv")


if __name__ == "__main__":
    main()
