"""Plot absolute RUL residuals against predicted RUL with lifecycle stages marked.

    python scripts/plot_abs_residual_vs_predicted_rul.py --subset FD001

Reads results/<subset>/conformal_windows.csv and writes
results/<subset>/fig_abs_residual_vs_predicted_rul.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


STAGES = (
    (0, 30, "Late (<30)", "#e45756"),
    (30, 100, "Mid (30–100)", "#f2cf5b"),
    (100, None, "Early (>100)", "#72b7b2"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="FD001", help="C-MAPSS subset (default: FD001)")
    parser.add_argument("--input", type=Path, help="Override the conformal_windows.csv path")
    parser.add_argument("--output", type=Path, help="Override the output PNG path")
    args = parser.parse_args()

    input_path = args.input or Path("results") / args.subset / "conformal_windows.csv"
    output_path = args.output or input_path.with_name("fig_abs_residual_vs_predicted_rul.png")
    data = pd.read_csv(input_path)
    required = {"yhat", "abs_res"}
    missing = required - set(data.columns)
    if missing:
        raise SystemExit(f"{input_path} is missing required columns: {', '.join(sorted(missing))}")
    data = data[["yhat", "abs_res"]].dropna().sort_values("yhat")
    if data.empty:
        raise SystemExit(f"No valid prediction/residual rows in {input_path}")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for low, high, label, color in STAGES:
        ax.axvspan(low, high if high is not None else max(125, data.yhat.max()), color=color, alpha=0.15)
        center = (low + high) / 2 if high is not None else max(125, data.yhat.max()) - 12.5
        ax.text(center, 0.98, label, color=color, ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=9, weight="bold")
    ax.scatter(data.yhat, data.abs_res, s=9, alpha=0.22, color="#315a7d", edgecolors="none", rasterized=True)
    ax.axvline(30, color="#9b3d3d", linestyle="--", linewidth=1.2)
    ax.axvline(100, color="#8a6d1d", linestyle="--", linewidth=1.2)
    ax.set(xlim=(0, max(125, data.yhat.max())), ylim=(0, None),
           xlabel="Predicted RUL (cycles)", ylabel="Absolute residual |y − ŷ| (cycles)",
           title=f"Absolute RUL residual vs. predicted RUL — {args.subset}")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
