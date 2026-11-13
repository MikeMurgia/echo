"""Collect every results/<subset>/*_metrics.json into one comparison table.

    python scripts/summarize_models.py --subset FD002

Prints and saves results/<subset>/model_comparison.csv (mean and std across seeds), and names the
checkpoint to use for the conformal step. The choice is made on VALIDATION RMSE only, never on test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["val_rmse", "test_rmse_capped", "test_nasa_score_capped", "test_rmse_raw", "test_nasa_score_raw"]
SEQUENCE_MODELS = {"transformer", "mamba"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=("FD001", "FD002"), default="FD002")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    folder = args.results_dir / args.subset
    runs = [json.loads(p.read_text()) for p in sorted(folder.glob("*_metrics.json"))]
    if not runs:
        raise SystemExit(f"No *_metrics.json files in {folder}. Run scripts/train.py / train_baselines.py first.")
    frame = pd.DataFrame(runs)
    if "tag" not in frame.columns:
        frame["tag"] = "default"
    frame["tag"] = frame["tag"].fillna("default")
    cols = [c for c in COLUMNS if c in frame.columns]
    keys = ["model", "tag"]
    table = frame.groupby(keys)[cols].agg(["mean", "std"])
    table.columns = [f"{a}_{b}" for a, b in table.columns]
    table.insert(0, "runs", frame.groupby(keys).size())
    table = table.sort_values("val_rmse_mean").reset_index()
    table.to_csv(folder / "model_comparison.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.2f}".format):
        print(table.to_string(index=False))

    seq = frame[frame.model.isin(SEQUENCE_MODELS)]
    if seq.empty:
        print("\nNo sequence-model runs yet, so there is no checkpoint to calibrate.")
        return
    best_model, best_tag = seq.groupby(keys)["val_rmse"].mean().idxmin()
    best = seq[(seq.model == best_model) & (seq.tag == best_tag)].sort_values("val_rmse").iloc[0]
    tag_part = "" if best_tag == "default" else f"_{best_tag}"
    checkpoint = args.results_dir / "checkpoints" / f"{best_model}_{args.subset}{tag_part}_seed{int(best.seed)}_best.pt"
    print(f"\nLowest mean validation RMSE among sequence models: {best_model} ({best_tag} recipe)")
    print(f"Use for conformal: {checkpoint}  (seed {int(best.seed)}, validation RMSE {best.val_rmse:.3f})")


if __name__ == "__main__":
    main()