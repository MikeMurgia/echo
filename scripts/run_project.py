#!/usr/bin/env python3
"""Run the complete ECHO reporting pipeline from raw C-MAPSS data to results."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBSETS = ("FD001", "FD002")
SEEDS = (1, 2, 3)


def build_steps(device: str, include_mamba: bool) -> list[tuple[str, list[str]]]:
    py = sys.executable
    steps: list[tuple[str, list[str]]] = []

    def add(label: str, *args: str) -> None:
        steps.append((label, [py, *args]))

    add("Download NASA C-MAPSS data", "scripts/get_data.py", "--subsets", *SUBSETS)
    add("Remove constant sensors", "scripts/clean_data.py")
    add("Create capped and raw RUL labels", "scripts/label_data.py")
    add("Normalize sensor data", "scripts/normalize_data.py")

    transformer_args = (
        "--model", "transformer", "--tag", "tuned", "--patience", "3",
        "--lr", "3e-4", "--d-model", "64", "--layers", "2",
        "--dropout", "0.3", "--weight-decay", "1e-3", "--device", device,
    )
    for subset in SUBSETS:
        for seed in SEEDS:
            add(f"Train Transformer {subset}, seed {seed}", "scripts/train.py",
                "--subset", subset, *transformer_args, "--seed", str(seed))
    if include_mamba:
        for seed in SEEDS:
            add(f"Train optional Mamba FD002, seed {seed}", "scripts/train.py",
                "--subset", "FD002", "--model", "mamba", "--tag", "tuned",
                "--patience", "3", "--lr", "3e-4", "--d-model", "64",
                "--layers", "2", "--dropout", "0.3", "--weight-decay", "1e-3",
                "--device", device, "--seed", str(seed))

    for subset in SUBSETS:
        for seed in SEEDS:
            add(f"Train Random Forest and XGBoost {subset}, seed {seed}",
                "scripts/train_baselines.py", "--subset", subset, "--seed", str(seed))
        summary_runs = ["transformer:tuned", "random_forest:default", "xgboost:default"]
        if include_mamba and subset == "FD002":
            summary_runs.append("mamba:tuned")
        summary_args = [arg for run in summary_runs for arg in ("--run", run)]
        add(f"Summarize current pipeline metrics {subset}", "scripts/summarize_models.py",
            "--subset", subset, *summary_args)

    calibration_seed = SEEDS[-1]
    for subset in SUBSETS:
        checkpoint = f"results/checkpoints/transformer_{subset}_tuned_seed{calibration_seed}_best.pt"
        add(f"Run 20-split conformal calibration {subset}", "scripts/conformal_calibration.py",
            "--checkpoint", checkpoint, "--repeats", "20", "--final", "--device", device)
        add(f"Run 20-split ECHO decision analysis {subset}", "scripts/decision_layer.py",
            "--subset", subset, "--repeats", "20", "--write-final-model")
        add(f"Plot residual stages {subset}", "scripts/plot_abs_residual_vs_predicted_rul.py",
            "--subset", subset)

    add("Create combined cost and residual-source figures", "scripts/plot_slide_sensitivity.py")
    add("Create repeated-split decision sensitivity figure", "scripts/plot_repeated_decision_sensitivity.py")
    return steps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto",
                        help="device for Transformer and conformal inference (default: auto)")
    parser.add_argument("--include-mamba", action="store_true",
                        help="also train the optional FD002 Mamba experiment for seeds 1–3")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the full command list without downloading, training, or writing results")
    args = parser.parse_args()

    steps = build_steps(args.device, args.include_mamba)
    for index, (label, command) in enumerate(steps, start=1):
        print(f"\n[{index}/{len(steps)}] {label}", flush=True)
        print("$ " + " ".join(command), flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, cwd=ROOT, check=True)

    if args.dry_run:
        print(f"\nDry run complete: {len(steps)} commands listed; no project files were changed.")
    else:
        print("\nECHO pipeline complete. See results/FD001, results/FD002, and results/ for combined figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
