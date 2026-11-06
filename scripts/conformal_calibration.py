"""Calibrate and assess RUL dependent Mondrian conformal intervals"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import SlidingWindowDataset, read_normalized, sensor_columns, split_engine_ids  # noqa: E402
from src.models import build_model  # noqa: E402


@torch.no_grad()
def predict(model, dataset, device: torch.device, batch_size: int):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions, targets = [], []
    model.eval()
    for x, y in loader:
        predictions.append(model(x.to(device)).reshape(-1).cpu().numpy())
        targets.append(y.numpy())
    if not targets:
        raise ValueError("Dataset has no windows")
    return np.concatenate(predictions), np.concatenate(targets)


def window_metadata(dataset: SlidingWindowDataset) -> pd.DataFrame:
    rows = []
    for engine_idx, start in dataset.index:
        unit, _, rul = dataset.engines[engine_idx]
        cycle = min(start + dataset.window_size, len(rul))
        rows.append({"unit": unit, "cycle": cycle, "last_cycle": len(rul)})
    frame = pd.DataFrame(rows)
    return frame


def stage_for_rul(rul: np.ndarray, early_threshold: float, late_threshold: float) -> np.ndarray:

    return np.where(rul > early_threshold, "early", np.where(rul >= late_threshold, "mid", "late"))


def conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    n = len(scores)
    if n == 0:
        raise ValueError("Calibration split is empty")
    rank = int(np.ceil((n + 1) * coverage))
    if rank > n:
        return float("inf")
    return float(np.partition(scores, rank - 1)[rank - 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results/checkpoints/transformer_FD001_best.pt")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--subset", choices=("FD001", "FD002"), default="FD001")
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument("--early-rul", type=float, default=100.0,
                        help="early stage is RUL above this threshold")
    parser.add_argument("--late-rul", type=float, default=30.0,
                        help="late stage is RUL below this threshold")
    parser.add_argument("--calibration-fraction", type=float, default=0.5,
                        help="fraction of validation engines used to calibrate the interval")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--output", type=Path, default=ROOT / "results/conformal_stage_summary.csv")
    parser.add_argument("--plot", type=Path, default=ROOT / "results/mondrian_intervals.png")
    parser.add_argument("--representative-engines", type=int, default=3,
                        help="maximum assessment engines plotted")
    args = parser.parse_args()
    if not 0 < args.coverage < 1:
        parser.error("--coverage must be between 0 and 1")
    if not 0 < args.calibration_fraction < 1:
        parser.error("--calibration-fraction must be between 0 and 1")
    if args.late_rul >= args.early_rul:
        parser.error("--late-rul must be less than --early-rul")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "mps" if args.device == "auto" and torch.backends.mps.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS requested but unavailable")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    subset = checkpoint.get("subset", args.subset)
    if subset != args.subset:
        parser.error(f"Checkpoint is for {subset}; requested subset is {args.subset}")
    frame = read_normalized(args.data_dir, "train", subset)
    _, validation_ids = split_engine_ids(frame, args.validation_fraction, args.seed)
    val = frame[frame.unit.isin(validation_ids)]
    units = np.asarray(sorted(validation_ids))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(units)
    n_cal = min(len(units) - 1, max(1, int(round(len(units) * args.calibration_fraction))))
    cal_ids, eval_ids = set(units[:n_cal]), set(units[n_cal:])
    features = checkpoint.get("feature_columns") or sensor_columns(frame)
    common = dict(window_size=int(checkpoint["window_size"]), feature_columns=features, stride=args.stride)
    cal_ds = SlidingWindowDataset(val[val.unit.isin(cal_ids)], **common)
    eval_ds = SlidingWindowDataset(val[val.unit.isin(eval_ids)], **common)

    model = build_model(
        checkpoint.get("model", "transformer"), input_size=int(checkpoint["input_size"]),
        d_model=int(checkpoint.get("d_model", 128)),
        num_layers=int(checkpoint.get("num_layers", 3)),
        dropout=float(checkpoint.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    cal_pred, cal_y = predict(model, cal_ds, device, args.batch_size)
    cal_scores = np.abs(cal_pred - cal_y)
    cal_stages = stage_for_rul(cal_pred, args.early_rul, args.late_rul)
    quantiles = {
        stage: conformal_quantile(cal_scores[cal_stages == stage], args.coverage)
        for stage in ("early", "mid", "late")
        if np.any(cal_stages == stage)
    }
    missing_stages = set(("early", "mid", "late")) - set(quantiles)
    if missing_stages:
        parser.error(f"No calibration predictions fell in stage(s): {', '.join(sorted(missing_stages))}")

    eval_pred, eval_y = predict(model, eval_ds, device, args.batch_size)
    metadata = window_metadata(eval_ds)
    eval_stages = stage_for_rul(eval_pred, args.early_rul, args.late_rul)
    widths = np.asarray([2 * quantiles[stage] for stage in eval_stages])
    stage_quantiles = np.asarray([quantiles[stage] for stage in eval_stages])
    summary = metadata.assign(
        prediction=eval_pred, target=eval_y,
        stage=eval_stages,
        lower=eval_pred - stage_quantiles, upper=eval_pred + stage_quantiles,
        covered=np.abs(eval_pred - eval_y) <= stage_quantiles,
        width=widths,
    ).groupby("stage", observed=False).agg(
        windows=("target", "size"), engines=("unit", "nunique"),
        empirical_coverage=("covered", "mean"), mean_width=("width", "mean"),
        median_rul=("target", "median"),
    ).reset_index()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)

    plot_data = metadata.assign(prediction=eval_pred, target=eval_y,
                                half_width=stage_quantiles, predicted_stage=eval_stages)
    engine_ids = sorted(plot_data.unit.unique())
    if len(engine_ids) > args.representative_engines:
        # Evenly sample assessment engines for a reproducible, readable figure.
        picks = np.linspace(0, len(engine_ids) - 1, args.representative_engines, dtype=int)
        engine_ids = [engine_ids[i] for i in picks]
    fig, axes = plt.subplots(len(engine_ids), 1, figsize=(10, max(3, 2.8 * len(engine_ids))), squeeze=False)
    colors = {"early": "#4c78a8", "mid": "#f2a541", "late": "#e45756"}
    for ax, unit in zip(axes[:, 0], engine_ids):
        engine = plot_data[plot_data.unit == unit].sort_values("cycle")
        ax.plot(engine.cycle, engine.target, color="black", linewidth=1.5, label="True RUL")
        ax.plot(engine.cycle, engine.prediction, color="#4c78a8", linewidth=1.2, label="Prediction")
        ax.fill_between(engine.cycle.to_numpy(),
                        (engine.prediction - engine.half_width).to_numpy(),
                        (engine.prediction + engine.half_width).to_numpy(),
                        color="#4c78a8", alpha=0.18, label="Mondrian interval")
        for stage, color in colors.items():
            mask = engine.predicted_stage.to_numpy() == stage
            if mask.any():
                ax.scatter(engine.cycle.to_numpy()[mask], engine.prediction.to_numpy()[mask],
                           s=8, color=color, label=f"{stage} prediction stage")
        ax.set(title=f"Assessment engine {unit}", xlabel="Cycle", ylabel="RUL (cycles)")
        ax.grid(alpha=0.2)
        ax.legend(ncol=3, fontsize="small")
    fig.tight_layout()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=160)
    plt.close(fig)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Target coverage: {args.coverage:.1%}; calibration windows: {len(cal_y)} across {len(cal_ids)} engines")
    for stage in ("early", "mid", "late"):
        count = int(np.sum(cal_stages == stage))
        print(f"{stage}: q={quantiles[stage]:.3f}, nominal width={2 * quantiles[stage]:.3f}, calibration n={count}")
    print(f"Assessment: {len(eval_y)} windows across {len(eval_ids)} engines")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
