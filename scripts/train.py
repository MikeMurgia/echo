"""Train a Transformer or Mamba regressor on normalized C-MAPSS sequences.

Each run writes, under results/<subset>/:
    <model>_seed<seed>_history.csv   per-epoch train/validation RMSE
    <model>_seed<seed>_metrics.json  final metrics (test RMSE / NASA score, capped and uncapped)
and the best checkpoint under results/checkpoints/ (gitignored).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import make_dataloaders, read_normalized  # noqa: E402
from src.models import build_model  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(predictions: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    """Return RMSE and the summed NASA asymmetric score (lower is better)."""
    delta = predictions - targets
    rmse = torch.sqrt(torch.mean(delta.square())).item()
    score = torch.where(delta < 0, torch.expm1(-delta / 13.0), torch.expm1(delta / 10.0)).sum().item()
    return rmse, score


@torch.no_grad()
def predict(model: nn.Module, loader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Predicted and true (capped) RUL for every window in the loader, in loader order."""
    model.eval()
    predictions, targets = [], []
    for x, y in loader:
        predictions.append(model(x.to(device)).reshape(-1).cpu())
        targets.append(y.cpu())
    if not targets:
        raise ValueError("Evaluation loader is empty")
    return torch.cat(predictions), torch.cat(targets)


def evaluate(model: nn.Module, loader, device: torch.device) -> tuple[float, float, float]:
    yhat, y = predict(model, loader, device)
    rmse, nasa_score = metrics(yhat, y)
    return float(torch.mean((yhat - y).square())), rmse, nasa_score


def train_epoch(model: nn.Module, loader, optimizer, criterion, device: torch.device) -> float:
    model.train()
    squared_error, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x).reshape(-1)
        loss = criterion(prediction, y)
        loss.backward()
        optimizer.step()
        squared_error += loss.item() * y.numel()
        n += y.numel()
    return math.sqrt(squared_error / n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=("FD001", "FD002"), default="FD002")
    parser.add_argument("--model", choices=("transformer", "mamba"), default="transformer")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7,
                        help="epochs without validation RMSE improvement before stopping")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--stride", type=int, default=1,
                        help="window stride for training/validation; use 2 or 3 to speed up CPU training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="", help="label for this recipe, e.g. tuned; keeps its files separate from other recipes")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--save-dir", type=Path, default=ROOT / "results" / "checkpoints")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0 or args.weight_decay < 0 or args.patience < 1:
        parser.error("--epochs and --patience must be positive; --lr must be positive and --weight-decay non-negative")

    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS was requested but is not available")

    train_loader, val_loader, test_loader, features = make_dataloaders(
        subset=args.subset,
        data_dir=ROOT / "data",
        window_size=args.window_size,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        stride=args.stride,
    )
    model = build_model(
        args.model,
        input_size=len(features),
        d_model=args.d_model,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    out_dir = args.results_dir / args.subset
    out_dir.mkdir(parents=True, exist_ok=True)
    tag_part = f"_{args.tag}" if args.tag else ""
    run_name = f"{args.model}{tag_part}_seed{args.seed}"
    checkpoint_path = args.save_dir / f"{args.model}_{args.subset}{tag_part}_seed{args.seed}_best.pt"
    best_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    start = time.time()

    print(f"Training {args.model} on {args.subset} with {len(features)} features, "
          f"{n_params:,} parameters, {len(train_loader.dataset)} train windows, using {device}")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_rmse = train_epoch(model, train_loader, optimizer, criterion, device)
        val_mse, val_rmse, val_score = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_rmse": train_rmse, "val_rmse": val_rmse,
                        "val_nasa_score": val_score, "seconds": time.time() - t0})
        print(
            f"Epoch {epoch:03d}/{args.epochs} ({time.time() - t0:.0f}s) "
            f"train_RMSE={train_rmse:.3f} val_MSE={val_mse:.3f} "
            f"val_RMSE={val_rmse:.3f} val_NASA_score={val_score:.3f}"
        )
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model": args.model,
                    "subset": args.subset,
                    "seed": args.seed,
                    "input_size": len(features),
                    "feature_columns": features,
                    "window_size": args.window_size,
                    "d_model": args.d_model,
                    "num_layers": args.layers,
                    "dropout": args.dropout,
                    "weight_decay": args.weight_decay,
                    "epoch": epoch,
                    "val_rmse": val_rmse,
                    "val_nasa_score": val_score,
                },
                checkpoint_path,
            )
            print(f"  saved best checkpoint: {checkpoint_path}")
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"  no validation RMSE improvement for {epochs_without_improvement}/{args.patience} epoch(s)")
            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping triggered at epoch {epoch}; "
                    f"best validation RMSE was {best_rmse:.3f} at epoch {best_epoch}."
                )
                break
    train_seconds = time.time() - start
    pd.DataFrame(history).to_csv(out_dir / f"{run_name}_history.csv", index=False)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Restored best checkpoint from epoch {checkpoint['epoch']} (validation RMSE={checkpoint['val_rmse']:.3f}).")

    # Official test set, one window per engine at its last observed cycle.
    yhat, y_capped = predict(model, test_loader, device)
    rmse_c, score_c = metrics(yhat, y_capped)
    result = {
        "model": args.model, "tag": args.tag or "default", "subset": args.subset, "seed": args.seed,
        "device": str(device),
        "best_epoch": int(checkpoint["epoch"]), "epochs_run": len(history),
        "val_rmse": float(checkpoint["val_rmse"]), "val_nasa_score": float(checkpoint["val_nasa_score"]),
        "test_rmse_capped": rmse_c, "test_nasa_score_capped": score_c,
        "n_params": n_params, "train_seconds": train_seconds,
        "window_size": args.window_size, "d_model": args.d_model, "layers": args.layers,
        "dropout": args.dropout, "lr": args.lr, "weight_decay": args.weight_decay, "stride": args.stride,
    }
    test_frame = read_normalized(ROOT / "data", "test", args.subset)
    if "RUL_raw" in test_frame.columns:
        y_raw = torch.tensor(test_frame.groupby("unit")["RUL_raw"].last().to_numpy(), dtype=torch.float32)
        if len(y_raw) == len(yhat):
            rmse_r, score_r = metrics(yhat, y_raw)
            result.update({"test_rmse_raw": rmse_r, "test_nasa_score_raw": score_r})
        else:
            print("Warning: engine count mismatch; skipping uncapped test metrics")
    else:
        print("Warning: no RUL_raw column; rerun scripts/label_data.py and scripts/normalize_data.py")
    (out_dir / f"{run_name}_metrics.json").write_text(json.dumps(result, indent=2))
    print(
        f"Best epoch {checkpoint['epoch']}: test_RMSE(capped)={rmse_c:.3f} test_NASA(capped)={score_c:.3f}"
        + (f" test_RMSE(raw)={result['test_rmse_raw']:.3f} test_NASA(raw)={result['test_nasa_score_raw']:.3f}"
           if "test_rmse_raw" in result else "")
    )
    print(f"Training took {train_seconds / 60:.1f} min. Wrote {out_dir / (run_name + '_metrics.json')}")


if __name__ == "__main__":
    main()