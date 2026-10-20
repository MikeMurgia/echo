"""Sliding-window datasets and DataLoaders for normalized NASA C-MAPSS data.

Run ``python -m src.dataset --subset FD001 --window-size 30`` from the repo
root to inspect the resulting split and tensor shapes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"
META_COLUMNS = {"unit", "cycle", "RUL", "regime"}


def sensor_columns(frame: pd.DataFrame) -> list[str]:
    """Return sensor feature columns in their dataframe order."""
    return [c for c in frame.columns if c.startswith("s") and c[1:].isdigit()]


def read_normalized(data_dir: str | Path, split: str, subset: str) -> pd.DataFrame:
    """Read a normalized, RUL-labeled train or test table."""
    path = Path(data_dir) / f"norm_rul_clean_{split}_{subset}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Normalized {split} data not found: {path}. Run the preprocessing scripts first.")
    frame = pd.read_csv(path, sep=r"\s+")
    required = {"unit", "cycle", "RUL"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
    return frame.sort_values(["unit", "cycle"], kind="stable").reset_index(drop=True)


class SlidingWindowDataset(Dataset):

    def __init__(
        self,
        frame: pd.DataFrame,
        window_size: int = 30,
        feature_columns: Sequence[str] | None = None,
        stride: int = 1,
        final_window_only: bool = False,
    ) -> None:
        if window_size < 1 or stride < 1:
            raise ValueError("window_size and stride must be positive integers")
        if feature_columns is None:
            feature_columns = sensor_columns(frame)
        self.feature_columns = list(feature_columns)
        if not self.feature_columns:
            raise ValueError("No input features found; provide feature_columns or sensor columns")
        missing = set(self.feature_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.final_window_only = final_window_only
        self.engines: list[tuple[int, np.ndarray, np.ndarray]] = []
        self.index: list[tuple[int, int]] = []

        for unit, group in frame.groupby("unit", sort=True):
            x = group[self.feature_columns].to_numpy(dtype=np.float32, copy=True)
            y = group["RUL"].to_numpy(dtype=np.float32, copy=True)
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f"Non-finite feature or RUL value for engine {unit}")
            engine_idx = len(self.engines)
            self.engines.append((int(unit), x, y))
            n = len(group)
            if final_window_only:
                self.index.append((engine_idx, max(0, n - self.window_size)))
            else:
                starts = range(0, max(1, n - self.window_size + 1), self.stride)
                self.index.extend((engine_idx, start) for start in starts)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        engine_idx, start = self.index[item]
        _, values, rul = self.engines[engine_idx]
        end = min(start + self.window_size, len(values))
        window = values[start:end]
        if len(window) < self.window_size:
            # Left pad with the first available observation, preserving the
            # alignment of the target with the final observed cycle.
            pad = np.repeat(window[:1], self.window_size - len(window), axis=0)
            window = np.concatenate((pad, window), axis=0)
        return torch.from_numpy(window), torch.tensor(rul[end - 1], dtype=torch.float32)


def split_engine_ids(frame: pd.DataFrame, validation_fraction: float = 0.2, seed: int = 42):
    """Deterministically split training engine IDs before window generation."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    units = np.asarray(sorted(frame["unit"].unique()))
    if len(units) < 2:
        raise ValueError("At least two training engines are required for a validation split")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(units)
    n_val = min(len(units) - 1, max(1, int(round(len(units) * validation_fraction))))
    return set(shuffled[n_val:].tolist()), set(shuffled[:n_val].tolist())


def make_dataloaders(
    subset: str,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    window_size: int = 30,
    batch_size: int = 64,
    validation_fraction: float = 0.2,
    seed: int = 42,
    stride: int = 1,
    num_workers: int = 0,
    feature_columns: Sequence[str] | None = None,
):
    """Return train, validation, and test loaders plus selected feature names."""
    train_frame = read_normalized(data_dir, "train", subset)
    test_frame = read_normalized(data_dir, "test", subset)
    if feature_columns is None:
        feature_columns = sensor_columns(train_frame)
    feature_columns = list(feature_columns)
    train_ids, val_ids = split_engine_ids(train_frame, validation_fraction, seed)
    train_part = train_frame[train_frame.unit.isin(train_ids)]
    val_part = train_frame[train_frame.unit.isin(val_ids)]

    common = dict(window_size=window_size, feature_columns=feature_columns, stride=stride)
    train_ds = SlidingWindowDataset(train_part, **common)
    val_ds = SlidingWindowDataset(val_part, **common)
    test_ds = SlidingWindowDataset(test_frame, **common, final_window_only=True)
    loaders = (
        DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            generator=torch.Generator().manual_seed(seed),
        ),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
    return (*loaders, feature_columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--subset", choices=("FD001", "FD002"), default="FD001")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    train, val, test, features = make_dataloaders(
        args.subset, args.data_dir, args.window_size, args.batch_size,
        args.validation_fraction, args.seed, args.stride,
    )
    x, y = next(iter(train))
    print(f"{args.subset}: {len(features)} sensor features ({', '.join(features)})")
    print(f"windows: train={len(train.dataset)}, validation={len(val.dataset)}, test={len(test.dataset)}")
    print(f"training batch: x={tuple(x.shape)}, y={tuple(y.shape)}")


if __name__ == "__main__":
    main()
