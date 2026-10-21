"""Transformer and Mamba sequence-to-one regression models."""
from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed positional encoding for batch-first sequences."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(positions * scale)
        encoding[:, 1::2] = torch.cos(positions * scale[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.encoding.shape[1]:
            raise ValueError(f"Sequence length {x.shape[1]} exceeds configured max_len {self.encoding.shape[1]}")
        return x + self.encoding[:, : x.shape[1]].to(dtype=x.dtype)


class TransformerRegressor(nn.Module):
    """Standard Transformer encoder mapping [batch, time, features] to RUL."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        max_len: int = 512,
        pooling: str = "last",
    ) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        if pooling not in {"last", "mean"}:
            raise ValueError("pooling must be 'last' or 'mean'")
        self.pooling = pooling
        self.input_projection = nn.Linear(input_size, d_model)
        self.position = SinusoidalPositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.regression_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, features], received shape {tuple(x.shape)}")
        hidden = self.encoder(self.position(self.input_projection(x)))
        pooled = hidden[:, -1] if self.pooling == "last" else hidden.mean(dim=1)
        return self.regression_head(self.norm(pooled)).squeeze(-1)


class MambaRegressor(nn.Module):
    """Mamba sequence model mapping [batch, time, features] to RUL.

    Install the optional ``mamba-ssm`` package to use this backbone. Mamba's
    CUDA-oriented package may not support every platform or PyTorch build.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        num_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as exc:
            raise ImportError(
                "The Mamba model requires the optional 'mamba-ssm' package. "
                "Install it following the package's platform-specific instructions, "
                "or choose --model transformer."
            ) from exc
        self.input_projection = nn.Linear(input_size, d_model)
        self.layers = nn.ModuleList(
            nn.ModuleDict({"mamba": Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand), "norm": nn.LayerNorm(d_model)})
            for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.regression_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, features], received shape {tuple(x.shape)}")
        hidden = self.input_projection(x)
        for layer in self.layers:
            hidden = layer["norm"](hidden + self.dropout(layer["mamba"](hidden)))
        return self.regression_head(hidden[:, -1]).squeeze(-1)


def build_model(name: str, input_size: int, **kwargs: object) -> nn.Module:
    """Construct a supported sequence regressor by CLI model name."""
    models = {"transformer": TransformerRegressor, "mamba": MambaRegressor}
    try:
        model_type = models[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown model {name!r}; choose one of {', '.join(models)}") from exc
    return model_type(input_size=input_size, **kwargs)
