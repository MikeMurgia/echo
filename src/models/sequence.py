"""Transformer and Mamba sequence-to-one regression models."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
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


class MambaBlock(nn.Module):
    """Mamba selective state-space block in pure PyTorch (Gu & Dao, 2023).

    Follows the reference computation: input projection into a main branch and a gate,
    causal depthwise convolution, input-dependent step size / B / C, discretization
    (exact for A, Euler for B), a sequential scan over time, the D skip connection and
    SiLU gating. It does NOT use the fused CUDA kernel of the ``mamba-ssm`` package, so it
    runs on CPU, any GPU, Windows, macOS and Linux. Windows here are short (about 30 steps),
    so the Python loop over time is cheap. Weights are not interchangeable with ``mamba-ssm``.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ) -> None:
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner,
                                padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Step size init: softplus(bias) is log-uniform in [dt_min, dt_max] (as in the reference).
        nn.init.uniform_(self.dt_proj.weight, -self.dt_rank ** -0.5, self.dt_rank ** -0.5)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        # State matrix A = -exp(A_log) (real, negative), initialised to 1..d_state per channel.
        a_init = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a_init))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def selective_scan(self, u, delta, A, B, C, D):
        """u, delta: [b, l, d_inner]; A: [d_inner, n]; B, C: [b, l, n]; D: [d_inner] -> [b, l, d_inner]."""
        batch, length, d_inner = u.shape
        delta_a = torch.exp(delta.unsqueeze(-1) * A)                       # [b, l, d_inner, n]
        delta_b_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)  # [b, l, d_inner, n]
        state = u.new_zeros(batch, d_inner, self.d_state)
        outputs = []
        for t in range(length):
            state = delta_a[:, t] * state + delta_b_u[:, t]                # h_t = exp(dA) h_{t-1} + dB u_t
            outputs.append((state * C[:, t].unsqueeze(1)).sum(dim=-1))     # y_t = C_t h_t
        return torch.stack(outputs, dim=1) + u * D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, time, d_model] -> [batch, time, d_model] (causal in time)."""
        length = x.shape[1]
        main, gate = self.in_proj(x).chunk(2, dim=-1)
        main = self.conv1d(main.transpose(1, 2))[:, :, :length].transpose(1, 2)  # causal: drop right padding
        main = F.silu(main)
        dt, B, C = self.x_proj(main).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))
        y = self.selective_scan(main, delta, -torch.exp(self.A_log.float()), B, C, self.D.float())
        return self.out_proj(y * F.silu(gate))


class MambaRegressor(nn.Module):
    """Stack of pure-PyTorch Mamba blocks mapping [batch, time, features] to RUL.

    Runs everywhere (CPU, CUDA, MPS); no optional packages needed.
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
        self.input_projection = nn.Linear(input_size, d_model)
        self.layers = nn.ModuleList(
            nn.ModuleDict({"mamba": MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand),
                           "norm": nn.LayerNorm(d_model)})
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