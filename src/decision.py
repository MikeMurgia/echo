"""Decision layer: failure probability from calibration residuals, and the breakeven cost-ratio rule.

No torch or plotting imports, so the dashboard can use this module directly.

Setup. A model predicts RUL `yhat` for a window. The signed residual is e = yhat - y (positive means
RUL was overestimated, the dangerous direction). Given `yhat`, the true RUL is y = yhat - e, so for a
planning horizon H cycles

    p = P(y <= H | yhat) = P(e >= yhat - H),

estimated from the calibration residuals of the window's predicted RUL stage (Mondrian) or from all
residuals pooled (global). These are the same residuals that the conformal intervals are built from.
`p` is smoothed as (count + 1) / (n + 2) so it is never exactly 0 or 1.

Costs. Acting now (planned maintenance) costs Cs. Waiting costs Cu if the engine fails within the
horizon and nothing otherwise. Act now when p * Cu > Cs, i.e. when the cost ratio rho = Cu / Cs
exceeds the breakeven ratio T = 1 / p.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

STAGES = ("early", "mid", "late")
MARGIN_GRID = np.arange(-20.0, 81.0, 1.0)  # safety margins (cycles) tried by the margin baselines
GRID_POINTS = 501  # quantile grid stored in the JSON model for the dashboard


def stage_for_rul(rul, early_threshold: float = 100.0, late_threshold: float = 30.0) -> np.ndarray:
    """early: RUL > early_threshold; late: RUL < late_threshold; mid otherwise (same as the conformal script)."""
    rul = np.asarray(rul, dtype=float)
    return np.where(rul > early_threshold, "early", np.where(rul >= late_threshold, "mid", "late"))


def breakeven_ratio(p) -> np.ndarray:
    """T = 1 / p: the smallest Cu/Cs at which acting now is worth it."""
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore"):
        return 1.0 / p


@dataclass
class ResidualModel:
    """Sorted calibration residuals (predicted - true RUL) for each stage, plus a pooled set."""

    residuals: dict            # "global" and each stage -> sorted np.ndarray
    counts: dict               # original sample size behind each array
    fallback: dict             # stage -> True if it uses the pooled residuals
    min_group: int = 30
    meta: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, residual, stage, min_group: int = 30, meta: dict | None = None) -> "ResidualModel":
        residual = np.asarray(residual, dtype=float)
        stage = np.asarray(stage)
        if residual.size == 0:
            raise ValueError("No calibration residuals")
        pooled = np.sort(residual)
        residuals, counts, fallback = {"global": pooled}, {"global": len(pooled)}, {}
        for s in STAGES:
            r = residual[stage == s]
            if len(r) >= min_group:
                residuals[s], counts[s], fallback[s] = np.sort(r), len(r), False
            else:
                residuals[s], counts[s], fallback[s] = pooled, len(pooled), True
        return cls(residuals, counts, fallback, min_group, meta or {})

    def failure_prob(self, yhat, stage, horizon: float, pooled: bool = False) -> np.ndarray:
        """P(true RUL <= horizon) for each window, from the residuals of its stage (or all, if pooled)."""
        yhat = np.atleast_1d(np.asarray(yhat, dtype=float))
        stage = np.broadcast_to(np.asarray(stage), yhat.shape)
        if not pooled and not np.isin(stage, STAGES).all():
            raise ValueError(f"Unknown stage label(s): {sorted(set(stage.tolist()) - set(STAGES))}")
        threshold = yhat - horizon
        p = np.empty(len(yhat))
        keys = ["global"] if pooled else list(STAGES)
        for key in keys:
            mask = np.ones(len(yhat), bool) if pooled else (stage == key)
            if not mask.any():
                continue
            arr, n = self.residuals[key], self.counts[key]
            at_least = len(arr) - np.searchsorted(arr, threshold[mask], side="left")  # residuals >= threshold
            p[mask] = (at_least / len(arr) * n + 1.0) / (n + 2.0)
        return p

    def to_json(self) -> dict:
        grid = np.linspace(0.0, 1.0, GRID_POINTS)
        return {
            "convention": "residual = predicted - true RUL; p = P(true RUL <= horizon) = P(residual >= yhat - horizon)",
            "min_group": self.min_group, "counts": self.counts, "fallback_to_global": self.fallback,
            "quantile_grid_points": GRID_POINTS,
            "residual_quantiles": {k: np.quantile(v, grid).round(3).tolist() for k, v in self.residuals.items()},
            "meta": self.meta,
        }

    @classmethod
    def from_json(cls, payload: dict) -> "ResidualModel":
        return cls({k: np.sort(np.asarray(v, dtype=float)) for k, v in payload["residual_quantiles"].items()},
                   {k: int(v) for k, v in payload["counts"].items()}, dict(payload["fallback_to_global"]),
                   int(payload.get("min_group", 30)), dict(payload.get("meta", {})))


@dataclass
class LocalResidualModel:
    """Failure probability from the calibration residuals of windows with a SIMILAR predicted RUL.

    Instead of pooling a whole stage, only calibration windows whose prediction is within +-bandwidth cycles of
    the query's prediction are used. This matters because prediction errors grow with predicted RUL, so a pooled
    stage overstates the risk for windows near the low end of the stage. Where fewer than `min_local`
    calibration windows are that close, it falls back to the stage-conditional `ResidualModel`.
    """

    yhat: np.ndarray           # calibration predictions, sorted
    residual: np.ndarray       # matching residuals (predicted - true)
    bandwidth: float
    fallback: ResidualModel
    min_local: int = 50

    @classmethod
    def fit(cls, yhat, residual, bandwidth: float, fallback: ResidualModel, min_local: int = 50) -> "LocalResidualModel":
        yhat, residual = np.asarray(yhat, dtype=float), np.asarray(residual, dtype=float)
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        order = np.argsort(yhat, kind="stable")
        return cls(yhat[order], residual[order], float(bandwidth), fallback, int(min_local))

    def failure_prob(self, yhat, stage, horizon: float) -> np.ndarray:
        yhat = np.atleast_1d(np.asarray(yhat, dtype=float))
        stage = np.broadcast_to(np.asarray(stage), yhat.shape)
        unique, inverse = np.unique(np.round(yhat * 4.0) / 4.0, return_inverse=True)  # 0.25-cycle resolution
        lo = np.searchsorted(self.yhat, unique - self.bandwidth, side="left")
        hi = np.searchsorted(self.yhat, unique + self.bandwidth, side="right")
        p_unique = np.full(len(unique), np.nan)
        for j, (a, b) in enumerate(zip(lo, hi)):
            n = int(b - a)
            if n >= self.min_local:
                at_least = int((self.residual[a:b] >= unique[j] - horizon).sum())
                p_unique[j] = (at_least + 1.0) / (n + 2.0)
        p = p_unique[inverse]
        missing = np.isnan(p)
        if missing.any():
            p[missing] = self.fallback.failure_prob(yhat[missing], stage[missing], horizon)
        return p


    def to_json(self, decimals: int = 2) -> dict:
        """Calibration pairs sorted by predicted RUL (rounded, to keep the file small)."""
        return {"bandwidth": self.bandwidth, "min_local": self.min_local,
                "yhat": np.round(self.yhat, decimals).tolist(), "residual": np.round(self.residual, decimals).tolist()}

    @classmethod
    def from_json(cls, payload: dict, fallback: ResidualModel, bandwidth: float | None = None) -> "LocalResidualModel":
        return cls.fit(np.asarray(payload["yhat"], dtype=float), np.asarray(payload["residual"], dtype=float),
                       float(bandwidth if bandwidth is not None else payload["bandwidth"]), fallback,
                       int(payload.get("min_local", 50)))


def save_model(model: ResidualModel, path: Path, local: "LocalResidualModel | None" = None,
               bandwidths=None) -> None:
    """Write the stage model and, if given, the local model (its calibration pairs) to one JSON file."""
    payload = model.to_json()
    if local is not None:
        payload["local"] = local.to_json()
        payload["local"]["bandwidths_available"] = [float(w) for w in (bandwidths or [local.bandwidth])]
    Path(path).write_text(json.dumps(payload))


def load_model(path: Path) -> ResidualModel:
    return ResidualModel.from_json(json.loads(Path(path).read_text()))


def load_local_model(path: Path, bandwidth: float | None = None) -> "LocalResidualModel | None":
    """The local model stored in the file (None if it was saved without one). bandwidth overrides the saved default."""
    payload = json.loads(Path(path).read_text())
    if "local" not in payload:
        return None
    return LocalResidualModel.from_json(payload["local"], ResidualModel.from_json(payload), bandwidth)


def decide(model: ResidualModel, yhat, stage, horizon: float, cost_ratio: float, pooled: bool = False,
           local: "LocalResidualModel | None" = None) -> dict:
    """Both decisions for each window: ECHO (uncertainty-aware) and the point-prediction rule.

    ECHO acts when p * cost_ratio > 1 (cost_ratio = Cu / Cs). The point rule acts when yhat <= horizon.
    If a LocalResidualModel is given, p comes from it (residuals of windows with similar predicted RUL).
    """
    yhat = np.atleast_1d(np.asarray(yhat, dtype=float))
    p = local.failure_prob(yhat, stage, horizon) if local is not None else model.failure_prob(yhat, stage, horizon, pooled=pooled)
    return {"p": p, "breakeven_ratio": breakeven_ratio(p), "echo_act": p * cost_ratio > 1.0,
            "point_act": yhat <= horizon}


def fit_margins(A: np.ndarray, failure: np.ndarray, stage: np.ndarray, rho: float, grid: np.ndarray = MARGIN_GRID) -> dict:
    """Best constant safety margin for the point rule "act when yhat <= horizon + m", chosen on calibration windows.

    A[i, j] is True when calibration window i would act with margin grid[j]; failure[i] is True when it really
    fails within the horizon. Acting costs 1, waiting costs rho on a failure. Returns the single best margin
    ("global") and the best margin per predicted stage. Ties go to the margin closest to zero.
    """
    cost = np.where(A, 1.0, rho * failure[:, None].astype(float))

    def best(mask) -> float:
        if not mask.any():
            return float("nan")
        c = cost[mask].mean(axis=0)
        ties = np.flatnonzero(np.isclose(c, c.min()))
        return float(grid[ties[np.argmin(np.abs(grid[ties]))]])

    m_global = best(np.ones(len(failure), bool))
    per_stage = {s: best(stage == s) for s in STAGES}
    per_stage = {s: (m if m == m else m_global) for s, m in per_stage.items()}
    return {"global": m_global, "stage": per_stage}


def bootstrap_cost_difference(units, cost_a, cost_b, rng, n_boot: int = 1000):
    """Engine-level bootstrap of mean(cost_a) - mean(cost_b), paired on the same resampled engines.

    Windows of one engine are strongly dependent, so whole engines are resampled. Returns
    (difference, ci_low, ci_high, share of resamples where a is cheaper than b).
    """
    units = np.asarray(units)
    cost_a, cost_b = np.asarray(cost_a, float), np.asarray(cost_b, float)
    ids, inverse = np.unique(units, return_inverse=True)
    k = len(ids)
    observed = float(cost_a.mean() - cost_b.mean())
    if k < 3 or n_boot < 1:
        return observed, float("nan"), float("nan"), float("nan")
    sum_a = np.bincount(inverse, weights=cost_a, minlength=k)
    sum_b = np.bincount(inverse, weights=cost_b, minlength=k)
    count = np.bincount(inverse, minlength=k).astype(float)
    idx = rng.integers(0, k, size=(n_boot, k))
    denom = count[idx].sum(axis=1)
    diff = sum_a[idx].sum(axis=1) / denom - sum_b[idx].sum(axis=1) / denom
    return observed, float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5)), float((diff < 0).mean())