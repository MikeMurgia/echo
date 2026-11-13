"""Data and numbers behind the dashboard. No streamlit, torch or plotting imports, so it is easy to test.

Reads what the pipeline already wrote for one dataset:
    results/<subset>/conformal_windows.csv     every test window: prediction, true RUL, residual, stage, split
    results/<subset>/decision_model_final.json stage residual model (+ local model) fitted on all engines
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.decision import (
    DEFAULT_BAND, STAGES, LocalResidualModel, ResidualModel, advice_shares, decide_triage, load_local_model,
    load_model, p_reliable, stage_for_rul)

WINDOW_COLUMNS = ["unit", "cycle", "last_cycle", "y", "yhat", "residual", "abs_res", "stage", "split", "hw_mondrian"]
DEFAULT_LEVEL = 0.9
DEFAULT_BANDWIDTH = 10.0


def conformal_halfwidths(windows: pd.DataFrame, level: float = DEFAULT_LEVEL, min_group: int = 30) -> dict:
    """Mondrian split-conformal half-width per predicted stage: the ceil((n+1)*level)-th smallest |residual|.

    Stages with fewer than `min_group` windows use all windows instead (same rule as the calibration script).
    """
    everything = windows["abs_res"].to_numpy(dtype=float)
    out = {}
    for s in STAGES:
        a = windows.loc[windows["stage"] == s, "abs_res"].to_numpy(dtype=float)
        if len(a) < min_group:
            a = everything
        n = len(a)
        k = min(int(np.ceil((n + 1) * level)), n)
        out[s] = float(np.sort(a)[k - 1])
    return out


def load_bundle(root, subset: str, bandwidth: float = DEFAULT_BANDWIDTH):
    """Everything the page needs for one dataset, or None if its result files are not there yet."""
    folder = Path(root) / "results" / subset
    windows_path, model_path = folder / "conformal_windows.csv", folder / "decision_model_final.json"
    if not windows_path.exists() or not model_path.exists():
        return None
    windows = pd.read_csv(windows_path)
    missing = [c for c in WINDOW_COLUMNS if c not in windows.columns]
    if missing:
        raise ValueError(f"{windows_path} is missing columns {missing}; rerun scripts/conformal_calibration.py")
    model = load_model(model_path)
    local = load_local_model(model_path, bandwidth)
    cal = windows[windows["split"] == "cal"]
    cal_model = cal_local = None
    if len(cal) and (windows["split"] == "eval").any():
        cal_model = ResidualModel.fit(cal["residual"].to_numpy(), cal["stage"].to_numpy())
        cal_local = LocalResidualModel.fit(cal["yhat"].to_numpy(), cal["residual"].to_numpy(), bandwidth, cal_model)
    sweep_path = folder / "decision_sweep.csv"
    return {
        "subset": subset, "windows": windows, "model": model, "local": local,
        "cal_model": cal_model, "cal_local": cal_local,
        "halfwidth": conformal_halfwidths(windows), "units": sorted(int(u) for u in windows["unit"].unique()),
        "sweep": pd.read_csv(sweep_path) if sweep_path.exists() else None,
    }


def most_urgent_unit(bundle: dict) -> int:
    """The engine whose latest window has the lowest predicted RUL (an interesting default to open the page on)."""
    last = bundle["windows"].sort_values("cycle").groupby("unit")["yhat"].last()
    return int(last.idxmin())


def engine_frame(bundle: dict, unit: int, horizon: float, ratio: float, band: float = DEFAULT_BAND) -> pd.DataFrame:
    """One row per observed cycle of one engine: prediction, interval, p and the three-way advice."""
    w = bundle["windows"]
    g = w[w["unit"] == unit].sort_values("cycle").reset_index(drop=True)
    if g.empty:
        raise KeyError(f"No windows for engine {unit}")
    yhat = g["yhat"].to_numpy(dtype=float)
    stage = stage_for_rul(yhat)
    out = decide_triage(bundle["model"], yhat, stage, horizon, ratio, band, local=bundle["local"])
    hw = np.array([bundle["halfwidth"][s] for s in stage])
    truth = g["y_raw"].to_numpy(dtype=float) if "y_raw" in g else g["y"].to_numpy(dtype=float)
    return pd.DataFrame({
        "cycle": g["cycle"].to_numpy(), "yhat": yhat, "true_capped": g["y"].to_numpy(dtype=float), "true_raw": truth,
        "stage": stage, "lo": np.maximum(yhat - hw, 0.0), "hi": yhat + hw,
        "p": out["p"], "score": out["score"], "advice": out["advice"], "echo_act": out["echo_act"],
        "point_act": out["point_act"],
        "reliable": p_reliable(bundle["model"], yhat, stage, bundle["local"]),
        "n_nearby": bundle["local"].n_nearby(yhat) if bundle["local"] is not None else np.full(len(yhat), np.nan),
    })


def latest_table(bundle: dict, horizon: float, ratio: float, band: float = DEFAULT_BAND) -> pd.DataFrame:
    """One row per engine at its latest observed cycle: prediction, interval, p and advice (for review and download)."""
    last = bundle["windows"].sort_values("cycle").groupby("unit").tail(1).sort_values("unit").reset_index(drop=True)
    yhat = last["yhat"].to_numpy(dtype=float)
    stage = stage_for_rul(yhat)
    out = decide_triage(bundle["model"], yhat, stage, horizon, ratio, band, local=bundle["local"])
    hw = np.array([bundle["halfwidth"][s] for s in stage])
    return pd.DataFrame({
        "engine": last["unit"].to_numpy(), "last_cycle": last["cycle"].to_numpy(),
        "predicted_rul": np.round(yhat, 1), "interval_low": np.round(np.maximum(yhat - hw, 0.0), 1),
        "interval_high": np.round(yhat + hw, 1), "stage": stage, "p_fail_within_H": np.round(out["p"], 4),
        "advice": out["advice"],
    })


def holds_up(bundle: dict, stage: str, horizon: float, ratio: float, band: float = DEFAULT_BAND) -> dict:
    """Checks on the held-out half of the test engines (models fitted on the other half, as in the report).

    Returns interval coverage and error for `stage`, and the cost of ECHO vs the point rule over all held-out
    windows (acting costs 1, waiting costs `ratio` if the engine really fails within `horizon`). This is one
    split with no confidence interval; the report has the repeated-split numbers for the validated horizons.
    """
    w = bundle["windows"]
    ev = w[w["split"] == "eval"]
    if ev.empty or bundle["cal_model"] is None:
        return {"available": False}
    in_stage = ev[ev["stage"] == stage]
    out = {"available": True, "windows": int(len(ev)), "engines": int(ev["unit"].nunique()),
           "stage_windows": int(len(in_stage))}
    if len(in_stage):
        out["coverage"] = float((in_stage["abs_res"] <= in_stage["hw_mondrian"]).mean())
        out["rmse"] = float(np.sqrt((in_stage["residual"] ** 2).mean()))
        out["mae"] = float(in_stage["abs_res"].mean())
    else:
        out.update(coverage=float("nan"), rmse=float("nan"), mae=float("nan"))
    truth = ev["y_raw"] if "y_raw" in ev else ev["y"]
    failure = (truth.to_numpy(dtype=float) <= horizon)
    decided = decide_triage(bundle["cal_model"], ev["yhat"].to_numpy(), ev["stage"].to_numpy(), horizon, ratio, band,
                            local=bundle["cal_local"])

    def mean_cost(act):
        return float(np.where(act, 1.0, ratio * failure).mean())

    out["cost_echo"], out["cost_point"] = mean_cost(decided["echo_act"]), mean_cost(decided["point_act"])
    out["change_pct"] = (100.0 * (out["cost_echo"] - out["cost_point"]) / out["cost_point"]
                         if out["cost_point"] > 0 else float("nan"))
    out["investigate_share"] = advice_shares(decided["advice"])["INVESTIGATE"]
    return out