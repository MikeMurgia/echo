"""Phase 3: decision layer and decision-flip experiment, from the conformal calibration outputs.

    python scripts/decision_layer.py --subset FD002 --repeats 20 --write-final-model

Reads results/<subset>/conformal_windows.csv (written by scripts/conformal_calibration.py). Windows marked
split == "cal" provide the residuals that define p (and fit the margin baselines); windows marked
split == "eval" (different engines) are used to score every decision, so nothing is assessed on the data
that fitted it.

For each planning horizon H and cost ratio rho = Cu / Cs it compares, at every assessment window:
    point         act when the predicted RUL is <= H                   (ignores uncertainty)
    margin        act when predicted RUL <= H + m, with the single best constant margin m for this rho
    margin_stage  the same, with a separate best margin for each predicted stage
    interval      act when the lower end of the 90% Mondrian interval is <= H
    echo          act when p * rho > 1, i.e. rho > T = 1 / p           (p = P(true RUL <= H), stage residuals)
    echo local<w> the same rule, with p from calibration windows whose predicted RUL is within +-w cycles
                  of the window's own prediction (ablation; w = 5, 10, 20 by default, all reported)
    oracle        act exactly when the engine really fails within H    (best possible)
plus the always-act and never-act extremes. The margins are fitted on the calibration engines only.
Costs are per decision point: acting costs 1, waiting costs rho if the engine fails within H and 0
otherwise. Decision points are windows, not engine lifetimes (test engines are cut off before failure, so a
run-to-failure simulation is not possible on them).

A "flip" is a window where echo and the point rule disagree. Writes to results/<subset>/ :
    decision_sweep.csv          flips and costs by horizon, cost ratio, p source (stage or pooled) and scope
    decision_bootstrap.csv      echo minus each baseline, with 95% engine-bootstrap CIs (key cost ratios)
    decision_reliability.csv    predicted p vs observed failure frequency, with Brier scores
    decision_repeats*.csv       (with --repeats N) the key comparisons over N random engine splits
    decision_model_final.json   stage residuals and the local model (calibration pairs) fitted on ALL windows,
                                for the dashboard (with --write-final-model)
    fig_decision_flips.png, fig_decision_cost.png, fig_decision_reliability.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision import (  # noqa: E402
    MARGIN_GRID, STAGES, LocalResidualModel, ResidualModel, bootstrap_cost_difference, fit_margins, save_model)

REQUIRED = ["unit", "cycle", "last_cycle", "y", "yhat", "residual", "abs_res", "stage", "split", "hw_mondrian"]
SCOPES = ("all", "early", "mid", "late", "final_cycle")
KEY_RHOS = [2.0, 5.0, 10.0, 20.0, 50.0]
BASELINES = ("point", "margin", "margin_stage", "interval")
COST_COLUMNS = ["cost_echo", "cost_point", "cost_margin", "cost_margin_stage", "cost_interval",
                "cost_oracle", "cost_always", "cost_never"]
P_EDGES = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0001]
TRAIN_CAP = 125


def scope_masks(ev: pd.DataFrame) -> dict:
    masks = {"all": np.ones(len(ev), bool)}
    masks.update({s: (ev["stage"] == s).to_numpy() for s in STAGES})
    masks["final_cycle"] = (ev["cycle"] == ev["last_cycle"]).to_numpy()  # the official test point per engine
    return masks


def calibration_pack(cal: pd.DataFrame, truth_col: str, horizon: float) -> dict:
    """What the margin baselines need from the calibration windows for one horizon."""
    yhat = cal["yhat"].to_numpy()
    return {"A": yhat[:, None] <= horizon + MARGIN_GRID[None, :],
            "failure": (cal[truth_col] <= horizon).to_numpy(), "stage": cal["stage"].to_numpy()}


def window_costs(yhat, stage, hw, failure, p, horizon, rho, margins):
    """Per-window action and cost of every rule at one horizon and cost ratio."""
    m_stage = np.select([stage == "early", stage == "mid"], [margins["stage"]["early"], margins["stage"]["mid"]],
                        default=margins["stage"]["late"])
    acts = {"echo": p * rho > 1.0, "point": yhat <= horizon, "margin": yhat <= horizon + margins["global"],
            "margin_stage": yhat <= horizon + m_stage, "interval": (yhat - hw) <= horizon}
    wait_cost = rho * failure
    costs = {f"cost_{name}": np.where(a, 1.0, wait_cost) for name, a in acts.items()}
    costs.update({"cost_oracle": failure.astype(float), "cost_always": np.ones(len(p)), "cost_never": wait_cost})
    return acts, costs


def make_sources(model, cal, bandwidths, min_local, pooled=True) -> dict:
    """Ways of estimating p: stage residuals (the main method), pooled residuals, and local-in-predicted-RUL residuals."""
    sources = {"stage": lambda y, s, h: model.failure_prob(y, s, h)}
    if pooled:
        sources["pooled"] = lambda y, s, h: model.failure_prob(y, s, h, pooled=True)
    for w in bandwidths:
        local = LocalResidualModel.fit(cal["yhat"].to_numpy(), cal["residual"].to_numpy(), w, model, min_local)
        sources[f"local{w:g}"] = local.failure_prob
    return sources


def decision_rows(ev, p, source, horizon, rhos, failure, masks, margins_by_rho) -> list:
    yhat, stage, hw = ev["yhat"].to_numpy(), ev["stage"].to_numpy(), ev["hw_mondrian"].to_numpy()
    rows = []
    for rho in rhos:
        margins = margins_by_rho[rho]
        acts, costs = window_costs(yhat, stage, hw, failure, p, horizon, rho, margins)
        echo, point = acts["echo"], acts["point"]
        for scope, mask in masks.items():
            n = int(mask.sum())
            if n == 0:
                continue
            e, q = echo[mask], point[mask]
            row = {"horizon": horizon, "rho": float(rho), "p_source": source, "scope": scope, "windows": n,
                   "engines": int(ev["unit"][mask].nunique()), "failures": int(failure[mask].sum()),
                   "flip_rate": float((e != q).mean()), "wait_to_act": int((e & ~q).sum()),
                   "act_to_wait": int((~e & q).sum()), "echo_act_rate": float(e.mean()),
                   "point_act_rate": float(q.mean()), "margin_global": margins["global"],
                   **{f"margin_{s}": margins["stage"][s] for s in STAGES}}
            row.update({k: float(v[mask].mean()) for k, v in costs.items()})
            rows.append(row)
    return rows


def margins_for(cal_pack, rhos) -> dict:
    return {rho: fit_margins(cal_pack["A"], cal_pack["failure"], cal_pack["stage"], rho) for rho in rhos}


def bootstrap_rows(ev, p_by_source, horizon, rhos, failure, margins_by_rho, masks, rng, n_boot) -> list:
    yhat, stage, hw = ev["yhat"].to_numpy(), ev["stage"].to_numpy(), ev["hw_mondrian"].to_numpy()
    rows = []
    for rho in rhos:
        costs_by_source = {src: window_costs(yhat, stage, hw, failure, p, horizon, rho, margins_by_rho[rho])[1]
                           for src, p in p_by_source.items()}
        for source, costs in costs_by_source.items():
            baselines = {b: costs[f"cost_{b}"] for b in BASELINES}
            if source != "stage":
                baselines["echo_stage"] = costs_by_source["stage"]["cost_echo"]
            for scope in ("all", "final_cycle"):
                mask = masks[scope]
                if not mask.any():
                    continue
                units = ev["unit"].to_numpy()[mask]
                for base, base_cost in baselines.items():
                    diff, lo, hi, share = bootstrap_cost_difference(units, costs["cost_echo"][mask], base_cost[mask], rng, n_boot)
                    rows.append({"horizon": horizon, "rho": rho, "p_source": source, "scope": scope, "baseline": base,
                                 "engines": int(len(np.unique(units))), "cost_echo": float(costs["cost_echo"][mask].mean()),
                                 "cost_baseline": float(base_cost[mask].mean()), "echo_minus_baseline": diff,
                                 "ci_low": lo, "ci_high": hi, "share_echo_cheaper": share})
    return rows


def reliability_rows(ev, p, source, horizon, failure) -> list:
    prevalence = float(failure.mean())
    brier = float(np.mean((p - failure) ** 2))
    base = float(np.mean((prevalence - failure) ** 2))
    bins = np.digitize(p, P_EDGES[1:-1])
    rows = []
    for b in range(len(P_EDGES) - 1):
        m = bins == b
        if not m.any():
            continue
        rows.append({"horizon": horizon, "p_source": source, "bin_low": P_EDGES[b], "bin_high": min(P_EDGES[b + 1], 1.0),
                     "windows": int(m.sum()), "engines": int(ev["unit"][m].nunique()),
                     "mean_p": float(p[m].mean()), "observed_rate": float(failure[m].mean()),
                     "brier": brier, "brier_constant_baseline": base, "prevalence": prevalence})
    return rows


def repeated_splits(win: pd.DataFrame, truth_col: str, args, horizons, rhos) -> pd.DataFrame:
    """Re-split the engines at random (as the conformal script does) and redo the key comparisons each time."""
    from scripts.conformal_calibration import fit_quantiles, stratified_engine_split  # heavy imports, only needed here

    final = win.groupby("unit")[truth_col].last()
    rows = []
    for k in range(args.repeats):
        seed = args.seed + k
        cal_units, _ = stratified_engine_split(final.index.to_numpy(), final.to_numpy(), seed)
        is_cal = win["unit"].isin(cal_units)
        cal, ev = win[is_cal], win[~is_cal].reset_index(drop=True)
        model = ResidualModel.fit(cal["residual"].to_numpy(), cal["stage"].to_numpy(), args.min_group)
        fq = fit_quantiles(cal["abs_res"].to_numpy(), cal["stage"].to_numpy(), args.interval_coverage, args.min_group)
        ev = ev.assign(hw_mondrian=ev["stage"].map(fq["stage"]).to_numpy(dtype=float))
        masks = {s: m for s, m in scope_masks(ev).items() if s in ("all", "final_cycle")}
        sources = make_sources(model, cal, args.bandwidths, args.min_local, pooled=False)
        for horizon in horizons:
            failure = (ev[truth_col] <= horizon).to_numpy()
            margins = margins_for(calibration_pack(cal, truth_col, horizon), rhos)
            for label, fn in sources.items():
                p = fn(ev["yhat"].to_numpy(), ev["stage"].to_numpy(), horizon)
                for r in decision_rows(ev, p, label, horizon, rhos, failure, masks, margins):
                    rows.append({"repeat": k, "seed": seed, **r})
    return pd.DataFrame(rows)


def summarize_repeats(by_split: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (horizon, rho, scope, source), g in by_split.groupby(["horizon", "rho", "scope", "p_source"]):
        row = {"horizon": horizon, "rho": rho, "scope": scope, "p_source": source, "splits": len(g)}
        for col in COST_COLUMNS:
            row[f"{col}_mean"], row[f"{col}_std"] = g[col].mean(), g[col].std()
        for base in BASELINES:
            row[f"echo_cheaper_than_{base}"] = int((g["cost_echo"] < g[f"cost_{base}"] - 1e-12).sum())
        row["flip_rate_mean"] = g["flip_rate"].mean()
        out.append(row)
    return pd.DataFrame(out)


def print_summary(sweep, boot, horizon) -> None:
    d = sweep[(sweep.horizon == horizon) & (sweep.p_source == "stage") & sweep.rho.isin(KEY_RHOS)]
    cols = ["rho", "scope", "windows", "flip_rate", "cost_echo", "cost_point", "cost_margin", "cost_margin_stage",
            "cost_interval", "cost_oracle", "cost_never"]
    with pd.option_context("display.width", 230, "display.max_columns", None, "display.float_format", "{:.3f}".format):
        print(f"\nHorizon H = {horizon:g} cycles, stage-conditional p (cost per decision point; lower is better):")
        print(d[d.scope.isin(["all", "final_cycle"])][cols].sort_values(["scope", "rho"]).to_string(index=False))
        m = sweep[(sweep.horizon == horizon) & (sweep.p_source == "stage") & (sweep.scope == "all") & sweep.rho.isin(KEY_RHOS)]
        print("\nBest margins fitted on the calibration engines (cycles added to H; 0 = the plain point rule):")
        print(m[["rho", "margin_global", "margin_early", "margin_mid", "margin_late"]].sort_values("rho").to_string(index=False))
        b = boot[(boot.horizon == horizon) & (boot.scope == "all") & (boot.p_source == "stage")]
        print("\nECHO cost minus baseline cost, all windows, 95% engine-bootstrap CI (negative = ECHO cheaper):")
        print(b[["rho", "baseline", "echo_minus_baseline", "ci_low", "ci_high", "share_echo_cheaper"]]
              .sort_values(["baseline", "rho"]).to_string(index=False))
        bf = boot[(boot.horizon == horizon) & (boot.scope == "final_cycle") & (boot.p_source == "stage")]
        if len(bf):
            print("\nSame at the official final cycle only (one window per engine):")
            print(bf[["rho", "baseline", "engines", "echo_minus_baseline", "ci_low", "ci_high"]]
                  .sort_values(["baseline", "rho"]).to_string(index=False))


def print_repeats(by_split: pd.DataFrame, horizon: float) -> None:
    d = by_split[(by_split.horizon == horizon) & (by_split.scope == "all")]
    piv = d.pivot_table(index=["repeat", "rho"], columns="p_source", values="cost_echo")
    base = d[d.p_source == "stage"].set_index(["repeat", "rho"])[["cost_point", "cost_margin", "cost_interval"]]
    piv = piv.join(base)
    sources = [c for c in piv.columns if c not in ("cost_point", "cost_margin", "cost_interval")]
    n = d["repeat"].nunique()
    with pd.option_context("display.width", 230, "display.max_columns", None, "display.float_format", "{:.4f}".format):
        print(f"\nOver {n} random engine splits, H = {horizon:g}, all windows: mean cost per decision point")
        print("(columns after cost_interval are ECHO with that source of p):")
        print(piv.groupby("rho").mean()[["cost_point", "cost_margin", "cost_interval"] + sources].to_string())
        wins = {}
        for src in sources:
            wins[f"{src}<margin"] = (piv[src] < piv["cost_margin"]).groupby("rho").sum()
            if src != "stage":
                wins[f"{src}<stage"] = (piv[src] < piv["stage"]).groupby("rho").sum()
        print(f"\nNumber of the {n} splits in which each ECHO variant was cheaper than the best margin / than stage ECHO:")
        print(pd.DataFrame(wins).to_string())


def print_ablation(sweep, boot, horizon) -> None:
    d = sweep[(sweep.horizon == horizon) & (sweep.scope == "all") & sweep.rho.isin(KEY_RHOS)]
    piv = d.pivot_table(index="rho", columns="p_source", values="cost_echo")
    ref = d[d.p_source == "stage"].set_index("rho")[["cost_point", "cost_margin"]]
    b = boot[(boot.horizon == horizon) & (boot.scope == "all") & boot.p_source.str.startswith("local")]
    with pd.option_context("display.width", 230, "display.max_columns", None, "display.float_format", "{:.4f}".format):
        print(f"\nAblation, H = {horizon:g}, all windows: ECHO cost by source of p (point and best-margin shown for reference)")
        print(ref.join(piv).to_string())
        for base in ("echo_stage", "margin"):
            x = b[b.baseline == base]
            if len(x):
                print(f"\nLocal ECHO minus {'stage ECHO' if base == 'echo_stage' else 'best-margin rule'} "
                      f"(negative = local cheaper), 95% engine-bootstrap CI:")
                print(x[["p_source", "rho", "echo_minus_baseline", "ci_low", "ci_high"]].sort_values(["p_source", "rho"]).to_string(index=False))


def plot_flips(sweep, horizon, out):
    d = sweep[(sweep.horizon == horizon) & (sweep.p_source == "stage")]
    fig, ax = plt.subplots(figsize=(7, 4))
    for scope, color in zip(("all", "early", "mid", "late"), ("k", "#4c78a8", "#f58518", "#e45756")):
        s = d[d.scope == scope].sort_values("rho")
        if len(s):
            ax.plot(s.rho, 100 * s.flip_rate, label=scope, color=color, lw=2 if scope == "all" else 1.3)
    ax.set(xscale="log", xlabel="cost ratio  Cu / Cs", ylabel="windows where ECHO differs from point rule (%)",
           title=f"Decision flips, horizon {horizon:g} cycles")
    ax.grid(alpha=0.25)
    ax.legend(title="predicted stage", fontsize="small")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_cost(sweep, horizon, out):
    d = sweep[(sweep.horizon == horizon) & (sweep.p_source == "stage") & (sweep.scope == "all")].sort_values("rho")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    styles = (("cost_never", "never act", "gray", ":"), ("cost_always", "always act", "gray", "--"),
              ("cost_point", "point rule", "#f58518", "-"), ("cost_margin", "point + best margin", "#b279a2", "-"),
              ("cost_margin_stage", "point + best margin per stage", "#9d755d", "-"),
              ("cost_interval", "interval lower bound", "#54a24b", "-"),
              ("cost_echo", "ECHO (breakeven)", "#4c78a8", "-"), ("cost_oracle", "oracle", "k", "-"))
    for col, label, color, ls in styles:
        ax.plot(d.rho, d[col], label=label, color=color, ls=ls, lw=2.2 if col == "cost_echo" else 1.3)
    ax.set(xscale="log", yscale="log", xlabel="cost ratio  Cu / Cs", ylabel="mean cost per decision point",
           title=f"Cost of each rule, horizon {horizon:g} cycles")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize="x-small")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_reliability(rel, horizon, out):
    d = rel[rel.horizon == horizon]
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.plot([0, 1], [0, 1], "k:", label="perfect")
    palette = {"stage": "#4c78a8", "pooled": "gray"}
    locals_ = [s for s in d.p_source.unique() if s.startswith("local")]
    for source, color in zip(locals_, ("#f58518", "#e45756", "#b279a2", "#54a24b")):
        palette[source] = color
    for source, color in palette.items():
        s = d[d.p_source == source]
        if len(s):
            ax.plot(s.mean_p, s.observed_rate, "o-", ms=4, color=color, label=f"{source} residuals")
    ax.set(xlabel="predicted probability of failure within horizon", ylabel="observed frequency",
           title=f"Reliability of p, horizon {horizon:g}", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.25)
    ax.legend(fontsize="x-small")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", choices=("FD001", "FD002"), default="FD002")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--horizons", type=float, nargs="+", default=[15.0, 30.0, 50.0],
                        help="planning horizons H in cycles (each must be below the 125 training cap)")
    parser.add_argument("--main-horizon", type=float, default=30.0, help="horizon used for the figures and printout")
    parser.add_argument("--rho-min", type=float, default=1.0)
    parser.add_argument("--rho-max", type=float, default=100.0)
    parser.add_argument("--rho-points", type=int, default=40)
    parser.add_argument("--min-group", type=int, default=30,
                        help="stages with fewer calibration windows use the pooled residuals")
    parser.add_argument("--bandwidths", type=float, nargs="+", default=[5.0, 10.0, 20.0],
                        help="half-widths (cycles of predicted RUL) for the local-residual ablation")
    parser.add_argument("--final-bandwidth", type=float, default=10.0,
                        help="default local bandwidth stored in decision_model_final.json (the other --bandwidths stay selectable)")
    parser.add_argument("--min-local", type=int, default=50,
                        help="local residual sets smaller than this fall back to the stage residuals")
    parser.add_argument("--n-boot", type=int, default=1000, help="engine-bootstrap resamples for the confidence intervals")
    parser.add_argument("--repeats", type=int, default=1,
                        help="number of random engine splits for the stability check (1 = the saved split only)")
    parser.add_argument("--seed", type=int, default=42, help="first seed for --repeats and the bootstrap")
    parser.add_argument("--interval-coverage", type=float, default=0.90,
                        help="coverage of the interval rule in --repeats (single split uses the saved intervals)")
    parser.add_argument("--write-final-model", action="store_true",
                        help="also fit on ALL windows and write decision_model_final.json for the dashboard")
    args = parser.parse_args()
    if any(h <= 0 or h >= TRAIN_CAP for h in args.horizons):
        parser.error(f"every horizon must be between 0 and {TRAIN_CAP}")
    if args.main_horizon not in args.horizons:
        args.horizons.append(args.main_horizon)
    if not 0 < args.rho_min < args.rho_max:
        parser.error("need 0 < --rho-min < --rho-max")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    out_dir = args.results_dir / args.subset
    path = out_dir / "conformal_windows.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run scripts/conformal_calibration.py first.")
    win = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in win.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns {missing}; rerun scripts/conformal_calibration.py")
    truth_col = "y_raw" if "y_raw" in win.columns else "y"
    cal, ev = win[win.split == "cal"], win[win.split == "eval"].reset_index(drop=True)
    if cal.empty or ev.empty:
        raise SystemExit("Need both calibration and assessment windows in conformal_windows.csv")

    model = ResidualModel.fit(cal["residual"].to_numpy(), cal["stage"].to_numpy(), args.min_group)
    print(f"{args.subset}: residuals from {len(cal)} calibration windows / {cal['unit'].nunique()} engines; "
          f"decisions scored on {len(ev)} windows / {ev['unit'].nunique()} different engines")
    for s in STAGES:
        n_s = int((cal["stage"] == s).sum())
        note = "  <- too few, using pooled residuals" if model.fallback[s] else ""
        print(f"  {s:5s}: {n_s} calibration residuals, mean {cal.loc[cal['stage'] == s, 'residual'].mean():+.2f}{note}")

    rhos = np.union1d(np.geomspace(args.rho_min, args.rho_max, args.rho_points), KEY_RHOS)
    rhos = [float(r) for r in rhos if args.rho_min <= r <= args.rho_max]
    key_rhos = [r for r in KEY_RHOS if r in rhos]
    masks = scope_masks(ev)
    rng = np.random.default_rng(args.seed)

    sources = make_sources(model, cal, args.bandwidths, args.min_local)
    sweep_rows, rel_rows, boot_rows = [], [], []
    for horizon in args.horizons:
        failure = (ev[truth_col] <= horizon).to_numpy()
        margins = margins_for(calibration_pack(cal, truth_col, horizon), rhos)
        p_by_source = {label: fn(ev["yhat"].to_numpy(), ev["stage"].to_numpy(), horizon) for label, fn in sources.items()}
        for label, p in p_by_source.items():
            sweep_rows += decision_rows(ev, p, label, horizon, rhos, failure, masks, margins)
            rel_rows += reliability_rows(ev, p, label, horizon, failure)
        boot_rows += bootstrap_rows(ev, {k: v for k, v in p_by_source.items() if k != "pooled"}, horizon, key_rhos,
                                    failure, margins, masks, rng, args.n_boot)
    sweep, rel, boot = pd.DataFrame(sweep_rows), pd.DataFrame(rel_rows), pd.DataFrame(boot_rows)
    sweep.to_csv(out_dir / "decision_sweep.csv", index=False)
    rel.to_csv(out_dir / "decision_reliability.csv", index=False)
    boot.to_csv(out_dir / "decision_bootstrap.csv", index=False)

    print_summary(sweep, boot, args.main_horizon)
    print_ablation(sweep, boot, args.main_horizon)
    brier = rel.groupby(["horizon", "p_source"])[["brier", "brier_constant_baseline", "prevalence"]].first()
    with pd.option_context("display.float_format", "{:.4f}".format):
        print("\nBrier score of p (lower is better; the baseline predicts the overall failure rate for every window):")
        print(brier.to_string())

    plot_flips(sweep, args.main_horizon, out_dir / "fig_decision_flips.png")
    plot_cost(sweep, args.main_horizon, out_dir / "fig_decision_cost.png")
    plot_reliability(rel, args.main_horizon, out_dir / "fig_decision_reliability.png")

    if args.repeats > 1:
        print(f"\nRepeating the comparison over {args.repeats} random engine splits ...")
        by_split = repeated_splits(win, truth_col, args, args.horizons, key_rhos)
        by_split.to_csv(out_dir / "decision_repeats_by_split.csv", index=False)
        summary = summarize_repeats(by_split)
        summary.to_csv(out_dir / "decision_repeats.csv", index=False)
        print_repeats(by_split, args.main_horizon)

    if args.write_final_model:
        final = ResidualModel.fit(win["residual"].to_numpy(), win["stage"].to_numpy(), args.min_group,
                                  meta={"subset": args.subset, "windows": int(len(win)), "engines": int(win["unit"].nunique())})
        final_local = LocalResidualModel.fit(win["yhat"].to_numpy(), win["residual"].to_numpy(), args.final_bandwidth,
                                             final, args.min_local)
        bandwidths = sorted(set(args.bandwidths) | {args.final_bandwidth})
        target = out_dir / "decision_model_final.json"
        save_model(final, target, local=final_local, bandwidths=bandwidths)
        print(f"\nWrote {target} ({target.stat().st_size / 1024:.0f} KB): stage residuals and the local model "
              f"(default bandwidth +-{args.final_bandwidth:g}), fitted on all {win['unit'].nunique()} engines")
    print(f"\nWrote results to {out_dir}")


if __name__ == "__main__":
    main()