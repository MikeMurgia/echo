"""Global vs Mondrian (RUL-stage-conditional) split conformal intervals for a trained RUL model.

Calibration and assessment both use the OFFICIAL TEST ENGINES at every cycle (never seen in
training or checkpoint selection), split BY ENGINE into a calibration half and an assessment
half. The split is stratified by each engine's final RUL so both halves contain engines
near failure.

    python scripts/conformal_calibration.py --checkpoint results/checkpoints/transformer_FD002_seed1_best.pt

Extra options:
    --repeats N   repeat the engine split with N different seeds and report mean/min/max coverage
                  per stage (writes conformal_repeats.csv, conformal_repeats_quantiles.csv). With few
                  late-stage engines one split is noisy; this shows how noisy.
    --final       calibrate on ALL test engines and write conformal_quantiles_final.json for the
                  dashboard / decision layer. Its coverage cannot be assessed on the same engines;
                  the repeated-split results are the evidence that the method works.

Writes to results/<subset>/ :
    conformal_windows.csv        every window: prediction, true RUL, signed residual, intervals, split
    conformal_quantiles.json     thresholds, quantiles, calibration counts (read by the decision layer)
    conformal_summary.csv        coverage / width / bootstrap CI by method and slice
    conformal_coverage_sweep.csv coverage and width at several target coverage levels
    fig_conformal_bands.png, fig_conformal_coverage.png

Groups are defined from PREDICTED RUL only (known at deployment), never true RUL.
Coverage by TRUE-RUL bin is reported as a diagnostic: it shows where a flat interval fails.
Signed residual = predicted - true, so positive means RUL was overestimated (a late warning).
"""
from __future__ import annotations

import argparse
import json
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

from src.dataset import SlidingWindowDataset, read_normalized, sensor_columns  # noqa: E402
from src.models import build_model  # noqa: E402

STAGES = ("early", "mid", "late")
TRUE_BINS = [(0, 15), (16, 30), (31, 60), (61, 100), (101, 10_000)]
TRUE_BIN_LABELS = ["0-15", "16-30", "31-60", "61-100", ">100"]


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


def stage_for_rul(rul: np.ndarray, early_threshold: float, late_threshold: float) -> np.ndarray:
    """early: RUL > early_threshold; late: RUL < late_threshold; mid otherwise."""
    return np.where(rul > early_threshold, "early", np.where(rul >= late_threshold, "mid", "late"))


def conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    """Finite-sample conformal quantile (ceil((n+1)*coverage)-th smallest score)."""
    n = len(scores)
    if n == 0:
        raise ValueError("Calibration split is empty")
    rank = int(np.ceil((n + 1) * coverage))
    if rank > n:
        return float("inf")
    return float(np.partition(scores, rank - 1)[rank - 1])


def fit_quantiles(abs_res: np.ndarray, stage: np.ndarray, coverage: float, min_group: int) -> dict:
    """One global quantile plus one per stage; a stage with < min_group windows falls back to global."""
    q_global = conformal_quantile(abs_res, coverage)
    q_stage, fallback = {}, {}
    for s in STAGES:
        scores = abs_res[stage == s]
        if len(scores) >= min_group:
            q_stage[s], fallback[s] = conformal_quantile(scores, coverage), False
        else:
            q_stage[s], fallback[s] = q_global, True
    return {"global": q_global, "stage": q_stage, "fallback": fallback}


def stratified_engine_split(units: np.ndarray, final_rul: np.ndarray, seed: int):
    """Sort engines by final RUL, then send one engine of each consecutive pair to each half."""
    order = np.argsort(final_rul, kind="stable")
    units = units[order]
    rng = np.random.default_rng(seed)
    cal, ev = [], []
    for i in range(0, len(units) - 1, 2):
        pair = units[i:i + 2].copy()
        rng.shuffle(pair)
        cal.append(pair[0])
        ev.append(pair[1])
    if len(units) % 2:
        cal.append(units[-1])
    return set(int(u) for u in cal), set(int(u) for u in ev)


def build_windows(frame: pd.DataFrame, dataset: SlidingWindowDataset) -> pd.DataFrame:
    """Per-window metadata (engine, end cycle, regime, uncapped RUL) aligned with dataset order."""
    per_unit = {}
    for unit, g in frame.groupby("unit", sort=True):
        per_unit[int(unit)] = (g["cycle"].to_numpy(), g["regime"].to_numpy() if "regime" in g else np.zeros(len(g), int),
                               g["RUL_raw"].to_numpy() if "RUL_raw" in g else g["RUL"].to_numpy())
    rows = []
    for engine_idx, start in dataset.index:
        unit, _, rul = dataset.engines[engine_idx]
        end = min(start + dataset.window_size, len(rul))
        cycles, regimes, raw = per_unit[int(unit)]
        rows.append((int(unit), int(cycles[end - 1]), len(rul), int(regimes[end - 1]), float(raw[end - 1])))
    return pd.DataFrame(rows, columns=["unit", "cycle", "last_cycle", "regime", "y_raw"])


def engine_bootstrap(unit: np.ndarray, covered: np.ndarray, rng, n_boot: int):
    """95% CI of coverage, resampling ENGINES (windows of one engine are strongly dependent)."""
    units, inverse = np.unique(unit, return_inverse=True)
    k = len(units)
    if k < 3 or n_boot < 1:
        return float("nan"), float("nan")
    hits = np.bincount(inverse, weights=covered.astype(float), minlength=k)
    counts = np.bincount(inverse, minlength=k).astype(float)
    idx = rng.integers(0, k, size=(n_boot, k))
    cov = hits[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(np.percentile(cov, 2.5)), float(np.percentile(cov, 97.5))


def summarize(ev: pd.DataFrame, methods: dict, rng, n_boot: int) -> pd.DataFrame:
    """Coverage / width / CI for each method over: overall, predicted stage, true-RUL bin, regime."""
    slices = [("overall", "all", np.ones(len(ev), bool))]
    slices += [("predicted_stage", s, (ev["stage"] == s).to_numpy()) for s in STAGES]
    slices += [("true_rul_bin", label, ((ev["y"] >= lo) & (ev["y"] <= hi)).to_numpy())
               for (lo, hi), label in zip(TRUE_BINS, TRUE_BIN_LABELS)]
    slices += [("regime", str(r), (ev["regime"] == r).to_numpy()) for r in sorted(ev["regime"].unique())]
    rows = []
    for method, hw in methods.items():
        covered = (ev["abs_res"].to_numpy() <= hw)
        for kind, name, mask in slices:
            if not mask.any():
                continue
            lo, hi = engine_bootstrap(ev["unit"].to_numpy()[mask], covered[mask], rng, n_boot)
            rows.append({"method": method, "slice_type": kind, "slice": name, "windows": int(mask.sum()),
                         "engines": int(ev["unit"][mask].nunique()), "coverage": float(covered[mask].mean()),
                         "ci_low": lo, "ci_high": hi, "mean_width": float(np.mean(2 * hw[mask]))})
    return pd.DataFrame(rows)


def repeated_split_check(win: pd.DataFrame, final: pd.Series, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat the stratified engine split with seeds seed, seed+1, ...; return per-split coverage and quantiles."""
    rows, qrows = [], []
    for k in range(args.repeats):
        seed = args.seed + k
        cal_units, _ = stratified_engine_split(final.index.to_numpy(), final.to_numpy(), seed)
        is_cal = win["unit"].isin(cal_units)
        cal, ev = win[is_cal], win[~is_cal]
        fq = fit_quantiles(cal["abs_res"].to_numpy(), cal["stage"].to_numpy(), args.coverage, args.min_group)
        qrows.append({"repeat": k, "seed": seed, "q_global": fq["global"], **{f"q_{s}": fq["stage"][s] for s in STAGES}})
        halfwidths = {"global": np.full(len(ev), fq["global"]),
                      "mondrian": ev["stage"].map(fq["stage"]).to_numpy(dtype=float)}
        slices = [("overall", np.ones(len(ev), bool))] + [(s, (ev["stage"] == s).to_numpy()) for s in STAGES]
        for method, hw in halfwidths.items():
            covered = ev["abs_res"].to_numpy() <= hw
            for name, mask in slices:
                if mask.any():
                    rows.append({"repeat": k, "seed": seed, "method": method, "slice": name,
                                 "engines": int(ev["unit"][mask].nunique()), "coverage": float(covered[mask].mean()),
                                 "mean_width": float(np.mean(2 * hw[mask]))})
    return pd.DataFrame(rows), pd.DataFrame(qrows)


def aggregate_repeats(rows: pd.DataFrame, target: float) -> pd.DataFrame:
    """Mean / std / min / max coverage over the repeated splits, and how many fell below the target."""
    agg = rows.groupby(["method", "slice"]).agg(
        splits=("coverage", "size"), coverage_mean=("coverage", "mean"), coverage_std=("coverage", "std"),
        coverage_min=("coverage", "min"), coverage_max=("coverage", "max"),
        splits_below_target=("coverage", lambda c: int((c < target).sum())),
        mean_width=("mean_width", "mean"), assessment_engines_mean=("engines", "mean")).reset_index()
    order = {"overall": 0, "early": 1, "mid": 2, "late": 3}
    return agg.sort_values(["method", "slice"], key=lambda col: col.map(order) if col.name == "slice" else col,
                           ignore_index=True)


def final_quantiles(win: pd.DataFrame, args, subset: str, model_name) -> dict:
    """Quantiles from ALL test engines, for several coverage levels (for the dashboard's coverage choice)."""
    levels = sorted(set([args.coverage, *args.sweep]))
    out = {"checkpoint": str(args.checkpoint), "subset": subset, "model": model_name,
           "default_coverage": args.coverage, "early_rul": args.early_rul, "late_rul": args.late_rul,
           "min_group": args.min_group, "calibration_engines_total": int(win["unit"].nunique()),
           "calibration_windows": {s: int((win["stage"] == s).sum()) for s in STAGES},
           "calibration_engines": {s: int(win.loc[win["stage"] == s, "unit"].nunique()) for s in STAGES},
           "levels": {},
           "note": ("calibrated on every test engine; coverage is evidenced by the repeated split-assessment, "
                    "not measured on these engines. signed residual = predicted - true RUL (capped).")}
    for level in levels:
        fq = fit_quantiles(win["abs_res"].to_numpy(), win["stage"].to_numpy(), level, args.min_group)
        out["levels"][f"{level:g}"] = {"q_global": fq["global"], "q_stage": fq["stage"],
                                       "stage_fallback_to_global": fq["fallback"]}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument("--sweep", type=float, nargs="+", default=[0.80, 0.90, 0.95],
                        help="target coverage levels for the sensitivity table")
    parser.add_argument("--early-rul", type=float, default=100.0, help="early stage: predicted RUL above this")
    parser.add_argument("--late-rul", type=float, default=30.0, help="late stage: predicted RUL below this")
    parser.add_argument("--min-group", type=int, default=30,
                        help="stages with fewer calibration windows fall back to the global quantile")
    parser.add_argument("--seed", type=int, default=42, help="seed for the engine split and the bootstrap")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--representative-engines", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1,
                        help="number of random engine splits for the coverage-stability check (1 = single split)")
    parser.add_argument("--final", action="store_true",
                        help="also calibrate on ALL test engines and write conformal_quantiles_final.json")
    args = parser.parse_args()
    if not 0 < args.coverage < 1 or any(not 0 < c < 1 for c in args.sweep):
        parser.error("coverage levels must be between 0 and 1")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.late_rul >= args.early_rul:
        parser.error("--late-rul must be less than --early-rul")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    subset = checkpoint["subset"]
    frame = read_normalized(args.data_dir, "test", subset)
    features = checkpoint.get("feature_columns") or sensor_columns(frame)
    ds = SlidingWindowDataset(frame, window_size=int(checkpoint["window_size"]), feature_columns=features,
                              stride=args.stride)
    model = build_model(
        checkpoint.get("model", "transformer"), input_size=int(checkpoint["input_size"]),
        d_model=int(checkpoint.get("d_model", 128)), num_layers=int(checkpoint.get("num_layers", 3)),
        dropout=float(checkpoint.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"{subset}: predicting {len(ds)} windows from {frame['unit'].nunique()} test engines on {device} ...")
    yhat, y = predict(model, ds, device, args.batch_size)
    win = build_windows(frame, ds)
    win["y"], win["yhat"] = y, yhat
    win["residual"] = win["yhat"] - win["y"]            # >0: RUL overestimated (dangerous direction)
    win["abs_res"] = win["residual"].abs()
    win["stage"] = stage_for_rul(yhat, args.early_rul, args.late_rul)

    final = win.groupby("unit")["y_raw"].last()
    cal_units, eval_units = stratified_engine_split(final.index.to_numpy(), final.to_numpy(), args.seed)
    win["split"] = np.where(win["unit"].isin(cal_units), "cal", "eval")
    cal, ev = win[win.split == "cal"], win[win.split == "eval"].copy()
    print(f"calibration: {len(cal)} windows / {len(cal_units)} engines;  assessment: {len(ev)} windows / {len(eval_units)} engines")

    fq = fit_quantiles(cal["abs_res"].to_numpy(), cal["stage"].to_numpy(), args.coverage, args.min_group)
    print(f"\nTarget coverage {args.coverage:.0%}.  Global half-width q = {fq['global']:.2f}")
    for s in STAGES:
        n_s = int((cal["stage"] == s).sum())
        e_s = cal.loc[cal["stage"] == s, "unit"].nunique()
        note = "  <- too few windows, using global quantile" if fq["fallback"][s] else ""
        print(f"  {s:5s}: q = {fq['stage'][s]:.2f}  (calibration windows {n_s}, engines {e_s}){note}")

    # Per-window half-widths for every method, on all windows (cal rows are in-sample, flagged by 'split').
    hw_global_all = np.full(len(win), fq["global"])
    hw_mond_all = win["stage"].map(fq["stage"]).to_numpy(dtype=float)
    win["hw_global"], win["hw_mondrian"] = hw_global_all, hw_mond_all
    win["covered_global"] = win["abs_res"] <= win["hw_global"]
    win["covered_mondrian"] = win["abs_res"] <= win["hw_mondrian"]
    ev = win[win.split == "eval"].copy()

    rng = np.random.default_rng(args.seed)
    summary = summarize(ev, {"global": ev["hw_global"].to_numpy(), "mondrian": ev["hw_mondrian"].to_numpy()},
                        rng, args.bootstrap)
    out_dir = args.results_dir / subset
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "conformal_summary.csv", index=False)
    win.to_csv(out_dir / "conformal_windows.csv", index=False)

    # Sensitivity to the target coverage level.
    sweep_rows = []
    for level in args.sweep:
        f = fit_quantiles(cal["abs_res"].to_numpy(), cal["stage"].to_numpy(), level, args.min_group)
        for method, hw in (("global", np.full(len(ev), f["global"])),
                           ("mondrian", ev["stage"].map(f["stage"]).to_numpy(dtype=float))):
            cov = ev["abs_res"].to_numpy() <= hw
            late = (ev["y"] <= 30).to_numpy()
            sweep_rows.append({"target": level, "method": method, "coverage": float(cov.mean()),
                               "coverage_true_rul_le_30": float(cov[late].mean()) if late.any() else float("nan"),
                               "mean_width": float(np.mean(2 * hw))})
    pd.DataFrame(sweep_rows).to_csv(out_dir / "conformal_coverage_sweep.csv", index=False)

    meta = {
        "checkpoint": str(args.checkpoint), "subset": subset, "model": checkpoint.get("model"),
        "coverage": args.coverage, "early_rul": args.early_rul, "late_rul": args.late_rul, "min_group": args.min_group,
        "q_global": fq["global"], "q_stage": fq["stage"], "stage_fallback_to_global": fq["fallback"],
        "calibration_windows": {s: int((cal["stage"] == s).sum()) for s in STAGES},
        "calibration_engines": {s: int(cal.loc[cal["stage"] == s, "unit"].nunique()) for s in STAGES},
        "seed": args.seed, "calibration_units": sorted(cal_units), "assessment_units": sorted(eval_units),
        "note": "signed residual = predicted - true RUL (capped); positive = RUL overestimated",
    }
    (out_dir / "conformal_quantiles.json").write_text(json.dumps(meta, indent=2))

    show = summary[summary.slice_type.isin(["overall", "predicted_stage", "true_rul_bin"])]
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.3f}".format):
        print("\nAssessment engines (coverage with 95% engine-bootstrap CI):")
        print(show.to_string(index=False))
    sparse = summary[(summary.slice_type != "overall") & (summary.engines < 10)]
    if len(sparse):
        print(f"\nWarning: {sparse['slice'].nunique()} slice(s) have fewer than 10 assessment engines; read their CIs, not their point estimates.")

    # Figure 1: flat vs Mondrian bands for a few assessment engines.
    counts = ev.groupby("unit").size().sort_values()
    picks = np.linspace(0, len(counts) - 1, min(args.representative_engines, len(counts)), dtype=int)
    engines = [int(counts.index[i]) for i in picks]
    fig, axes = plt.subplots(len(engines), 1, figsize=(10, max(3, 2.9 * len(engines))), squeeze=False)
    for ax, unit in zip(axes[:, 0], engines):
        e = ev[ev.unit == unit].sort_values("cycle")
        c = e["cycle"].to_numpy()
        ax.fill_between(c, e.yhat - e.hw_global, e.yhat + e.hw_global, color="gray", alpha=0.2, label="flat (global) interval")
        ax.fill_between(c, e.yhat - e.hw_mondrian, e.yhat + e.hw_mondrian, color="#4c78a8", alpha=0.3, label="Mondrian interval")
        ax.plot(c, e.y, "k", lw=1.4, label="true RUL (capped)")
        ax.plot(c, e.yhat, color="#e45756", lw=1.1, label="prediction")
        ax.set(title=f"Assessment engine {unit}", xlabel="cycle", ylabel="RUL (cycles)")
        ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize="small")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "fig_conformal_bands.png", dpi=150)
    plt.close(fig)

    # Figure 2: coverage by true-RUL bin with engine-bootstrap error bars.
    bins = summary[summary.slice_type == "true_rul_bin"]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.38
    for j, (method, color) in enumerate((("global", "gray"), ("mondrian", "#4c78a8"))):
        d = bins[bins.method == method].set_index("slice").reindex(TRUE_BIN_LABELS).dropna(subset=["coverage"])
        x = np.array([TRUE_BIN_LABELS.index(s) for s in d.index]) + (j - 0.5) * width
        err = np.vstack([d.coverage - d.ci_low, d.ci_high - d.coverage]).clip(min=0)
        ax.bar(x, d.coverage, width, yerr=err, capsize=3, color=color, label=method)
    ax.axhline(args.coverage, color="k", ls=":", label=f"target {args.coverage:.0%}")
    ax.set_xticks(range(len(TRUE_BIN_LABELS)), TRUE_BIN_LABELS)
    ax.set(xlabel="true RUL bin (diagnostic only)", ylabel="coverage", ylim=(0, 1.05))
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_conformal_coverage.png", dpi=150)
    plt.close(fig)
    if args.repeats > 1:
        rep_rows, rep_q = repeated_split_check(win, final, args)
        rep_summary = aggregate_repeats(rep_rows, args.coverage)
        rep_rows.to_csv(out_dir / "conformal_repeats_by_split.csv", index=False)
        rep_summary.to_csv(out_dir / "conformal_repeats.csv", index=False)
        rep_q.to_csv(out_dir / "conformal_repeats_quantiles.csv", index=False)
        with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.3f}".format):
            print(f"\nCoverage over {args.repeats} random engine splits (target {args.coverage:.0%}):")
            print(rep_summary.to_string(index=False))
            print("\nQuantile spread over the same splits:")
            print(rep_q.drop(columns=["repeat"]).describe().loc[["mean", "min", "max"]].to_string())
    if args.final:
        final_meta = final_quantiles(win, args, subset, checkpoint.get("model"))
        (out_dir / "conformal_quantiles_final.json").write_text(json.dumps(final_meta, indent=2))
        q90 = final_meta["levels"][f"{args.coverage:g}"]
        print(f"\nFinal quantiles from all {final_meta['calibration_engines_total']} engines at {args.coverage:.0%}: "
              f"global {q90['q_global']:.2f}, " + ", ".join(f"{k} {v:.2f}" for k, v in q90["q_stage"].items()))
        print(f"Wrote {out_dir / 'conformal_quantiles_final.json'}")
    print(f"\nWrote results to {out_dir}")


if __name__ == "__main__":
    main()