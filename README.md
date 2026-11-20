# ECHO

ECHO predicts remaining useful life (RUL) for NASA C-MAPSS turbofan engines, builds global and life-stage Mondrian conformal intervals, and converts estimated failure risk into maintenance decisions.

## Run the complete project

Requirements: Python 3.10+, internet access for the initial NASA data download, and enough disk space for the data, checkpoints, and generated results. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/run_project.py
```

The runner handles both FD001 and FD002. It downloads and preprocesses the data, trains the tuned Transformer and Random Forest/XGBoost baselines for seeds 1–3, summarizes metrics, runs 20-split conformal calibration and ECHO decision analysis, and regenerates the pipeline figures. It uses the seed-3 Transformer checkpoint for conformal analysis, matching the documented workflow. Training can take a while, especially on CPU.

To inspect the commands without running them:

```bash
python scripts/run_project.py --dry-run
```

Choose a device with `--device auto|cpu|cuda|mps`. To include the optional, slower FD002 Mamba experiment:

```bash
python scripts/run_project.py --include-mamba
```

The runner regenerates files under `data/` and `results/` and overwrites same-named outputs. It does not delete unrelated files. NASA data and model checkpoints are ignored by Git; generated reports/CSVs/figures under `results/` may be tracked by the repository.

## What the pipeline does

1. Downloads FD001 and FD002, removes constant sensors, adds capped and raw RUL labels, and normalizes sensors within operating regimes. FD001 drops sensors 1, 5, 6, 10, 16, 18, and 19; FD002 drops sensors 1, 5, 18, and 19.
2. Trains three tuned Transformer seeds per subset using 30-cycle sensor windows and the 80/20 engine-level train/validation split. The same seeds and engine split helper are used by the tree baselines.
3. Reports capped and raw test metrics for one final window per official test engine. The NASA score is an asymmetric sum, so it scales with test-engine count.
4. Calibrates 90% global and predicted-life-stage Mondrian intervals using half of the official test engines and assesses on the other half. The 20 repeats use different stratified engine splits. Stages are based on predicted RUL: early >100, mid 30–100, and late <30 cycles.
5. Estimates failure probability from calibration residuals and compares ECHO's break-even rule with point, margin, interval, and oracle rules. Costs are per decision window: maintenance costs 1; waiting costs the failure-cost ratio if failure occurs within the planning horizon.

## Outputs

- `results/FD001/` and `results/FD002/`: seed metrics and histories, model comparison, conformal tables and figures, decision sweeps, bootstrap intervals, repeated-split summaries, and final dashboard models.
- `results/checkpoints/`: trained PyTorch checkpoints (ignored by Git).
- `results/`: combined cost sensitivity and residual-source figures.
- `data/`: downloaded source tables, labeled/normalized tables, and fitted preprocessors (ignored by Git).

The dashboard reads the saved conformal windows and final decision model:

```bash
streamlit run app/dashboard.py
```

The live dashboard is [echorulprediction.streamlit.app](https://echorulprediction.streamlit.app/). Uploaded raw engine files are supported only by a local dashboard with the matching checkpoint and `data/preproc_<subset>.joblib`.

## Run individual steps

The full runner is the recommended path. For a single step, run these from the repository root:

```bash
python scripts/get_data.py
python scripts/clean_data.py
python scripts/label_data.py
python scripts/normalize_data.py
python scripts/train.py --subset FD002 --model transformer --seed 3 --tag tuned --patience 3 --lr 3e-4 --d-model 64 --layers 2 --dropout 0.3 --weight-decay 1e-3
python scripts/train_baselines.py --subset FD002 --seed 3
python scripts/conformal_calibration.py --checkpoint results/checkpoints/transformer_FD002_tuned_seed3_best.pt --repeats 20 --final
python scripts/decision_layer.py --subset FD002 --repeats 20 --write-final-model
```

`scripts/summarize_models.py --subset FD002` compares available metrics and selects a sequence-model recipe by mean validation RMSE, not test performance. The full runner filters this comparison to the model/tag combinations it trained; use repeatable `--run MODEL:TAG` options to filter a manual summary. `scripts/plot_abs_residual_vs_predicted_rul.py --subset FD001` creates the residual-by-stage plot.

## Tests

```bash
python -m unittest discover tests
```

## Repository notes

- Do not commit raw NASA data, virtual environments, or model checkpoints.
- The lightweight dashboard dependencies are listed separately in `app/requirements.txt`; the root `requirements.txt` is for training and analysis.
