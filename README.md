# ECHO

Remaining-useful-life prediction with conformal uncertainty and a breakeven cost-ratio decision layer, on NASA C-MAPSS turbofan data.

Projects in Data Science IDC4290

## Setup
Run commands from the repository root with the virtual environment active.
Requires Python 3.10+.

```bash
git clone <repo-url>
cd <repo-folder>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy pandas scikit-learn matplotlib xgboost jupyter torch transformers
```

## 1. Set up the Data (ONLY ONCE)

The NASA files are not stored in this repo. One command downloads them and sets up `data/`:

```bash
python src/data_setup.py
```

This downloads FD001/FD002, drops uninformative sensors (per subset), adds capped `RUL` and uncapped
`RUL_raw` labels, normalizes sensors within operating regimes using training rows only, and saves the
fitted regimes and scalers to `data/preproc_<subset>.joblib` for later use by the dashboard.

The files have no header row. Columns are: unit, cycle, 3 operating settings, 21 sensors.

## 2. Sequence data loaders

After cleaning, labeling and normalizing FD001/FD002, inspect the sequence
windows and loader shapes with:

```bash
python -m src.dataset --subset FD001 --window-size 30 --batch-size 64
python -m src.dataset --subset FD002 --window-size 50
```

## 3. Train the models

All commands run from the repo root with the virtual environment active. FD002 is the main
dataset; FD001 is the replication (change `--subset`).

Sequence model (Transformer). On a CPU use `--stride 2` to halve the epoch time:

```bash
python scripts/train.py --subset FD002 --model transformer --seed 1
```

Three seeds (PowerShell, then bash):

```powershell
foreach ($s in 1,2,3) { python scripts/train.py --subset FD002 --seed $s }
```
```bash
for s in 1 2 3; do python scripts/train.py --subset FD002 --seed $s; done
```

Mamba is a pure-PyTorch implementation of the selective state-space block (Gu & Dao, 2023), so it
runs on any machine with no extra packages (it is not the fused `mamba-ssm` CUDA kernel, and its
weights are not interchangeable with that package). It is slower than the Transformer because of the
loop over time steps; use a GPU (see below) and, if needed, `--batch-size 128`:

```bash
python scripts/train.py --subset FD002 --model mamba --seed 1
```

On Windows, `pip install torch` gives a CPU-only build. For an NVIDIA GPU install the CUDA build from
pytorch.org, then check `python -c "import torch; print(torch.cuda.is_available())"` prints `True`.

Baselines (random forest and XGBoost on the same windows and split; use the same seeds):

```bash
python scripts/train_baselines.py --subset FD002 --seed 1
```

Each run writes `results/<subset>/<model>_seed<seed>_metrics.json` (validation RMSE, test RMSE and
NASA score both capped and uncapped) and a history CSV. The checkpoint goes to
`results/checkpoints/` (gitignored). Compare everything, and get the checkpoint to calibrate:

```bash
python scripts/summarize_models.py --subset FD002
```

The checkpoint is chosen on validation RMSE only, never on test.

## 4. Conformal calibration (global vs Mondrian)

Calibrates 90% intervals on half of the official test engines (every cycle, split by engine and
stratified by final RUL) and assesses them on the other half. These engines were never used for
training or checkpoint selection. Compares one global (flat) quantile with a separate quantile per
predicted-RUL stage (early above 100, mid 30 to 100, late below 30):

```bash
python scripts/conformal_calibration.py --checkpoint results/checkpoints/transformer_FD002_seed1_best.pt
```

Outputs in `results/<subset>/`: `conformal_windows.csv` (every window with prediction, true RUL,
signed residual and both intervals), `conformal_quantiles.json`, `conformal_summary.csv` (coverage,
width and 95% engine-bootstrap CIs by predicted stage, true-RUL bin and regime),
`conformal_coverage_sweep.csv` (80/90/95% targets) and two figures.

Notes: groups use predicted RUL only (known at deployment). Coverage by true-RUL bin is a diagnostic
and is not guaranteed by the method. A stage with fewer than 30 calibration windows falls back to the
global quantile. Overlapping windows are dependent, so CIs resample engines, not windows.

## 5. Tests

```bash
python -m unittest discover tests
```

## Folders and files

| Path | What it is |
|---|---|
| `data/` | Raw NASA files. Gitignored, so you add them yourself. |
| `scripts/` | Helper scripts. `get_data.py` downloads and sets up the NASA data. |
| `src/` | Data loading and sequence regression models. |
| `notebooks/` | Jupyter notebooks for exploration and building figures. |
| `results/` | Small CSVs and figures produced by the pipeline. These go into the report. |
| `docs/` | Proposal deck, report and presentation slides. |
| `tests/` | Unit tests for the conformal code and the models. |
| `.gitignore` | Keeps data, virtual environments, model files and caches out of git. |
| `README.md` | This file. |

## Working in the repo

- Do not commit raw data or large model files.
- Make one branch per task and merge through a pull request.
- Keep the random seed fixed so results are reproducible.