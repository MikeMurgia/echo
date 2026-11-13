# ECHO

Remaining-useful-life (RUL) prediction with conformal uncertainty and a breakeven cost-ratio decision layer, on NASA C-MAPSS turbofan data.

Projects in Data Science IDC4290

## Setup

Run every command from the repository root with the virtual environment active. Requires Python 3.10+.

```bash
git clone <repo-url>
cd <repo-folder>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy pandas scikit-learn matplotlib xgboost jupyter torch joblib
```

On Windows, `pip install torch` gives a CPU-only build. For an NVIDIA GPU, install the CUDA build from
pytorch.org, then check that `python -c "import torch; print(torch.cuda.is_available())"` prints `True`.
On a Mac, the default build works and uses the CPU or Apple GPU (MPS) automatically.

## 1. Set up the data (once)

The NASA files are not stored in this repo. This downloads them and prepares `data/`:

```bash
python src/data_setup.py
```

It downloads FD001 and FD002, drops uninformative sensors (per subset), adds a capped `RUL` and an
uncapped `RUL_raw` label, normalizes sensors within operating regimes using training rows only, and
saves the fitted regimes and scalers to `data/preproc_<subset>.joblib`.

Everything in `data/` is gitignored. Every teammate runs this step themselves.

## 2. Train the models

FD002 is the main dataset and FD001 is the replication (change `--subset`).

The recipe used for all reported results (a `--tag` keeps each recipe's files separate):

```bash
python scripts/train.py --subset FD002 --model transformer --seed 1 --tag tuned --patience 3 --lr 3e-4 --d-model 64 --layers 2 --dropout 0.3 --weight-decay 1e-3
```

Three seeds (PowerShell, then bash):

```powershell
foreach ($s in 1,2,3) { python scripts/train.py --subset FD002 --model transformer --seed $s --tag tuned --patience 3 --lr 3e-4 --d-model 64 --layers 2 --dropout 0.3 --weight-decay 1e-3 }
```
```bash
for s in 1 2 3; do python scripts/train.py --subset FD002 --model transformer --seed $s --tag tuned --patience 3 --lr 3e-4 --d-model 64 --layers 2 --dropout 0.3 --weight-decay 1e-3; done
```

Mamba uses the same flags with `--model mamba`. It is a pure-PyTorch implementation of the selective
state-space block (Gu & Dao, 2023), so it runs on any machine with no extra packages. It is not the
fused `mamba-ssm` CUDA kernel, and its weights are not interchangeable with that package. It is much
slower than the Transformer (about a minute per epoch on a laptop GPU), so train it on a GPU and use
the saved checkpoints elsewhere.

Baselines (random forest and XGBoost on the same windows and engine split; use the same seeds):

```bash
python scripts/train_baselines.py --subset FD002 --seed 1
```

Each run writes `results/<subset>/<model>_<tag>_seed<seed>_metrics.json` (validation RMSE, test RMSE and
NASA score, capped and uncapped) and a history CSV. Checkpoints go to `results/checkpoints/`.
Compare everything and see which checkpoint to calibrate:

```bash
python scripts/summarize_models.py --subset FD002
```

The checkpoint is chosen on mean validation RMSE only, never on test.

## 3. Conformal calibration (global vs Mondrian)

Calibrates 90% intervals on half of the official test engines (every cycle, split by engine and
stratified by final RUL) and assesses them on the other half. These engines are never used for training
or checkpoint selection. It compares one global (flat) quantile with a separate quantile for each
predicted-RUL stage (early above 100, mid 30 to 100, late below 30). The subset is read from the
checkpoint, so there is no `--subset` flag.

```bash
python scripts/conformal_calibration.py --checkpoint results/checkpoints/transformer_FD002_tuned_seed3_best.pt --repeats 20 --final
```

- `--repeats N` repeats the engine split N times and reports mean, min and max coverage per stage.
  With few late-stage engines a single split is noisy, so use this for any number you report.
- `--final` calibrates on all test engines and writes `conformal_quantiles_final.json`, the file the
  decision layer and dashboard read. Its coverage is evidenced by the repeated splits, not measured on
  the same engines.

Outputs in `results/<subset>/`: `conformal_windows.csv` (every window with prediction, true RUL, signed
residual and both intervals), `conformal_quantiles.json` (single split), `conformal_quantiles_final.json`,
`conformal_summary.csv` (coverage, width and 95% engine-bootstrap CIs by stage, true-RUL bin and regime),
`conformal_repeats.csv`, `conformal_repeats_by_split.csv`, `conformal_repeats_quantiles.csv`,
`conformal_coverage_sweep.csv` (80/90/95% targets) and two figures.

Notes: groups use predicted RUL only (known at deployment). Signed residual is predicted minus true, so
positive means RUL was overestimated (the dangerous direction). Coverage by true-RUL bin is a
diagnostic and is not guaranteed by the method. A stage with fewer than 30 calibration windows falls
back to the global quantile. Overlapping windows are dependent, so CIs resample engines, not windows.

## 4. Tests

```bash
python -m unittest discover tests
```

## Sharing trained models

Checkpoints (`*.pt`) are gitignored, so they are not shared by default. They are small, so the ones the
team needs are committed on purpose with `git add -f`:

```bash
git add -f results/checkpoints/transformer_FD002_tuned_seed3_best.pt
```

To confirm your data setup matches the one a checkpoint was trained on, run the conformal command above
without `--repeats`. The printed FD002 global q should be 27.63. If it differs, your preprocessing differs.

## Folders and files

| Path | What it is |
|---|---|
| `data/` | NASA files and the saved preprocessing. Gitignored; created by `src/data_setup.py`. |
| `scripts/` | Pipeline scripts: data download and cleaning, training, baselines, model comparison, conformal calibration. |
| `src/` | Data loading, windowing, and the Transformer and Mamba models. |
| `notebooks/` | Jupyter notebooks for exploration and building figures. |
| `results/<subset>/` | Metrics, histories, conformal tables and figures for FD001 and FD002. These go into the report. |
| `results/checkpoints/` | Trained model weights. Gitignored except the files added on purpose. |
| `docs/` | Proposal, report and presentation slides. |
| `tests/` | Unit tests for the conformal code and the models. |
| `.gitignore` | Keeps data, virtual environments, model files and caches out of git. |
| `README.md` | This file. |

## Working in the repo

- Do not commit raw data or large model files.
- Make one branch per task and merge through a pull request.
- Keep the random seed fixed so results are reproducible.
