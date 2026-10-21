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

The files have no header row. Columns are: unit, cycle, 3 operating settings, 21 sensors.

## 2. Sequence data loaders

After cleaning, labeling and normalizing FD001/FD002, inspect the sequence
windows and loader shapes with:

```bash
python -m src.dataset --subset FD001 --window-size 30 --batch-size 64
python -m src.dataset --subset FD002 --window-size 50
```

## 3. Train sequence models

Train the Transformer encoder (the default backbone) with dropout `0.2`, AdamW
weight decay `1e-4`, and early stopping after seven epochs without improved
validation RMSE. The best checkpoint is saved under `results/checkpoints/`:

```bash
python scripts/train.py --subset FD001 --model transformer --epochs 50 --lr 0.001 --window-size 30
```
## 4. Split conformal calibration

## Folders and files

| Path | What it is |
|---|---|
| `data/` | Raw NASA files. Gitignored, so you add them yourself. |
| `scripts/` | Helper scripts. `get_data.py` downloads and sets up the NASA data. |
| `src/` | Data loading and sequence regression models. |
| `notebooks/` | Jupyter notebooks for exploration and building figures. |
| `results/` | Small CSVs and figures produced by the pipeline. These go into the report. |
| `docs/` | Proposal deck, report and presentation slides. |
| `tests/` | Unit tests (empty for now). |
| `.gitignore` | Keeps data, virtual environments, model files and caches out of git. |
| `README.md` | This file. |

## Working in the repo

- Do not commit raw data or large model files.
- Make one branch per task and merge through a pull request.
- Keep the random seed fixed so results are reproducible.
