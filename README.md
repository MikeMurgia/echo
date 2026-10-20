# ECHO

Remaining-useful-life prediction with conformal uncertainty and a breakeven cost-ratio decision layer, on NASA C-MAPSS turbofan data.

Projects in Data Science IDC4290

## Setup

Requires Python 3.10+.

```bash
git clone <repo-url>
cd <repo-folder>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy pandas scikit-learn matplotlib xgboost jupyter torch transformers
```

## Set up the Data (ONLY ONCE)

The NASA files are not stored in this repo. One command downloads them and sets up `data/`:

```bash
python src/data_setup.py
```

The files have no header row. Columns are: unit, cycle, 3 operating settings, 21 sensors.

## Sequence data loaders

After cleaning, labeling and normalizing FD001/FD002, inspect the sequence
windows and loader shapes with:

```bash
python -m src.dataset --subset FD001 --window-size 30 --batch-size 64
python -m src.dataset --subset FD002 --window-size 50
```

`src.dataset.make_dataloaders(...)` returns train, validation and test
`DataLoader`s plus the feature column names. Inputs default to the normalized
sensor columns, with each batch shaped `[batch, window, features]`; targets are
the capped RUL at each window's final cycle. Validation engines are held out
before windows are built. The test loader emits one final window per engine.

Run commands from the repository root with the virtual environment active.



## Folders and files

| Path | What it is |
|---|---|
| `data/` | Raw NASA files. Gitignored, so you add them yourself. |
| `scripts/` | Helper scripts. `get_data.py` downloads and sets up the NASA data. |
| `src/` | Project code (empty for now): data loading, preprocessing, models, conformal calibration, decision analysis, plotting. |
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
