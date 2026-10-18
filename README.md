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
pip install numpy pandas scikit-learn matplotlib xgboost jupyter
```

## Get the data

The NASA files are not stored in this repo. One command downloads them and sets up `data/`:

```bash
python scripts/get_data.py
```

This fetches the C-MAPSS zip, extracts FD001 and FD002 into `data/`, and skips files you already have. Options:

```bash
python scripts/get_data.py --subsets FD001 FD002 FD003 FD004   # choose subsets
python scripts/get_data.py --force                             # re-download
python scripts/get_data.py --url <direct-zip-link>             # if the NASA link moves
```

If the download fails, the script prints manual steps: download the Turbofan Engine Degradation Simulation data set from the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) and copy the `train_`, `test_` and `RUL_` `.txt` files into `data/`.

The files have no header row. Columns are: unit, cycle, 3 operating settings, 21 sensors.

## Run

The repo is currently just the folder structure, `.gitignore` and this README. There is no code yet, so there is nothing to run. As code is added, put the run commands here, for example:

```bash
python -m src.run --data-dir data --subset FD002
```

Run everything from the repo root with the virtual environment active.

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