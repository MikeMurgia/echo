# ECHO

Remaining-useful-life prediction with conformal uncertainty and a breakeven cost-ratio decision layer, on NASA C-MAPSS turbofan data.

Projects in Data Science IDC4290, Florida State University

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

The NASA files are not stored in this repo.

1. Download the C-MAPSS Turbofan Engine Degradation Simulation Data Set from the NASA Prognostics Data Repository:
   https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
2. Unzip it and copy these files into the `data/` folder:
   - `train_FD002.txt`, `test_FD002.txt`, `RUL_FD002.txt`
   - `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`

The files have no header row. Columns are: unit, cycle, 3 operating settings, 21 sensors.

## Run

Run scripts from the repo root with the virtual environment active.

```bash
# run the main pipeline on FD002
python -m src.run --data-dir data --subset FD002

# run on FD001
python -m src.run --data-dir data --subset FD001

# run tests
python -m unittest discover tests

# open notebooks
jupyter notebook notebooks/
```

Outputs are written to `results/`.

## Folders and files

| Path | What it is |
|---|---|
| `data/` | Raw NASA files. Gitignored, so you add them yourself. |
| `src/` | Project code: data loading, preprocessing, models, conformal calibration, decision analysis, plotting. |
| `notebooks/` | Jupyter notebooks for exploration and building figures. |
| `results/` | Small CSVs and figures produced by the pipeline. These go into the report. |
| `docs/` | Proposal deck, report and presentation slides. |
| `tests/` | Unit tests. |
| `.gitignore` | Keeps data, virtual environments, model files and caches out of git. |
| `README.md` | This file. |

## Working in the repo

- Do not commit raw data or large model files.
- Make one branch per task and merge through a pull request.
- Keep the random seed fixed so results are reproducible.