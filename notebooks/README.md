# Notebooks

Exploratory and reference notebooks. The package code under `src/divar/` is the source of truth for **reproducible** training and serving — these notebooks document the analysis that led to those choices and remain useful for ad-hoc exploration.

## Loading data inside a notebook

```python
from divar.data import load_divar

df = load_divar()   # reads DATA_DIR / DIVAR_CSV from .env
```

Copy `.env.example` to `.env` at the repo root if your CSV is not at `data/raw/Divar.csv`.

## Use the CLI instead for training

The training paths in `ml_prediction_price.ipynb` and `ml_prediction_credit.ipynb` are kept as a reference for the modeling decisions (Optuna hyperparameter searches, feature ablations) but they should **not** be the way you re-train. The CLI pipelines below produce versioned, registry-aware artifacts that match the production serving layer:

```bash
pip install -e ".[mlops]"
divar-prepare-price
divar-train-price  --from-processed --mlflow
divar-prepare-credit
divar-train-credit --from-processed --mlflow
```

Or run the whole DAG via DVC: `dvc repro train_price && dvc repro train_credit`.

## Notebook index

| Notebook | Purpose | Status |
|----------|---------|--------|
| `EDA.ipynb` | Exploratory data analysis on the raw Divar export — distributions, missingness, segment counts | Reference |
| `statistical_analysis.ipynb` | Statistical tests and geo summaries; uses `iran_city_classification.csv` | Reference |
| `ml_recommender_system.ipynb` | Geo clustering (KMeans, DBSCAN) for a neighborhood recommender | Reference, not yet ported to the package |
| `ml_prediction_price.ipynb` | Sale-price modeling — feature engineering, Optuna tuning, comparison of RF and LightGBM | Reference; current production path is `divar-train-price` |
| `ml_prediction_credit.ipynb` | Rent/credit modeling — derived target, three-way split, Optuna tuning | Reference; current production path is `divar-train-credit` |

## What lives where

If you want to change behavior, edit the package, not the notebook:

| Want to change… | Edit |
|---|---|
| Feature engineering | `src/divar/features/{price,credit,common}.py` |
| Model hyperparameters | `configs/{price,credit}.yaml` |
| Outlier bounds | `configs/{price,credit}.yaml` → `outliers:` |
| Train/val/test split | `configs/{price,credit}.yaml` → `split:` |
| MLflow registry gate | `MLFLOW_MIN_VAL_R2` env var, or `configs/mlflow.yaml` |
| API endpoints / Pydantic schemas | `src/divar/serve/` |
| Drift report contents | `src/divar/monitoring/drift.py` |

## Why notebooks stay in the repo

- They are the **archaeology** of the modeling decisions — every hyperparameter in `configs/` traces back to a sweep run in one of these notebooks.
- They are still the right tool for one-off exploration: distribution checks, quick segment analyses, prototyping a new feature before promoting it into `src/divar/features/`.
- The geo-clustering notebook has not been ported to the package yet; it remains the canonical implementation until it earns a CLI of its own.
