# 📊 Divar Project

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 🌟 Project Description

This project analyzes data from the **Divar** platform (an advertising company in Iran), including Exploratory Data Analysis (EDA), statistical analysis, recommender system, and price/rent prediction. The main goal is to use machine learning techniques to better understand the data and provide predictive models.

**Phase 0** adds an installable Python package, unified data paths, YAML configs, and CLI pipelines extracted from the original notebooks.

**Phase 1** adds **DVC** pipelines, **Pandera** data validation, and **MLflow** experiment tracking.

**Phase 2** adds **sklearn inference pipelines**, a **FastAPI** serving layer, **Evidently** drift reports, and **LightGBM** for credit prediction.

## 👥 Contributors

- [Mahdi](https://github.com/mehdirt) 👨‍💻
- [Ramin](https://github.com/raminBadri) 👨‍💻

## 📂 Dataset

The dataset is not stored in Git (~1M rows, 64 columns). Place files under `data/raw/`:

| File | Purpose |
|------|---------|
| `Divar.csv` | Main ads dataset (required for ML pipelines) |
| `iran_city_classification.csv` | Optional, for statistical analysis |

See [data/README.md](data/README.md) for setup.

## 🗂️ Project Structure

```text
├── configs/              # YAML: price.yaml, credit.yaml
├── data/raw/             # Place Divar.csv here (gitignored)
├── data/processed/       # Generated parquet splits
├── notebooks/            # Original Jupyter notebooks
├── src/divar/            # Installable Python package
│   ├── data/             # load_divar()
│   ├── features/         # price & credit feature engineering
│   ├── models/           # encoding, training, metrics
│   ├── validation/       # Pandera schemas
│   ├── tracking/         # MLflow helpers
│   ├── serve/            # FastAPI inference API
│   ├── monitoring/       # Evidently drift reports
│   └── pipelines/        # CLI + DVC stages
├── dvc.yaml              # DVC pipeline definition
├── params.yaml           # DVC experiment parameters
├── metrics/              # DVC-tracked evaluation metrics (JSON)
├── tests/
├── models/               # Saved model artifacts (DVC outs)
├── mlruns/               # MLflow local tracking (gitignored)
├── reports/drift/        # Evidently HTML reports (DVC outs)
└── requirements.txt
```

### Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/EDA.ipynb` | Exploratory analysis |
| `notebooks/statistical_analysis.ipynb` | Statistical tests |
| `notebooks/ml_recommender_system.ipynb` | Geo clustering |
| `notebooks/ml_prediction_price.ipynb` | Sale price (reference) |
| `notebooks/ml_prediction_credit.ipynb` | Rent/credit (reference) |

## 🛠️ Technologies

- **Core ML**: pandas, scikit-learn, LightGBM, category-encoders
- **Notebooks / EDA**: matplotlib, seaborn, plotly, geopandas, folium
- **Tuning** (credit notebook): Optuna
- **MLOps**: DVC, MLflow, Pandera, Evidently
- **Serving**: FastAPI, uvicorn

## 📋 Prerequisites

- Python 3.11+
- Raw data in `data/raw/Divar.csv` (or configure `.env`)

## 🚀 Setup and Run

```bash
git clone https://github.com/username/divar_project.git
cd divar_project
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # optional: customize paths
```

### Environment variables

| Variable | Default |
|----------|---------|
| `DATA_DIR` | `./data` |
| `DIVAR_CSV` | `./data/raw/Divar.csv` |

### DVC pipeline (recommended)

```bash
pip install -e ".[mlops]"
dvc add data/raw/Divar.csv   # once, then commit *.dvc
dvc repro train_price        # validate → prepare → train (+ MLflow)
dvc repro train_credit       # credit path
dvc metrics show
dvc exp show                 # compare experiment params
```

### CLI (sale price)

```bash
divar-prepare-price
divar-train-price --from-processed --mlflow
mlflow ui                    # open http://127.0.0.1:5000
```

### CLI (rent/credit)

```bash
divar-prepare-credit
divar-train-credit --from-processed --mlflow   # RF + LightGBM
```

Set `MLFLOW_ENABLE=true` in `.env` to log runs without `--mlflow`.

### Inference API (Phase 2)

After training, start the API (expects `*_pipeline.joblib` under `models/`):

```bash
pip install -e ".[serving]"
divar-serve
# GET  http://127.0.0.1:8000/schema/price
# POST http://127.0.0.1:8000/predict/price  {"model":"lightgbm","records":[{...}]}
```

Feature rows must match the **processed** schema (see `/schema/{task}`). Each saved model includes encoding + regressor in one sklearn `Pipeline`.

### Drift monitoring

```bash
pip install -e ".[monitoring]"
divar-monitor-drift --task price
# or: dvc repro monitor_drift_price
# → reports/drift/price_drift.html
```

### Use in Python

```python
from divar.data import load_divar
from divar.features import prepare_price_dataset

df = load_divar()
train, val = prepare_price_dataset(df)
```

### Tests

```bash
pytest -q
```

## ⚠️ Notes

- Original notebooks used inconsistent CSV paths; use `load_divar()` or `.env` instead.
- Clustering/recommender notebook is not yet extracted to the package.
- Configure a DVC remote (`dvc remote add`) for team data sharing when ready.
- For questions or collaboration, use Issues or Pull Requests.

## 📜 License

MIT License.
