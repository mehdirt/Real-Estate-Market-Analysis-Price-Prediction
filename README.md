# 📊 Divar Project

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 🌟 Project Description

This project analyzes data from the **Divar** platform (an advertising company in Iran), including Exploratory Data Analysis (EDA), statistical analysis, recommender system, and price/rent prediction. The main goal is to use machine learning techniques to better understand the data and provide predictive models.

**Phase 0 (MLOps foundation)** adds an installable Python package, unified data paths, YAML configs, and CLI pipelines extracted from the original notebooks.

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
│   └── pipelines/        # CLI: prepare & train
├── tests/
├── models/               # Saved model artifacts (gitignored)
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

### Pipelines (sale price)

```bash
divar-prepare-price          # writes data/processed/price_{train,val}.parquet
divar-train-price --from-processed   # trains RF + LightGBM → models/price/
```

### Pipelines (rent/credit)

```bash
divar-prepare-credit         # writes credit_{train,val,test}.parquet
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
- Clustering/recommender notebook is not yet extracted to the package (Phase 1+).
- For questions or collaboration, use Issues or Pull Requests.

## 📜 License

MIT License.
