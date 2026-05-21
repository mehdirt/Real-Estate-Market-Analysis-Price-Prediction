# Notebooks

Exploratory and original analysis notebooks live here. For reproducible training, use the `divar` Python package and CLI commands instead of running notebooks end-to-end.

## Loading data in notebooks

```python
from divar.data import load_divar

df = load_divar()  # uses DATA_DIR / DIVAR_CSV from .env
```

Copy `.env.example` to `.env` and set paths if your CSV is not under `data/raw/Divar.csv`.

## Pipelines (recommended)

```bash
pip install -e .
divar-prepare-price
divar-train-price --from-processed
python -m divar.pipelines.prepare_credit
```

## Notebook index

| Notebook | Purpose |
|----------|---------|
| `EDA.ipynb` | Exploratory data analysis |
| `statistical_analysis.ipynb` | Statistical tests and geo analysis |
| `ml_recommender_system.ipynb` | Geo clustering (KMeans, DBSCAN) |
| `ml_prediction_price.ipynb` | Sale price prediction (reference) |
| `ml_prediction_credit.ipynb` | Rent/credit prediction (reference) |
