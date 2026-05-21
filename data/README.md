# Data directory

Place raw datasets here (not committed to Git due to size). Use **DVC** to version large files.

## Expected layout

```text
data/
├── raw/
│   ├── Divar.csv                      # dvc add (see below)
│   ├── Divar.csv.dvc                  # pointer file (commit to Git)
│   └── iran_city_classification.csv   # optional
└── processed/                         # DVC pipeline outputs (parquet)
    ├── price_train.parquet
    ├── price_val.parquet
    ├── credit_train.parquet
    ├── credit_val.parquet
    └── credit_test.parquet
```

## Setup

1. Download `Divar.csv` from your data source.
2. Copy it to `data/raw/Divar.csv`, or set `DIVAR_CSV` in `.env`.
3. Track raw data with DVC:

```bash
pip install -e ".[mlops]"
dvc add data/raw/Divar.csv
git add data/raw/Divar.csv.dvc data/raw/.gitignore
git commit -m "Track Divar.csv with DVC"
```

4. Run the full pipeline:

```bash
dvc repro train_price    # or: dvc repro
```

## Without DVC

```bash
pip install -e .
divar-prepare-price
divar-train-price --from-processed --mlflow
```
