# Divar Project — Claude guide

End-to-end ML platform for Iranian Divar classified-ad listings. Two regression tasks:
- **price** — sale-price prediction (`price_value`) over sell listings
- **credit** — rent/credit prediction (`total_credit`) over rental listings

Both tasks train Random Forest + LightGBM regressors. Pipeline is DVC-orchestrated, MLflow-tracked, Pandera-validated, Evidently-monitored, FastAPI-served.

## Layout

```
configs/        price.yaml, credit.yaml, mlflow.yaml — per-task feature lists, outlier bounds, splits, model hyperparams
data/raw/       Divar.csv (DVC-tracked, not in git)
data/processed/ train/val/test parquet (DVC outputs)
src/divar/      main package
  config.py        load_config(name), path resolution via env vars
  data/load.py     load_divar(), load_iran_city_classification()
  features/        price.py, credit.py, common.py (Persian-digit conversion, luxury scoring, building age)
  validation/      Pandera schemas (raw + processed) + run.py runner
  models/          train_price.py, train_credit.py, sklearn_pipeline.py (DivarFeatureTransformer), encoding.py, metrics.py
  tracking/        MLflow utils, registry (R² gate), naming, deployment manifests
  serve/           FastAPI app (app.py), ModelRegistry (predictor.py), Pydantic schemas
  monitoring/      drift.py — Evidently DataDriftPreset HTML reports
  pipelines/       CLI entry points wired to dvc.yaml stages
models/         {task}/{algo}_pipeline.joblib + deployment.json manifest per task
metrics/        DVC-tracked price.json / credit.json
mlruns/         local MLflow tracking (gitignored)
reports/drift/  Evidently HTML drift reports
tests/          pytest suite — feature engineering, sklearn pipeline, serve API, MLflow integration
.github/workflows/ci.yml  ruff + pytest on push/PR
dvc.yaml        pipeline DAG
params.yaml     DVC param overrides
```

## DVC pipeline DAG (`dvc.yaml`)

```
validate_raw → prepare_{price,credit} → validate_{price,credit} → train_{price,credit} → monitor_drift_{price,credit}
```

## Common commands

```bash
# Install (full stack)
pip install -e ".[dev,mlops,serving,monitoring]"

# Full pipeline (preferred)
dvc repro train_price
dvc repro train_credit
dvc metrics show
dvc exp show --all-commits

# Standalone CLIs (entry points in pyproject.toml)
divar-prepare-price
divar-train-price --from-processed --mlflow --metrics-file metrics/price.json
divar-monitor-drift --task price
divar-promote-model --task price --model lightgbm   # manual → Production

# Serve
divar-serve                                          # 127.0.0.1:8000, /docs for Swagger
MODEL_SOURCE=mlflow divar-serve                      # load from MLflow registry instead of joblib

# Tests + lint
pytest -q
ruff check src tests
```

## API surface (`src/divar/serve/app.py`)

- `GET /health` → status, model_source, loaded_pipelines, deployment manifests
- `GET /schema/{task}` → feature_columns, target_column
- `POST /predict/{task}` → body `{"model": "lightgbm"|"random_forest", "records": [{...}]}` → predictions[]

Uses `fastapi-offline-docs` to serve Swagger without CDN (air-gapped friendly).

## Config & env

`divar.config.load_config(name)` reads `configs/{name}.yaml` with `@lru_cache`. Env vars (see `.env.example`):
- `DATA_DIR`, `DIVAR_CSV` — data paths
- `MLFLOW_MIN_VAL_R2` — registry gate (default 0.65)
- `MODEL_SOURCE` — `local` (joblib, default) or `mlflow` (Production stage)
- `SERVE_HOST`, `SERVE_PORT`

## Conventions & gotchas

- **Two models per task, not an ensemble.** API caller picks `model` per request.
- **Auto-registration gate:** training only registers a model to MLflow Staging if val R² ≥ `MLFLOW_MIN_VAL_R2`. Promotion to Production is manual via `divar-promote-model`.
- **MLflow naming:** experiment `divar/{task}-prediction`; run name embeds val R² as `r2-0p7100` (`.` → `p`). See `src/divar/tracking/naming.py`.
- **Target encoding requires y at fit time.** `DivarFeatureTransformer.fit(X, y)` — inference never refits.
- **Persian-data quirks:** `has_balcony` raw values are `"unselect"`/`"true"`/`"false"`; `features/common.py:add_luxury_scores(fix_balcony=True)` normalizes. Persian digits handled in `common.py`. Building-age reference year is 1404 (Persian calendar); pre-1370 mapped to "1370".
- **Drift report drops the target** (`price_value`, `total_credit`) before generating — features only, no target leak.
- **Credit uses a 3-way split** (test 0.15, val 0.1275); price uses 2-way. See respective configs.
- **Clustering notebook not yet ported.** `notebooks/ml_recommender_system.ipynb` (geo clustering) stays in notebooks/ — not in pipelines/.
- **Phase status:** Phase 2 features (FastAPI serving, offline docs, drift monitoring, sklearn inference pipeline, LightGBM for credit) are landed as of `55c62fb`.

## Style

- `ruff` with line-length 100, `E501` ignored, `select = E,F,I,W` (see `pyproject.toml`).
- Tests live under `tests/`; `conftest.py` provides synthetic sell/credit dataframe fixtures (~30 rows).
