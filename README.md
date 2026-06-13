# Divar — Real-Estate Market Analysis & Price Prediction

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![CI](https://img.shields.io/badge/CI-ruff%20%2B%20pytest-success.svg)](.github/workflows/ci.yml)

An end-to-end machine-learning platform built on classified-ad listings from **Divar**, Iran's largest online marketplace. The project covers two regression tasks over real-estate listings, an EDA layer, geo-clustering recommenders, and a production-style MLOps stack with reproducible pipelines, model registry, serving API, drift monitoring, and an interactive UI.

| Task | Target | Listings | Models |
|------|--------|----------|--------|
| **Price prediction** | `price_value` (sale price, IRR) | Sell ads | Random Forest, LightGBM |
| **Credit prediction** | `total_credit` (rent + deposit normalized) | Rent/credit ads | Random Forest, LightGBM |

---

## Table of contents

- [Architecture overview](#architecture-overview)
- [Why these design choices](#why-these-design-choices)
- [Repository layout](#repository-layout)
- [Quickstart](#quickstart)
- [DVC pipeline](#dvc-pipeline)
- [MLflow: tracking, registry, and promotion](#mlflow-tracking-registry-and-promotion)
- [Inference API](#inference-api)
- [Streamlit UI](#streamlit-ui)
- [Drift monitoring](#drift-monitoring)
- [Testing and CI](#testing-and-ci)
- [Environment variables](#environment-variables)
- [Contributors](#contributors)
- [License](#license)

---

## Architecture overview

```
                   ┌──────────────────────────┐
                   │  data/raw/Divar.csv      │  (DVC-tracked, ~1M rows)
                   └────────────┬─────────────┘
                                │
                       validate_raw (Pandera)
                                │
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
        prepare_price                       prepare_credit
              │                                   │
        validate_price                      validate_credit
              │                                   │
        train_price ──► models/price/      train_credit ──► models/credit/
              │            metrics/price.json         │            metrics/credit.json
              │            (MLflow run + registry)    │            (MLflow run + registry)
              ▼                                       ▼
      monitor_drift_price                   monitor_drift_credit
              │                                       │
              └──────────► reports/drift/ ◄───────────┘

                  FastAPI  ◄──► ModelRegistry  ◄──► joblib | MLflow Production
                     ▲
                     │
                 Streamlit UI
```

Every solid line is a **DVC stage** with declared deps and outputs, so any input change re-runs only the affected branch. MLflow captures the trained-model lineage; Pandera enforces schema contracts at every step; Evidently produces drift reports against the training reference.

---

## Why these design choices

This section documents the *rationale* behind the non-obvious decisions in the codebase — the kind of context that does not survive a `git blame`.

### Two models per task, no ensemble

Random Forest and LightGBM are trained **independently** on each task and saved side-by-side. The inference API exposes a `model` field so the caller picks one per request.

- **Why both?** RF is a strong, interpretable, low-tuning baseline that resists noisy categorical splits; LightGBM achieves higher accuracy on this dataset but is more sensitive to overfitting. Keeping them separate gives an honest A/B at inference time and lets product decide which to ship based on latency, interpretability, or accuracy.
- **Why not stack/ensemble?** Stacking adds calibration risk and obscures which feature drove a prediction — costly in a domain where users will ask "why this price?" Two clear models with comparable input/output contracts are easier to audit and roll back.

### Hyperparameters chosen per algorithm and per task

Random Forest hyperparameters are intentionally **identical across tasks** (`n_estimators=355`, `max_depth=22`, `min_samples_split=13`, `min_samples_leaf=5`) — RF is robust enough that the cost of separate tuning was not worth the marginal accuracy gain, and a shared config simplifies reasoning about variance.

LightGBM is **tuned per task** because boosting amplifies dataset-specific signal:

| Hyperparameter | Price | Credit | Reason for the gap |
|---|---|---|---|
| `n_estimators` | 1147 | 1438 | Credit's wider total-credit distribution benefits from more rounds |
| `learning_rate` | 0.013 | 0.027 | Faster LR for credit pairs with higher `min_child_samples` to stay regularized |
| `num_leaves` | 129 | 141 | Both deep; credit slightly wider |
| `max_depth` | 22 | 15 | Credit kept shallower to avoid overfit on amenity-system one-hots |
| `min_child_samples` | 97 | 41 | Price needs stronger leaf regularization (heavier tail) |
| `subsample` / `colsample_bytree` | 0.64 / 0.62 | 0.75 / 0.63 | Tuned per task via the notebook's Optuna runs |

Values were produced by Optuna sweeps in the original notebooks and frozen into `configs/{price,credit}.yaml`. The configs are the single source of truth — training scripts read them directly so a tuning re-run is a config edit, not a code change.

### Split shapes differ between tasks

- **Price** uses an **80/20 train+val split** (`test_size: 0.2`). There is no held-out test set because validation R² is the registry-gate metric and the production ground truth comes from the API itself once deployed.
- **Credit** uses an explicit **three-way split** (`test_size: 0.15`, `val_size: 0.1275` → ~70.5 / 12.75 / 15). The credit target is **derived** (`(deposit + rent×100/3) × 6`), which makes test-set leakage harder to detect during tuning — the third split exists specifically to verify the derived target generalizes.

### Preprocessing decisions

The feature pipelines (`src/divar/features/`) encode several domain quirks that would silently degrade model quality if ignored:

- **Persian numerals → English digits.** Raw `construction_year` arrives with Persian glyphs in a non-trivial fraction of rows. `common.persian_to_english()` normalizes them before parsing.
- **Persian-calendar building age.** Reference year is **1404** (Persian calendar, ~2025 Gregorian). Values pre-1370 are clamped to "1370" because pre-Revolution data is sparse and noisy. Age is then binned into `[new, relatively_new, mid_age, old, very_old]` using `[0, 5, 10, 20, 30]` years — coarse enough to be robust, fine enough to separate decades that materially affect price.
- **Balcony normalization.** `has_balcony` raw values are the strings `"unselect"`, `"true"`, `"false"` (not booleans). `add_luxury_scores(fix_balcony=True)` maps them explicitly. Skipping this silently breaks the luxury-score aggregation.
- **Luxury & non-luxury scoring.** Five amenity flags (pool, jacuzzi, sauna, BBQ, security guard) are aggregated into a 4-bin **luxury score**; four "table-stakes" amenities (balcony, elevator, warehouse, parking) form a **non-luxury score**. Binning into ordinal categories lets tree models exploit the monotonic signal without exploding the one-hot width.
- **Rare-category collapse (credit only).** Categorical levels with frequency < 1 % are folded into `"Other"`. Credit listings have a longer tail of niche `cat3_slug` values; collapsing them stabilizes target encoding.
- **Location-aware imputation.** Missing `location_latitude` / `location_longitude` are filled by **city-mean computed on training only**, with a global-mean fallback. This keeps train and val on the same imputation distribution and prevents target leakage through location.
- **Outlier bounds in config, not code.** Building size, price, and credit are clipped by configurable bounds (e.g. `price_value_min: 5e8`, `price_value_max: 8e10` IRR). DVC tracks these as parameters, so an outlier-policy experiment is one `params.yaml` edit + `dvc exp run`.

### Encoding: target encoding + one-hot, fit on train only

`DivarFeatureTransformer` (`src/divar/models/encoding.py`) is a single sklearn-compatible step that:

1. Maps booleans to `{1, 0, -1}` (the `-1` is an explicit "missing" signal, not silent NaN).
2. **Target-encodes** `neighborhood_slug` and `city_slug` with smoothing = 10. These two columns have thousands of levels — one-hot would explode dimensionality, and label encoding would inject false ordinality. Smoothing pulls rare-neighborhood estimates toward the city mean to control variance.
3. **One-hot encodes** the remaining ~10–15 low-cardinality categoricals with `handle_unknown="ignore"` so unseen categories at inference become all-zero rows instead of crashes.

`fit(X, y)` *requires* `y` — target encoding is supervised, and forcing the API at the type level prevents anyone from accidentally fitting an encoder on validation data. Inference reuses the fitted transformer; it never refits.

### MLflow registry gate (validation R² ≥ 0.65)

Training always saves local joblib artifacts, but the **MLflow Model Registry has a quality gate**: a model is registered to Staging only if its validation R² meets `MLFLOW_MIN_VAL_R2` (default 0.65). Promotion to Production is **manual** (`divar-promote-model`).

- **Why a gate at all?** Without it, every bad experiment pollutes the registry, and "latest registered" stops being a meaningful default.
- **Why 0.65?** Calibrated against historical notebook runs: random-forest baselines on this data routinely clear 0.70–0.80, so 0.65 catches catastrophic regressions (bad feature, broken split) without rejecting legitimate experiments.
- **Why manual production promotion?** Promotion changes what the API serves. A manual step forces a human to read the metrics page before users are affected.

Override per environment with `MLFLOW_MIN_VAL_R2`.

### Dual model source: `local` vs `mlflow`

The serving layer reads `MODEL_SOURCE`:

| Mode | Source | When to use |
|------|--------|-------------|
| `local` (default) | `models/{task}/{model}_pipeline.joblib` | Local dev, CI, air-gapped demos — no MLflow server needed |
| `mlflow` | `models:/divar-{task}-{model}/Production` | Production — version-pinned, audit-trail, controlled promotion |

The API is identical in both modes; only the loader changes. This means local development and production serve from the same FastAPI code and the same sklearn pipeline contract — no "works on my machine" surprises.

### Pipelines bundle encoders + regressors

Every saved artifact is a full sklearn `Pipeline(features → regressor)`. The API never re-encodes; it just feeds the request DataFrame through the pipeline. Two consequences:

- **Offline metrics match online predictions** by construction.
- **Adding a feature is a single change** — the encoder and the regressor evolve together in one artifact.

### Tooling stack: why each piece earns its place

| Tool | Role | Why it (and not something simpler) |
|------|------|-------------------------------------|
| **DVC** | Pipeline + data versioning | The full pipeline is a DAG with deterministic dep tracking; a feature-engineering tweak re-runs only downstream stages instead of the whole notebook |
| **MLflow** | Experiment tracking + registry | Compare runs by params/metrics and gate model promotion behind an R² threshold |
| **Pandera** | Schema validation | Catches malformed inputs at stage boundaries with readable errors, not at random in feature code |
| **Evidently** | Drift monitoring | Generates per-task HTML drift reports against the training reference; reports are themselves DVC outputs so they are versioned |
| **FastAPI** | Serving | Async, type-checked, OpenAPI by default; `fastapi-offline-docs` lets `/docs` work without CDN access |
| **Streamlit** | Interactive UI | Lets non-engineers explore the feature space and see live predictions without writing JSON requests |
| **Ruff** | Lint + import sort | One tool replaces flake8 + isort + autoflake; configured in `pyproject.toml` |

---

## Repository layout

```text
.
├── configs/
│   ├── price.yaml            # Features, splits, outliers, RF + LightGBM hyperparams
│   ├── credit.yaml           # Same shape; differs by task (see "Why" above)
│   └── mlflow.yaml           # Tracking URI, registry gate, naming
├── data/
│   ├── raw/                  # Divar.csv lives here (DVC-tracked, gitignored)
│   └── processed/            # DVC pipeline outputs: train/val/test parquet
├── notebooks/                # Original EDA + reference notebooks (see notebooks/README.md)
├── src/divar/                # Installable package (`pip install -e .`)
│   ├── config.py             # load_config() + path resolution from env
│   ├── data/                 # load_divar(), load_iran_city_classification()
│   ├── features/             # price.py, credit.py, common.py (shared transforms)
│   ├── validation/           # Pandera schemas (raw + processed)
│   ├── models/               # train_price.py, train_credit.py, sklearn_pipeline.py, encoding.py, metrics.py
│   ├── tracking/             # MLflow utils, registry gate, naming, deployment manifests
│   ├── serve/                # FastAPI app + ModelRegistry + Pydantic schemas
│   ├── monitoring/           # Evidently drift report generator
│   ├── ui/                   # Streamlit app + launcher
│   └── pipelines/            # CLI entry points (one per DVC stage)
├── models/                   # {task}/{algo}_pipeline.joblib + deployment.json (DVC outs)
├── metrics/                  # DVC-tracked JSON metrics (cache: false, so they live in Git)
├── reports/drift/            # Evidently HTML reports (DVC outs)
├── mlruns/                   # Local MLflow tracking store (gitignored)
├── tests/                    # pytest suite (~30-row synthetic fixtures in conftest.py)
├── dvc.yaml                  # Pipeline DAG
├── params.yaml               # DVC parameter overrides for experiments
├── pyproject.toml            # Package metadata, extras, CLI entry points, ruff config
└── .github/workflows/ci.yml  # Ruff + pytest on push/PR
```

---

## Quickstart

```bash
git clone https://github.com/mehdirt/Real-Estate-Market-Analysis-Price-Prediction.git
cd Real-Estate-Market-Analysis-Price-Prediction
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev,mlops,serving,monitoring,ui]"
cp .env.example .env                 # adjust paths if needed
```

Place `data/raw/Divar.csv` in the repo (see [data/README.md](data/README.md) for setup and DVC tracking), then:

```bash
dvc repro train_price    # validate → prepare → validate → train → drift
dvc repro train_credit
dvc metrics show
```

---

## DVC pipeline

The pipeline DAG is declared in `dvc.yaml`. Each stage is also runnable as a standalone CLI for debugging.

```bash
dvc repro                        # run the whole DAG
dvc repro train_price            # only the price branch
dvc repro -f train_credit        # force a re-run even if deps are unchanged
dvc exp run -S 'configs/price.yaml:outliers.price_value_max=1.0e11'
dvc exp show --all-commits       # compare experiments
```

Standalone CLIs (defined in `pyproject.toml`):

```bash
divar-validate-raw
divar-prepare-price
divar-prepare-credit
divar-train-price   --from-processed --mlflow --metrics-file metrics/price.json
divar-train-credit  --from-processed --mlflow --metrics-file metrics/credit.json
divar-monitor-drift --task price
divar-promote-model --task price --model lightgbm   # manual Staging → Production
divar-serve                                         # FastAPI server
divar-streamlit                                     # UI launcher
```

---

## MLflow: tracking, registry, and promotion

Every training run logs params, metrics, and the full sklearn `Pipeline` artifact to MLflow. Naming is deterministic so runs are easy to grep:

| Item | Pattern | Example |
|------|---------|---------|
| Experiment | `divar/{task}-prediction` | `divar/price-prediction` |
| Run | `{task}-{timestamp}-rf-r2-…-lgb-r2-…` | `price-20250521T143052-rf-r2-0p7100-lgb-r2-0p8800` |
| Registered model | `divar-{task}-{algorithm}` | `divar-price-lightgbm` |

The decimal point in R² becomes `p` (`0.71 → 0p7100`) to keep run names filesystem-safe.

**Lifecycle:**

1. **Training** — always writes `models/{task}/{algo}_pipeline.joblib`. The latest local copy is what `MODEL_SOURCE=local` serves.
2. **Staging** — if `val_R² ≥ MLFLOW_MIN_VAL_R2` (default 0.65), the model is registered and auto-transitioned to Staging.
3. **Production** — promote manually: `divar-promote-model --task price --model lightgbm`. After promotion, `MODEL_SOURCE=mlflow divar-serve` will load that version.

Check `models/{task}/deployment.json` after every training run for the run id, val R² per algorithm, and registered versions.

---

## Inference API

After training, start the FastAPI server (defaults to `127.0.0.1:8000`):

```bash
divar-serve
# Swagger UI: http://127.0.0.1:8000/docs   (offline-friendly, no CDN)
```

Endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Status, `model_source`, loaded pipelines, deployment manifests |
| `GET` | `/schema/{task}` | Required feature columns + target name for a task |
| `POST` | `/predict/{task}` | Body: `{"model": "lightgbm" \| "random_forest", "records": [{...}]}` → `predictions[]` |

The request payload uses the **processed** feature schema (post-engineering). Use `GET /schema/{task}` to discover the exact field names and types. Set `SERVE_RELOAD=true` for dev auto-reload.

---

## Streamlit UI

A typed-input form for exploring predictions interactively:

```bash
pip install -e ".[ui]"
divar-streamlit
```

The UI:

- Loads a random validation row, or lets you pick one, as a starting point.
- Renders typed widgets for every feature in the task schema (numeric inputs for size/floors, selects for categoricals, toggles for amenity flags).
- Sends the modified record to `/predict/{task}` and displays both RF and LightGBM predictions side-by-side.
- Persists results across reruns so you can A/B feature changes without losing context.

The UI assumes the API is running on `SERVE_HOST:SERVE_PORT`.

---

## Drift monitoring

```bash
divar-monitor-drift --task price
# or: dvc repro monitor_drift_price
# → reports/drift/price_drift.html
```

The report compares the **train split** (reference) against the **validation split** (current) using Evidently's `DataDriftPreset`. The target column is dropped before comparison — drift is reported on features only, never on the label.

For production drift, point the same generator at a snapshot of recent prediction requests instead of the validation split.

---

## Testing and CI

```bash
pytest -q                    # unit + integration tests
ruff check src tests         # lint + import sort
```

`tests/conftest.py` provides ~30-row synthetic sell and credit fixtures so the feature, encoding, training, and API tests run end-to-end in CI without needing the real dataset.

CI (`.github/workflows/ci.yml`) runs ruff and pytest on every push and PR to `main` / `mahdi`. Pip cache is keyed on `pyproject.toml` + `requirements.txt` for faster installs.

---

## Environment variables

Defaults in `.env.example`; override per environment.

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATA_DIR` | `./data` | Base data directory |
| `DIVAR_CSV` | `./data/raw/Divar.csv` | Raw CSV path |
| `MLFLOW_TRACKING_URI` | `./mlruns` | MLflow tracking store (local file URI by default) |
| `MLFLOW_MIN_VAL_R2` | `0.65` | Registry gate threshold |
| `MLFLOW_ENABLE` | `false` | If `true`, training logs to MLflow even without `--mlflow` |
| `MODEL_SOURCE` | `local` | `local` (joblib) or `mlflow` (Production registry) |
| `SERVE_HOST` | `127.0.0.1` | FastAPI bind host |
| `SERVE_PORT` | `8000` | FastAPI bind port |
| `SERVE_RELOAD` | `false` | Uvicorn auto-reload for dev |

---

## Contributors

- [Mahdi](https://github.com/mehdirt)
- [Ramin](https://github.com/raminBadri)

Issues and pull requests welcome.

## License

MIT — see [LICENSE](LICENSE).
