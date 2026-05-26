"""Train sale price models (Random Forest + LightGBM)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from divar.config import MODELS_DIR, load_config
from divar.models.encoding import fit_feature_matrices, split_xy
from divar.models.metrics import regression_metrics
from divar.models.sklearn_pipeline import save_task_pipelines


def train_price_models(
    train: pd.DataFrame,
    val: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Train Random Forest and LightGBM on prepared sale-price data.

    Returns
    -------
    dict
        Models, encoders, ``feature_columns``, ``sample_X``, and ``metrics``
        (validation R²/MAE/RMSE per algorithm).
    """
    cfg = config or load_config("price")
    target = cfg["target_column"]

    X_train, y_train, X_val, y_val = split_xy(train, val, target)
    X_train_final, X_val_final, te, ohe = fit_feature_matrices(X_train, y_train, X_val, cfg)

    rf_params = cfg["models"]["random_forest"]
    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_train_final, y_train)

    lgb_params = cfg["models"]["lightgbm"]
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train_final, y_train)

    rf_val = regression_metrics(y_val, rf.predict(X_val_final))
    lgb_val = regression_metrics(y_val, lgb_model.predict(X_val_final))

    return {
        "random_forest": rf,
        "lightgbm": lgb_model,
        "target_encoder": te,
        "one_hot_encoder": ohe,
        "feature_columns": list(X_train_final.columns),
        "sample_X": X_train_final.head(100),
        "metrics": {"random_forest": rf_val, "lightgbm": lgb_val},
    }


def save_price_artifacts(artifacts: dict[str, Any], output_dir: str | Path | None = None) -> None:
    """
    Save joblib artifacts and sklearn inference pipelines under ``models/price/``.

    Writes ``random_forest.joblib``, ``lightgbm.joblib``, encoders, and
    ``{model}_pipeline.joblib`` files used by the API in local mode.
    """
    out = MODELS_DIR / "price" if output_dir is None else Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config("price")
    joblib.dump(artifacts["random_forest"], out / "random_forest.joblib")
    joblib.dump(artifacts["lightgbm"], out / "lightgbm.joblib")
    joblib.dump(artifacts["target_encoder"], out / "target_encoder.joblib")
    joblib.dump(artifacts["one_hot_encoder"], out / "one_hot_encoder.joblib")
    save_task_pipelines(artifacts, "price", cfg, out)
