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


def train_price_models(
    train: pd.DataFrame,
    val: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Train RF and LightGBM on prepared train/val frames.

    Returns dict with models, encoders, and validation metrics.
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
        "metrics": {"random_forest_val": rf_val, "lightgbm_val": lgb_val},
    }


def save_price_artifacts(artifacts: dict[str, Any], output_dir: str | Path | None = None) -> None:
    """Persist models and encoders with joblib."""
    out = MODELS_DIR / "price" if output_dir is None else Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts["random_forest"], out / "random_forest.joblib")
    joblib.dump(artifacts["lightgbm"], out / "lightgbm.joblib")
    joblib.dump(artifacts["target_encoder"], out / "target_encoder.joblib")
    joblib.dump(artifacts["one_hot_encoder"], out / "one_hot_encoder.joblib")
