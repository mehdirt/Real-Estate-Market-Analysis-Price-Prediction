"""Train rent/credit (total_credit) models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from divar.config import MODELS_DIR, load_config
from divar.models.encoding import fit_feature_matrices, split_xy, transform_features
from divar.models.metrics import regression_metrics
from divar.models.sklearn_pipeline import save_task_pipelines


def train_credit_models(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Train Random Forest and LightGBM for ``total_credit`` prediction.

    Returns
    -------
    dict
        Models, encoders, and ``metrics`` with ``val`` and ``test`` scores per algorithm.
    """
    cfg = config or load_config("credit")
    target = cfg["target_column"]

    X_train, y_train, X_val, y_val = split_xy(train, val, target)
    X_test = test.drop(columns=[target])
    y_test = test[target]

    X_train_final, X_val_final, te, ohe = fit_feature_matrices(X_train, y_train, X_val, cfg)
    X_test_final = transform_features(X_test, cfg, te, ohe)

    rf_params = cfg["models"]["random_forest"]
    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_train_final, y_train)

    lgb_params = cfg["models"]["lightgbm"]
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train_final, y_train)

    metrics = {
        "val": {
            "random_forest": regression_metrics(y_val, rf.predict(X_val_final)),
            "lightgbm": regression_metrics(y_val, lgb_model.predict(X_val_final)),
        },
        "test": {
            "random_forest": regression_metrics(y_test, rf.predict(X_test_final)),
            "lightgbm": regression_metrics(y_test, lgb_model.predict(X_test_final)),
        },
    }

    return {
        "random_forest": rf,
        "lightgbm": lgb_model,
        "target_encoder": te,
        "one_hot_encoder": ohe,
        "feature_columns": list(X_train_final.columns),
        "sample_X": X_train_final.head(100),
        "metrics": metrics,
    }


def save_credit_artifacts(artifacts: dict[str, Any], output_dir: str | Path | None = None) -> None:
    """Save joblib models, encoders, and inference pipelines under ``models/credit/``."""
    out = MODELS_DIR / "credit" if output_dir is None else Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config("credit")
    joblib.dump(artifacts["random_forest"], out / "random_forest.joblib")
    joblib.dump(artifacts["lightgbm"], out / "lightgbm.joblib")
    joblib.dump(artifacts["target_encoder"], out / "target_encoder.joblib")
    joblib.dump(artifacts["one_hot_encoder"], out / "one_hot_encoder.joblib")
    save_task_pipelines(artifacts, "credit", cfg, out)
