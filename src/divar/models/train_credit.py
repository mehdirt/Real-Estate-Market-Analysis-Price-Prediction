"""Train rent/credit (total_credit) models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from divar.config import MODELS_DIR, load_config
from divar.models.encoding import fit_feature_matrices, split_xy, transform_features
from divar.models.metrics import regression_metrics


def train_credit_models(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train Random Forest and return metrics on val and test."""
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

    metrics = {
        "val": regression_metrics(y_val, rf.predict(X_val_final)),
        "test": regression_metrics(y_test, rf.predict(X_test_final)),
    }

    return {
        "random_forest": rf,
        "target_encoder": te,
        "one_hot_encoder": ohe,
        "feature_columns": list(X_train_final.columns),
        "sample_X": X_train_final.head(100),
        "metrics": metrics,
    }


def save_credit_artifacts(artifacts: dict[str, Any], output_dir: str | Path | None = None) -> None:
    """Persist credit model and encoders."""
    out = MODELS_DIR / "credit" if output_dir is None else Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts["random_forest"], out / "random_forest.joblib")
    joblib.dump(artifacts["target_encoder"], out / "target_encoder.joblib")
    joblib.dump(artifacts["one_hot_encoder"], out / "one_hot_encoder.joblib")
