"""MLflow experiment tracking helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow
from mlflow.models import infer_signature

from divar.config import PROJECT_ROOT, load_config


def setup_mlflow(experiment_name: str) -> None:
    """Configure tracking URI and experiment."""
    uri = os.getenv("MLFLOW_TRACKING_URI", str(PROJECT_ROOT / "mlruns"))
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)


def _flatten_params(cfg: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested config dict for MLflow params (string values only)."""
    flat: dict[str, Any] = {}
    for key, value in cfg.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            flat.update(_flatten_params(value, full_key))
        elif isinstance(value, (list, tuple)):
            flat[full_key] = str(value)
        else:
            flat[full_key] = value
    return flat


def _metrics_to_flat(metrics: dict[str, dict[str, float]], prefix: str) -> dict[str, float]:
    flat = {}
    for model_name, model_metrics in metrics.items():
        for metric_name, value in model_metrics.items():
            flat[f"{prefix}{model_name}_{metric_name}"] = float(value)
    return flat


def log_price_training_run(
    artifacts: dict[str, Any],
    config: dict[str, Any] | None = None,
    *,
    run_name: str | None = None,
    models_dir: Path | str | None = None,
) -> str:
    """
    Log price training to MLflow: params, metrics, sklearn models, encoders.

    Returns the MLflow run id.
    """
    cfg = config or load_config("price")
    setup_mlflow("divar-price")

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(_flatten_params({"outliers": cfg["outliers"], "split": cfg["split"]}))
        mlflow.log_params(_flatten_params(cfg["models"], prefix="models"))

        flat_metrics = _metrics_to_flat(artifacts["metrics"], prefix="val_")
        mlflow.log_metrics(flat_metrics)

        rf = artifacts["random_forest"]
        lgb_model = artifacts["lightgbm"]
        sample = artifacts.get("sample_X")
        if sample is not None:
            signature = infer_signature(sample, rf.predict(sample))
        else:
            signature = None

        reg = os.getenv("MLFLOW_REGISTER_MODELS", "false").lower() == "true"
        rf_kwargs = {"signature": signature}
        lgb_kwargs = {"signature": signature}
        if reg:
            rf_kwargs["registered_model_name"] = "divar-price-rf"
            lgb_kwargs["registered_model_name"] = "divar-price-lgbm"
        mlflow.sklearn.log_model(rf, name="random_forest", **rf_kwargs)
        mlflow.sklearn.log_model(lgb_model, name="lightgbm", **lgb_kwargs)

        if models_dir is not None:
            mlflow.log_artifacts(str(models_dir), artifact_path="joblib_artifacts")

        return run.info.run_id


def log_credit_training_run(
    artifacts: dict[str, Any],
    config: dict[str, Any] | None = None,
    *,
    run_name: str | None = None,
    models_dir: Path | str | None = None,
) -> str:
    """Log credit training to MLflow."""
    cfg = config or load_config("credit")
    setup_mlflow("divar-credit")

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(_flatten_params({"outliers": cfg["outliers"], "split": cfg["split"]}))
        mlflow.log_params(_flatten_params(cfg["models"], prefix="models"))

        for split_name, split_metrics in artifacts.get("metrics", {}).items():
            for metric_name, value in split_metrics.items():
                mlflow.log_metric(f"{split_name}_{metric_name}", float(value))

        rf = artifacts["random_forest"]
        reg = os.getenv("MLFLOW_REGISTER_MODELS", "false").lower() == "true"
        kwargs = {}
        if reg:
            kwargs["registered_model_name"] = "divar-credit-rf"
        mlflow.sklearn.log_model(rf, name="random_forest", **kwargs)

        if models_dir is not None:
            mlflow.log_artifacts(str(models_dir), artifact_path="joblib_artifacts")

        return run.info.run_id


def write_dvc_metrics(metrics: dict[str, float], path: Path) -> None:
    """Write flat metrics JSON for DVC (e.g. metrics/price.json)."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
