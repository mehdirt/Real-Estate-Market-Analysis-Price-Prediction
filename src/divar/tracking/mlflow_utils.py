"""MLflow experiment tracking helpers."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
from mlflow.models import infer_signature

from divar.config import PROJECT_ROOT, load_config
from divar.tracking.deployment import write_deployment_manifest
from divar.tracking.naming import experiment_name, run_name, run_tags
from divar.tracking.registry import (
    get_val_r2,
    load_mlflow_config,
    qualifies_for_registry,
    register_pipeline_version,
)

logger = logging.getLogger(__name__)

TaskName = str


def setup_mlflow(task: TaskName) -> str:
    """Configure tracking URI and experiment; return experiment name."""
    uri = os.getenv("MLFLOW_TRACKING_URI", str(PROJECT_ROOT / "mlruns"))
    mlflow.set_tracking_uri(uri)
    mlflow_cfg = load_mlflow_config()
    exp = experiment_name(task, mlflow_cfg)
    mlflow.set_experiment(exp)
    return exp


def _flatten_params(cfg: dict[str, Any], prefix: str = "") -> dict[str, Any]:
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


def _metrics_to_flat(metrics: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten nested (credit) or flat (price) metrics for MLflow."""
    flat: dict[str, float] = {}
    if "val" in metrics and isinstance(metrics["val"], dict):
        for split_name, split_block in metrics.items():
            if not isinstance(split_block, dict):
                continue
            for model_name, model_metrics in split_block.items():
                if isinstance(model_metrics, dict):
                    for metric_name, value in model_metrics.items():
                        flat[f"{split_name}_{model_name}_{metric_name}"] = float(value)
    else:
        for model_name, model_metrics in metrics.items():
            if isinstance(model_metrics, dict):
                for metric_name, value in model_metrics.items():
                    flat[f"{prefix}{model_name}_{metric_name}"] = float(value)
    return flat


def _should_register() -> bool:
    return os.getenv("MLFLOW_REGISTER_QUALIFIED", "true").lower() == "true"


def log_training_run(
    task: TaskName,
    artifacts: dict[str, Any],
    config: dict[str, Any] | None = None,
    *,
    models_dir: Path | str | None = None,
) -> str:
    """
    Log training with structured experiment/run names and conditional registry.

    Registers to Model Registry only when validation R² >= min (default 0.65).
    Qualified versions go to **Staging**; promote to **Production** via CLI.
    """
    cfg = config or load_config(task)
    mlflow_cfg = load_mlflow_config()
    models_path = Path(models_dir) if models_dir else None

    exp = setup_mlflow(task)
    trained = [m for m in ("random_forest", "lightgbm") if m in artifacts]

    with mlflow.start_run(run_name=run_name(task, artifacts["metrics"])) as run:
        mlflow.set_tags(run_tags(task, trained))
        mlflow.log_params(_flatten_params({"outliers": cfg["outliers"], "split": cfg["split"]}))
        mlflow.log_params(_flatten_params(cfg["models"], prefix="models"))
        mlflow.log_param("mlflow.experiment", exp)

        mlflow.log_metrics(_metrics_to_flat(artifacts["metrics"], prefix="val_"))

        sample = artifacts.get("sample_X")
        signature = None
        if sample is not None and "random_forest" in artifacts:
            signature = infer_signature(sample, artifacts["random_forest"].predict(sample))

        registry_versions: dict[str, int | None] = {}
        val_r2_map: dict[str, float | None] = {}
        min_r2 = mlflow_cfg["registry_validation"]["min_val_r2"]
        mlflow.log_param("registry.min_val_r2", min_r2)

        for model in trained:
            if models_path is None:
                continue
            pipeline_file = models_path / f"{model}_pipeline.joblib"
            if not pipeline_file.is_file():
                logger.warning("Pipeline not found for %s", model)
                continue

            pipe = joblib.load(pipeline_file)
            artifact_path = f"pipeline_{model}"
            val_r2 = get_val_r2(artifacts["metrics"], model, split="val")
            val_r2_map[model] = val_r2

            log_kwargs: dict[str, Any] = {}
            if signature is not None:
                log_kwargs["signature"] = signature
            mlflow.sklearn.log_model(pipe, artifact_path=artifact_path, **log_kwargs)

            qualified = qualifies_for_registry(val_r2, mlflow_cfg)
            mlflow.set_tag(f"{model}.registry_qualified", str(qualified).lower())
            if val_r2 is not None:
                mlflow.set_tag(f"{model}.val_r2", f"{val_r2:.6f}")

            if _should_register() and qualified:
                registry_versions[model] = register_pipeline_version(
                    task=task,
                    model=model,
                    run_id=run.info.run_id,
                    val_r2=val_r2,
                    pipeline_artifact_path=artifact_path,
                    mlflow_cfg=mlflow_cfg,
                )
            elif _should_register():
                logger.info("Skipped registry for %s (val_r2=%s < %.2f)", model, val_r2, min_r2)
                registry_versions[model] = None

        if models_path is not None:
            mlflow.log_artifacts(str(models_path), artifact_path="joblib_artifacts")
            write_deployment_manifest(
                task,
                models_dir=models_path,
                run_id=run.info.run_id,
                registry_versions=registry_versions,
                val_r2=val_r2_map,
            )

        return run.info.run_id


def log_price_training_run(
    artifacts: dict[str, Any],
    config: dict[str, Any] | None = None,
    *,
    run_name: str | None = None,
    models_dir: Path | str | None = None,
) -> str:
    del run_name
    return log_training_run("price", artifacts, config, models_dir=models_dir)


def log_credit_training_run(
    artifacts: dict[str, Any],
    config: dict[str, Any] | None = None,
    *,
    run_name: str | None = None,
    models_dir: Path | str | None = None,
) -> str:
    del run_name
    return log_training_run("credit", artifacts, config, models_dir=models_dir)


def write_dvc_metrics(metrics: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
