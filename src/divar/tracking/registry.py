"""Model Registry validation, registration, and stage transitions."""

from __future__ import annotations

import logging
import os
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from divar.tracking.naming import (
    ModelName,
    TaskName,
    model_version_description,
    registered_model_name,
)

logger = logging.getLogger(__name__)


class RegistryValidationError(Exception):
    """Model did not meet registry quality thresholds."""


def load_mlflow_config() -> dict[str, Any]:
    from divar.config import load_config

    return load_config("mlflow")


def get_val_r2(
    metrics: dict[str, Any],
    model: ModelName,
    *,
    split: str = "val",
) -> float | None:
    """Extract validation R² for an algorithm from training metrics."""
    if model in metrics and isinstance(metrics[model], dict):
        return metrics[model].get("r2")
    if split in metrics and model in metrics[split]:
        return metrics[split][model].get("r2")
    return None


def qualifies_for_registry(
    val_r2: float | None,
    mlflow_cfg: dict[str, Any] | None = None,
) -> bool:
    """Return True if validation R² meets the registry minimum."""
    cfg = mlflow_cfg or load_mlflow_config()
    if val_r2 is None:
        return False
    minimum = float(
        os.getenv("MLFLOW_MIN_VAL_R2", cfg["registry_validation"]["min_val_r2"])
    )
    return val_r2 >= minimum


def ensure_registered_model(client: MlflowClient, name: str) -> None:
    """Create registered model if it does not exist."""
    try:
        client.get_registered_model(name)
    except Exception:
        client.create_registered_model(name)


def register_pipeline_version(
    *,
    task: TaskName,
    model: ModelName,
    run_id: str,
    val_r2: float,
    pipeline_artifact_path: str,
    mlflow_cfg: dict[str, Any] | None = None,
) -> int | None:
    """
    Register a pipeline artifact from a run if validation R² passes.

    Returns the new version number, or None if skipped.
    """
    cfg = mlflow_cfg or load_mlflow_config()
    if not qualifies_for_registry(val_r2, cfg):
        logger.warning(
            "Skipping registry for %s/%s: val_r2=%.4f < min %.2f",
            task,
            model,
            val_r2,
            cfg["registry_validation"]["min_val_r2"],
        )
        return None

    name = registered_model_name(task, model, cfg)
    client = MlflowClient()
    ensure_registered_model(client, name)

    source = f"runs:/{run_id}/{pipeline_artifact_path}"
    desc = model_version_description(task, model, val_r2, run_id)
    mv = mlflow.register_model(model_uri=source, name=name)
    version = int(mv.version)
    client.update_model_version(name=name, version=version, description=desc)
    client.set_model_version_tag(name, version, "task", task)
    client.set_model_version_tag(name, version, "algorithm", model)
    client.set_model_version_tag(name, version, "val_r2", f"{val_r2:.6f}")
    client.set_model_version_tag(name, version, "qualified", "true")

    if cfg.get("auto_promote_to_staging", True):
        staging = cfg["stages"]["staging"]
        client.transition_model_version_stage(
            name=name,
            version=version,
            stage=staging,
            archive_existing_versions=False,
        )
        logger.info("Registered %s v%s → %s", name, version, staging)

    return version


def promote_to_production(
    task: TaskName,
    model: ModelName,
    version: int | None = None,
    *,
    mlflow_cfg: dict[str, Any] | None = None,
) -> int:
    """
    Promote a registry version to Production.

    If ``version`` is None, promotes the latest version in Staging for that model.
    """
    cfg = mlflow_cfg or load_mlflow_config()
    name = registered_model_name(task, model, cfg)
    client = MlflowClient()
    production = cfg["stages"]["production"]

    if version is None:
        versions = client.search_model_versions(f"name='{name}'")
        staging = cfg["stages"]["staging"]
        staging_versions = [
            int(v.version) for v in versions if v.current_stage == staging
        ]
        if not staging_versions:
            raise RegistryValidationError(
                f"No {staging} version for {name}. Train and qualify a model first."
            )
        version = max(staging_versions)

    client.transition_model_version_stage(
        name=name,
        version=version,
        stage=production,
        archive_existing_versions=True,
    )
    logger.info("Promoted %s v%s → %s", name, version, production)
    return version


def production_model_uri(task: TaskName, model: ModelName, mlflow_cfg: dict[str, Any] | None = None) -> str:
    """URI for the Production-stage registered model."""
    name = registered_model_name(task, model, mlflow_cfg or load_mlflow_config())
    return f"models:/{name}/Production"
