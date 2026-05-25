"""Consistent names for MLflow experiments, runs, and registry models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

TaskName = str
ModelName = str


def experiment_name(task: TaskName, mlflow_cfg: dict[str, Any]) -> str:
    """MLflow experiment path, e.g. ``divar/price-prediction``."""
    return mlflow_cfg["experiments"][task]


def registered_model_name(task: TaskName, model: ModelName, mlflow_cfg: dict[str, Any]) -> str:
    """Registry name, e.g. ``divar-price-lightgbm``."""
    prefix = mlflow_cfg["registry"]["name_prefix"]
    algo = mlflow_cfg["registry"]["algorithms"][model]
    return f"{prefix}-{task}-{algo}"


def _format_r2(r2: float) -> str:
    """Compact R² for run names (0.8721 → r2-0p8721)."""
    return f"r2-0p{int(round(r2 * 10000)):04d}"


def _metrics_block(metrics: dict[str, Any]) -> dict[str, Any]:
    """Normalize price (flat) vs credit (nested val/test) metrics."""
    if "val" in metrics and isinstance(metrics["val"], dict):
        first = next(iter(metrics["val"].values()), None)
        if isinstance(first, dict) and "r2" in first:
            return metrics["val"]
    return metrics


def run_name(task: TaskName, metrics: dict[str, Any]) -> str:
    """
    Run name: ``price-20250521T143052-rf-r2-0p7100-lgb-r2-0p8800``.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    parts = [task, ts]
    block = _metrics_block(metrics)
    for algo, short in (("random_forest", "rf"), ("lightgbm", "lgb")):
        if algo in block and isinstance(block[algo], dict) and "r2" in block[algo]:
            parts.extend([short, _format_r2(block[algo]["r2"])])
    return "-".join(parts)


def model_version_description(
    task: TaskName,
    model: ModelName,
    val_r2: float,
    run_id: str,
) -> str:
    """Human-readable Registry version description."""
    return (
        f"{task}/{model} val_r2={val_r2:.4f} "
        f"run={run_id[:8]} @ {datetime.now(timezone.utc).isoformat(timespec='seconds')}"
    )


def run_tags(task: TaskName, model_names: list[str]) -> dict[str, str]:
    """Standard tags attached to every training run."""
    return {
        "project": "divar",
        "task": task,
        "models_trained": ",".join(model_names),
    }
