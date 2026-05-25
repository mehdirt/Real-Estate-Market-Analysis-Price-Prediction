"""Load and run saved inference pipelines (local disk or MLflow Production)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from divar.config import MODELS_DIR
from divar.models.sklearn_pipeline import ModelName, TaskName, inference_feature_columns
from divar.tracking.deployment import read_deployment_manifest
from divar.tracking.registry import production_model_uri

logger = logging.getLogger(__name__)


def model_source() -> str:
    """``local`` = joblib under ``models/``; ``mlflow`` = Registry Production stage."""
    return os.getenv("MODEL_SOURCE", "local").lower()


class ModelRegistry:
    """Lazy-load sklearn pipelines from local files or MLflow Model Registry."""

    def __init__(self, models_root: Path | None = None):
        self.models_root = models_root or MODELS_DIR
        self._cache: dict[tuple[str, str], Pipeline] = {}
        self._source = model_source()

    @property
    def source(self) -> str:
        return self._source

    def list_available(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for task in ("price", "credit"):
            if self._source == "mlflow":
                out[task] = self._list_mlflow_models(task)
            else:
                task_dir = self.models_root / task
                if not task_dir.is_dir():
                    out[task] = []
                    continue
                out[task] = sorted(
                    p.stem.replace("_pipeline", "")
                    for p in task_dir.glob("*_pipeline.joblib")
                )
        return out

    def _list_mlflow_models(self, task: TaskName) -> list[str]:
        try:
            from mlflow.tracking import MlflowClient

            from divar.tracking.naming import registered_model_name
            from divar.tracking.registry import load_mlflow_config

            cfg = load_mlflow_config()
            client = MlflowClient()
            found = []
            for model in ("random_forest", "lightgbm"):
                name = registered_model_name(task, model, cfg)
                try:
                    versions = client.get_latest_versions(name, stages=["Production"])
                    if versions:
                        found.append(model)
                except Exception:
                    continue
            return found
        except Exception as exc:
            logger.warning("MLflow list failed: %s", exc)
            return []

    def deployment_info(self, task: TaskName) -> dict | None:
        """Local deployment manifest (``models/{task}/deployment.json``)."""
        if self._source != "local":
            return {"source": "mlflow", "note": "Loads Production stage from Model Registry"}
        return read_deployment_manifest(task, self.models_root)

    def load(self, task: TaskName, model: ModelName) -> Pipeline:
        key = (task, model)
        if key not in self._cache:
            if self._source == "mlflow":
                self._cache[key] = self._load_mlflow(task, model)
            else:
                self._cache[key] = self._load_local(task, model)
        return self._cache[key]

    def _load_local(self, task: TaskName, model: ModelName) -> Pipeline:
        path = self.models_root / task / f"{model}_pipeline.joblib"
        if not path.is_file():
            raise FileNotFoundError(
                f"No pipeline at {path}. Run divar-train-{task} or set MODEL_SOURCE=mlflow."
            )
        pipe = joblib.load(path)
        logger.info("Loaded local pipeline %s", path)
        return pipe

    def _load_mlflow(self, task: TaskName, model: ModelName) -> Pipeline:
        import mlflow

        uri = os.getenv("MLFLOW_TRACKING_URI")
        if uri:
            mlflow.set_tracking_uri(uri)
        model_uri = production_model_uri(task, model)
        try:
            pipe = mlflow.sklearn.load_model(model_uri)
        except Exception as exc:
            raise FileNotFoundError(
                f"No Production model at {model_uri}. "
                f"Train, qualify (val R² >= 0.65), then: divar-promote-model --task {task} --model {model}"
            ) from exc
        logger.info("Loaded MLflow Production model %s", model_uri)
        return pipe

    def predict(
        self,
        task: TaskName,
        model: ModelName,
        records: list[dict],
    ) -> list[float]:
        expected = inference_feature_columns(task)
        df = pd.DataFrame(records)
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        pipe = self.load(task, model)
        preds = pipe.predict(df[expected])
        return [float(p) for p in preds]


registry = ModelRegistry()
