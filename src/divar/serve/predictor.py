"""Load and run saved inference pipelines."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from divar.config import MODELS_DIR
from divar.models.sklearn_pipeline import ModelName, TaskName, inference_feature_columns

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Lazy-load sklearn inference pipelines from disk."""

    def __init__(self, models_root: Path | None = None):
        self.models_root = models_root or MODELS_DIR
        self._cache: dict[tuple[str, str], Pipeline] = {}

    def list_available(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for task in ("price", "credit"):
            task_dir = self.models_root / task
            if not task_dir.is_dir():
                out[task] = []
                continue
            out[task] = sorted(
                p.stem.replace("_pipeline", "")
                for p in task_dir.glob("*_pipeline.joblib")
            )
        return out

    def load(self, task: TaskName, model: ModelName) -> Pipeline:
        key = (task, model)
        if key not in self._cache:
            path = self.models_root / task / f"{model}_pipeline.joblib"
            if not path.is_file():
                raise FileNotFoundError(
                    f"No pipeline at {path}. Train with divar-train-{task} first."
                )
            self._cache[key] = joblib.load(path)
            logger.info("Loaded %s", path)
        return self._cache[key]

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
