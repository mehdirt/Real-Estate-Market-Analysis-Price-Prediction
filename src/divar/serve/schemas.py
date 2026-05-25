"""Pydantic request/response models for the inference API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

TaskName = Literal["price", "credit"]
ModelName = Literal["random_forest", "lightgbm"]


class PredictRequest(BaseModel):
    """One or more listings with processed feature columns (post feature engineering)."""

    records: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="Feature rows matching the trained model schema (see /schema/{task})",
    )
    model: ModelName = Field(
        default="lightgbm",
        description="Which trained model to use",
    )


class PredictResponse(BaseModel):
    task: TaskName
    model: ModelName
    predictions: list[float]
    model_source: str = Field(description="local joblib files or mlflow Production registry")


class HealthResponse(BaseModel):
    status: str
    model_source: str
    models_loaded: dict[str, list[str]]
    deployment: dict[str, Any] | None = None


class SchemaResponse(BaseModel):
    task: TaskName
    feature_columns: list[str]
    target_column: str
    model_source: str
