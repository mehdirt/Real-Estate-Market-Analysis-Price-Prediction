"""FastAPI application for Divar ML inference."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException

from divar.config import load_config
from divar.models.sklearn_pipeline import inference_feature_columns
from divar.serve.predictor import model_source, registry
from divar.serve.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    SchemaResponse,
    TaskName,
)

app = FastAPI(
    title="Divar ML API",
    description="Sale price and rent/credit prediction using trained sklearn pipelines.",
    version="0.2.1",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Service health, model source, loaded pipelines, and deployment manifest."""
    deployment = {}
    for task in ("price", "credit"):
        info = registry.deployment_info(task)
        if info:
            deployment[task] = info
    return HealthResponse(
        status="ok",
        model_source=model_source(),
        models_loaded=registry.list_available(),
        deployment=deployment or None,
    )


@app.get("/schema/{task}", response_model=SchemaResponse)
def schema(task: TaskName) -> SchemaResponse:
    """Return required input feature columns and target name for a task."""
    cfg = load_config(task)
    return SchemaResponse(
        task=task,
        feature_columns=inference_feature_columns(task),
        target_column=cfg["target_column"],
        model_source=model_source(),
    )


@app.post("/predict/{task}", response_model=PredictResponse)
def predict(task: TaskName, body: PredictRequest) -> PredictResponse:
    """Predict sale price or total credit for one or more listings."""
    try:
        predictions = registry.predict(task, body.model, body.records)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictResponse(
        task=task,
        model=body.model,
        predictions=predictions,
        model_source=model_source(),
    )


def main() -> None:
    """Start the FastAPI server (default ``127.0.0.1:8000``)."""
    import uvicorn

    host = os.getenv("SERVE_HOST", "127.0.0.1")
    port = int(os.getenv("SERVE_PORT", "8000"))
    uvicorn.run(
        "divar.serve.app:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
