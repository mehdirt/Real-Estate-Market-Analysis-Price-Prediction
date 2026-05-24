"""FastAPI application for Divar ML inference."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from divar.config import load_config
from divar.models.sklearn_pipeline import inference_feature_columns
from divar.serve.predictor import registry
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
    version="0.2.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", models_loaded=registry.list_available())


@app.get("/schema/{task}", response_model=SchemaResponse)
def schema(task: TaskName) -> SchemaResponse:
    cfg = load_config(task)
    return SchemaResponse(
        task=task,
        feature_columns=inference_feature_columns(task),
        target_column=cfg["target_column"],
    )


@app.post("/predict/{task}", response_model=PredictResponse)
def predict(task: TaskName, body: PredictRequest) -> PredictResponse:
    try:
        predictions = registry.predict(task, body.model, body.records)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictResponse(task=task, model=body.model, predictions=predictions)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "divar.serve.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
