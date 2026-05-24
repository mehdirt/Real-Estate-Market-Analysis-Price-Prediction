import json

import pytest
from fastapi.testclient import TestClient

from divar.config import load_config
from divar.features.price import prepare_price_dataset
from divar.models.train_price import train_price_models
from divar.serve.app import app
from divar.serve.predictor import ModelRegistry


@pytest.fixture
def client_with_models(sample_price_df, tmp_path, monkeypatch):
    train, val = prepare_price_dataset(sample_price_df)
    artifacts = train_price_models(train, val, load_config("price"))
    model_dir = tmp_path / "price"
    model_dir.mkdir()
    from divar.models.sklearn_pipeline import save_task_pipelines

    save_task_pipelines(artifacts, "price", load_config("price"), model_dir)

    registry = ModelRegistry(models_root=tmp_path)
    monkeypatch.setattr("divar.serve.predictor.registry", registry)
    monkeypatch.setattr("divar.serve.app.registry", registry)

    return TestClient(app), val


def test_health(client_with_models):
    client, _ = client_with_models
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "random_forest" in r.json()["models_loaded"]["price"]


def test_predict_price(client_with_models):
    client, val = client_with_models
    cfg = load_config("price")
    target = cfg["target_column"]
    records = json.loads(val.drop(columns=[target]).iloc[0:1].to_json(orient="records"))
    r = client.post(
        "/predict/price",
        json={"records": records, "model": "random_forest"},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 1
    assert body["predictions"][0] > 0
