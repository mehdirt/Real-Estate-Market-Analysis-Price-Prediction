from divar.tracking.naming import (
    experiment_name,
    registered_model_name,
    run_name,
)
from divar.tracking.registry import load_mlflow_config


def test_experiment_names():
    cfg = load_mlflow_config()
    assert experiment_name("price", cfg) == "divar/price-prediction"
    assert experiment_name("credit", cfg) == "divar/credit-prediction"


def test_registered_model_names():
    cfg = load_mlflow_config()
    assert registered_model_name("price", "lightgbm", cfg) == "divar-price-lightgbm"
    assert registered_model_name("credit", "random_forest", cfg) == "divar-credit-random-forest"


def test_run_name_includes_task_and_r2():
    metrics = {
        "random_forest": {"r2": 0.71, "rmse": 1e6},
        "lightgbm": {"r2": 0.88, "rmse": 9e5},
    }
    name = run_name("price", metrics)
    assert name.startswith("price-")
    assert "rf" in name
    assert "lgb" in name
