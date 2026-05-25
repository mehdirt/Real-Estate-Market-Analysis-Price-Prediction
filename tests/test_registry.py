from divar.tracking.registry import get_val_r2, load_mlflow_config, qualifies_for_registry


def test_qualifies_for_registry_threshold():
    cfg = load_mlflow_config()
    assert qualifies_for_registry(0.70, cfg) is True
    assert qualifies_for_registry(0.65, cfg) is True
    assert qualifies_for_registry(0.64, cfg) is False
    assert qualifies_for_registry(None, cfg) is False


def test_get_val_r2_price_metrics():
    metrics = {"random_forest": {"r2": 0.8}, "lightgbm": {"r2": 0.9}}
    assert get_val_r2(metrics, "lightgbm") == 0.9


def test_get_val_r2_credit_metrics():
    metrics = {"val": {"random_forest": {"r2": 0.7}, "lightgbm": {"r2": 0.85}}}
    assert get_val_r2(metrics, "lightgbm", split="val") == 0.85
