from divar.tracking.mlflow_utils import _flatten_params, _metrics_to_flat


def test_flatten_params():
    flat = _flatten_params({"split": {"test_size": 0.2, "random_state": 42}})
    assert flat["split.test_size"] == 0.2
    assert flat["split.random_state"] == 42


def test_metrics_to_flat():
    metrics = {"random_forest": {"r2": 0.9, "rmse": 1e6}}
    flat = _metrics_to_flat(metrics, prefix="val_")
    assert flat["val_random_forest_r2"] == 0.9
    assert flat["val_random_forest_rmse"] == 1e6
