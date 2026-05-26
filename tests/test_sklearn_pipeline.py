import joblib
from sklearn.ensemble import RandomForestRegressor

from divar.config import load_config
from divar.features.price import prepare_price_dataset
from divar.models.sklearn_pipeline import (
    build_regression_pipeline,
    inference_feature_columns,
    pipeline_from_fitted_encoders,
    save_task_pipelines,
)
from divar.models.train_price import train_price_models


def test_inference_feature_columns_price():
    cols = inference_feature_columns("price")
    assert "price_value" not in cols
    assert "building_size" in cols


def test_sklearn_pipeline_roundtrip(sample_price_df, tmp_path):
    train, val = prepare_price_dataset(sample_price_df)
    artifacts = train_price_models(train, val, load_config("price"))

    cfg = load_config("price")
    target = cfg["target_column"]
    X_val = val.drop(columns=[target])

    pipe = pipeline_from_fitted_encoders(
        "price",
        artifacts["random_forest"],
        artifacts["target_encoder"],
        artifacts["one_hot_encoder"],
        cfg,
        feature_columns=artifacts["feature_columns"],
    )
    preds_pipe = pipe.predict(X_val)
    assert len(preds_pipe) == len(val)

    out = tmp_path / "price"
    save_task_pipelines(artifacts, "price", cfg, out)
    loaded = joblib.load(out / "random_forest_pipeline.joblib")
    assert loaded.predict(X_val).shape == (len(val),)


def test_build_regression_pipeline_fits(sample_price_df):
    train, val = prepare_price_dataset(sample_price_df)
    cfg = load_config("price")
    target = cfg["target_column"]
    X_train = train.drop(columns=[target])
    y_train = train[target]

    pipe = build_regression_pipeline(RandomForestRegressor(n_estimators=10, random_state=42), "price")
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_train)
    assert len(preds) == len(X_train)
