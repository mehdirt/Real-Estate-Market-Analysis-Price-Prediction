"""Sklearn-compatible encoding + bundled inference pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd
from category_encoders import TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from divar.config import load_config
from divar.features.common import encode_boolean_features

TaskName = Literal["price", "credit"]
ModelName = Literal["random_forest", "lightgbm"]


class DivarFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Fit target encoding + one-hot on training data; transform for inference.

    Requires ``y`` in ``fit(X, y)``.
    """

    def __init__(self, task: TaskName = "price"):
        """
        Parameters
        ----------
        task
            ``"price"`` or ``"credit"`` — selects encoding config and column sets.
        """
        self.task = task
        self.te_: TargetEncoder | None = None
        self.ohe_: OneHotEncoder | None = None
        self.cfg_: dict[str, Any] | None = None
        self.categorical_cols_: list[str] = []
        self.feature_columns_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        """Fit target encoder and one-hot encoder on processed feature columns."""
        if y is None:
            raise ValueError("DivarFeatureTransformer.fit requires y")
        self.cfg_ = load_config(self.task)
        enc = self.cfg_["encoding"]
        self.categorical_cols_ = list(enc["categorical_columns"])

        X = encode_boolean_features(X, enc["boolean_columns"])
        self.te_ = TargetEncoder(
            cols=enc["target_encoder_cols"],
            smoothing=enc["target_encoder_smoothing"],
        )
        X_enc = self.te_.fit_transform(X, y)

        self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.ohe_.fit(X_enc[self.categorical_cols_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform raw processed features to the numeric matrix used by the model."""
        if self.te_ is None or self.ohe_ is None or self.cfg_ is None:
            raise RuntimeError("Transformer is not fitted")
        enc = self.cfg_["encoding"]
        X = encode_boolean_features(X, enc["boolean_columns"])
        X_enc = self.te_.transform(X)
        X_cat = pd.DataFrame(
            self.ohe_.transform(X_enc[self.categorical_cols_]),
            columns=self.ohe_.get_feature_names_out(self.categorical_cols_),
            index=X_enc.index,
        )
        encoded = pd.concat([X_enc.drop(columns=self.categorical_cols_), X_cat], axis=1)
        if self.feature_columns_ is not None:
            return encoded[self.feature_columns_]
        return encoded


def build_regression_pipeline(
    regressor: BaseEstimator,
    task: TaskName,
) -> Pipeline:
    """Create sklearn Pipeline: feature encoding → regressor."""
    return Pipeline(
        [
            ("features", DivarFeatureTransformer(task=task)),
            ("model", regressor),
        ]
    )


def inference_feature_columns(task: TaskName) -> list[str]:
    """Feature column names expected at inference (processed schema, no target)."""
    cfg = load_config(task)
    target = cfg["target_column"]
    if task == "price":
        return [
            "cat2_slug",
            "cat3_slug",
            "city_slug",
            "building_size",
            "deed_type",
            "has_business_deed",
            "rooms_count",
            "is_rebuilt",
            "floor_material",
            "building_age",
            "luxury_score",
            "non_luxury_score",
            "neighborhood_slug",
            "location_latitude",
            "location_longitude",
            "floor_cat",
            "total_floors_cat",
        ]
    return [
        c
        for c in [
            "cat2_slug",
            "cat3_slug",
            "city_slug",
            "building_size",
            "rooms_count",
            "is_rebuilt",
            "floor",
            "total_floors_count",
            "building_age",
            "luxury_items",
            "non_luxury_items",
            "unit_per_floor",
            "neighborhood_slug",
            "location_latitude",
            "location_longitude",
            "has_warm_water_provider",
            "has_heating_system",
            "has_cooling_system",
            "building_direction",
            "floor_material",
            "has_restroom",
        ]
        if c != target
    ]


def pipeline_from_fitted_encoders(
    task: TaskName,
    regressor: BaseEstimator,
    te: TargetEncoder,
    ohe: OneHotEncoder,
    config: dict[str, Any],
    feature_columns: list[str] | None = None,
) -> Pipeline:
    """
    Assemble a fitted ``features → model`` pipeline without refitting.

    Reuses encoders and regressor from training so inference matches offline metrics.
    """
    transformer = DivarFeatureTransformer(task=task)
    transformer.cfg_ = config
    transformer.te_ = te
    transformer.ohe_ = ohe
    transformer.categorical_cols_ = list(config["encoding"]["categorical_columns"])
    transformer.feature_columns_ = feature_columns
    return Pipeline([("features", transformer), ("model", regressor)])


def save_task_pipelines(
    artifacts: dict[str, Any],
    task: TaskName,
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    """Write ``{random_forest,lightgbm}_pipeline.joblib`` for local/API inference."""
    from pathlib import Path as PathLib

    import joblib

    out = PathLib(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    te = artifacts["target_encoder"]
    ohe = artifacts["one_hot_encoder"]
    for model_name in ("random_forest", "lightgbm"):
        if model_name not in artifacts:
            continue
        pipe = pipeline_from_fitted_encoders(
            task,
            artifacts[model_name],
            te,
            ohe,
            config,
            feature_columns=artifacts.get("feature_columns"),
        )
        joblib.dump(pipe, out / f"{model_name}_pipeline.joblib")
