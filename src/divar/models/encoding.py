"""Feature encoding: target encoding + one-hot (notebook parity)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder

from divar.config import load_config
from divar.features.common import encode_boolean_features


def split_xy(
    train: pd.DataFrame,
    val: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split train/val frames into X and y."""
    X_train = train.drop(columns=[target_column])
    y_train = train[target_column]
    X_val = val.drop(columns=[target_column])
    y_val = val[target_column]
    return X_train, y_train, X_val, y_val


def _encode_frame(
    X: pd.DataFrame,
    y: pd.Series | None,
    cfg: dict[str, Any],
    te: TargetEncoder,
    ohe: OneHotEncoder,
    *,
    fit: bool,
) -> pd.DataFrame:
    enc = cfg["encoding"]
    X = encode_boolean_features(X, enc["boolean_columns"])
    if fit:
        if y is None:
            raise ValueError("y required when fit=True")
        X_enc = te.fit_transform(X, y)
    else:
        X_enc = te.transform(X)

    categorical_cols = enc["categorical_columns"]
    if fit:
        X_cat = pd.DataFrame(
            ohe.fit_transform(X_enc[categorical_cols]),
            columns=ohe.get_feature_names_out(categorical_cols),
            index=X_enc.index,
        )
    else:
        X_cat = pd.DataFrame(
            ohe.transform(X_enc[categorical_cols]),
            columns=ohe.get_feature_names_out(categorical_cols),
            index=X_enc.index,
        )
    return pd.concat([X_enc.drop(columns=categorical_cols), X_cat], axis=1)


def fit_feature_matrices(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_other: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, TargetEncoder, OneHotEncoder]:
    """Fit encoders on training data; transform train and other split."""
    cfg = config or load_config("price")
    enc = cfg["encoding"]

    te = TargetEncoder(cols=enc["target_encoder_cols"], smoothing=enc["target_encoder_smoothing"])
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    X_train_final = _encode_frame(X_train, y_train, cfg, te, ohe, fit=True)
    X_other_final = _encode_frame(X_other, None, cfg, te, ohe, fit=False)

    return X_train_final, X_other_final, te, ohe


def transform_features(
    X: pd.DataFrame,
    config: dict[str, Any],
    te: TargetEncoder,
    ohe: OneHotEncoder,
) -> pd.DataFrame:
    """Apply fitted encoders to a feature matrix."""
    return _encode_frame(X, None, config, te, ohe, fit=False)
