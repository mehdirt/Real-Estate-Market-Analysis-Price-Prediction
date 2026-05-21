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


def fit_feature_matrices(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, TargetEncoder, OneHotEncoder]:
    """
    Fit encoders on training data and return final numeric matrices for train and val.
    """
    cfg = config or load_config("price")
    enc = cfg["encoding"]

    X_train = encode_boolean_features(X_train, enc["boolean_columns"])
    X_val = encode_boolean_features(X_val, enc["boolean_columns"])

    te = TargetEncoder(cols=enc["target_encoder_cols"], smoothing=enc["target_encoder_smoothing"])
    X_train_enc = te.fit_transform(X_train, y_train)
    X_val_enc = te.transform(X_val)

    categorical_cols = enc["categorical_columns"]
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    X_train_cat = pd.DataFrame(
        ohe.fit_transform(X_train_enc[categorical_cols]),
        columns=ohe.get_feature_names_out(categorical_cols),
        index=X_train_enc.index,
    )
    X_val_cat = pd.DataFrame(
        ohe.transform(X_val_enc[categorical_cols]),
        columns=ohe.get_feature_names_out(categorical_cols),
        index=X_val_enc.index,
    )

    X_train_final = pd.concat([X_train_enc.drop(columns=categorical_cols), X_train_cat], axis=1)
    X_val_final = pd.concat([X_val_enc.drop(columns=categorical_cols), X_val_cat], axis=1)

    return X_train_final, X_val_final, te, ohe
