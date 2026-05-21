"""Sale price prediction dataset preparation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from divar.config import load_config
from divar.features.common import (
    add_building_age,
    add_luxury_scores,
    fill_location_from_train,
    map_rooms_count,
)

LUXURY_COLS = ["has_security_guard", "has_barbecue", "has_pool", "has_jacuzzi", "has_sauna"]
NON_LUXURY_COLS = ["has_balcony", "has_elevator", "has_warehouse", "has_parking"]


def _apply_floor_features(train: pd.DataFrame, val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bin floor and total_floors_count using statistics from train only."""
    train_out = train.copy()
    val_out = val.copy()

    for frame in (train_out, val_out):
        frame["floor"] = pd.to_numeric(frame["floor"], errors="coerce")
        frame["floor"] = frame["floor"].apply(lambda x: 0 if pd.notna(x) and x < 0 else x)

    floor_mode = train_out["floor"].mode()[0]
    train_out["floor"] = train_out["floor"].fillna(floor_mode).astype(int)
    val_out["floor"] = val_out["floor"].fillna(floor_mode).astype(int)

    floor_bins = [-1, 0, 5, 10, 20, float("inf")]
    floor_labels = ["basement", "low-height", "mid-height", "high-height", "very-high"]
    train_out["floor_cat"] = pd.cut(train_out["floor"], bins=floor_bins, labels=floor_labels, right=True)
    val_out["floor_cat"] = pd.cut(val_out["floor"], bins=floor_bins, labels=floor_labels, right=True)

    for frame in (train_out, val_out):
        frame["total_floors_count"] = pd.to_numeric(frame["total_floors_count"], errors="coerce")
        frame["total_floors_count"] = frame["total_floors_count"].apply(
            lambda x: 0 if pd.notna(x) and x < 0 else x
        )

    total_mode = train_out["total_floors_count"].mode()[0]
    train_out["total_floors_count"] = train_out["total_floors_count"].fillna(total_mode).astype(int)
    val_out["total_floors_count"] = val_out["total_floors_count"].fillna(total_mode).astype(int)

    tf_bins = [0, 5, 15, 20, float("inf")]
    tf_labels = ["low", "medium", "high", "very-high"]
    train_out["total_floors_cat"] = pd.cut(
        train_out["total_floors_count"], bins=tf_bins, labels=tf_labels, right=True
    )
    val_out["total_floors_cat"] = pd.cut(
        val_out["total_floors_count"], bins=tf_bins, labels=tf_labels, right=True
    )

    return (
        train_out.drop(columns=["floor", "total_floors_count"]),
        val_out.drop(columns=["floor", "total_floors_count"]),
    )


def prepare_price_features(df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """
    Clean and engineer features for sale price modeling (before train/val split).
    """
    cfg = config or load_config("price")
    outliers = cfg["outliers"]
    sell_slugs = cfg["sell_cat2_slugs"]
    feature_cols = cfg["feature_columns"] + [cfg["target_column"]]

    data = df[df[cfg["target_column"]].notna()].copy()
    data = data[[c for c in feature_cols if c in data.columns]]

    data = add_building_age(
        data,
        reference_year=cfg["building_age"]["reference_year"],
        pre_1370_replacement=cfg["building_age"]["pre_1370_replacement"],
        bins=[0, *cfg["building_age"]["bins"][1:], np.inf],
        labels=cfg["building_age"]["labels"],
    )
    data = add_luxury_scores(data, luxury_cols=LUXURY_COLS, non_luxury_cols=NON_LUXURY_COLS)
    data["rooms_count"] = map_rooms_count(data["rooms_count"])

    data = data[data["cat2_slug"].isin(sell_slugs)].copy()
    data["cat2_slug"] = data["cat2_slug"].fillna("unselect")
    data = data[~data["cat3_slug"].str.contains("-rent", na=False)].copy()
    data["cat3_slug"] = data["cat3_slug"].fillna("unselect")
    data["deed_type"] = data["deed_type"].fillna("unselect")
    data["floor_material"] = data["floor_material"].fillna("unselect")

    data = data.dropna(subset=["building_size", "city_slug"])
    data = data[
        (data["building_size"] >= outliers["building_size_min"])
        & (data["building_size"] <= outliers["building_size_max"])
        & (data[cfg["target_column"]] >= outliers["price_value_min"])
        & (data[cfg["target_column"]] <= outliers["price_value_max"])
    ]
    return data.reset_index(drop=True)


def prepare_price_dataset(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full price pipeline: feature prep, split, floor bins, location imputation.

    Returns (train_data, val_data) with target column included.
    """
    cfg = config or load_config("price")
    prepared = prepare_price_features(df, cfg)

    split_cfg = cfg["split"]
    train_data, val_data = train_test_split(
        prepared,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
    )
    train_data = train_data.dropna(subset=["location_latitude", "location_longitude"])
    train_data, val_data = _apply_floor_features(train_data, val_data)
    train_data, val_data = fill_location_from_train(train_data, val_data)

    return train_data.reset_index(drop=True), val_data.reset_index(drop=True)
