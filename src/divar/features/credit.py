"""Rent/credit (total_credit) prediction dataset preparation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from divar.config import load_config
from divar.features.common import (
    bin_luxury_score,
    calculate_total_credit,
    group_rare_categories,
    map_rooms_count,
    persian_to_english,
)

LUXURY_COLS = ["has_pool", "has_sauna", "has_barbecue", "has_jacuzzi", "has_security_guard"]
NON_LUXURY_COLS = ["has_balcony", "has_elevator", "has_warehouse", "has_parking"]

DROP_ALWAYS = [
    "title",
    "description",
    "created_at_month",
    "user_type",
    "location_radius",
    "rent_mode",
    "rent_to_single",
    "rent_type",
    "price_mode",
    "credit_mode",
    "rent_credit_transform",
    "transformable_price",
    "transformable_credit",
    "transformed_credit",
    "transformable_rent",
    "transformed_rent",
    "has_water",
    "has_electricity",
    "has_gas",
    "regular_person_capacity",
    "extra_person_capacity",
    "cost_per_extra_person",
    "rent_price_on_regular_days",
    "rent_price_on_special_days",
    "rent_price_at_weekends",
    "property_type",
    "deed_type",
    "has_business_deed",
    "land_size",
]


def _clean_floor_value(x: Any) -> float:
    if pd.isna(x):
        return np.nan
    if x == "unselect":
        return np.nan
    s = str(x).strip()
    if s.endswith("+"):
        return float(s.replace("+", ""))
    return float(s)


def _clean_unit_per_floor(x: Any) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "unselect":
        return np.nan
    if s == "more_than_8":
        return 9.0
    try:
        return float(s)
    except ValueError:
        return np.nan


def prepare_credit_features(df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Clean and engineer features for total_credit modeling (before splits)."""
    cfg = config or load_config("credit")
    outliers = cfg["outliers"]
    rare_cfg = cfg["rare_category"]

    data = df.loc[df["price_value"].isna()].copy()
    data = data.drop(columns=["price_value"], errors="ignore")
    data = data.dropna(subset=["rent_value", "credit_value"])
    data["total_credit"] = data.apply(
        lambda row: calculate_total_credit(row["rent_value"], row["credit_value"]),
        axis=1,
    )
    data = data.drop(columns=["rent_value", "credit_value"])

    drop_cols = [c for c in DROP_ALWAYS if c in data.columns]
    data = data.drop(columns=drop_cols)

    data["floor"] = data["floor"].apply(_clean_floor_value)
    data["total_floors_count"] = data["total_floors_count"].apply(_clean_floor_value)

    floor_bins = [-1, 0, 3, 7, 12, 20, 30]
    floor_labels = ["underground", "low_floors", "mid_low_floors", "mid_floors", "high_floors", "very_high"]
    total_bins = [0, 3, 7, 12, 20, 30]
    total_labels = ["low_floors", "mid_low_floors", "mid_floors", "high_floors", "very_high"]

    data["floor"] = pd.cut(data["floor"], bins=floor_bins, labels=floor_labels)
    data["total_floors_count"] = pd.cut(data["total_floors_count"], bins=total_bins, labels=total_labels)
    data["floor"] = data["floor"].cat.add_categories("unselect").fillna("unselect")
    data["total_floors_count"] = (
        data["total_floors_count"].cat.add_categories("unselect").fillna("unselect")
    )

    data["rooms_count"] = map_rooms_count(data["rooms_count"])
    data["unit_per_floor"] = data["unit_per_floor"].apply(_clean_unit_per_floor)
    unit_bins = [0, 1, 2, 3, 4, 6, 8, np.inf]
    unit_labels = ["1", "2", "3", "4", "5-6", "7-8", "9+"]
    data["unit_per_floor"] = pd.cut(data["unit_per_floor"], bins=unit_bins, labels=unit_labels, include_lowest=True)
    data["unit_per_floor"] = data["unit_per_floor"].cat.add_categories("unselect").fillna("unselect")

    cy = data["construction_year"].apply(lambda x: persian_to_english(x) if isinstance(x, str) else x)
    cy = cy.replace(cfg["building_age"]["pre_1370_replacement"], cfg["building_age"]["pre_1370_replacement"])
    cy = pd.to_numeric(cy, errors="coerce")
    data["building_age"] = pd.cut(
        cfg["building_age"]["reference_year"] - cy,
        bins=[0, *cfg["building_age"]["bins"][1:], np.inf],
        labels=cfg["building_age"]["labels"],
    )
    data = data.drop(columns=["construction_year"])
    data["building_age"] = data["building_age"].cat.add_categories("unselect").fillna("unselect")

    data["cat3_slug"] = group_rare_categories(
        data["cat3_slug"], min_freq=rare_cfg["min_freq"], other_label=rare_cfg["other_label"]
    )
    data["cat2_slug"] = group_rare_categories(
        data["cat2_slug"], min_freq=rare_cfg["min_freq"], other_label=rare_cfg["other_label"]
    )

    if "has_balcony" in data.columns:
        data["has_balcony"] = data["has_balcony"].replace({"true": True, "false": False, "unselect": False})
    data["luxury_items"] = data[LUXURY_COLS].sum(axis=1)
    data["non_luxury_items"] = data[NON_LUXURY_COLS].sum(axis=1)
    data = data.drop(columns=[c for c in LUXURY_COLS + NON_LUXURY_COLS if c in data.columns])
    data["luxury_items"] = data["luxury_items"].apply(bin_luxury_score)
    data["non_luxury_items"] = data["non_luxury_items"].apply(bin_luxury_score)

    data = data[
        (data["total_credit"] >= outliers["total_credit_min"])
        & (data["total_credit"] <= outliers["total_credit_max"])
    ]
    data = data[
        (data["building_size"] >= outliers["building_size_min"])
        & (data["building_size"] <= outliers["building_size_max"])
    ]
    return data.reset_index(drop=True)


def prepare_credit_dataset(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full credit pipeline: features, train/val/test split, imputation.

    Returns (train_data, val_data, test_data) with target column included.
    """
    cfg = config or load_config("credit")
    target = cfg["target_column"]
    prepared = prepare_credit_features(df, cfg)

    X = prepared.drop(columns=[target])
    y = prepared[target]

    split = cfg["split"]
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=split["test_size"], random_state=split["random_state"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=split["val_size"],
        random_state=split["random_state"],
    )

    train_data = pd.concat([X_train, y_train], axis=1)
    val_data = pd.concat([X_val, y_val], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    for frame in (train_data, val_data, test_data):
        frame.dropna(subset=["building_size"], inplace=True)

    from divar.features.common import fill_location_from_train

    train_data, val_data = fill_location_from_train(train_data, val_data)
    _, test_data = fill_location_from_train(train_data, test_data)

    fill_cols = [
        "has_warm_water_provider",
        "has_heating_system",
        "has_cooling_system",
        "building_direction",
        "floor_material",
        "has_restroom",
    ]
    for col in fill_cols:
        if col in train_data.columns:
            for frame in (train_data, val_data, test_data):
                frame[col] = frame[col].fillna("unselect")

    train_data = train_data.dropna(subset=["city_slug"])
    val_data = val_data.dropna(subset=["city_slug"])
    test_data = test_data.dropna(subset=["city_slug"])

    return (
        train_data.reset_index(drop=True),
        val_data.reset_index(drop=True),
        test_data.reset_index(drop=True),
    )
