"""Shared feature engineering helpers."""

from __future__ import annotations

import pandas as pd

ROOM_COUNT_MAP = {
    "بدون اتاق": "0_rooms",
    "یک": "1_room",
    "دو": "2_rooms",
    "سه": "3_rooms",
    "چهار": "4_rooms",
    "پنج یا بیشتر": "5plus_rooms",
}


def persian_to_english(value: str) -> str:
    """Convert Persian digits in a string to English digits."""
    persian_digits = "۰۱۲۳۴۵۶۷۸۹"
    english_digits = "0123456789"
    return value.translate(str.maketrans(persian_digits, english_digits))


def map_rooms_count(series: pd.Series) -> pd.Series:
    """Map Persian room labels to normalized codes; missing -> unselect."""
    return series.map(ROOM_COUNT_MAP).fillna("unselect")


def bin_luxury_score(value: float | int) -> str:
    """Bin a luxury/non-luxury count into categorical labels."""
    if pd.isna(value):
        return "unselect"
    if value == 0:
        return "none"
    if value <= 2:
        return "low"
    if value <= 4:
        return "medium"
    return "high"


def add_luxury_scores(
    df: pd.DataFrame,
    *,
    luxury_cols: list[str],
    non_luxury_cols: list[str],
    luxury_name: str = "luxury_score",
    non_luxury_name: str = "non_luxury_score",
    fix_balcony: bool = True,
) -> pd.DataFrame:
    """Aggregate amenity flags into binned luxury score columns."""
    out = df.copy()
    if fix_balcony and "has_balcony" in out.columns:
        out["has_balcony"] = (
            out["has_balcony"]
            .replace({"unselect": False, "true": True, "false": False})
        )
    out[luxury_name] = out[luxury_cols].sum(axis=1).astype("int64")
    out[non_luxury_name] = out[non_luxury_cols].sum(axis=1).astype("int64")
    out = out.drop(columns=luxury_cols + non_luxury_cols)
    out[luxury_name] = out[luxury_name].apply(bin_luxury_score)
    out[non_luxury_name] = out[non_luxury_name].apply(bin_luxury_score)
    return out


def add_building_age(
    df: pd.DataFrame,
    *,
    reference_year: int = 1404,
    pre_1370_replacement: str = "1370",
    bins: list[float],
    labels: list[str],
) -> pd.DataFrame:
    """Derive binned building_age from construction_year."""
    out = df.copy()
    cy = out["construction_year"].apply(lambda x: persian_to_english(x) if isinstance(x, str) else x)
    cy = cy.replace(pre_1370_replacement, pre_1370_replacement)
    cy = pd.to_numeric(cy, errors="coerce")
    out["building_age"] = pd.cut(reference_year - cy, bins=bins, labels=labels)
    out["building_age"] = out["building_age"].cat.add_categories("unselect").fillna("unselect")
    return out.drop(columns=["construction_year"])


def encode_boolean_features(df: pd.DataFrame, boolean_columns: list[str]) -> pd.DataFrame:
    """Encode boolean columns as 1/0 with -1 for missing."""
    out = df.copy()
    for col in boolean_columns:
        if col not in out.columns:
            continue
        out[col] = (
            out[col]
            .map({True: 1, False: 0})
            .fillna(-1)
            .astype(int)
        )
    return out


def group_rare_categories(series: pd.Series, min_freq: float = 0.01, other_label: str = "Other") -> pd.Series:
    """Collapse rare categorical levels into ``other_label``."""
    freq = series.value_counts(normalize=True)
    rare = freq[freq < min_freq].index
    return series.replace(rare, other_label)


def calculate_total_credit(rent: float, credit: float) -> float:
    """Compute total credit from rent and deposit (notebook formula)."""
    if rent < 0 or credit < 0:
        return 0.0
    credit_total = credit + (rent * 100) / 3
    return credit_total * 6


def fill_location_from_train(
    train: pd.DataFrame,
    other: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Impute latitude/longitude on train (group means) and apply train city means to other splits.
    """
    train_out = train.copy()
    other_out = other.copy()

    overall_lat = train_out["location_latitude"].mean()
    overall_long = train_out["location_longitude"].mean()

    train_out["location_latitude"] = train_out.groupby("city_slug")["location_latitude"].transform(
        lambda x: x.fillna(x.mean())
    )
    train_out["location_longitude"] = train_out.groupby("city_slug")["location_longitude"].transform(
        lambda x: x.fillna(x.mean())
    )
    train_out["location_latitude"] = train_out["location_latitude"].fillna(overall_lat)
    train_out["location_longitude"] = train_out["location_longitude"].fillna(overall_long)

    city_lat = train_out.groupby("city_slug")["location_latitude"].mean()
    city_long = train_out.groupby("city_slug")["location_longitude"].mean()

    other_out["location_latitude"] = other_out.apply(
        lambda row: city_lat.get(row["city_slug"], overall_lat)
        if pd.isna(row["location_latitude"])
        else row["location_latitude"],
        axis=1,
    )
    other_out["location_longitude"] = other_out.apply(
        lambda row: city_long.get(row["city_slug"], overall_long)
        if pd.isna(row["location_longitude"])
        else row["location_longitude"],
        axis=1,
    )
    return train_out, other_out
