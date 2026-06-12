"""Pandera schemas for raw and processed datasets."""

from __future__ import annotations

from typing import Any

from pandera.pandas import Check, Column, DataFrameSchema

# Columns expected in the raw Divar export (subset used by pipelines).
RAW_REQUIRED_COLUMNS = [
    "cat2_slug",
    "cat3_slug",
    "city_slug",
    "neighborhood_slug",
    "building_size",
    "construction_year",
    "price_value",
    "rent_value",
    "credit_value",
    "location_latitude",
    "location_longitude",
]


def raw_divar_schema() -> DataFrameSchema:
    """Schema for raw CSV structure (not row-level business rules)."""
    columns = {name: Column(nullable=True) for name in RAW_REQUIRED_COLUMNS}
    for col in ("price_value", "rent_value", "credit_value", "building_size", "location_latitude", "location_longitude"):
        columns[col] = Column(nullable=True, coerce=True)
    return DataFrameSchema(columns, strict=False, coerce=True)


def processed_price_schema(cfg: dict[str, Any]) -> DataFrameSchema:
    """Schema for prepared price train/val frames."""
    outliers = cfg["outliers"]
    target = cfg["target_column"]
    return DataFrameSchema(
        {
            target: Column(
                nullable=False,
                coerce=True,
                checks=[
                    Check.greater_than_or_equal_to(outliers["price_value_min"]),
                    Check.less_than_or_equal_to(outliers["price_value_max"]),
                ],
            ),
            "building_size": Column(
                nullable=False,
                coerce=True,
                checks=[
                    Check.greater_than_or_equal_to(outliers["building_size_min"]),
                    Check.less_than_or_equal_to(outliers["building_size_max"]),
                ],
            ),
            "city_slug": Column(str),
            "cat2_slug": Column(str),
        },
        strict=False,
        coerce=True,
    )


def processed_credit_schema(cfg: dict[str, Any]) -> DataFrameSchema:
    """Schema for prepared credit train/val/test frames."""
    outliers = cfg["outliers"]
    target = cfg["target_column"]
    return DataFrameSchema(
        {
            target: Column(
                nullable=False,
                coerce=True,
                checks=[
                    Check.greater_than_or_equal_to(outliers["total_credit_min"]),
                    Check.less_than_or_equal_to(outliers["total_credit_max"]),
                ],
            ),
            "building_size": Column(
                nullable=False,
                coerce=True,
                checks=[
                    Check.greater_than_or_equal_to(outliers["building_size_min"]),
                    Check.less_than_or_equal_to(outliers["building_size_max"]),
                ],
            ),
            "city_slug": Column(str),
        },
        strict=False,
        coerce=True,
    )
