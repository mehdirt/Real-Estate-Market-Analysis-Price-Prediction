"""Shared pytest fixtures."""

import pandas as pd
import pytest


@pytest.fixture
def sample_sell_row() -> dict:
    """Minimal sell listing row for price pipeline tests."""
    return {
        "cat2_slug": "residential-sell",
        "cat3_slug": "apartment-sell",
        "city_slug": "tehran",
        "neighborhood_slug": "test-hood",
        "land_size": 100.0,
        "building_size": 80.0,
        "deed_type": "single",
        "has_business_deed": False,
        "floor": "2",
        "rooms_count": "دو",
        "total_floors_count": "5",
        "unit_per_floor": None,
        "has_balcony": False,
        "has_elevator": True,
        "has_warehouse": False,
        "has_parking": True,
        "construction_year": "1395",
        "is_rebuilt": False,
        "has_warm_water_provider": None,
        "has_heating_system": None,
        "has_cooling_system": None,
        "has_restroom": None,
        "has_security_guard": False,
        "has_barbecue": False,
        "building_direction": None,
        "has_pool": False,
        "has_jacuzzi": False,
        "has_sauna": False,
        "floor_material": "ceramic",
        "property_type": "apartment",
        "location_latitude": 35.7,
        "location_longitude": 51.4,
        "price_value": 2_000_000_000.0,
        "rent_value": None,
        "credit_value": None,
    }


@pytest.fixture
def sample_price_df(sample_sell_row) -> pd.DataFrame:
    """Small sell-only dataframe."""
    rows = []
    for i in range(30):
        row = sample_sell_row.copy()
        row["price_value"] = 1_000_000_000 + i * 50_000_000
        row["building_size"] = 60 + i
        row["location_latitude"] = 35.7 + i * 0.001
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def sample_credit_df(sample_sell_row) -> pd.DataFrame:
    """Small rent/credit dataframe."""
    rows = []
    for i in range(30):
        row = sample_sell_row.copy()
        row["price_value"] = None
        row["rent_value"] = 10_000_000 + i * 100_000
        row["credit_value"] = 100_000_000 + i * 1_000_000
        rows.append(row)
    return pd.DataFrame(rows)
