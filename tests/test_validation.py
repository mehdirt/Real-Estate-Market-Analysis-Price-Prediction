import pytest

from divar.config import load_config
from divar.validation.run import DataValidationError, validate_processed_price, validate_raw_divar


def test_validate_raw_divar_passes(sample_price_df):
    # Add rent/credit columns for raw schema
    df = sample_price_df.copy()
    df["rent_value"] = None
    df["credit_value"] = None
    validate_raw_divar(df)


def test_validate_processed_price_passes(sample_price_df):
    from divar.features.price import prepare_price_dataset

    train, val = prepare_price_dataset(sample_price_df)
    validate_processed_price(train, val, load_config("price"))


def test_validate_processed_price_fails_on_bad_target(sample_price_df):
    from divar.features.price import prepare_price_dataset

    train, val = prepare_price_dataset(sample_price_df)
    train.loc[0, "price_value"] = 1.0  # below minimum
    with pytest.raises(DataValidationError):
        validate_processed_price(train, val, load_config("price"))
