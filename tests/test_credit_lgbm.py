from divar.config import load_config
from divar.features.credit import prepare_credit_dataset
from divar.models.train_credit import train_credit_models


def test_credit_trains_lightgbm(sample_credit_df):
    train, val, test = prepare_credit_dataset(sample_credit_df)
    artifacts = train_credit_models(train, val, test, load_config("credit"))
    assert "lightgbm" in artifacts
    assert "random_forest" in artifacts
    assert "lightgbm" in artifacts["metrics"]["val"]
    assert artifacts["metrics"]["val"]["lightgbm"]["r2"] is not None
