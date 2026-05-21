from divar.config import load_config
from divar.features.price import prepare_price_dataset, prepare_price_features


def test_prepare_price_features(sample_price_df):
    cfg = load_config("price")
    out = prepare_price_features(sample_price_df, cfg)
    assert "price_value" in out.columns
    assert "building_age" in out.columns
    assert "luxury_score" in out.columns
    assert len(out) > 0


def test_prepare_price_dataset(sample_price_df):
    train, val = prepare_price_dataset(sample_price_df)
    assert len(train) > 0
    assert len(val) > 0
    assert "price_value" in train.columns
