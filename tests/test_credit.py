from divar.features.credit import prepare_credit_dataset, prepare_credit_features


def test_prepare_credit_features(sample_credit_df):
    out = prepare_credit_features(sample_credit_df)
    assert "total_credit" in out.columns
    assert "rent_value" not in out.columns
    assert len(out) > 0


def test_prepare_credit_dataset(sample_credit_df):
    train, val, test = prepare_credit_dataset(sample_credit_df)
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
