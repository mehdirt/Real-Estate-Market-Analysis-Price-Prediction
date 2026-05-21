from divar.features.common import calculate_total_credit, persian_to_english


def test_persian_to_english():
    assert persian_to_english("۱۳۹۵") == "1395"


def test_calculate_total_credit():
    assert calculate_total_credit(3_000_000, 100_000_000) > 0
    assert calculate_total_credit(-1, 100) == 0
