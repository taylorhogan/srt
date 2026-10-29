"""The ticker's dew margin (hardware_control/dew.py, 2026-10-29)."""
from hardware_control.dew import dew_margin_c


def test_margin_matches_the_nights_headers():
    # 10-28 05:26, Pegasus: 6.13 C, 97% -> N.I.N.A's DEWPOINT 5.70, margin 0.43
    assert abs(dew_margin_c(6.13, 97.0) - 0.43) < 0.05
    # 10-29 11:15: 14.1 C, 62% -> 7.2
    assert abs(dew_margin_c(14.1, 62.0) - 7.2) < 0.1
    assert abs(dew_margin_c(10.0, 100.0)) < 1e-9          # saturated: no margin


def test_bad_readings_give_nothing():
    for t, rh in ((None, 50), (10, None), (10, 0), (10, 120)):
        assert dew_margin_c(t, rh) is None
