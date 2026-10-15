"""The `tonight` weather line judges the hours the planner counts (2026-10-14).

ngc6946's window opened at 7:53 pm; the 7 pm forecast hour was 45% cloud and
seven minutes of it flagged the whole window "not good", while every hour the
scheduler counted (20:00 on) was under 30% and the night was planned.
"""
import pytest

v = pytest.importorskip("iris_astronomy.astro_dso_visibility")

CLOUD = ["cloud cover over 40%"]


def _k(h, d=14):
    return (2026, 10, d, h)


def test_an_edge_hour_that_starts_before_the_window_is_not_judged():
    per_hour = {_k(19): CLOUD, _k(20): [], _k(21): [], _k(22): [], _k(23): []}
    whole = {_k(20), _k(21), _k(22), _k(23)}
    assert v.window_weather_verdict(per_hour, whole) == (True, [], [])


def test_a_cloudy_counted_hour_is_named():
    per_hour = {_k(19): CLOUD, _k(20): [], _k(21): CLOUD, _k(22): CLOUD + ["a chance of rain"]}
    whole = {_k(20), _k(21), _k(22)}
    ok, issues, bad = v.window_weather_verdict(per_hour, whole)
    assert not ok and bad == [_k(21), _k(22)]
    assert issues == ["a chance of rain", "cloud cover over 40%"]


def test_a_window_inside_one_hour_falls_back_to_the_hour_it_touches():
    assert v.window_weather_verdict({_k(20): CLOUD}, set()) == (False, CLOUD, [_k(20)])
    assert v.window_weather_verdict({}, set()) == (True, [], [])


def test_hour_label():
    assert [v._hour_label(h) for h in (0, 7, 12, 19, 23)] == ["12am", "7am", "12pm", "7pm", "11pm"]


# ------------------------------------------------------------ the ranking's weather (2026-10-15)

from datetime import datetime, timedelta, timezone   # noqa: E402

TZ = timezone(timedelta(hours=-4))


def test_pre_dawn_hours_use_tonights_forecast_not_this_mornings():
    """Run after midnight on 2026-10-15: this morning's 02-05 h were 25-29%
    cloud, tonight's (10-16) 49-82%. The ranking must judge tonight's."""
    forecast = {(2026, 10, 15, h): (25.0, 0.0, 10.0, 50.0) for h in range(2, 24)}
    forecast.update({(2026, 10, 16, h): (80.0 if h < 6 else 20.0, 0.0, 10.0, 50.0) for h in range(24)})
    forecast[(2026, 10, 15, 23)] = (40.0, 0.0, 10.0, 50.0)
    dark = [datetime(2026, 10, 15, 20, tzinfo=TZ) + timedelta(hours=i) for i in range(10)]   # 20:00 .. 05:00
    wx = v.weather_ok_by_dark_hour(dark, forecast)
    assert wx[20] is True and wx[23] is True            # 40% is not over 40
    assert wx[2] is False and wx[5] is False              # tonight's 80%, not this morning's 25%
    assert len(wx) == 10


def test_an_hour_missing_from_the_forecast_reads_bad():
    dark = [datetime(2026, 10, 15, 20, tzinfo=TZ), datetime(2026, 10, 15, 21, tzinfo=TZ)]
    wx = v.weather_ok_by_dark_hour(dark, {(2026, 10, 15, 20): (10.0, 0.0, 5.0, 50.0)})
    assert wx == {20: True} and wx.get(21, False) is False
