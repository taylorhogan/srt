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
