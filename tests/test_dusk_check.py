"""sentry/dusk_check.py: the verdict rule and what it says to the chat.

Pure functions only; the reading itself needs the sky log, a plate solution and
astropy, which CI does not have.
"""
import importlib
import sys
import types

import pytest


@pytest.fixture
def dc(monkeypatch):
    cfg = types.SimpleNamespace(data=lambda: {"location": {"timezone": "America/New_York"}})
    import configs
    monkeypatch.setitem(sys.modules, "configs.config", cfg)
    monkeypatch.setattr(configs, "config", cfg, raising=False)
    sys.modules.pop("sentry.dusk_check", None)
    return importlib.import_module("sentry.dusk_check")


def _reading(rel, moonlit=False):
    return {"rel": rel, "transparency": rel * 0.75, "frames": 3, "sun_alt_deg": -12.1,
            "moonlit": moonlit, "moon_illum": 0.86 if moonlit else 0.1}


def test_verdict_bands(dc):
    s = dict(dc.DEFAULTS)
    assert dc.verdict(_reading(0.83), s) == "CLEAR"
    assert dc.verdict(_reading(0.65), s) == "CLEAR"
    assert dc.verdict(_reading(0.40), s) == "UNSURE"
    assert dc.verdict(_reading(0.12), s) == "CLOUDY"


def test_moonlight_never_reads_cloudy(dc):
    """The moon only lowers the reading, so a low one under a bright moon is
    not evidence of cloud (2026-09-29: 8% all night under an 86% moon)."""
    s = dict(dc.DEFAULTS)
    assert dc.verdict(_reading(0.10, moonlit=True), s) == "UNSURE"
    assert dc.verdict(_reading(0.80, moonlit=True), s) == "CLEAR"


def test_skipped_night_that_is_clear_is_a_disagreement(dc):
    plan = {"image": False, "dso": "ngc7380", "mode": "manual"}
    msg, disagree = dc.compose(_reading(0.83), "CLEAR", plan)
    assert disagree and "skipped" in msg and "clear" in msg


def test_planned_night_that_is_cloudy_is_a_disagreement(dc):
    plan = {"image": True, "dso": "ngc7380", "mode": "auto"}
    msg, disagree = dc.compose(_reading(0.10), "CLOUDY", plan)
    assert disagree and "ngc7380" in msg


def test_agreement_and_unsure_do_not_push(dc):
    go = {"image": True, "dso": "m33", "mode": "auto"}
    skip = {"image": False, "dso": "m33", "mode": "auto"}
    assert dc.compose(_reading(0.9), "CLEAR", go)[1] is False
    assert dc.compose(_reading(0.1), "CLOUDY", skip)[1] is False
    assert dc.compose(_reading(0.4), "UNSURE", go)[1] is False
    assert dc.compose(_reading(0.4), "UNSURE", skip)[1] is False
    assert dc.compose(_reading(0.9), "CLEAR", None)[1] is False
