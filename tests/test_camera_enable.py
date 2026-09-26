"""A camera switched off in the Kasa app is switched back on by the reader.

kasa_ptz.ensure_enabled is the one place that asks the cloud; the readers
(kasa_state._grab, north_roof._grab) call it only after every plain attempt
gave nothing. Cloud calls are faked here; nothing is reached.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

kp = pytest.importorskip("scripts.kasa_ptz")


def _fake(monkeypatch, on_now, set_result=True, device_ok=True):
    calls = []
    monkeypatch.setattr(kp.time, "sleep", lambda s: calls.append(("sleep", s)))
    if device_ok:
        monkeypatch.setattr(kp, "_device", lambda name: {"deviceId": "d", "alias": name})
    else:
        def boom(name):
            raise kp.PTZError("%r is offline" % name)
        monkeypatch.setattr(kp, "_device", boom)
    monkeypatch.setattr(kp, "enabled", lambda dev: on_now)
    monkeypatch.setattr(kp, "set_enabled",
                        lambda dev, on: calls.append(("set", on)) or set_result)
    kp.last_switched_on.clear()
    return calls


def test_a_camera_already_on_is_left_alone(monkeypatch):
    calls = _fake(monkeypatch, on_now=True)
    assert kp.ensure_enabled("Iris cam") == "on"
    assert calls == [] and kp.last_switched_on == {}


def test_a_camera_off_in_the_app_is_switched_on_and_given_time_to_stream(monkeypatch):
    calls = _fake(monkeypatch, on_now=False)
    assert kp.ensure_enabled("Iris cam", settle_s=7) == "switched_on"
    assert calls == [("set", True), ("sleep", 7)]
    assert "Iris cam" in kp.last_switched_on


def test_a_switch_that_does_not_take_is_reported_off_not_on(monkeypatch):
    calls = _fake(monkeypatch, on_now=False, set_result=False)
    assert kp.ensure_enabled("Iris cam") == "off"
    assert ("sleep", kp.ENABLE_SETTLE_S) not in calls


def test_a_cloud_failure_is_unknown_and_never_raises(monkeypatch):
    _fake(monkeypatch, on_now=False, device_ok=False)
    assert kp.ensure_enabled("Iris cam") == "unknown"
