"""Phase 2b: the roof moves through iris/hardware/roof.py and nowhere else (2026-10-10).

A text scan, stdlib only, so it runs in CI's pytest-only environment. It
fails when any other file fires the roof relay or switches the roof motor
plug ON. Switching the plug OFF is allowed everywhere: every shutdown path
does it, and off is the safe direction.
"""
import os
import re

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SKIP_DIRS = {".venv", "venv", ".git", "__pycache__", "not_used_archive", "local", "tests", "node_modules"}
ALLOWED = {
    os.path.join("iris", "hardware", "roof.py"),
    os.path.join("hardware_control", "utl_shelly.py"),     # where fire_roof_relay is defined
}
FIRE = re.compile(r"fire_roof_relay\s*\(")
MOTOR_ON = re.compile(r"""["']Roof motor["']\s*:\s*["']on["']""")


def _sources():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            if f.endswith(".py"):
                path = os.path.join(dirpath, f)
                yield os.path.relpath(path, ROOT), path


def _offenders(pattern):
    out = []
    for rel, path in _sources():
        if rel in ALLOWED:
            continue
        with open(path, encoding="utf-8", errors="replace") as fh:
            for n, line in enumerate(fh, 1):
                code = line.split("#", 1)[0]
                if pattern.search(code):
                    out.append("%s:%d: %s" % (rel, n, line.strip()))
    return out


def test_only_the_roof_module_fires_the_relay():
    assert _offenders(FIRE) == []


def test_only_the_roof_module_powers_the_motor():
    assert _offenders(MOTOR_ON) == []


def test_the_scan_would_catch_a_bypass(tmp_path):
    assert FIRE.search("    utl_shelly.fire_roof_relay()")
    assert MOTOR_ON.search("""inst = {"Roof motor": 'on'}""")
    assert not MOTOR_ON.search("""inst = {"Roof motor": 'off'}""")


def test_switch_motor_switches_the_plug_and_others_in_one_call(monkeypatch):
    pytest.importorskip("hardware_control.kasa_utils")
    from hardware_control import kasa_utils as ku
    from iris.hardware import roof
    seen = []

    async def fake_do(dev_map, inst):
        seen.append(dict(inst))
        return {k: True for k in inst}
    monkeypatch.setattr(ku, "kasa_do", fake_do)
    res = roof.switch_motor({}, True, also={"Telescope mount": "off", "Iris inside light": "on"})
    assert seen == [{"Telescope mount": "off", "Iris inside light": "on", "Roof motor": "on"}]
    assert res == {"Telescope mount": True, "Iris inside light": True, "Roof motor": True}
    roof.switch_motor({}, False)
    assert seen[-1] == {"Roof motor": "off"}
    with pytest.raises(ValueError):
        roof.switch_motor({}, True, also={"Roof motor": "off"})


@pytest.fixture
def toggle_env(monkeypatch):
    """toggle_roof with every hardware touch faked and recorded."""
    suc = pytest.importorskip("cmd_processing.super_user_commands")
    from hardware_control import kasa_utils as ku, utl_shelly
    from sentry import roof_evidence, kasa_audio
    calls = []

    async def fake_do(dev_map, inst):
        calls.append(("kasa", dict(inst)))
        return {k: True for k in inst}
    monkeypatch.setattr(ku, "kasa_do", fake_do)
    monkeypatch.setattr(roof_evidence, "record_motion_possible", lambda why: calls.append("motion_possible"))
    monkeypatch.setattr(suc, "_wait_for_roof_relay", lambda: calls.append("relay_boot") or True)
    monkeypatch.setattr(suc.rcs, "start_background_capture", lambda **k: None)
    monkeypatch.setattr(kasa_audio, "start_capture_async", lambda **k: None)
    monkeypatch.setattr(suc, "_wait_for_roof_travel", lambda *a, **k: calls.append("travel"))
    return suc, utl_shelly, calls


def test_toggle_roof_goes_through_the_roof_module(toggle_env, monkeypatch):
    suc, utl_shelly, calls = toggle_env
    monkeypatch.setattr(utl_shelly, "_get", lambda url, timeout=10: calls.append("fire") or "ok")
    suc.toggle_roof({}, capture_direction="open")
    assert calls == ["motion_possible", ("kasa", {"Roof motor": "on"}), "relay_boot",
                     "motion_possible", "fire", "travel", ("kasa", {"Roof motor": "off"})]


def test_toggle_roof_relay_failure_cuts_the_motor_and_raises(toggle_env, monkeypatch):
    suc, utl_shelly, calls = toggle_env
    monkeypatch.setattr(utl_shelly, "_get", lambda url, timeout=10: calls.append("fire") or None)
    with pytest.raises(suc.RoofFireError):
        suc.toggle_roof({}, capture_direction="close")
    assert calls[-2:] == ["fire", ("kasa", {"Roof motor": "off"})] and "travel" not in calls
