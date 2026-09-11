"""The stop! sequence's decisions and its ending, without hardware.

On 2026-09-10 a stop! read the roof open and the scope parked (vision), but
PWI4 was not running: get_is_parked() returned False, the sequence tried to
park through a dead PWI4, park_scope() raised, and the job died before the
roof close. The roof was closed by hand and the dehumidifier and Pegasus ports
were never touched. These tests pin the three fixes: the park reading is
three-way and PWI4 unreachable is not a denial; a failure inside the sequence
is reported rather than swallowed; and a confirmed roof close is followed by
dehumidifier-on + Pegasus-off, verified by read-back.

Needs the private config to import the command module; skipped on a bare CI
runner like the other hardware-adjacent tests.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

suc = pytest.importorskip("cmd_processing.super_user_commands")
pegasus = pytest.importorskip("hardware_control.pegasus")
pwi4_utils = pytest.importorskip("hardware_control.pwi4_utils")


# ------------------------------------------------------------- park rule

def test_park_rule_is_vision_confirmed_plus_pwi4_not_denied():
    assert suc._scope_confirmed_parked("parked", True) is True
    assert suc._scope_confirmed_parked("unknown", True) is True      # PWI4 down: tag decides
    assert suc._scope_confirmed_parked("not_parked", True) is False  # PWI4 denial wins
    assert suc._scope_confirmed_parked("parked", False) is False     # vision must confirm
    assert suc._scope_confirmed_parked("unknown", False) is False


def test_mount_park_state_is_three_way(monkeypatch):
    class Dead:
        def status(self):
            raise ConnectionRefusedError("no PWI4")

    class Alive:
        def status(self):
            return object()

    monkeypatch.setattr(pwi4_utils, "PWI4", Dead)
    assert pwi4_utils.mount_park_state() == "unknown"
    monkeypatch.setattr(pwi4_utils, "PWI4", Alive)
    monkeypatch.setattr(pwi4_utils, "get_is_parked", lambda: True)
    assert pwi4_utils.mount_park_state() == "parked"
    monkeypatch.setattr(pwi4_utils, "get_is_parked", lambda: False)
    assert pwi4_utils.mount_park_state() == "not_parked"


# ------------------------------------------------------------- pegasus

def test_pegasus_power_off_is_verified_by_read_back(monkeypatch):
    sent = []
    monkeypatch.setattr(pegasus, "_send_raw_command", lambda cmd: sent.append(cmd) or cmd)
    monkeypatch.setattr(pegasus, "read_power_ports",
                        lambda: {1: 0, 2: 0, 3: 0, 4: 0, 5: 100, 6: 0})
    ok, levels = pegasus.power_off_imaging_train()
    assert sent == ["P1:0", "P2:0", "P3:0"]
    assert ok is True and levels == {1: 0, 2: 0, 3: 0}


def test_pegasus_power_off_not_verified_when_a_port_stays_on(monkeypatch):
    monkeypatch.setattr(pegasus, "_send_raw_command", lambda cmd: cmd)
    monkeypatch.setattr(pegasus, "read_power_ports", lambda: {1: 0, 2: 100, 3: 0})
    ok, levels = pegasus.power_off_imaging_train()
    assert ok is False and levels[2] == 100


def test_pegasus_power_off_not_verified_when_box_unreadable(monkeypatch):
    # An echoed command is not evidence the port is off.
    monkeypatch.setattr(pegasus, "_send_raw_command", lambda cmd: cmd)
    monkeypatch.setattr(pegasus, "read_power_ports", lambda: None)
    assert pegasus.power_off_imaging_train() == (False, None)


# ------------------------------------------------------------- status block

def test_status_hardware_block_lists_pwi4_and_the_three_ports():
    ports = {1: ("camera", 0), 2: ("gemini", 100), 3: ("fan", 40), 4: ("Output4", 0)}
    out = suc.format_hardware_status("reachable, mount connected", ports)
    assert "PWI4      : reachable, mount connected" in out
    assert "Pegasus 1 : camera OFF" in out
    assert "Pegasus 2 : gemini ON" in out
    assert "Pegasus 3 : fan ON 40%" in out
    assert "Output4" not in out


def test_status_hardware_block_when_pegasus_unreachable():
    out = suc.format_hardware_status("unreachable", None)
    assert "PWI4      : unreachable" in out and "Pegasus   : unreachable" in out


def test_status_hardware_lines_never_raise(monkeypatch):
    def boom():
        raise OSError("unity down")
    monkeypatch.setattr(suc.pegasus, "read_power_port_details", boom)
    monkeypatch.setattr(suc, "pwi4_reach_state", lambda: "unreachable")
    assert "Pegasus   : unreachable" in suc.hardware_status_lines()


# ------------------------------------------------------------- ending

def test_power_down_after_close_does_both_and_pegasus_last(monkeypatch):
    order = []
    monkeypatch.setattr(suc.utl_shelly, "set_dehumidifier", lambda on: order.append(("dehum", on)))
    monkeypatch.setattr(suc.pegasus, "power_off_imaging_train",
                        lambda: order.append("pegasus") or (True, {1: 0, 2: 0, 3: 0}))
    posts = []
    monkeypatch.setattr(suc.social_server, "post_social_message", posts.append)
    suc._power_down_after_close()
    assert order == [("dehum", True), "pegasus"]
    assert any("Dehumidifier on" in p for p in posts)
    assert any("verified" in p for p in posts)


def test_power_down_reports_an_unverified_pegasus(monkeypatch):
    monkeypatch.setattr(suc.utl_shelly, "set_dehumidifier", lambda on: None)
    monkeypatch.setattr(suc.pegasus, "power_off_imaging_train", lambda: (False, {1: 0, 2: 100, 3: 0}))
    posts = []
    monkeypatch.setattr(suc.social_server, "post_social_message", posts.append)
    assert suc._power_off_pegasus_train() is False
    assert any("NOT verified" in p and "port 2=100" in p for p in posts)


def test_dehumidifier_failure_does_not_stop_the_pegasus_step(monkeypatch):
    def boom(on):
        raise OSError("relay unreachable")
    called = []
    monkeypatch.setattr(suc.utl_shelly, "set_dehumidifier", boom)
    monkeypatch.setattr(suc.pegasus, "power_off_imaging_train",
                        lambda: called.append(1) or (True, {1: 0, 2: 0, 3: 0}))
    monkeypatch.setattr(suc.social_server, "post_social_message", lambda m: None)
    suc._power_down_after_close()
    assert called == [1]


def test_stop_sequence_reports_a_crash_instead_of_dying_silently(monkeypatch):
    def body():
        raise ConnectionRefusedError("PWI4 refused")
    monkeypatch.setattr(suc, "_emergency_stop_body", body)
    posts, pushes = [], []
    monkeypatch.setattr(suc.social_server, "post_social_message", posts.append)
    monkeypatch.setattr(suc.pushover, "push_message",
                        lambda msg, *a, **kw: pushes.append((msg, kw.get("priority"))))
    suc._emergency_stop_sequence()          # must not raise
    assert any("EMERGENCY STOP FAILED" in p and "PWI4 refused" in p for p in posts)
    assert pushes and pushes[0][1] == 2
