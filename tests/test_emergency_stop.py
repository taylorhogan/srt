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


# ------------------------------------------------------- image!! refusal

def test_image_cmd_refusal_is_loud_and_returns_false(monkeypatch):
    """2026-09-13: the first auto night was refused because N.I.N.A was still
    open, and the only trace was a Pushover message. A refusal must log,
    post to the feed, and tell the caller, so the scheduler never reads an
    unclaimed imaging state as a completed run."""
    posted, pushed = [], []
    monkeypatch.setattr(suc, "is_imaging", lambda: False)
    monkeypatch.setattr(suc, "is_nina_running", lambda: True)
    monkeypatch.setattr(suc.social_server, "post_social_message", lambda m: posted.append(m))
    monkeypatch.setattr(suc.pushover, "push_message", lambda m: pushed.append(m))
    monkeypatch.setattr(suc.jobs, "spawn", lambda fn: pytest.fail("must not launch"))
    assert suc.image_cmd(["", "image!!", "1"], "iris") is False
    assert posted and "REFUSED" in posted[0] and "N.I.N.A" in posted[0]
    assert pushed and "REFUSED" in pushed[0]


# ------------------------------------------------------- blind park (2026-09-17)

from datetime import datetime, timedelta, timezone

T0 = datetime(2026, 9, 17, 0, 33, 37, tzinfo=timezone.utc)
CLEAR = dict(mount_state="not_parked",
             mount_motion={"connected": True, "moving": True, "alt": 72.4},
             mount_powered=True, camera_ok=True,
             frame_roof_verdicts=["unknown"] * 3,
             open_confirmed=T0 - timedelta(hours=5, minutes=36),     # 18:57
             motion_possible=T0 - timedelta(hours=5, minutes=40),    # the open's own fire
             roof_plug=0, roof_lock_held=False, now=T0)


def test_blind_park_allowed_on_the_2026_09_17_evidence():
    assert suc.blind_park_refusal(**CLEAR) is None


def test_blind_park_allowed_with_no_fire_on_record():
    assert suc.blind_park_refusal(**dict(CLEAR, motion_possible=None)) is None


@pytest.mark.parametrize("change, word", [
    (dict(mount_state="unknown"), "PWI4"),
    (dict(mount_state="parked"), "PWI4"),
    (dict(mount_motion={"connected": False, "moving": True, "alt": 72.4}), "not connected"),
    (dict(mount_motion={"connected": True, "moving": False, "alt": 72.4}), "not be homed"),
    (dict(mount_motion={"connected": True, "moving": True, "alt": -48.7}), "not plausible"),
    (dict(mount_motion={}), "not connected"),
    (dict(mount_powered=None), "mount plug"),
    (dict(mount_powered=False), "mount plug"),
    (dict(camera_ok=False), "no frames"),
    (dict(frame_roof_verdicts=["unknown", "shut", "unknown"]), "SHUT"),
    (dict(open_confirmed=None), "no record"),
    (dict(open_confirmed=T0 - timedelta(hours=17)), "h ago"),
    (dict(motion_possible=T0 - timedelta(minutes=10)), "after it was last confirmed open"),
    (dict(roof_plug=1), "plug not confirmed OFF"),
    (dict(roof_plug=None), "plug not confirmed OFF"),
    (dict(roof_lock_held=True), "in progress"),
])
def test_blind_park_refuses(change, word):
    why = suc.blind_park_refusal(**dict(CLEAR, **change))
    assert why and word in why, why


@pytest.fixture
def stop_env(tmp_path, monkeypatch):
    """_emergency_stop_body with every hardware touch recorded, nothing real."""
    from fits_processing import frame_watcher
    from sentry import kasa_state
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(suc.utils, "set_install_dir", lambda: None)
    monkeypatch.setattr(suc, "ABORT_FLAG_PATH", str(tmp_path / "abort.flag"))
    monkeypatch.setattr(suc, "_kill_nina", lambda: None)
    monkeypatch.setattr(frame_watcher, "stop", lambda: None)
    monkeypatch.setattr(suc.time, "sleep", lambda s: None)
    monkeypatch.setattr(suc, "turn_inside_light_on", lambda d: None)
    monkeypatch.setattr(suc, "set_imaging_state", lambda s: None)
    monkeypatch.setattr(suc, "_power_down_after_close", lambda: calls.append("power_down"))

    async def fake_map(*a, **k):
        return {"Telescope mount": "m", "Roof motor": "r", "Iris inside light": "l"}

    async def fake_do(dev_map, inst):
        calls.append(("kasa", dict(inst)))
        return {k: True for k in inst}

    calls, posts, pushes = [], [], []
    monkeypatch.setattr(suc.ku, "make_discovery_map", fake_map)
    monkeypatch.setattr(suc.ku, "kasa_do", fake_do)
    monkeypatch.setattr(suc.ku, "legacy_relay", lambda host: {"m": 1, "r": 0}[host])
    monkeypatch.setattr(suc.social_server, "post_social_message", posts.append)
    monkeypatch.setattr(suc.pushover, "push_message", lambda m, *a, **k: pushes.append(m))
    monkeypatch.setattr(suc, "_stop_mount_motion", lambda: calls.append("stop") or "stopped")
    monkeypatch.setattr(suc.pwi4_utils, "park_scope", lambda: pytest.fail("blind park must not connect"))
    monkeypatch.setattr(suc, "_park_connected_mount", lambda: calls.append("park") or True)
    monkeypatch.setattr(suc, "_read_mount_motion",
                        lambda: calls.append("read") or {"connected": True, "moving": True, "alt": 72.4})
    monkeypatch.setattr(suc.end, "do_main", lambda: calls.append("close") or True)
    monkeypatch.setattr(kasa_state, "last_detail",
                        {"camera": True, "per_frame": [{"roof": "unknown"}] * 3})
    now = datetime.now().astimezone()
    monkeypatch.setattr(suc.roof_evidence, "read", lambda: {
        "open_confirmed": now - timedelta(hours=5), "motion_possible": now - timedelta(hours=5, minutes=2)})

    def script(vision, mount):
        v, m = iter(vision), iter(mount)
        monkeypatch.setattr(suc, "get_status_with_lights", lambda: next(v))
        monkeypatch.setattr(suc.pwi4_utils, "mount_park_state", lambda: next(m))

    return calls, posts, pushes, script


BLIND = (False, False, False, None)


def test_stop_blind_parks_then_closes_on_a_fresh_read(stop_env):
    calls, posts, pushes, script = stop_env
    script([BLIND, (True, False, True, None)], ["not_parked", "parked"])
    suc._emergency_stop_body()
    assert calls.index("read") < calls.index("stop") < calls.index("park") < calls.index("close")
    assert "power_down" in calls
    assert any("parking scope" in p for p in posts)


def test_stop_blind_refusal_stops_tracking_and_pushes_once(stop_env, monkeypatch):
    calls, posts, pushes, script = stop_env
    monkeypatch.setattr(suc.ku, "legacy_relay", lambda host: {"m": 1, "r": 1}[host])   # roof plug ON
    script([BLIND], ["not_parked"])
    suc._emergency_stop_body()
    assert "stop" in calls and "park" not in calls and "close" not in calls
    assert len(pushes) == 1 and "roof motor plug not confirmed OFF" in pushes[0]
