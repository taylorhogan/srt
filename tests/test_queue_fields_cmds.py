"""E3 of docs/AUTOMATION_PLAN.md (2026-10-30): the queue's hand-set fields as chat commands.

control/queue_fields.parse is pure (CI); the commands write through
instructions in the venv (the module needs astropy and the chat server).
"""
import json

import pytest

from control import queue_fields as Q


def test_parse_hold():
    assert Q.parse("hold", ["ngc7790"]) == {"dso": "ngc7790", "field": "status", "value": "held", "error": None}
    assert Q.parse("hold", ["ngc", "7790", "off"])["value"] == "waiting"
    assert Q.parse("hold", ["ngc", "7790", "off"])["dso"] == "ngc 7790"
    assert Q.parse("hold", [])["error"].startswith("Usage: hold")


def test_parse_moon_maxhours_type():
    assert Q.parse("moon", ["ic410", "any"]) == {"dso": "ic410", "field": "moon", "value": "any", "error": None}
    assert Q.parse("moon", ["m", "31", "dark"])["dso"] == "m 31"
    assert Q.parse("moon", ["ic410", "clear"])["value"] is None
    assert "say what to set" in Q.parse("moon", ["ic410"])["error"]
    assert "say what to set" in Q.parse("moon", ["ic410", "sometimes"])["error"]    # an unknown word is part of the name
    p = Q.parse("maxhours", ["hubblev1", "1.5"])
    assert p["field"] == "max_hours" and p["value"] == 1.5
    assert Q.parse("maxhours", ["hubblev1", "clear"])["value"] is None
    assert "between" in Q.parse("maxhours", ["hubblev1", "20"])["error"]
    assert "between" in Q.parse("maxhours", ["hubblev1", "0"])["error"]
    assert Q.parse("type", ["sh2-92", "nebula"]) == {"dso": "sh2-92", "field": "obj_type", "value": "nebula", "error": None}
    assert Q.parse("type", ["sh2-92", "galaxy"])["value"] == "galaxy"
    assert "say what to set" in Q.parse("type", ["sh2-92", "cluster"])["error"]
    assert Q.parse("bogus", ["x"])["error"].startswith("unknown command")


def test_describe_says_what_each_setting_means():
    assert "until `hold ngc7790 off`" in Q.describe("hold", "ngc7790", "held")
    assert "back to waiting" in Q.describe("hold", "ngc7790", "waiting")
    assert "never held" in Q.describe("moon", "ic410", "any") and "whenever the moon is up" in Q.describe("moon", "x", "dark")
    assert "default: avoid" in Q.describe("moon", "ic410", None)
    assert "at most 1.5 h" in Q.describe("maxhours", "h", 1.5) and "no cap" in Q.describe("maxhours", "h", None)
    assert "narrowband" in Q.describe("type", "s", "nebula") and "L+RGB" in Q.describe("type", "s", "galaxy")


# ------------------------------------------------------------ the commands (venv)

@pytest.fixture
def world(tmp_path, monkeypatch):
    suc = pytest.importorskip("cmd_processing.super_user_commands")
    instr = pytest.importorskip("control.instructions")
    p = tmp_path / "q.json"
    p.write_text(json.dumps([{"dso": "ngc7790", "status": "waiting"},
                             {"dso": "ic410", "status": "waiting", "moon": "any"},
                             {"dso": "m1", "status": "completed"}]))
    monkeypatch.setattr(instr, "_INSTRUCTIONS_PATH", str(p))
    posted, notes = [], []
    monkeypatch.setattr(suc.social_server, "post_social_message", lambda m: posted.append(m))
    monkeypatch.setattr(suc.instructions, "create_instructions_table", lambda: None)
    import iris.client as client
    monkeypatch.setattr(client, "post_event", lambda ev, src, data=None, **k: notes.append((ev, data)) or {})
    return suc, p, posted, notes


def _rows(p):
    return {r["dso"]: r for r in json.loads(p.read_text())}


def test_hold_and_release(world):
    suc, p, posted, notes = world
    suc.hold_cmd(["", "hold", "ngc7790"], "taylor")
    assert _rows(p)["ngc7790"]["status"] == "held" and "held" in posted[-1]
    assert notes[-1] == ("QUEUE_FIELD", {"dso": "ngc7790", "field": "status", "value": "held", "by": "taylor"})
    suc.hold_cmd(["", "hold", "ngc7790"], "taylor")
    assert "is held, not waiting" in posted[-1]
    suc.hold_cmd(["", "hold", "ngc7790", "off"], "taylor")
    assert _rows(p)["ngc7790"]["status"] == "waiting" and "back to waiting" in posted[-1]
    suc.hold_cmd(["", "hold", "m1"], "taylor")
    assert "not in the queue (completed)" in posted[-1]


def test_moon_maxhours_type_write_the_fields(world):
    suc, p, posted, notes = world
    suc.moon_cmd(["", "moon", "ngc7790", "dark"], "taylor")
    assert _rows(p)["ngc7790"]["moon"] == "dark"
    suc.moon_cmd(["", "moon", "ic410", "clear"], "taylor")
    assert "moon" not in _rows(p)["ic410"] and "default: avoid" in posted[-1]
    suc.maxhours_cmd(["", "maxhours", "ngc7790", "1"], "taylor")
    assert _rows(p)["ngc7790"]["max_hours"] == 1.0 and "at most 1 h" in posted[-1]
    suc.maxhours_cmd(["", "maxhours", "ngc7790", "99"], "taylor")
    assert "between" in posted[-1] and _rows(p)["ngc7790"]["max_hours"] == 1.0
    suc.type_cmd(["", "type", "ic410", "nebula"], "taylor")
    assert _rows(p)["ic410"]["obj_type"] == "nebula" and "narrowband" in posted[-1]
    suc.type_cmd(["", "type", "nope", "nebula"], "taylor")
    assert "not in the queue" in posted[-1]
    assert len([n for n in notes if n[0] == "QUEUE_FIELD"]) == 4


# ------------------------------------------------------------ a change replans tonight (2026-11-21)

def _rec(check, dsos, outcome="good"):
    return {"check": check, "outcome": outcome,
            "plan": {"slots": [{"dso": d, "start": "2026-11-20T23:00-05:00", "end": "2026-11-21T04:00-05:00"} for d in dsos]}}


def test_replan_verdict_is_pure_and_says_why():
    """2026-11-20: `filters ic405 ...` after noon rewrote the sequence quietly while the
    push and the board kept the advisor's O-III; a change to tonight's target replans."""
    noon = {"noon": _rec("noon", ["ross248", "ic405"]), "sunset": None}
    do, text = Q.replan_verdict(noon, "ic405")
    assert do and "tonight's plan (23:00-04:00)" in text and "replanning" in text
    assert Q.replan_verdict(noon, "IC 405")[0]                       # the name squashes
    assert Q.replan_verdict(noon, "ngc1055") == (False, "")           # not tonight's business
    assert Q.replan_verdict(None, "ic405") == (False, "")             # no conductor
    assert Q.replan_verdict({"noon": None, "sunset": None}, "ic405") == (False, "")
    assert Q.replan_verdict({"noon": _rec("noon", ["ic405"], outcome="bad"), "sunset": None}, "ic405") == (False, "")
    after = {"noon": _rec("noon", ["ic405"]), "sunset": _rec("sunset", ["ic405"])}
    do, text = Q.replan_verdict(after, "ic405")
    assert not do and "sunset check has already decided" in text and "tomorrow" in text
    assert Q.replan_verdict(after, "ngc1055") == (False, "")
    # the machine's state: REPLAN_REQUESTED from SLOT_IMAGING would re-seat the slot
    busy = {"noon": _rec("noon", ["ic405"]), "sunset": None, "state": "SLOT_IMAGING"}
    do, text = Q.replan_verdict(busy, "ic405")
    assert not do and "conductor is in SLOT_IMAGING" in text
    assert Q.replan_verdict(dict(busy, state="ARMED"), "ic405")[0] and Q.replan_verdict(dict(busy, state="IDLE_DAY"), "ic405")[0]


def test_type_on_tonights_target_requests_a_replan(world, monkeypatch):
    suc, p, posted, notes = world
    import iris.client as client
    asked = []
    plan = {"noon": _rec("noon", ["ic410"]), "sunset": None, "state": "IDLE_DAY"}
    monkeypatch.setattr(client, "drives_checks", lambda: True)
    monkeypatch.setattr(client, "get_plan", lambda *a, **k: plan)
    monkeypatch.setattr(client, "request_replan", lambda source="chat": asked.append(source) or {"accepted": True})
    suc.type_cmd(["", "type", "ic410", "nebula"], "taylor")
    assert _rows(p)["ic410"]["obj_type"] == "nebula"
    assert asked == ["chat:type"]
    assert any("replanning so the sequence" in m for m in posted) and "Replanning:" in posted[-1]
    # a target that is not in tonight's plan changes silently
    suc.moon_cmd(["", "moon", "ngc7790", "dark"], "taylor")
    assert asked == ["chat:type"] and "whenever the moon" in posted[-1]
    # after the sunset check the change is tomorrow's
    plan["sunset"] = _rec("sunset", ["ic410"])
    suc.type_cmd(["", "type", "ic410", "galaxy"], "taylor")
    assert asked == ["chat:type"] and "applies from tomorrow" in posted[-1]
    # the conductor not driving the checks: nothing extra
    monkeypatch.setattr(client, "drives_checks", lambda: False)
    plan["sunset"] = None
    suc.type_cmd(["", "type", "ic410", "nebula"], "taylor")
    assert asked == ["chat:type"] and "narrowband" in posted[-1]
