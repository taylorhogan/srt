"""The flat panel's light through its driver: the parser, the words, the
retry on a busy port, and the flats routine's off step."""
import pytest

from hardware_control import flat_panel as fp


def test_the_drivers_answer_parses_to_on_off_or_unreadable():
    assert fp.parse('{"state":1,"brightness":0,"max":255}')["light"] == "off"
    lit = fp.parse('PowerShell chatter\n{"state":3,"brightness":255,"max":255}')
    assert lit["ok"] and lit["light"] == "on" and lit["brightness"] == 255
    assert fp.parse('{"state":2,"brightness":20,"max":255}')["light"] == "on"   # settling
    gone = fp.parse('{"state":0,"brightness":0,"max":0}')
    assert not gone["ok"] and "not present" in gone["why"]
    err = fp.parse('{"error":"Access to the port \'COM5\' is denied."}')
    assert not err["ok"] and fp.port_busy(err)
    assert not fp.parse("")["ok"]
    assert not fp.parse("{not json")["ok"]
    assert not fp.port_busy(fp.parse('{"error":"Class not registered"}'))


def test_describe_is_one_line_for_a_push():
    assert fp.describe(fp.parse('{"state":1,"brightness":0,"max":255}')) == "flat panel light off"
    assert fp.describe(fp.parse('{"state":3,"brightness":255,"max":255}')) == (
        "flat panel light ON at 255 of 255")
    assert fp.describe(fp.parse('{"error":"boom"}')) == "flat panel unreadable: boom"
    assert fp.describe(None).startswith("flat panel unreadable")


def test_ensure_off_waits_out_a_busy_port_and_gives_up_on_anything_else():
    answers = [fp.parse('{"error":"port COM5 is denied"}')] * 2 + [
        fp.parse('{"state":1,"brightness":0,"max":255}')]
    calls, naps = [], []
    st = fp.ensure_off(run=lambda action, cfg: (calls.append(action), answers.pop(0))[1],
                       sleep=naps.append, retries=6, pause_s=5)
    assert st["light"] == "off" and len(calls) == 3 and naps == [5, 5]
    assert "CalibratorOn(1)" in calls[0] and "CalibratorOff()" in calls[0]

    calls.clear()
    st = fp.ensure_off(run=lambda a, c: (calls.append(a), fp.parse('{"error":"Class not registered"}'))[1],
                       sleep=lambda s: pytest.fail("no retry on a driver fault"))
    assert not st["ok"] and len(calls) == 1

    # a port that never frees: retries+1 tries, then the last reading
    busy = fp.parse('{"error":"port busy"}')
    n = []
    st = fp.ensure_off(run=lambda a, c: (n.append(1), busy)[1], sleep=lambda s: None, retries=3)
    assert not st["ok"] and len(n) == 4


def test_off_the_observatory_pc_every_call_is_unreadable_and_nothing_raises(monkeypatch):
    monkeypatch.setattr(fp.platform, "system", lambda: "Linux")
    assert not fp.status()["ok"]
    assert not fp.ensure_off(sleep=lambda s: None)["ok"]


def test_the_panel_words():
    suc = pytest.importorskip("cmd_processing.super_user_commands")
    panel_words = suc.panel_words
    assert panel_words(["panel"]) == ("status", None)
    assert panel_words(["panel", "OFF"]) == ("off", None)
    assert panel_words(["panel", "on"])[0] is None
    assert panel_words(["panel", "off", "now"])[0] is None


def test_the_flats_routine_says_so_when_the_panel_is_not_confirmed_off(monkeypatch):
    suc = pytest.importorskip("cmd_processing.super_user_commands")
    posts = []
    monkeypatch.setattr(suc.social_server, "post_social_message", posts.append)
    monkeypatch.setattr(suc.flat_panel, "ensure_off",
                        lambda: fp.parse('{"state":1,"brightness":0,"max":255}'))
    assert suc._flat_panel_off()["light"] == "off"
    assert posts == []
    monkeypatch.setattr(suc.flat_panel, "ensure_off",
                        lambda: fp.parse('{"error":"port COM5 is denied"}'))
    suc._flat_panel_off()
    assert len(posts) == 1 and "NOT confirmed off" in posts[0] and "denied" in posts[0]
