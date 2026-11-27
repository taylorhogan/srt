"""scripts/seestar_watch: the Seestar's presence on the wifi (2026-11-27).
The decision is pure: a fresh state takes the first verdict silently, a drop
is declared after ``down_after`` misses and dated from the first miss, a
single blip heals without a line, a return is announced with the gap's
length, and the report reads the log. Nothing here touches the network.
"""
from datetime import datetime, timedelta

import pytest

sw = pytest.importorskip("scripts.seestar_watch")

T0 = datetime(2026, 11, 27, 20, 0, 0)


def _minutes(n):
    return T0 + timedelta(minutes=n)


def test_first_probe_sets_state_without_an_event():
    s, ev = sw.decide({}, True, T0, "192.168.87.250")
    assert ev is None and s["up"] is True and s["ip"] == "192.168.87.250" and s["since"] == T0.isoformat()
    s2, ev2 = sw.decide({}, False, T0, None)
    assert ev2 is None and s2["up"] is False


def test_one_miss_is_a_blip_not_a_drop():
    s, _ = sw.decide({}, True, T0, "10.0.0.5")
    s, ev = sw.decide(s, False, _minutes(1), None, down_after=2)
    assert ev is None and s["up"] is True and s["misses"] == 1
    s, ev = sw.decide(s, True, _minutes(2), "10.0.0.5", down_after=2)
    assert ev is None and s["misses"] == 0 and s["blips"] == 1 and s["up"] is True


def test_two_misses_declare_down_dated_from_the_first():
    s, _ = sw.decide({}, True, T0, "10.0.0.5")
    s, _ = sw.decide(s, True, _minutes(90), "10.0.0.5")
    s, ev = sw.decide(s, False, _minutes(91), None, down_after=2)
    assert ev is None
    s, ev = sw.decide(s, False, _minutes(92), None, down_after=2)
    assert ev["event"] == "down" and ev["at"] == _minutes(91).isoformat()
    assert ev["after_s"] == 91 * 60 and ev["ip"] == "10.0.0.5"
    assert s["up"] is False and s["since"] == _minutes(91).isoformat() and s["misses"] == 0
    # staying down is silent
    s, ev = sw.decide(s, False, _minutes(93), None, down_after=2)
    assert ev is None and s["up"] is False


def test_return_is_announced_with_the_gap():
    s, _ = sw.decide({}, False, T0, None)
    s, ev = sw.decide(s, True, _minutes(14), "10.0.0.7")
    assert ev["event"] == "up" and ev["after_s"] == 14 * 60 and ev["ip"] == "10.0.0.7"
    assert s["up"] is True and s["since"] == _minutes(14).isoformat()
    assert sw.line_for(ev) == "Seestar back on the wifi at 10.0.0.7, after 14 min off"


def test_down_after_one_declares_at_once():
    s, _ = sw.decide({}, True, T0, "10.0.0.5")
    s, ev = sw.decide(s, False, _minutes(1), None, down_after=1)
    assert ev["event"] == "down" and ev["after_s"] == 60


def test_lines_and_spans():
    assert sw.span(45) == "45 s"
    assert sw.span(3 * 3600 + 12 * 60) == "3 h 12 min"
    assert sw.span(2 * 86400 + 5 * 3600) == "2 d 5 h"
    assert sw.line_for({"event": "down", "ip": "10.0.0.5", "after_s": 3 * 3600 + 12 * 60}) == \
        "Seestar off the wifi at 10.0.0.5, after 3 h 12 min up"
    assert sw.line_for({"event": "down", "ip": None, "after_s": 60}) == "Seestar off the wifi, after 1 min up"


def test_report_counts_sessions_and_the_open_state():
    events = [
        {"event": "down", "at": _minutes(100).isoformat(), "ip": "10.0.0.5", "after_s": 6000},
        {"event": "up", "at": _minutes(110).isoformat(), "ip": "10.0.0.5", "after_s": 600},
        {"event": "down", "at": _minutes(300).isoformat(), "ip": "10.0.0.5", "after_s": 11400},
        {"event": "up", "at": (T0 - timedelta(days=30)).isoformat(), "ip": "10.0.0.5", "after_s": 5},
    ]
    state = {"up": False, "since": _minutes(300).isoformat(), "ip": "10.0.0.5", "blips": 2}
    text = sw.report(events, state, _minutes(330), days=7)
    assert "2 drops, 1 return" in text
    assert "longest 3 h 10 min" in text and "shortest 1 h 40 min" in text
    assert "now: DOWN for 30 min at 10.0.0.5" in text
    assert "blips healed without a line: 2" in text
    assert (T0 - timedelta(days=30)).isoformat() not in text


def test_state_and_log_round_trip(tmp_path):
    st = tmp_path / "s.json"
    lg = tmp_path / "l.jsonl"
    sw.save_state({"up": True, "since": T0.isoformat()}, st)
    assert sw.load_state(st)["up"] is True
    sw.append_log({"event": "down", "at": T0.isoformat(), "ip": None, "after_s": 1}, lg)
    sw.append_log({"event": "up", "at": T0.isoformat(), "ip": None, "after_s": 2}, lg)
    assert [e["event"] for e in sw.read_log(lg)] == ["down", "up"]
    assert sw.load_state(tmp_path / "missing.json") == {} and sw.read_log(tmp_path / "missing") == []


def test_probe_reports_an_unresolvable_name():
    res = sw.probe("no-such-host.invalid", 445, 0.5)
    assert res["ok"] is False and res["ip"] is None and "does not resolve" in res["why"]
