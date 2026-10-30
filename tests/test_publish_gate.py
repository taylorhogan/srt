"""control/publish_gate (D3 of docs/AUTOMATION_PLAN.md, 2026-10-30): one tap publishes.

The card text, the decision, the journal's drafts and the decisions file are
pure and stdlib; the commands on the PC are tested in the venv (test_approve_cmds).
"""
import json
from datetime import date, datetime

from control import publish_gate as G

REC = {"dso": "ic410", "recipe": "SHO", "product": "learned", "status": "pending", "draft_hash": "a" * 64,
       "article_source": "fallback", "auto_publish_on": "2026-11-09", "qc": {"pass": True, "reasons": []},
       "image": "/f/ic410/ic410_SHO_learned_final.jpg"}
ART = {"title": "IC410 - Emission Nebula", "distance_text": "NOT ON RECORD", "type_text": "EMISSION NEBULA",
       "paragraphs": ["First.", "Second."], "time_travel_title": "", "time_travel": "", "source": "fallback"}
FACTS = {"meta_frames": "35 Ha + 25 S-II &times; 300 s", "hours": 5.0, "nights": ["2026-10-27", "2026-10-28"]}


def test_card_text_shows_the_draft_and_the_reply():
    t = G.card_text(REC, ART, FACTS, "shadow")
    assert t.startswith("PUBLISH DRAFT: IC410 - Emission Nebula   (SHO, learned; QC PASS)")
    assert "DISTANCE: NOT ON RECORD | TYPE: EMISSION NEBULA | 35 Ha + 25 S-II x 300 s | 5.0 HOURS OVER 2 NIGHTS" in t
    assert "First.\nSecond." in t and "On Earth" not in t
    assert "facts only" in t and "Reply `approve ic410` or `reject ic410 [why]`." in t
    assert "Would auto-publish 2026-11-09 (gate: shadow; nothing publishes yet)." in t
    live = G.card_text(dict(REC, article_source="api"), dict(ART, time_travel="Light left.", time_travel_title="On Earth: then"), FACTS, "live")
    assert "Written by: the model" in live and "On Earth: then: Light left." in live and "Auto-publishes 2026-11-09." in live
    f = G.fail_card_text("ic410", "SHO", {"pass": False, "reasons": ["sky at 0.3% grey"]})
    assert f.startswith("FINAL RENDER NEEDS A HAND: ic410 SHO -- QC FAIL: sky at 0.3% grey") and "no timer" in f


def test_decide_covers_every_outcome():
    today = date(2026, 11, 7)
    assert G.decide(None, None, today, True)["action"] == "none"
    assert G.decide(dict(REC, status="on_site"), None, today, True)["action"] == "none"
    assert G.decide(REC, None, today, True)["action"] == "wait"
    assert G.decide(REC, None, today, False)["action"] == "wait" and "not armed" in G.decide(REC, None, today, False)["why"]
    assert G.decide(REC, None, date(2026, 11, 9), True)["action"] == "publish"        # the timer, gate live
    assert G.decide(REC, None, date(2026, 11, 9), False)["action"] == "wait"          # the timer, gate shadow
    ok = {"decision": "approve", "draft_hash": "a" * 64, "by": "taylor", "at": "2026-11-07T09:00"}
    assert G.decide(REC, ok, today, False)["action"] == "publish"                    # an approve beats the gate's timer rule
    assert G.decide(REC, dict(ok, decision="reject", why="red stars"), today, True) == \
        {"action": "rejected", "why": "rejected by taylor: red stars"}
    assert G.decide(REC, dict(ok, draft_hash="b" * 64), today, True)["action"] == "stale"


def test_latest_drafts_follows_the_journal():
    entries = [{"event": "DRAFT_READY", "data": {"dso": "IC 410", "draft_hash": "a" * 64, "recipe": "SHO"}},
               {"event": "NIGHT_PLAN", "data": {}},
               {"event": "DRAFT_READY", "data": {"dso": "m1", "draft_hash": "c" * 64}},
               {"event": "PUBLISH_DONE", "data": {"dso": "m1"}},
               {"event": "DRAFT_READY", "data": {"dso": "ic410", "draft_hash": "b" * 64, "recipe": "HOO"}}]
    d = G.latest_drafts(entries)
    assert list(d) == ["ic410"] and d["ic410"]["draft_hash"] == "b" * 64 and d["ic410"]["recipe"] == "HOO"

    class E:                                                   # iris.core.journal.Entry shape
        def __init__(self, event, data):
            self.event, self.data = event, data
    assert G.latest_drafts([E("DRAFT_READY", {"dso": "x", "draft_hash": "h"})]) == {"x": {"dso": "x", "draft_hash": "h"}}


def test_read_journal_drafts_over_day_files(tmp_path):
    jd = tmp_path / "local" / "journal"
    jd.mkdir(parents=True)
    line = {"seq": 1, "ts": "2026-11-07T09:00:00-04:00", "kind": "note", "event": "DRAFT_READY", "source": "spark",
            "guard": None, "data": {"dso": "ic410", "draft_hash": "a" * 64}, "from": None, "to": None}
    (jd / "2026-11-07.jsonl").write_text(json.dumps(line) + "\n")
    assert G.read_journal_drafts(str(tmp_path))["ic410"]["draft_hash"] == "a" * 64
    assert G.read_journal_drafts(str(tmp_path / "nope")) == {}


def test_decisions_file_round_trip_and_drafts_lines(tmp_path):
    p = tmp_path / "local" / "publish_decisions.json"
    e = G.record_decision(p, "IC 410", "approve", "a" * 64, "taylor", now=datetime(2026, 11, 7, 9, 0))
    assert e == {"decision": "approve", "draft_hash": "a" * 64, "by": "taylor", "at": "2026-11-07T09:00", "why": ""}
    G.record_decision(p, "m1", "reject", "c" * 64, "taylor", "red stars")
    d = G.load_decisions(p)
    assert set(d) == {"ic410", "m1"} and d["m1"]["why"] == "red stars"
    assert G.load_decisions(tmp_path / "missing.json") == {}
    drafts = {"ic410": {"draft_hash": "a" * 64, "recipe": "SHO", "product": "learned", "article_source": "fallback",
                        "auto_publish_on": "2026-11-09"},
              "m1": {"draft_hash": "d" * 64, "recipe": "HOO", "product": "learned", "article_source": "api"}}
    lines = G.drafts_lines(drafts, d, date(2026, 11, 7), "shadow")
    assert lines[0].startswith("ic410  SHO learned  draft aaaaaaaaaaaa  facts only -- publish")
    assert lines[1].startswith("m1  HOO learned  draft dddddddddddd  model -- stale")
    assert G.drafts_lines({}, {}, date(2026, 11, 7), "shadow") == ["no drafts awaiting a decision"]
    assert G.gate_mode({}) == "shadow" and G.gate_mode({"publish": {"gate": "LIVE"}}) == "live" and G.gate_mode({"publish": {"gate": "x"}}) == "shadow"
