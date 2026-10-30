"""`approve`, `reject` and `drafts` on the PC (D3): venv only (the chat command module)."""
import json
from pathlib import Path

import pytest

suc = pytest.importorskip("cmd_processing.super_user_commands")
from control import publish_gate as G  # noqa: E402


@pytest.fixture
def world(tmp_path, monkeypatch):
    posted, notes = [], []
    monkeypatch.setattr(suc, "_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(suc.social_server, "post_social_message", lambda m: posted.append(m))
    monkeypatch.setattr(suc.config, "data", lambda: {"publish": {"gate": "shadow"}})
    import iris.client as client
    monkeypatch.setattr(client, "post_event", lambda ev, src, data=None, **k: notes.append((ev, data)) or {})
    jd = tmp_path / "local" / "journal"
    jd.mkdir(parents=True)
    (jd / "2026-11-07.jsonl").write_text(json.dumps(
        {"seq": 1, "ts": "t", "kind": "note", "event": "DRAFT_READY", "source": "spark", "guard": None,
         "data": {"dso": "ic410", "draft_hash": "a" * 64, "recipe": "SHO", "product": "learned",
                  "article_source": "fallback", "auto_publish_on": "2026-11-09"}, "from": None, "to": None}) + "\n")
    return tmp_path, posted, notes


def test_approve_records_the_hash_and_journals(world):
    root, posted, notes = world
    suc.approve_cmd(["", "approve", "ic410"], "taylor")
    d = G.load_decisions(root / "local" / "publish_decisions.json")
    assert d["ic410"]["decision"] == "approve" and d["ic410"]["draft_hash"] == "a" * 64 and d["ic410"]["by"] == "taylor"
    assert notes == [("PUBLISH_APPROVED", {"dso": "ic410", "draft_hash": "a" * 64, "by": "taylor", "why": ""})]
    assert "ic410 approved (draft aaaaaaaaaaaa, SHO learned)" in posted[-1] and "Gate is shadow" in posted[-1]


def test_reject_with_a_reason_and_a_two_word_name(world):
    root, posted, notes = world
    suc.reject_cmd(["", "reject", "ic", "410", "stars", "too", "red"], "taylor")
    d = G.load_decisions(root / "local" / "publish_decisions.json")
    assert d["ic410"]["decision"] == "reject" and d["ic410"]["why"] == "stars too red"
    assert notes[-1][0] == "PUBLISH_REJECTED" and "rejected (draft aaaaaaaaaaaa): stars too red" in posted[-1]


def test_unknown_draft_and_usage(world):
    root, posted, notes = world
    suc.approve_cmd(["", "approve", "m1"], "taylor")
    assert "no draft awaiting a decision" in posted[-1] and notes == []
    suc.approve_cmd(["", "approve"], "taylor")
    assert posted[-1].startswith("Usage: approve <dso>")


def test_drafts_lists_what_would_happen(world):
    root, posted, notes = world
    suc.drafts_cmd(["", "drafts"], "taylor")
    assert posted[-1].startswith("Drafts (gate: shadow):") and "ic410  SHO learned  draft aaaaaaaaaaaa  facts only -- wait" in posted[-1]
    suc.approve_cmd(["", "approve", "ic410"], "taylor")
    suc.drafts_cmd(["", "drafts"], "taylor")
    assert "-- publish: approved by taylor" in posted[-1]
