"""`reopen <dso> [for N]` (A4 of docs/AUTOMATION_PLAN.md, ADR 0016): the undo for auto-stop.

Needs control.instructions (astropy, the chat server): venv only.
"""
import json
from datetime import datetime

import pytest

instr = pytest.importorskip("control.instructions")
from control import auto_stop as A  # noqa: E402


@pytest.fixture
def queue(tmp_path, monkeypatch):
    p = tmp_path / "q.json"
    p.write_text(json.dumps([
        {"dso": "ngc1961", "status": "converged",
         "status_history": [{"from": "waiting", "to": "converged", "by": "auto_stop"}]},
        {"dso": "m31", "status": "completed"},
        {"dso": "ic410", "status": "waiting"}]))
    monkeypatch.setattr(instr, "_INSTRUCTIONS_PATH", str(p))
    return p


def test_set_field_db(queue):
    assert instr.set_field_db("NGC 1961", A.HOLD_KEY, "2026-11-05")
    assert json.loads(queue.read_text())[0][A.HOLD_KEY] == "2026-11-05"
    assert instr.set_field_db("ngc1961", A.HOLD_KEY, None)
    assert A.HOLD_KEY not in json.loads(queue.read_text())[0]
    assert not instr.set_field_db("m31", "x", 1)                 # completed rows are not touched
    assert not instr.set_field_db("nope", "x", 1)


def _cmd(monkeypatch):
    suc = pytest.importorskip("cmd_processing.super_user_commands")
    posted, notes = [], []
    monkeypatch.setattr(suc.social_server, "post_social_message", lambda m: posted.append(m))
    monkeypatch.setattr(suc.instructions, "create_instructions_table", lambda: None)
    monkeypatch.setattr(suc, "datetime", type("D", (), {"now": staticmethod(lambda: datetime(2026, 11, 6, 9, 0))}))
    import iris.client as client
    monkeypatch.setattr(client, "post_event", lambda ev, src, data=None, **k: notes.append((ev, data)) or {})
    return suc, posted, notes


def test_reopen_puts_a_converged_target_back_and_holds_auto_stop_off_it(queue, monkeypatch):
    suc, posted, notes = _cmd(monkeypatch)
    suc.reopen_cmd(["", "reopen", "ngc1961"], "taylor")
    row = json.loads(queue.read_text())[0]
    assert row["status"] == "waiting" and row[A.HOLD_KEY] == "2026-11-12"          # 7 nights from 11-06
    assert row["status_history"][-1] == {**row["status_history"][-1], "from": "converged", "to": "waiting",
                                         "by": "taylor", "why": "reopened by taylor"}
    assert notes == [("REOPENED", {"dso": "ngc1961", "from": "converged", "by": "taylor",
                                   "hold_until": "2026-11-12"})]
    assert "back in the queue as waiting" in posted[-1] and "7 nights, through 2026-11-12" in posted[-1]
    # the noon evaluation now leaves it alone
    rows = json.loads(queue.read_text())
    from datetime import date
    assert [v["dso"] for v in A.candidates(rows, lambda d: True, lambda d, r: {"action": "done", "have": {}},
                                           night=date(2026, 11, 12))] == ["ic410"]
    assert [v["dso"] for v in A.candidates(rows, lambda d: True, lambda d, r: {"action": "done", "have": {}},
                                           night=date(2026, 11, 13))] == ["ngc1961", "ic410"]


def test_reopen_for_n_and_the_edge_cases(queue, monkeypatch):
    suc, posted, notes = _cmd(monkeypatch)
    suc.reopen_cmd(["", "reopen", "ic410", "for", "1"], "taylor")
    row = json.loads(queue.read_text())[2]
    assert row["status"] == "waiting" and row[A.HOLD_KEY] == "2026-11-06"
    assert "is waiting; auto-stop leaves it alone for tonight only" in posted[-1]
    suc.reopen_cmd(["", "reopen", "m31"], "taylor")
    assert "completed" in posted[-1] and "image m31" in posted[-1]
    suc.reopen_cmd(["", "reopen", "nope"], "taylor")
    assert "not in the queue" in posted[-1]
    suc.reopen_cmd(["", "reopen", "ic410", "for", "x"], "taylor")
    assert "whole number" in posted[-1]
    suc.reopen_cmd(["", "reopen"], "taylor")
    assert posted[-1].startswith("Usage")
    assert len(notes) == 1
