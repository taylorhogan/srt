"""The journal: append/replay round-trips, monotonic seq, crash tolerance."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from iris.core.journal import Entry, Journal


def test_round_trip_preserves_everything(tmp_path):
    j = Journal(tmp_path)
    e = j.append("transition", "NINA_MAIN_DONE", "nina",
                 from_state="SLOT_IMAGING", to_state="FLATS",
                 data={"slot": 2, "dso": "sh2-129"})
    got = list(j.replay())
    assert got == [e]
    assert got[0].data["dso"] == "sh2-129"


def test_wire_format_uses_from_and_to(tmp_path):
    j = Journal(tmp_path)
    j.append("transition", "X_EV", "conductor", from_state="A", to_state="B")
    line = next(iter(tmp_path.glob("*.jsonl"))).read_text().strip()
    d = json.loads(line)
    assert d["from"] == "A" and d["to"] == "B"
    assert "from_state" not in d


def test_seq_is_monotonic_and_survives_reopen(tmp_path):
    j = Journal(tmp_path)
    for i in range(5):
        j.append("note", "N", "shadow")
    assert j.head() == 5
    j2 = Journal(tmp_path)              # new process, same directory
    assert j2.head() == 5
    e = j2.append("note", "N", "shadow")
    assert e.seq == 6


def test_entries_since_pages_in_order(tmp_path):
    j = Journal(tmp_path)
    for i in range(10):
        j.append("note", f"E{i}", "shadow")
    got = j.entries_since(4, limit=3)
    assert [e.seq for e in got] == [5, 6, 7]
    got = j.entries_since(7)
    assert [e.seq for e in got] == [8, 9, 10]


def test_truncated_final_line_is_dropped_not_fatal(tmp_path):
    j = Journal(tmp_path)
    j.append("note", "OK", "shadow")
    f = next(iter(tmp_path.glob("*.jsonl")))
    with open(f, "a", encoding="utf-8") as fh:
        fh.write('{"seq": 2, "ts": "2026-')       # crash mid-write
    j2 = Journal(tmp_path)
    assert [e.event for e in j2.replay()] == ["OK"]
    # and the next append REUSES seq 2 -- the half-written entry never happened
    assert j2.append("note", "NEXT", "shadow").seq == 2


def test_corruption_mid_file_raises(tmp_path):
    """A mangled line that is NOT the tail is real corruption; replaying past
    it silently would rebuild a state missing an arbitrary transition."""
    j = Journal(tmp_path)
    j.append("note", "A", "shadow")
    j.append("note", "B", "shadow")
    f = next(iter(tmp_path.glob("*.jsonl")))
    lines = f.read_text().splitlines()
    lines[0] = '{"broken'
    f.write_text("\n".join(lines) + "\n")
    with pytest.raises(Exception):
        list(Journal(tmp_path).replay())


def test_rejected_entries_carry_the_guard(tmp_path):
    j = Journal(tmp_path)
    j.append("rejected", "ROOF_OPEN_REQUESTED", "chat",
             from_state="PRE_FLIGHT",
             guard="mount_parked: scope not confirmed parked by vision (unknown)")
    e = list(j.replay())[0]
    assert e.kind == "rejected"
    assert "mount_parked" in e.guard
