"""roof_evidence: the two timestamps a blind stop! park rests on."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentry import roof_evidence as re_


def test_round_trip_and_ordering(tmp_path, monkeypatch):
    monkeypatch.setattr(re_, "PATH", str(tmp_path / "roof_evidence.json"))
    assert re_.read() == {"open_confirmed": None, "motion_possible": None}
    re_.record_motion_possible("fire")
    re_.record_open_confirmed("vision")
    ev = re_.read()
    assert ev["motion_possible"] <= ev["open_confirmed"]
    re_.record_motion_possible("fire again")
    assert re_.read()["motion_possible"] >= ev["open_confirmed"]


def test_corrupt_file_reads_as_no_evidence(tmp_path, monkeypatch):
    p = tmp_path / "roof_evidence.json"
    p.write_text("{not json")
    monkeypatch.setattr(re_, "PATH", str(p))
    assert re_.read() == {"open_confirmed": None, "motion_possible": None}
    re_.record_open_confirmed()                 # overwrites rather than raising
    assert re_.read()["open_confirmed"] is not None
