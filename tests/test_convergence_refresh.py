"""scripts/convergence_refresh -- which imaged targets have a stale convergence entry.

Pure: the module's live imports happen in main().
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import convergence_refresh as R  # noqa: E402


def _epoch(y, m, d, h=3):
    return datetime(y, m, d, h, 0).timestamp()


ENTRIES = {"m1": {"Ha": {"updated": "2026-10-28"}, "O-III": {"updated": "2026-10-27"}},
           "ic410": {"Ha": {"updated": "2026-10-29"}},
           "ngc7380": {"Ha": {}}}
NEWEST = {"m1": _epoch(2026, 10, 29), "ic410": _epoch(2026, 10, 29),
          "ngc7380": _epoch(2026, 10, 29), "trunk": _epoch(2026, 10, 29),
          "hubblev1": _epoch(2026, 10, 29)}


def test_an_entry_older_than_the_newest_frame_is_stale():
    out = dict(R.stale_targets(["m1"], ENTRIES, NEWEST))
    assert out == {"m1": "entry updated 2026-10-28, newest frame 2026-10-29"}


def test_an_entry_from_the_frames_day_is_current():
    assert R.stale_targets(["ic410"], ENTRIES, NEWEST) == []


def test_missing_or_undated_entries_are_stale():
    out = dict(R.stale_targets(["trunk", "ngc7380"], ENTRIES, NEWEST))
    assert out["trunk"] == "no convergence entry" and out["ngc7380"] == "entry carries no date"


def test_monitors_and_targets_without_frames_are_skipped():
    assert R.stale_targets(["hubblev1"], ENTRIES, NEWEST, monitors=["HubbleV1"]) == []
    assert R.stale_targets(["m101"], ENTRIES, NEWEST) == []


def test_names_match_the_way_convergence_keys_them():
    entries = {"ngc7380": {"Ha": {"updated": "2026-10-29"}}}
    assert R.stale_targets(["NGC 7380"], entries, {"NGC 7380": _epoch(2026, 10, 29)}) == []


def test_newest_light_reads_only_light_directories(tmp_path):
    light = tmp_path / "m1" / "2026-10-29" / "LIGHT"
    light.mkdir(parents=True)
    (light / "a.fits").write_bytes(b"x")
    flat = tmp_path / "m1" / "2026-10-29" / "FLAT"
    flat.mkdir()
    (flat / "f.fits").write_bytes(b"x")
    import os
    os.utime(flat / "f.fits", (2e9, 2e9))
    newest = R.newest_light(tmp_path / "m1")
    assert newest is not None and newest < 2e9
    assert R.newest_light(tmp_path / "nowhere") is None


def test_all_targets_lists_every_directory_with_light_frames(tmp_path):
    (tmp_path / "ic410" / "cdk17" / "2026-10-27" / "LIGHT").mkdir(parents=True)
    (tmp_path / "ic410" / "cdk17" / "2026-10-27" / "LIGHT" / "a.fits").write_bytes(b"")
    (tmp_path / "m1" / "FLAT").mkdir(parents=True)
    (tmp_path / "m1" / "FLAT" / "f.fits").write_bytes(b"")          # flats only: not a target
    (tmp_path / "notes.txt").write_text("x")
    assert R.all_targets(tmp_path) == ["ic410"]
    assert R.all_targets(tmp_path / "missing") == []
