"""Which targets did the night image (control/tonight_dsos)?

A synthetic archive: two DSOs with LIGHT frames on one night, in the order
a two-slot night writes them, plus a finished-products directory and an old
target that must not appear.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from control import tonight_dsos  # noqa: E402


def _frame(root, dso, name, age_s):
    p = root / dso / "cdk17" / "2026-09-08" / "LIGHT" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    t = time.time() - age_s
    os.utime(p, (t, t))
    return p


def test_two_slot_night_lists_both_in_shot_order(tmp_path):
    _frame(tmp_path, "ngc7380", "a.fits", 6 * 3600)
    _frame(tmp_path, "ngc7380", "b.fits", 5 * 3600)
    _frame(tmp_path, "m33", "c.fits", 3 * 3600)
    _frame(tmp_path, "m33", "d.fits", 1 * 3600)
    _frame(tmp_path, "squid", "old.fits", 30 * 3600)               # last week
    p = tmp_path / "Iris" / "m33" / "LIGHT" / "stack.fits"        # a product, not data
    p.parent.mkdir(parents=True); p.write_bytes(b"x")
    assert tonight_dsos.imaged(tmp_path, since=time.time() - 8 * 3600) == ["ngc7380", "m33"]


def test_non_light_directories_do_not_count(tmp_path):
    p = tmp_path / "m33" / "cdk17" / "2026-09-08" / "FLAT" / "f.fits"
    p.parent.mkdir(parents=True); p.write_bytes(b"x")
    assert tonight_dsos.imaged(tmp_path, since=0) == []


def test_hint_names_the_others_only(tmp_path):
    _frame(tmp_path, "ngc7380", "a.fits", 6 * 3600)
    _frame(tmp_path, "m33", "c.fits", 2 * 3600)
    (tmp_path / "imaging_start.txt").write_text("2000-01-01T00:00:00")
    hint = tonight_dsos.others_hint(tmp_path, "m33", tmp_path, "snr")
    assert "ngc7380" in hint and "`snr ngc7380`" in hint and "m33" not in hint.split("run")[1]
    assert tonight_dsos.others_hint(tmp_path, "ngc7380", tmp_path, "snr").count("m33") == 2
    _frame(tmp_path, "m33", "only.fits", 1 * 3600)
    (tmp_path / "ngc7380").rename(tmp_path / "zz_gone")
    assert tonight_dsos.others_hint(tmp_path, "m33", tmp_path, "snr").startswith("\nTonight also imaged zz_gone")


def test_missing_archive_is_empty_not_an_error(tmp_path):
    assert tonight_dsos.imaged(tmp_path / "nope") == []
    assert tonight_dsos.imaging_start(tmp_path) is None
