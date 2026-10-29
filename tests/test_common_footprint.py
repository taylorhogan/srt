"""The stacks of one target are cut to the area every filter's frames covered.

A stack built onto a shared reference is not cropped by the stacker (its box
would differ per filter, which is what the shared reference exists to
prevent); outside its footprint it is a flat fill at the global sky level.
At a vignetted edge that fill sits above the real sky beside it, and through
a palette gain it became a red bar along the top of every IC 410 render
(owner, 2026-10-28). The stacker now reports its coverage box either way and
the morning render cuts every channel to the intersection.
"""
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_coverage_box_is_the_well_covered_bounding_box():
    from stacking import stacker
    cov = np.zeros((40, 60), np.int32)
    cov[5:35, 8:50] = 10
    cov[20:25, 50:58] = 3                       # a corner only a few frames reached
    assert stacker.coverage_box(cov, 8) == (5, 35, 8, 50)
    assert stacker.coverage_box(cov, 2) == (5, 35, 8, 58)
    assert stacker.coverage_box(cov, 11) is None


def test_common_box_is_the_intersection_and_missing_boxes_count_as_full():
    import n2n_lrgb_render as r
    meta = {"channels": {"Ha": {"coverage": [10, 100, 5, 200]},
                         "O-III": {"coverage": [0, 90, 20, 210]},
                         "S-II": {}}}                               # cached before the box existed
    assert r.common_box(meta, ["Ha", "O-III", "S-II"], (120, 220)) == (10, 90, 20, 200)
    assert r.common_box(meta, ["S-II"], (120, 220)) == (0, 120, 0, 220)
    # an empty intersection falls back to the whole frame rather than nothing
    meta["channels"]["O-III"]["coverage"] = [100, 120, 0, 10]
    assert r.common_box(meta, ["Ha", "O-III"], (120, 220)) == (0, 120, 0, 220)


def test_crop_common_cuts_every_channel_and_says_so():
    import n2n_lrgb_render as r
    meta = {"channels": {"Ha": {"coverage": [2, 30, 3, 40]}, "O-III": {"coverage": [0, 28, 5, 44]}}}
    chans = {"Ha": np.ones((32, 44), np.float32), "O-III": np.ones((32, 44), np.float32)}
    lines = []
    out = r.crop_common(chans, meta, lines.append)
    assert out["Ha"].shape == (26, 35) and out["O-III"].shape == (26, 35)
    assert any("cut 2 top, 4 bottom, 5 left, 4 right" in m for m in lines)
    same = r.crop_common({"Ha": chans["Ha"]}, {"channels": {"Ha": {"coverage": [0, 32, 0, 44]}}}, lines.append)
    assert same["Ha"].shape == (32, 44)


def test_a_cached_stacks_record_is_found_in_any_recipes_meta(tmp_path, monkeypatch):
    # The morning's SHO rebuilt the stacks with coverage boxes; the HOO
    # after it read only its own meta from the day before and rendered
    # uncropped (Melotte 15, 2026-10-29). The record belongs to the stack:
    # the sidecar beside the npy, else the newest entry naming the file in
    # any recipe's meta of the target.
    import json
    import n2n_lrgb_render as R
    monkeypatch.setattr(R, "OUT", tmp_path)
    out = tmp_path / "x_Ha_raw.npy"
    old = {"file": "x_Ha_raw.npy", "built": "2026-10-28T08:00", "accepted": 60}
    new = {"file": "x_Ha_raw.npy", "built": "2026-10-29T08:00", "accepted": 73,
           "coverage": [76, 6381, 37, 9541]}
    (tmp_path / "meta_x_HOO.json").write_text(json.dumps({"channels": {"Ha": old}}))
    (tmp_path / "meta_x_SHO.json").write_text(json.dumps({"channels": {"Ha": new}}))
    (tmp_path / "meta_x_SHO.json").touch()
    assert R.stack_record("x", out) == new                   # the newest record of that file
    other = {"file": "x_O-III_raw.npy", "built": "2026-10-30T08:00"}
    (tmp_path / "meta_x_HOO.json").write_text(json.dumps({"channels": {"O-III": other}}))
    assert R.stack_record("x", out) == new                   # another file's entry is not it
    # a sidecar beside the npy wins over every meta
    side = {"file": "x_Ha_raw.npy", "built": "2026-10-31T08:00", "coverage": [1, 2, 3, 4]}
    out.with_suffix(".json").write_text(json.dumps(side))
    assert R.stack_record("x", out) == side
    assert R.stack_record("x", tmp_path / "x_S-II_raw.npy") == {}
