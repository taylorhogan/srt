"""Calibration matches the lights' camera gain (2026-10-24).

Bias and darks come from fixed directories and darks were matched on exposure
alone, so the first night at a new gain would have been calibrated with the
old gain's masters and nothing would have said so. Now the lights' majority
(GAIN, OFFSET) is the stack's gain, lights at another gain are rejected, and
calibration at another gain is dropped; a frame with no GAIN header (the 2024
archive) matches anything."""
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
fits = pytest.importorskip("astropy.io.fits")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stacking import stacker  # noqa: E402


def _fits(path: Path, gain=None, offset=None, filt=None, exptime=300.0, imagetyp="LIGHT"):
    h = fits.Header()
    if gain is not None:
        h["GAIN"] = gain
    if offset is not None:
        h["OFFSET"] = offset
    if filt:
        h["FILTER"] = filt
    h["EXPTIME"] = exptime
    h["IMAGETYP"] = imagetyp
    fits.PrimaryHDU(np.zeros((4, 4), dtype=np.uint16), header=h).writeto(path, overwrite=True)
    return path


def test_camera_gain_reads_the_header_and_is_none_when_silent(tmp_path):
    assert stacker.camera_gain(_fits(tmp_path / "a.fits", 26, 30)) == (26, 30)
    assert stacker.camera_gain(_fits(tmp_path / "b.fits")) == (None, None)
    assert stacker.camera_gain(_fits(tmp_path / "c.fits", 0)) == (0, None)


def test_the_majority_gain_wins_and_the_others_are_rejected(tmp_path):
    lights = [_fits(tmp_path / f"l{i}.fits", 26, 30) for i in range(3)]
    old = [_fits(tmp_path / f"o{i}.fits", 0, 10) for i in range(2)]
    gain, kept, other = stacker.select_gain(lights + old)
    assert gain == (26, 30) and kept == lights and other == old


def test_a_light_without_a_gain_header_rides_with_the_majority(tmp_path):
    lights = [_fits(tmp_path / f"l{i}.fits", 26, 30) for i in range(2)]
    archive = _fits(tmp_path / "arch.fits")
    gain, kept, other = stacker.select_gain(lights + [archive])
    assert gain == (26, 30) and kept == lights + [archive] and other == []


def test_calibration_at_another_gain_is_dropped_and_the_record_says_so(tmp_path):
    bias = [_fits(tmp_path / f"b{i}.fits", 26, 30, imagetyp="BIAS") for i in range(2)]
    bias_old = [_fits(tmp_path / f"bo{i}.fits", 0, 10, imagetyp="BIAS") for i in range(2)]
    dark_old = [_fits(tmp_path / f"d{i}.fits", 0, 10, imagetyp="DARK") for i in range(3)]
    flat_any = [_fits(tmp_path / f"f{i}.fits", filt="Ha", imagetyp="FLAT") for i in range(2)]   # 2024 archive style
    said = []
    b, d, f, note = stacker.match_calibration((26, 30), bias + bias_old, dark_old, flat_any, log=said.append)
    assert b == bias and d == [] and f == flat_any
    assert note["bias"] == {"kept": 2, "dropped": 2, "dropped_gains": ["gain 0/offset 10"]}
    assert note["dark"]["kept"] == 0 and note["dark"]["dropped"] == 3
    assert "flat" not in note
    assert any("NO dark at this gain" in m for m in said)


def test_config_calibration_is_matched_to_the_light_it_is_asked_for(tmp_path, monkeypatch):
    bdir, ddir, froot = tmp_path / "bias", tmp_path / "dark", tmp_path / "flats"
    (froot / "2026-10-24" / "FLAT").mkdir(parents=True)
    bdir.mkdir(); ddir.mkdir()
    _fits(bdir / "b0.fits", 0, 10, imagetyp="BIAS"); _fits(bdir / "b26.fits", 26, 30, imagetyp="BIAS")
    _fits(ddir / "d0.fits", 0, 10, imagetyp="DARK"); _fits(ddir / "d26.fits", 26, 30, imagetyp="DARK")
    _fits(froot / "2026-10-24" / "FLAT" / "f0.fits", 0, 10, filt="Ha", imagetyp="FLAT")
    _fits(froot / "2026-10-24" / "FLAT" / "f26.fits", 26, 30, filt="Ha", imagetyp="FLAT")
    from configs import config
    monkeypatch.setattr(config, "data", lambda: {"calibration": {
        "bias_dir": str(bdir), "dark_dir": str(ddir), "flat_root": str(froot)}})
    stacker._FLATS_BY_FILTER_CACHE.pop(str(froot), None)
    light = _fits(tmp_path / "light.fits", 26, 30, filt="Ha")
    b, d, f = stacker.calibration_paths_from_config("Ha", like=light)
    assert [p.name for p in b] == ["b26.fits"] and [p.name for p in d] == ["d26.fits"] and [p.name for p in f] == ["f26.fits"]
    b, d, f = stacker.calibration_paths_from_config("Ha")            # no light named: everything, as before
    assert len(b) == 2 and len(d) == 2 and len(f) == 2


def test_new_session_bias_and_darks_are_found_and_matched_by_gain(tmp_path, monkeypatch):
    # 2026-10-28: gain-26 bias/darks land in cdk17/<date>/BIAS|DARK, which a
    # single configured folder never looked in; they would have been missed
    # and the narrowband stacked uncalibrated.
    root = tmp_path / "cdk17"
    for sess, g, o in (("2026-07-25", 0, 10), ("2026-10-28", 26, 30)):
        for kind in ("BIAS", "DARK"):
            (root / sess / kind).mkdir(parents=True)
            _fits(root / sess / kind / ("%s_%d.fits" % (kind.lower(), g)), g, o, imagetyp=kind)
    from configs import config
    monkeypatch.setattr(config, "data", lambda: {"calibration": {
        "bias_dir": str(root / "2026-07-25" / "BIAS"), "dark_dir": str(root / "2026-07-25" / "DARK"),
        "flat_root": str(root)}})
    stacker._FLATS_BY_FILTER_CACHE.pop(str(root), None)
    nb = _fits(tmp_path / "ha.fits", 26, 30, filt="Ha")
    b, d, _f = stacker.calibration_paths_from_config("Ha", like=nb)
    assert [p.name for p in b] == ["bias_26.fits"] and [p.name for p in d] == ["dark_26.fits"]
    bb = _fits(tmp_path / "l.fits", 0, 10, filt="L")
    b, d, _f = stacker.calibration_paths_from_config("L", like=bb)
    assert [p.name for p in b] == ["bias_0.fits"] and [p.name for p in d] == ["dark_0.fits"]   # no duplicates


def test_flats_are_gain_matched_before_the_newest_sets_cut(tmp_path, monkeypatch):
    root = tmp_path / "cdk17"
    for sess, g, o in (("2026-10-20", 26, 30), ("2026-10-25", 0, 10), ("2026-10-26", 0, 10), ("2026-10-27", 0, 10)):
        (root / sess / "FLAT").mkdir(parents=True)
        _fits(root / sess / "FLAT" / "f.fits", g, o, filt="Ha", imagetyp="FLAT")
    from configs import config
    monkeypatch.setattr(config, "data", lambda: {"calibration": {"flat_root": str(root), "flat_sets": 3}})
    stacker._FLATS_BY_FILTER_CACHE.pop(str(root), None)
    _b, _d, f = stacker.calibration_paths_from_config("Ha", like=_fits(tmp_path / "ha.fits", 26, 30, filt="Ha"))
    assert [p.parent.parent.name for p in f] == ["2026-10-20"]       # the only gain-26 set, not cut away


def test_the_house_gain_wins_over_the_majority_when_any_light_is_at_it(tmp_path):
    """2026-11-21: IC 405's first gain-26 night put 33 Ha frames beside 47 at
    gain 0; the majority rule stacked the old ones and rejected every new one."""
    old = [_fits(tmp_path / f"o{i}.fits", 0, 10) for i in range(4)]
    new = [_fits(tmp_path / f"n{i}.fits", 26, 30) for i in range(2)]
    arch = _fits(tmp_path / "arch.fits")                       # no GAIN header: stacks either way
    gain, kept, other = stacker.select_gain(old + new + [arch], house=(26, 30))
    assert gain == (26, 30) and kept == new + [arch] and other == old
    # nothing at the house gain yet: the majority, as before
    gain, kept, other = stacker.select_gain(old + [arch], house=(26, 30))
    assert gain == (0, 10) and other == []
    # no house rule for this filter: the majority
    gain, kept, other = stacker.select_gain(old + new, house=None)
    assert gain == (0, 10) and other == new


def test_house_gain_reads_the_gain_table(monkeypatch):
    from configs import config
    monkeypatch.setattr(config, "data", lambda: {"nina": {"gain": {"Ha": [26, 30], "S-II": [26, -1],
                                                                   "O-III": 26, "B": "x"}}})
    assert stacker.house_gain("Ha") == (26, 30) and stacker.house_gain("ha") == (26, 30)
    assert stacker.house_gain("S-II") == (26, None) and stacker.house_gain("O-III") == (26, None)
    assert stacker.house_gain("B") is None and stacker.house_gain("L") is None
    assert stacker.house_gain(None) is None


def test_calibration_follows_the_stack_gain_not_the_first_light(tmp_path, monkeypatch):
    """2026-11-24: IC 405's lights ran gain 0 for five nights before Ha moved to
    26. Every caller asked for calibration like paths[0], the oldest frame, so
    the gain-0 masters were handed to a stack built at 26 and dropped there:
    the nebula rendered with no bias and no flat while gain-26 flats sat on
    disk. The list decides by the stack's own rule."""
    bdir, ddir, froot = tmp_path / "bias", tmp_path / "dark", tmp_path / "flats"
    (froot / "2026-11-23" / "FLAT").mkdir(parents=True)
    bdir.mkdir(); ddir.mkdir()
    _fits(bdir / "b0.fits", 0, 10, imagetyp="BIAS"); _fits(bdir / "b26.fits", 26, 30, imagetyp="BIAS")
    _fits(ddir / "d0.fits", 0, 10, imagetyp="DARK")
    _fits(froot / "2026-11-23" / "FLAT" / "f0.fits", 0, 10, filt="Ha", imagetyp="FLAT")
    _fits(froot / "2026-11-23" / "FLAT" / "f26.fits", 26, 30, filt="Ha", imagetyp="FLAT")
    from configs import config
    cfg = {"calibration": {"bias_dir": str(bdir), "dark_dir": str(ddir), "flat_root": str(froot)},
           "nina": {"gain": {"Ha": [26, 30]}}}
    monkeypatch.setattr(config, "data", lambda: cfg)
    stacker._FLATS_BY_FILTER_CACHE.pop(str(froot), None)
    old = [_fits(tmp_path / ("old%d.fits" % i), 0, 10, filt="Ha") for i in range(3)]
    new = [_fits(tmp_path / ("new%d.fits" % i), 26, 30, filt="Ha") for i in range(2)]
    # the house gain: 26 wins though the first light, and the majority, are at 0
    assert stacker.stack_like(old + new, "Ha") == new[0]
    b, d, f = stacker.calibration_paths_from_config("Ha", like=old + new)
    assert [p.name for p in b] == ["b26.fits"] and d == [] and [p.name for p in f] == ["f26.fits"]
    # no house gain for the filter: the majority, as stack() would choose
    cfg["nina"] = {}
    assert stacker.stack_like(old + new, "Ha") == old[0]
    b, d, f = stacker.calibration_paths_from_config("Ha", like=old + new)
    assert [p.name for p in b] == ["b0.fits"] and [p.name for p in d] == ["d0.fits"] and [p.name for p in f] == ["f0.fits"]
    # one path still works as before
    b, _d, _f = stacker.calibration_paths_from_config("Ha", like=new[0])
    assert [p.name for p in b] == ["b26.fits"]
