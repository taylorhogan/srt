"""Calibration matches the lights' camera gain (2026-10-24).

Bias and darks come from fixed directories and darks were matched on exposure
alone, so the first night at a new gain would have been calibrated with the
old gain's masters and nothing would have said so. Now the lights' majority
(GAIN, OFFSET) is the stack's gain, lights at another gain are rejected, and
calibration at another gain is dropped; a frame with no GAIN header (the 2024
archive) matches anything."""
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

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
