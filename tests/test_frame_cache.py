"""fits_processing.frame_cache — the first unit-tested seam of the refactor.

This module was chosen to be tested first because it is the function two
science packages used to reach into the god file for; its contract (empty map
on every failure mode, so the stacker measures normally) is exactly the kind
of quiet promise that breaks silently.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fits_processing.frame_cache import load_precomputed_fwhm_stars


def _write_cache(dso_dir, rows):
    (dso_dir / "frame_stats.json").write_text(json.dumps(rows))


def test_matches_by_normalised_path_and_converts_units(tmp_path):
    frame = tmp_path / "LIGHT" / "a.fits"
    frame.parent.mkdir()
    frame.write_bytes(b"")
    _write_cache(tmp_path, [
        # cache key spelled differently (case, separators) than the query path
        {"path": str(frame).upper().replace("\\", "/"),
         "fwhm_arcsec": 2.6, "star_count": 123},
    ])
    out = load_precomputed_fwhm_stars(tmp_path, [frame], arcsec_per_pixel=0.26)
    assert list(out) == [frame]
    fwhm_px, stars = out[frame]
    assert abs(fwhm_px - 10.0) < 1e-9          # 2.6" / 0.26"/px
    assert stars == 123


def test_only_requested_paths_returned(tmp_path):
    a, b = tmp_path / "a.fits", tmp_path / "b.fits"
    _write_cache(tmp_path, [
        {"path": str(a), "fwhm_arcsec": 2.0, "star_count": 10},
        {"path": str(b), "fwhm_arcsec": 3.0, "star_count": 20},
    ])
    out = load_precomputed_fwhm_stars(tmp_path, [a], 0.26)
    assert list(out) == [a]


def test_every_failure_mode_yields_empty_map(tmp_path):
    p = tmp_path / "x.fits"
    # missing cache
    assert load_precomputed_fwhm_stars(tmp_path, [p], 0.26) == {}
    # unreadable cache
    (tmp_path / "frame_stats.json").write_text("{not json")
    assert load_precomputed_fwhm_stars(tmp_path, [p], 0.26) == {}
    # zero pixel scale must not divide, must not match
    _write_cache(tmp_path, [{"path": str(p), "fwhm_arcsec": 2.0, "star_count": 5}])
    assert load_precomputed_fwhm_stars(tmp_path, [p], 0.0) == {}
    # malformed rows are skipped, not fatal
    _write_cache(tmp_path, ["nonsense", {"no_path": 1},
                            {"path": str(p), "fwhm_arcsec": 2.0, "star_count": 5}])
    out = load_precomputed_fwhm_stars(tmp_path, [p], 0.26)
    assert list(out) == [p]


def test_missing_fwhm_reports_zero_px_not_crash(tmp_path):
    p = tmp_path / "x.fits"
    _write_cache(tmp_path, [{"path": str(p), "star_count": 7}])
    out = load_precomputed_fwhm_stars(tmp_path, [p], 0.26)
    assert out[p] == (0.0, 7)
