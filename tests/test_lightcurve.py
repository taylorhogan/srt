"""photometry/lightcurve: the pure parts of the nightly light-curve point (2026-10-09)."""
import pytest

np = pytest.importorskip("numpy")
from photometry import lightcurve as lc  # noqa: E402


def test_zero_point_is_robust_to_one_bad_comparison():
    cat = [14.0, 15.0, 16.0, 14.5, 15.5, 16.5, 15.2]
    inst = [c - 20.0 for c in cat]
    inst[3] += 1.5                       # a comparison with a cosmic ray / neighbor
    zp, scatter, n = lc.zero_point(cat, inst)
    assert abs(zp - 20.0) < 0.02 and n == 6 and scatter < 0.02
    assert lc.zero_point([], []) == (None, None, 0)
    assert lc.zero_point([14.0, None], [None, -6.0]) == (None, None, 0)


def _star(ra, dec, g, var="NOT_AVAILABLE", ruwe=1.0):
    return {"ra": ra, "dec": dec, "gmag": g, "var": var, "ruwe": ruwe}


def test_select_comps_filters_and_orders_by_distance():
    t_ra, t_dec = 10.0, 41.0
    stars = [_star(10.0, 41.0, 19.0),                       # the target itself
             _star(10.01, 41.0, 15.0),                      # good, 27" away
             _star(10.05, 41.0, 15.5),                      # good, farther
             _star(10.02, 41.0, 12.0),                      # too bright
             _star(10.03, 41.0, 18.5),                      # too faint
             _star(10.04, 41.0, 15.0, var="VARIABLE"),      # variable
             _star(10.06, 41.0, 15.0, ruwe=2.0),            # bad astrometry
             _star(10.07, 41.0, 15.0), _star(10.07, 41.0015, 15.3)]   # a close pair: both crowded
    out = lc.select_comps(stars, t_ra, t_dec)
    assert [s["gmag"] for s in out] == [15.0, 15.5]
    assert out[0]["dist_arcsec"] < out[1]["dist_arcsec"]


def test_select_comps_caps_the_count():
    stars = [_star(10.0 + 0.01 * i, 41.0, 15.0 + (i % 5) * 0.3) for i in range(1, 60)]   # 36" apart
    assert len(lc.select_comps(stars, 10.0, 41.0, max_comps=10)) == 10


def test_nightly_point_error_is_the_larger_of_scatter_and_photometric():
    pt = lc.nightly_point([(2461000.1, 19.00, 0.05), (2461000.2, 19.10, 0.05), (2461000.3, 19.20, 0.05)])
    assert pt["n"] == 3 and abs(pt["mag"] - 19.1) < 1e-9 and abs(pt["jd"] - 2461000.2) < 1e-9
    assert abs(pt["err"] - 0.1 / 3 ** 0.5) < 1e-3          # scatter wins
    pt2 = lc.nightly_point([(1.0, 19.0, 0.3), (2.0, 19.0, 0.3)])
    assert abs(pt2["err"] - 0.3 / 2 ** 0.5) < 1e-3          # photometric wins
    assert lc.nightly_point([]) is None and lc.nightly_point([(1.0, None, 0.1)]) is None


def test_merge_record_replaces_same_night_and_filter():
    recs = [{"night": "2026-10-08", "filter": "L", "jd": 1.0, "mag": 19.0},
            {"night": "2026-10-08", "filter": "G", "jd": 1.1, "mag": 19.5}]
    out = lc.merge_record(recs, {"night": "2026-10-08", "filter": "L", "jd": 1.0, "mag": 19.05})
    assert len(out) == 2 and [r["mag"] for r in out if r["filter"] == "L"] == [19.05]
    out = lc.merge_record(out, {"night": "2026-10-09", "filter": "L", "jd": 2.0, "mag": 19.1})
    assert [r["night"] for r in out] == ["2026-10-08", "2026-10-08", "2026-10-09"]


def test_phase_and_night_of():
    from pathlib import Path
    assert abs(lc.phase(2455430.5 + 31.4 * 3.25, 31.4, 2455430.5) - 0.25) < 1e-9
    assert lc.night_of(Path("C:/x/Targets/hubblev1/cdk17/2026-10-08/LIGHT/f.fits")) == "2026-10-08"
    assert lc.night_of(Path("C:/x/f.fits")) == "unknown"
