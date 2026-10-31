"""photometry/lightcurve: the pure parts of the nightly light-curve point (2026-10-09)."""
import sys
from pathlib import Path

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


# --------------------------------------------------------------------------- short frames, several stars

def test_comp_range_follows_the_exposure():
    assert lc.comp_range(300) == lc.COMP_MAG
    assert lc.comp_range(None) == lc.COMP_MAG and lc.comp_range(0) == lc.COMP_MAG
    assert lc.comp_range("x") == lc.COMP_MAG
    assert lc.comp_range(10) == (9.8, 13.3)              # 2.5 log10(30) = 3.69 brighter
    assert lc.comp_range(20) == (10.6, 14.1)
    assert lc.comp_range(600) == (14.3, 17.8)            # longer frames reach fainter


def test_select_comps_keeps_the_other_program_stars_out():
    stars = [{"ra": 10.0 + i * 0.01, "dec": 41.0, "gmag": 15.0, "ruwe": 1.0, "var": "NOT_AVAILABLE"}
             for i in range(1, 8)]
    other = (stars[2]["ra"], stars[2]["dec"])
    out = lc.select_comps(stars, 10.0, 41.0, exclude=[other])
    assert len(out) == 6 and all(abs(s["ra"] - other[0]) > 1e-9 for s in out)


def test_stars_for_without_a_list_is_the_entry_itself():
    rec = {"period_days": 31.4, "epoch_jd": 2455430.5}
    (s,) = lc.stars_for("hubblev1", (10.36, 41.17), rec)
    assert s["key"] == "hubblev1" and s["name"] == "hubblev1" and s["period_days"] == 31.4


def test_stars_for_keys_each_listed_star_and_skips_bad_ones():
    rec = {"stars": [{"name": "CF Cas", "ra_deg": 359.57, "dec_deg": 61.22, "period_days": 4.875},
                     {"name": "CE Cas A+B", "ra_deg": 359.5387, "dec_deg": 61.2137},
                     {"name": "broken"}]}
    out = lc.stars_for("ngc7790", (359.9, 61.09), rec)
    assert [s["key"] for s in out] == ["ngc7790_cf_cas", "ngc7790_ce_cas_a_b"]
    assert out[0]["period_days"] == 4.875 and out[1]["period_days"] is None


def test_comps_cache_keeps_the_300s_name_and_shares_the_catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "LC_DIR", tmp_path)
    calls = []
    rows = [{"ra": 10.0 + i * 0.01, "dec": 41.0, "gmag": g, "ruwe": 1.0, "var": "NOT_AVAILABLE"}
            for i, g in enumerate([10.0, 11.0, 12.0, 14.0, 15.0, 16.0], start=1)]
    monkeypatch.setattr(lc, "gaia_comps", lambda ra, dec, g_limit=19.5: calls.append(g_limit) or rows)
    long_ = lc.comps_for("hubblev1", 10.0, 41.0)
    short = lc.comps_for("hubblev1", 10.0, 41.0, mag_range=lc.comp_range(10))
    assert (tmp_path / "hubblev1_comps.json").exists()
    assert (tmp_path / "hubblev1_comps_9.8_13.3.json").exists()
    assert [s["gmag"] for s in long_] == [14.0, 15.0, 16.0]
    assert sorted(s["gmag"] for s in short) == [10.0, 11.0, 12.0]
    assert calls == [19.5]                                # one catalog query for both ranges


def test_run_solves_each_frame_once_and_measures_every_star(tmp_path, monkeypatch):
    import types
    import configs
    fake_cfg = types.SimpleNamespace(data=lambda: {"nina": {"image_dir": str(tmp_path), "arc_sec_per_pixel": 0.5}})
    monkeypatch.setitem(sys.modules, "configs.config", fake_cfg)
    monkeypatch.setattr(configs, "config", fake_cfg, raising=False)
    monkeypatch.setattr(lc, "LC_DIR", tmp_path / "lc")
    rec = {"stars": [{"name": "CF Cas", "ra_deg": 359.57, "dec_deg": 61.22, "period_days": 4.875, "epoch_jd": 2437022.191},
                     {"name": "CG Cas", "ra_deg": 0.247, "dec_deg": 60.959}]}
    monkeypatch.setattr(lc, "_target_for", lambda d: ((359.911, 61.088), rec))
    frames = [Path("C:/x/ngc7790/cdk/2026-10-16/LIGHT/f%d.fits" % i) for i in range(3)]
    monkeypatch.setattr(lc, "_frames", lambda *a: {("2026-10-16", "L"): frames})
    solved = []
    monkeypatch.setattr(lc, "calibration_for", lambda fl, like, cb=None: None)
    monkeypatch.setattr(lc, "solve_frame",
                        lambda p, near, astap, cb, cal=None: solved.append(p) or lc.Solved("d", {"EXPTIME": 10.0}, "w", "raw"))
    seen_ranges = []

    def fake_comps(key, ra, dec, cb, rng, exclude=None):
        seen_ranges.append((key, rng, tuple(exclude)))
        return [{}] * 8
    monkeypatch.setattr(lc, "comps_for", fake_comps)
    mag = {359.57: 10.9, 0.247: 11.3}
    monkeypatch.setattr(lc, "measure_star", lambda d, h, w, p, t, c, aps, raw=None, site=None, **kw: {
        "jd": 2461330.5, "mag": mag[t[0]], "err": 0.01, "zp": 20.0, "zp_scatter": 0.02, "n_comps": len(c), "file": p.name,
        "peak_adu": 41000.0, "airmass": 1.21})
    monkeypatch.setattr(lc, "plot", lambda key, doc, star: tmp_path / ("%s.png" % key))
    out = lc.run("ngc7790")
    assert solved == frames                               # one solve per frame, not per star
    assert [s["key"] for s in out["stars"]] == ["ngc7790_cf_cas", "ngc7790_cg_cas"]
    assert [s["new"][0]["mag"] for s in out["stars"]] == pytest.approx([10.9, 11.3])
    assert all(rng == (9.8, 13.3) for _k, rng, _e in seen_ranges)
    assert ("ngc7790_cf_cas", (9.8, 13.3), ((0.247, 60.959),)) in seen_ranges
    assert (tmp_path / "lc" / "ngc7790_cg_cas.json").exists()
    rec0 = out["stars"][0]["new"][0]
    assert rec0["peak_adu"] == 41000 and rec0["airmass"] == 1.21 and rec0["n_saturated"] == 0 and rec0["calibrated"] is False
    again = lc.run("ngc7790")                             # nothing new the second time
    assert all(not s["new"] for s in again["stars"]) and len(solved) == 3
    assert "CF Cas" in lc.report("ngc7790", again)


def test_select_comps_measures_distance_across_ra_zero():
    stars = [{"ra": 359.99, "dec": 61.0, "gmag": 15.0, "ruwe": 1.0, "var": "NOT_AVAILABLE"},
             {"ra": 0.2, "dec": 61.0, "gmag": 15.0, "ruwe": 1.0, "var": "NOT_AVAILABLE"}]
    out = lc.select_comps(stars, 0.01, 61.0)
    assert out[0]["ra"] == 359.99 and out[0]["dist_arcsec"] < 40


def test_solved_unpacks_as_three_for_old_callers_and_keeps_the_raw_frame():
    sv = lc.Solved("data", {"EXPTIME": 60.0}, "wcs", "raw")
    data, hdr, wcs = sv
    assert (data, wcs) == ("data", "wcs") and sv.raw == "raw" and len(sv) == 3


def test_airmass_of_altitude():
    assert lc.airmass_of_alt(90.0) == pytest.approx(1.0, abs=1e-3)
    assert lc.airmass_of_alt(30.0) == pytest.approx(2.0, abs=0.01)
    assert lc.airmass_of_alt(0.0) is None and lc.airmass_of_alt(-5.0) is None
    assert lc.airmass(2461345.5, 359.911, 61.088, 41.8096, -72.8305) > 1.0


def test_night_extras_and_split_saturated():
    recs = [{"mag": 11.0, "peak_adu": 30000.0, "airmass": 1.2}, None,
            {"saturated": True, "peak_adu": 65535.0, "jd": 1.0, "file": "f"},
            {"mag": 11.1, "peak_adu": 42000.0, "airmass": 1.4}]
    meas, sat = lc.split_saturated(recs)
    assert len(meas) == 2 and len(sat) == 1
    ex = lc.night_extras(meas, sat, True)
    assert ex == {"n_saturated": 1, "calibrated": True, "peak_adu": 42000, "airmass": 1.3}
    assert lc.night_extras([], [], False) == {"n_saturated": 0, "calibrated": False}


def test_measure_star_drops_a_saturated_target_and_a_saturated_comparison(monkeypatch):
    """The peak is read on the raw frame; the target at the limit is a
    'saturated' answer, a comparison at the limit leaves the zero point."""
    comps = [{"ra": 1.0 + i * 0.01, "dec": 1.0, "gmag": 12.0 + i * 0.1} for i in range(6)]
    peaks = {"target": 65535.0}

    def fake_mags(data, wcs, positions, aps, peak_of=None):
        assert peak_of == "raw"
        out = [(-10.0, 0.01, peaks["target"])]
        for i in range(len(positions) - 1):
            # inst mag such that cat - inst = 20.0 for every comp; the first comp saturated
            out.append((comps[i]["gmag"] - 20.0, 0.01, 65000.0 if i == 0 else 30000.0))
        return out
    monkeypatch.setattr(lc, "_aperture_mags", fake_mags)
    hdr = {"EXPTIME": 60.0, "DATE-OBS": "2026-10-25T02:00:00.000", "AIRMASS": 1.33}
    m = lc.measure_star("d", hdr, "w", Path("f.fits"), (1.0, 1.0), comps, 0.5, raw="raw")
    assert m["saturated"] is True and m["peak_adu"] == 65535.0 and m["file"] == "f.fits"
    peaks["target"] = 40000.0
    m = lc.measure_star("d", hdr, "w", Path("f.fits"), (1.0, 1.0), comps, 0.5, raw="raw")
    assert m["mag"] == pytest.approx(10.0) and m["n_comps"] == 5 and m["comps_saturated"] == 1
    assert m["peak_adu"] == 40000.0 and m["airmass"] == 1.33          # header airmass when no site is given
    m = lc.measure_star("d", hdr, "w", Path("f.fits"), (1.0, 1.0), comps, 0.5, raw="raw", site=(41.8, -72.8))
    assert m["airmass"] is None or m["airmass"] > 1.0                  # computed from the site when given


def test_leavitt_distance_of_hubbles_v1():
    d = lc.leavitt_distance(31.4, 19.0, a_v=0.3, g_to_v=0.3)
    assert d["abs_mag"] == pytest.approx(-5.26, abs=0.01)
    assert d["modulus"] == pytest.approx(24.26, abs=0.01)
    assert d["kpc"] == pytest.approx(711, abs=2) and d["mly"] == pytest.approx(2.32, abs=0.01)
    assert d["pct_err"] == 10
    assert lc.leavitt_distance(None, 19.0) is None and lc.leavitt_distance(31.4, None) is None


def test_mean_magnitude_is_the_intensity_mean():
    recs = [{"filter": "L", "mag": 18.0}, {"filter": "L", "mag": 19.0}, {"filter": "G", "mag": 10.0}]
    m, n = lc.mean_magnitude(recs)
    assert n == 2 and 18.0 < m < 18.5 and m == pytest.approx(18.39, abs=0.02)   # brighter than the 18.5 mean of mags
    assert lc.mean_magnitude([], "L") == (None, 0)


def test_distance_text_needs_the_block_and_says_when_the_fold_is_incomplete():
    recs = [{"filter": "L", "mag": 18.6, "jd": 2461330.0 + i * 1.5, "night": "n%d" % i} for i in range(5)]
    star = {"period_days": 31.4, "epoch_jd": 2455430.5, "distance": {"a_v": 0.3, "g_to_v": 0.3}}
    text = lc.distance_text(recs, star)
    assert "Leavitt" in text and "kpc" in text and "million light years" in text
    assert "fold incomplete" in text
    assert lc.distance_text(recs, {"period_days": 31.4}) is None                 # no distance block: no guess
    assert lc.distance_text(recs, {"distance": {"a_v": 0.3}}) is None             # no period: no law
    full = [{"filter": "L", "mag": 18.6, "jd": 2461330.0 + i * 3.3, "night": "n%d" % i} for i in range(10)]
    assert "fold incomplete" not in lc.distance_text(full, star)


def test_stars_for_carries_the_distance_block():
    one = lc.stars_for("hubblev1", (10.36, 41.17), {"period_days": 31.4, "distance": {"a_v": 0.3}})
    assert one[0]["distance"] == {"a_v": 0.3}
    many = lc.stars_for("ngc7790", (0, 0), {"stars": [{"name": "CF Cas", "ra_deg": 1.0, "dec_deg": 2.0, "distance": {"a_v": 1.5}}]})
    assert many[0]["distance"] == {"a_v": 1.5}
