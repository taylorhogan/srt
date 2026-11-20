"""photometry/parallax: the catalogue, the geometry, the fits, and a planted
year of synthetic frames read back to the parallax that was put in."""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from photometry import parallax as P  # noqa: E402  (stdlib at import; numpy/astropy inside the functions)


# --------------------------------------------------------------------------- catalogue (no numpy)

def test_the_catalogue_names_the_two_chosen_stars_with_gaia_numbers_and_reference_files():
    stars = P.catalogue()
    chosen = [s for s in stars if s.get("chosen")]
    assert [s["key"] for s in chosen] == ["ross248", "wolf359"]
    for s in stars:
        g = s["gaia"]
        assert 100 < g["parallax"] < 600 and g["source_id"].isdigit() and "pmra" in g and "pmdec" in g
        assert set(s["plan"]["filters"]) == set(s["plan"]["exposure_s"])
        if s.get("refs"):
            doc = json.loads((P.REFS_DIR / ("%s.json" % s["refs"])).read_text())
            assert doc["catalogue"] == "Gaia DR3" and len(doc["rows"]) > 100
            assert any(r["id"] == g["source_id"] for r in doc["rows"]), "the target itself is in its reference file"
    assert P.star("Ross 248")["key"] == "ross248" and P.star("CN Leo")["key"] == "wolf359"
    assert P.star("barnard's star")["key"] == "barnard" and P.star("nope") is None


def test_queue_fields_point_at_the_star_today_not_at_its_2016_position():
    s = P.star("barnard")
    f = P.queue_fields(s)
    # 10.4"/yr north for ~10.9 years since 2016.0: about 113" = 0.031 deg
    assert f["dec_deg"] - s["gaia"]["dec"] == pytest.approx(10362.4 / 3.6e6 * ((P._now_jd() - P.J2016) / 365.25), abs=1e-5)
    assert f["parallax"] is True and f["parallax_star"] == "barnard" and f["obj_type"] == "star"
    assert f["filter_plan"] == {"R": 1} and f["exposure_s"] == {"R": 2} and f["max_hours"] == 0.25 and f["priority"] == 50
    assert f["gaia"]["parallax"] == pytest.approx(546.976)


def test_list_text_marks_queued_chosen_and_reserve():
    txt = P.list_text(queued={"ross248"})
    assert "ross248" in txt and "queued" in txt and "chosen" in txt and "reserve" in txt and "from 2027-03" in txt


def test_parallax_verdict_words():
    v = P.parallax_verdict("ross248", {"complete": True, "parallax_mas": 318.2, "err_mas": 4.1, "n": 61, "span_days": 380})
    assert v["done"] and v["why"] == ["parallax measured: 318.2 +/- 4.1 mas over 61 nights spanning 380 days"]
    v = P.parallax_verdict("ross248", {"complete": False, "why": ["120 of 365 days of baseline", "9 of 30 nights"]})
    assert not v["done"] and v["why"] == ["parallax monitor: 120 of 365 days of baseline; 9 of 30 nights"]


# --------------------------------------------------------------------------- geometry

def test_tangent_plane_is_arcsec_east_and_north():
    xi, eta = P.tangent(10.0 + 0.1 / math.cos(math.radians(41.0)), 41.0, 10.0, 41.0)
    assert xi == pytest.approx(360.0, abs=0.5) and abs(eta) < 0.5
    xi, eta = P.tangent(10.0, 41.1, 10.0, 41.0)
    assert abs(xi) < 1e-6 and eta == pytest.approx(360.0, abs=0.01)


def test_parallax_factors_trace_the_earths_orbit():
    pytest.importorskip("astropy")
    # at the ecliptic pole the ellipse is a circle of radius ~1 AU
    for jd in (2461365.5, 2461465.5, 2461550.5):
        pa, pd = P.parallax_factors(jd, 270.0, 66.56)
        assert math.hypot(pa, pd) == pytest.approx(1.0, abs=0.02)
    # a star on the ecliptic at quadrature moves ~1 AU in longitude, at conjunction ~0
    pa_q, _ = P.parallax_factors(2461485.5, 0.0, 0.0)      # 2027-03-21: sun at RA 0 -> factor ~0
    assert abs(pa_q) < 0.1


def test_propagate_applies_proper_motion_and_the_stars_own_parallax():
    row = {"ra": 100.0, "dec": 30.0, "pmra": 1000.0, "pmdec": -500.0, "plx": 100.0}
    jd = P.J2016 + 365.25 * 2
    ra, dec = P.propagate(row, jd)
    assert (ra - 100.0) * math.cos(math.radians(30.0)) * 3.6e6 == pytest.approx(2000.0, rel=1e-3)
    assert (dec - 30.0) * 3.6e6 == pytest.approx(-1000.0, rel=1e-3)
    ra2, dec2 = P.propagate(row, jd, factors=(0.5, -0.5))
    assert (ra2 - ra) * math.cos(math.radians(30.0)) * 3.6e6 == pytest.approx(50.0, rel=1e-3)
    assert (dec2 - dec) * 3.6e6 == pytest.approx(-50.0, rel=1e-3)


def test_usable_refs_drops_the_target_variables_bad_ruwe_nearby_and_far_stars():
    rows = [{"id": "t", "ra": 10.0, "dec": 40.0, "g": 10.0, "plx": 300.0},
            {"id": "a", "ra": 10.01, "dec": 40.0, "g": 14.0, "ruwe": 1.0},
            {"id": "b", "ra": 10.02, "dec": 40.0, "g": 14.0, "ruwe": 2.0},
            {"id": "c", "ra": 10.03, "dec": 40.0, "g": 14.0, "var": True},
            {"id": "d", "ra": 10.04, "dec": 40.0, "g": 18.0},
            {"id": "e", "ra": 10.05, "dec": 40.0, "g": 14.0, "plx": 50.0},
            {"id": "f", "ra": 10.5, "dec": 40.0, "g": 14.0}]
    assert [r["id"] for r in P.usable_refs(rows, "t", 10.0, 40.0)] == ["a"]


# --------------------------------------------------------------------------- fits

def test_fit_plate_recovers_an_affine_and_clips_one_bad_star():
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 4000, (40, 2))
    a, b, c, d = 0.26, 0.004, -0.004, 0.26           # 0.26"/px with a small rotation
    xi = a * (xy[:, 0] - 2000) + b * (xy[:, 1] - 1500) + 3.0
    eta = c * (xy[:, 0] - 2000) + d * (xy[:, 1] - 1500) - 2.0
    te = np.column_stack([xi, eta]) + rng.normal(0, 0.01, (40, 2))
    te[7] += (1.5, -2.0)                              # a blend
    plate = P.fit_plate(xy, te, order=1)
    assert plate["used"] == 39 and plate["rms"] < 0.02
    x, y = 1234.5, 2345.6
    assert P.apply_plate(plate, x, y)[0] == pytest.approx(a * (x - 2000) + b * (y - 1500) + 3.0, abs=0.01)
    assert P.apply_plate(plate, x, y)[1] == pytest.approx(c * (x - 2000) + d * (y - 1500) - 2.0, abs=0.01)
    assert P.fit_plate(xy[:4], te[:4]) is None


def _year(ra, dec, plx, pmra, pmdec, n=24, start=2461365.5, noise=0.004, seed=3, every=15):
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(seed)
    pts = []
    for i in range(n):
        jd = start + every * i
        yr = (jd - P.J2016) / 365.25
        pa, pd = P.parallax_factors(jd, ra, dec)
        pts.append({"jd": jd, "xi": (pmra * yr + plx * pa) / 1000.0 + rng.normal(0, noise),
                    "eta": (pmdec * yr + plx * pd) / 1000.0 + rng.normal(0, noise),
                    "sig_xi": noise, "sig_eta": noise, "night": "n%d" % i, "filter": "R"})
    return pts


def test_fit_motion_recovers_a_planted_parallax_and_motion():
    pytest.importorskip("numpy"); pytest.importorskip("astropy")
    ra, dec, plx, pmra, pmdec = 355.48, 44.17, 316.5, 112.5, -1591.6
    pts = _year(ra, dec, plx, pmra, pmdec)
    fit = P.fit_motion(pts, ra, dec)
    assert fit["parallax_mas"] == pytest.approx(plx, abs=3 * fit["parallax_err_mas"] + 1)
    assert fit["parallax_err_mas"] < 4
    assert fit["pmra_mas_yr"] == pytest.approx(pmra, abs=10) and fit["pmdec_mas_yr"] == pytest.approx(pmdec, abs=10)
    assert fit["distance_ly"] == pytest.approx(10.3, abs=0.3) and fit["chi2_red"] < 3
    assert P.fit_motion(pts[:2], ra, dec) is None
    with_dcr = P.fit_motion([dict(p, tanz=0.5 + 0.3 * (i % 3), parallactic=0.4) for i, p in enumerate(pts)], ra, dec, dcr=True)
    assert with_dcr["dcr"] and abs(with_dcr["dcr_mas"]) < 3 * with_dcr["dcr_err"] + 2
    assert with_dcr["parallax_mas"] == pytest.approx(plx, abs=3 * with_dcr["parallax_err_mas"] + 1)


def test_coverage_needs_a_year_thirty_nights_and_five_percent():
    pytest.importorskip("numpy"); pytest.importorskip("astropy")
    ra, dec, plx = 355.48, 44.17, 316.5
    full = _year(ra, dec, plx, 112.5, -1591.6, n=32, every=12)         # 372 days, 32 nights
    cov = P.coverage(full, ra, dec)
    assert cov["complete"] and cov["n"] == 32 and cov["rel_err"] < 0.05
    short = P.coverage(full[:12], ra, dec)
    assert not short["complete"] and any("days of baseline" in w for w in short["why"]) and any("of 30 nights" in w for w in short["why"])
    assert P.coverage([], ra, dec)["why"] == ["0 of 365 days of baseline", "0 of 30 nights", "no parallax fit yet"]
    noisy = P.coverage([dict(p, xi=p["xi"] + (0.3 if i % 2 else -0.3), eta=p["eta"] + (0.3 if i % 3 else -0.3),
                             sig_xi=0.3, sig_eta=0.3) for i, p in enumerate(full)], ra, dec)
    assert not noisy["complete"] and any("needs 5 %" in w for w in noisy["why"])


def test_nightly_epoch_clips_and_floors():
    pytest.importorskip("numpy")
    frames = [{"jd": 2461400.5 + i / 1000.0, "xi": 0.100 + 0.001 * (i % 3), "eta": -0.050, "tanz": 0.4, "n_refs": 30 + i} for i in range(10)]
    frames.append({"jd": 2461400.52, "xi": 0.9, "eta": -0.05, "tanz": 0.4, "n_refs": 12})
    e = P.nightly_epoch(frames)
    assert e["n"] == 10 and e["n_clipped"] == 1 and e["xi"] == pytest.approx(0.101, abs=1e-3)
    assert e["sig_xi"] == pytest.approx(0.003) and e["sig_eta"] == pytest.approx(0.003)    # the floor
    assert e["n_refs"] == 30 and e["tanz"] == pytest.approx(0.4)
    assert P.nightly_epoch(frames[:2]) is None


def test_merge_record_replaces_a_night_and_filter():
    recs = P.merge_record([{"night": "a", "filter": "R", "xi": 1}], {"night": "a", "filter": "R", "xi": 2})
    assert recs == [{"night": "a", "filter": "R", "xi": 2}]
    assert len(P.merge_record(recs, {"night": "b", "filter": "R"})) == 2


# --------------------------------------------------------------------------- a planted year of frames

def _wcs(ra0, dec0, nx=1200, ny=900, scale=1.0 / 3600.0, rot_deg=3.0):
    """A TAN frame of nx x ny at *scale* deg/px (1"/px: 20' x 15', so references
    within 10' of the centre fit on it)."""
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.crpix = [nx / 2 + 0.5, ny / 2 + 0.5]
    w.wcs.crval = [ra0, dec0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    r = math.radians(rot_deg)
    w.wcs.cd = [[-scale * math.cos(r), scale * math.sin(r)], [scale * math.sin(r), scale * math.cos(r)]]
    return w


def _put(img, x, y, flux, sig=4.0):
    import numpy as np
    h, w = img.shape
    r = 18
    x0, y0 = int(round(x)), int(round(y))
    ya, yb, xa, xb = max(y0 - r, 0), min(y0 + r + 1, h), max(x0 - r, 0), min(x0 + r + 1, w)
    if yb <= ya or xb <= xa:
        return
    ys, xs = np.mgrid[ya:yb, xa:xb]
    img[ya:yb, xa:xb] += flux / (2 * np.pi * sig ** 2) * np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sig ** 2))


def test_a_planted_year_of_frames_reads_back_the_parallax(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("astropy")
    rng = np.random.default_rng(11)
    ra0, dec0 = 355.480015, 44.170376
    target = {"ra": ra0, "dec": dec0, "pmra": 112.5, "pmdec": -1591.6, "source_id": "T"}
    plx = 316.5
    # 60 references within 10', each with a small motion and parallax of its own
    refs = []
    for i in range(60):
        d = rng.uniform(0, 9.5) / 60.0
        th = rng.uniform(0, 2 * math.pi)
        refs.append({"id": "r%d" % i, "ra": ra0 + d * math.cos(th) / math.cos(math.radians(dec0)), "dec": dec0 + d * math.sin(th),
                     "g": rng.uniform(12, 16), "pmra": rng.normal(0, 5), "pmdec": rng.normal(0, 5), "plx": rng.uniform(0.2, 3)})
    refs = P.usable_refs(refs, "T", ra0, dec0)
    assert len(refs) == 60
    nights = []
    for k in range(14):
        jd0 = 2461365.5 + 27 * k
        fac = P.parallax_factors(jd0, ra0, dec0)
        frames = []
        for f in range(4):
            jd = jd0 + f * 0.003
            # tonight's pointing wanders by a few pixels and the rotation by a degree
            w = _wcs(ra0 + rng.normal(0, 3) / 3600.0, dec0 + rng.normal(0, 3) / 3600.0, rot_deg=3.0 + rng.normal(0, 0.5))
            img = rng.normal(500.0, 8.0, (900, 1200)).astype(np.float32)
            for r in refs:
                rra, rdec = P.propagate(r, jd, fac)
                x, y = w.world_to_pixel_values(rra, rdec)
                _put(img, float(x), float(y), 10 ** (0.4 * (16.0 - r["g"])) * 3000)   # G 16 at SNR ~15, like 10 s R
            # the target: proper motion AND the parallax ellipse, from its 2016.0 place
            yr = (jd - P.J2016) / 365.25
            dxi = (target["pmra"] * yr + plx * fac[0]) / 1000.0
            deta = (target["pmdec"] * yr + plx * fac[1]) / 1000.0
            tra = ra0 + dxi / 3600.0 / math.cos(math.radians(dec0))
            tdec = dec0 + deta / 3600.0
            x, y = w.world_to_pixel_values(tra, tdec)
            _put(img, float(x), float(y), 2e6)
            m = P.measure_frame(img, w, jd, target, refs, raw=img, saturation=60000.0)
            assert m is not None and not m["saturated"] and m["n_refs"] >= 30 and m["order"] == 2
            assert m["xi"] == pytest.approx(dxi, abs=0.2) and m["eta"] == pytest.approx(deta, abs=0.2)   # 0.2 px at this scale
            frames.append(m)
        e = P.nightly_epoch(frames)
        nights.append(dict(e, night="n%d" % k, filter="R"))
    fit = P.fit_motion(nights, ra0, dec0)
    assert fit["parallax_mas"] == pytest.approx(plx, abs=max(3 * fit["parallax_err_mas"], 10))
    assert fit["pmra_mas_yr"] == pytest.approx(112.5, abs=25) and fit["pmdec_mas_yr"] == pytest.approx(-1591.6, abs=25)
    cov = P.coverage(nights, ra0, dec0)
    assert cov["parallax_mas"] == pytest.approx(fit["parallax_mas"]) and "of 30 nights" in " ".join(cov["why"])


def test_measure_frame_refuses_without_enough_references_and_flags_a_saturated_target():
    np = pytest.importorskip("numpy")
    pytest.importorskip("astropy")
    ra0, dec0 = 164.103190, 7.002727
    target = {"ra": ra0, "dec": dec0, "pmra": 0.0, "pmdec": 0.0}
    w = _wcs(ra0, dec0)
    img = np.full((900, 1200), 300.0, np.float32)
    assert P.measure_frame(img, w, 2461400.5, target, [], raw=img) is None
    refs = [{"id": "r%d" % i, "ra": ra0 + (i % 5 - 2) * 0.01, "dec": dec0 + (i // 5 - 1) * 0.01, "g": 14.0} for i in range(15)]
    for r in refs:
        x, y = w.world_to_pixel_values(r["ra"], r["dec"])
        _put(img, float(x), float(y), 3e5)
    x, y = w.world_to_pixel_values(ra0, dec0)
    _put(img, float(x), float(y), 1e7)                       # peak well past 60000
    m = P.measure_frame(img, w, 2461400.5, target, refs, raw=img, saturation=60000.0)
    assert m is not None and m["saturated"] and m["order"] == 1 and m["n_refs"] == 14    # r7 sits under the target


def test_run_measures_each_night_once_and_fits(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    pytest.importorskip("astropy")
    import types
    import configs
    fake_cfg = types.SimpleNamespace(data=lambda: {"nina": {"image_dir": str(tmp_path), "arc_sec_per_pixel": 0.26},
                                                   "location": {"latitude": 41.8, "longitude": -72.8}})
    monkeypatch.setitem(sys.modules, "configs.config", fake_cfg)
    monkeypatch.setattr(configs, "config", fake_cfg, raising=False)
    monkeypatch.setattr(P, "PLX_DIR", tmp_path / "plx")
    from photometry import lightcurve as lc
    rec = {"parallax": True, "parallax_star": "ross248", "ra_deg": 355.48, "dec_deg": 44.17}
    monkeypatch.setattr(lc, "_target_for", lambda d: ((355.48, 44.17), rec))
    frames = {("2026-11-21", "R"): [Path("C:/x/ross248/cdk/2026-11-21/LIGHT/f%d.fits" % i) for i in range(3)],
              ("2026-11-22", "R"): [Path("C:/x/ross248/cdk/2026-11-22/LIGHT/f%d.fits" % i) for i in range(3)]}
    monkeypatch.setattr(lc, "_frames", lambda *a: frames)
    monkeypatch.setattr(lc, "calibration_for", lambda fl, like, cb=None: None)
    monkeypatch.setattr(lc, "solve_frame", lambda p, near, astap, cb, cal=None: lc.Solved(
        "data", {"EXPTIME": 10.0, "DATE-OBS": "%sT02:00:00" % p.parts[-3]}, "wcs", "raw"))
    monkeypatch.setattr(P, "refs_for", lambda key, ra, dec, cb=None: {"rows": [{"id": "x", "ra": ra, "dec": dec + 0.01, "g": 14.0}] * 20,
                                                                      "source": "test", "has_pm": True})
    monkeypatch.setattr(P, "measure_frame", lambda data, wcs, jd, g, refs, raw=None, site=None: {
        "xi": 0.010, "eta": -0.020, "n_refs": 20, "plate_rms": 0.01, "order": 1, "peak": 3e4, "saturated": False, "jd": jd,
        "tanz": 0.3, "parallactic": 0.1, "airmass": 1.05})
    monkeypatch.setattr(P, "plot", lambda key, doc, entry, fit: tmp_path / "p.png")
    out = P.run("ross248")
    assert out["key"] == "ross248" and [r["night"] for r in out["new"]] == ["2026-11-21", "2026-11-22"]
    assert out["new"][0]["n"] == 3 and out["new"][0]["xi"] == pytest.approx(0.010)
    assert out["fit"] is None                                   # two nights: no fit yet
    assert (tmp_path / "plx" / "ross248.json").exists()
    assert "still needs" in P.report(out) and "2 nights on file" in P.report(out)
    again = P.run("ross248")
    assert again["new"] == []
