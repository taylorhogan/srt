"""Light curves for monitored stars: one calibrated point per night, any target.

    lightcurve <dso> [filter]            (web chat; also run at the end of a night
                                          for every imaged target whose queue entry
                                          has "lightcurve": true)

Built 2026-10-09 for Hubble's Cepheid V1 in M31, but nothing here knows about
V1: the target is wherever the queue entry's ra_deg/dec_deg point (or the name
resolves), the comparison stars come from Gaia DR3 around it, and an optional
"period_days" / "epoch_jd" on the entry adds a phased plot.

Per frame: plate-solve (ASTAP, the same helper the HR diagram uses), put a
3 arcsec aperture with a 6-10 arcsec sky annulus on the target and on each
comparison star, and zero-point the frame with a sigma-clipped median of
(Gaia G - instrumental) over the comparisons. The night's point is the mean of
the frames, its error the larger of the scatter between frames and the
per-frame photon+zero-point error. The scale is therefore "Gaia G" for every
filter -- consistent night to night, which is what a period needs; it is NOT
Johnson V, and a color term would be needed to compare with V-band literature.

Comparison stars are chosen once per target and cached
(local/lightcurves/<dso>_comps.json), so later nights do not depend on the
network; delete the cache to re-select. Raw frames are measured without
bias/dark/flat: the annulus removes the local background and the comparisons
share the frame, so what is left is the small-scale flat error, well below the
noise at magnitude 19.

Records: local/lightcurves/<dso>.json, one per (night, filter), replaced when
a night is re-run. Plot: local/lightcurves/<dso>.png.

Several stars in one field (2026-10-15, NGC 7790's Cepheids): a "stars" list
on the queue entry, each {"name", "ra_deg", "dec_deg"[, "period_days",
"epoch_jd"]}. Each frame is solved once and every star measured on it; each
star keeps its own files, keyed <dso>_<name> (e.g. ngc7790_cf_cas.json). No
list = one star at the entry's own coordinates, keyed <dso>, as before. The
program stars are never used as each other's comparisons.

Short frames (the entry's exposure_s): COMP_MAG is what is unsaturated and
well measured at 300 s, so the range moves by 2.5 log10(300 / EXPTIME) for
each frame -- 3.7 mag brighter at 10 s. The Gaia rows are cached
(<key>_gaia.json) and the comparison list per range (<key>_comps.json at
300 s, the name it always had; <key>_comps_<lo>_<hi>.json otherwise).
"""
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

AP_R_ARCSEC = 3.0
ANNULUS_ARCSEC = (6.0, 10.0)
COMP_MAG = (13.5, 17.0)          # Gaia G: unsaturated at 300 s, still well measured
REF_EXPOSURE_S = 300.0           # the exposure COMP_MAG was chosen for
CROWD_DEPTH_MAG = 2.5            # catalog depth past the faint comp limit, for the crowding test
COMP_RADIUS_ARCMIN = 9.0         # search radius around the target
COMP_MIN_SEP_ARCSEC = 8.0        # no neighbour of similar brightness inside this
MIN_COMPS = 5
MAX_COMPS = 25
FOV_HEIGHT_DEG = 0.47            # ASTAP's -fov is the frame height
SCALE = "Gaia G zero point"

_ROOT = Path(__file__).resolve().parent.parent
LC_DIR = _ROOT / "local" / "lightcurves"


# --------------------------------------------------------------------------- pure

def zero_point(cat_mags, inst_mags, clip: float = 2.5):
    """(zp, scatter, n_used): sigma-clipped median of cat - inst. None if too few."""
    import numpy as np
    d = np.array([c - i for c, i in zip(cat_mags, inst_mags) if c is not None and i is not None
                  and math.isfinite(c) and math.isfinite(i)], dtype=float)
    if d.size == 0:
        return None, None, 0
    keep = np.ones(d.size, bool)
    for _ in range(3):
        med = np.median(d[keep])
        mad = 1.4826 * np.median(np.abs(d[keep] - med)) or 0.02
        keep = np.abs(d - med) <= clip * mad
        if keep.sum() < 3:
            break
    return float(np.median(d[keep])), float(np.std(d[keep])) if keep.sum() > 1 else 0.0, int(keep.sum())


def comp_range(exptime_s) -> tuple:
    """Gaia G range of usable comparisons for a frame of *exptime_s*. Pure.

    COMP_MAG shifted by 2.5 log10(300 / exptime), rounded to 0.1 mag so the
    cache key is stable; COMP_MAG itself at (or within 0.05 mag of) 300 s or
    when the exposure is unknown.
    """
    try:
        e = float(exptime_s)
    except (TypeError, ValueError):
        return COMP_MAG
    if not (e > 0 and math.isfinite(e)):
        return COMP_MAG
    off = 2.5 * math.log10(REF_EXPOSURE_S / e)
    if abs(off) < 0.05:
        return COMP_MAG
    return (round(COMP_MAG[0] - off, 1), round(COMP_MAG[1] - off, 1))


def select_comps(stars: list, target_ra: float, target_dec: float, mag_range=COMP_MAG,
                 min_sep_arcsec: float = COMP_MIN_SEP_ARCSEC, exclude_arcsec: float = 5.0,
                 max_comps: int = MAX_COMPS, exclude: Optional[list] = None) -> list:
    """Comparison stars from catalog rows {ra, dec, gmag, ruwe, var}. Pure.

    Keeps stars in the magnitude range, not flagged variable, with sound
    astrometry (ruwe < 1.4), not the target itself, and with no catalog
    neighbour within *min_sep_arcsec* brighter than gmag + 2 (crowding spoils
    the aperture). Closest to the target first, so the same ones are used
    whatever the frame's edge did. *exclude* is more (ra, dec) positions to
    keep out the same way as the target: the other program stars.
    """
    def sep(a, b):
        dra = (a["ra"] - b["ra"] + 180.0) % 360.0 - 180.0     # across RA 0h (CG Cas is at 0.25 deg)
        return math.hypot(dra * math.cos(math.radians(a["dec"])), a["dec"] - b["dec"]) * 3600.0
    tgt = {"ra": target_ra, "dec": target_dec}
    good = []
    for s in stars:
        g = s.get("gmag")
        if g is None or not (mag_range[0] <= g <= mag_range[1]):
            continue
        if str(s.get("var") or "").upper() == "VARIABLE":
            continue
        if s.get("ruwe") is not None and s["ruwe"] >= 1.4:
            continue
        if sep(s, tgt) < exclude_arcsec:
            continue
        if any(sep(s, {"ra": e[0], "dec": e[1]}) < exclude_arcsec for e in (exclude or [])):
            continue
        crowded = any(o is not s and o.get("gmag") is not None and o["gmag"] < g + 2.0
                      and sep(s, o) < min_sep_arcsec for o in stars)
        if crowded:
            continue
        good.append(dict(s, dist_arcsec=sep(s, tgt)))
    good.sort(key=lambda s: s["dist_arcsec"])
    return good[:max_comps]


def nightly_point(frames: list) -> Optional[dict]:
    """One point from per-frame (jd, mag, err). Pure; None without frames."""
    import numpy as np
    fr = [(j, m, e) for j, m, e in frames if m is not None and math.isfinite(m)]
    if not fr:
        return None
    jd = np.array([f[0] for f in fr]); mag = np.array([f[1] for f in fr]); err = np.array([f[2] for f in fr])
    n = len(fr)
    scatter = float(np.std(mag, ddof=1)) if n > 1 else float(err[0])
    photometric = float(np.sqrt(np.mean(err ** 2)) / math.sqrt(n))
    return {"jd": float(np.mean(jd)), "mag": float(np.mean(mag)),
            "err": round(max(scatter / math.sqrt(n), photometric), 4), "n": n,
            "frame_scatter": round(scatter, 4)}


def merge_record(records: list, new: dict) -> list:
    """Replace the record with the same night and filter, keep the rest, sort. Pure."""
    out = [r for r in records if not (r["night"] == new["night"] and r["filter"] == new["filter"])]
    out.append(new)
    return sorted(out, key=lambda r: (r["jd"], r["filter"]))


def phase(jd: float, period: float, epoch: float) -> float:
    return ((jd - epoch) / period) % 1.0


COVERAGE_BINS = 10          # phase bins a curve is judged on
COVERAGE_MIN_BINS = 8       # bins with a point before the curve counts as drawn


def coverage(records: list, period, epoch, bins: int = COVERAGE_BINS, min_bins: int = COVERAGE_MIN_BINS,
             filt: str = "L") -> dict:
    """How much of the star's cycle the nightly points cover. Pure.

    {"bins", "hit", "nights", "missing", "complete"}: the points (the *filt*
    series when it has any, else every filter) folded on the period into
    *bins* phase bins; complete when at least *min_bins* of them hold a
    point. This is what "converged" means for a light-curve monitor (owner,
    2026-10-31): a curve it can draw, nothing to do with the stack's SNR.
    A star with no period cannot complete: its curve has no end.
    """
    recs = [r for r in records or [] if r.get("jd") is not None]
    mine = [r for r in recs if r.get("filter") == filt] or recs
    out = {"bins": bins, "hit": 0, "nights": len({r.get("night") or r["jd"] for r in mine}),
           "missing": list(range(bins)), "complete": False}
    if not period or float(period) <= 0:
        out["why"] = "no period on the entry"
        return out
    hit = {min(int(phase(float(r["jd"]), float(period), float(epoch or 0.0)) * bins), bins - 1) for r in mine}
    out["hit"] = len(hit)
    out["missing"] = sorted(set(range(bins)) - hit)
    out["complete"] = len(hit) >= min_bins
    return out


def night_of(path: Path) -> str:
    """The N.I.N.A session folder date: <dso>/<rig>/<YYYY-MM-DD>/LIGHT/frame."""
    for part in reversed(path.parts):
        if len(part) == 10 and part[4] == "-" and part[7] == "-":
            return part
    return "unknown"


# --------------------------------------------------------------------------- catalog

def gaia_comps(ra: float, dec: float, radius_arcmin: float = COMP_RADIUS_ARCMIN,
               g_limit: float = COMP_MAG[1] + CROWD_DEPTH_MAG) -> list:
    """Gaia DR3 stars around the target, brighter than *g_limit*, as plain dicts (network)."""
    from astroquery.vizier import Vizier
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    v = Vizier(columns=["Source", "RA_ICRS", "DE_ICRS", "Gmag", "BPmag", "RPmag", "RUWE", "VarFlag"],
               column_filters={"Gmag": "<%.1f" % g_limit}, row_limit=5000)
    res = v.query_region(SkyCoord(ra * u.deg, dec * u.deg), radius=radius_arcmin * u.arcmin, catalog="I/355/gaiadr3")
    if not res:
        return []
    out = []
    for r in res[0]:
        def f(k):
            try:
                x = float(r[k]); return x if math.isfinite(x) else None
            except Exception:
                return None
        out.append({"source": str(r["Source"]), "ra": f("RA_ICRS"), "dec": f("DE_ICRS"), "gmag": f("Gmag"),
                    "bp_rp": (f("BPmag") - f("RPmag")) if f("BPmag") is not None and f("RPmag") is not None else None,
                    "ruwe": f("RUWE"), "var": str(r["VarFlag"])})
    return out


def _gaia_rows(key: str, ra: float, dec: float, g_limit: float, progress_cb=None) -> list:
    """Gaia rows around (ra, dec) to *g_limit*, cached as <key>_gaia.json."""
    cache = LC_DIR / ("%s_gaia.json" % key)
    if cache.exists():
        doc = json.loads(cache.read_text())
        if float(doc.get("g_limit", 0)) >= g_limit:
            return doc["rows"]
    if progress_cb:
        progress_cb("choosing comparison stars from Gaia DR3…")
    rows = gaia_comps(ra, dec, g_limit=g_limit)
    cache.write_text(json.dumps({"ra": ra, "dec": dec, "g_limit": g_limit, "rows": rows}))
    return rows


def comps_for(dso: str, ra: float, dec: float, progress_cb=None, mag_range=COMP_MAG,
              exclude: Optional[list] = None) -> list:
    """Cached comparison list for *dso* (a star's key) and *mag_range*, selected on first use."""
    LC_DIR.mkdir(parents=True, exist_ok=True)
    mag_range = tuple(mag_range)
    if mag_range == tuple(COMP_MAG):
        cache = LC_DIR / ("%s_comps.json" % dso)
    else:
        cache = LC_DIR / ("%s_comps_%.1f_%.1f.json" % (dso, mag_range[0], mag_range[1]))
    if cache.exists():
        return json.loads(cache.read_text())
    rows = _gaia_rows(dso, ra, dec, max(COMP_MAG[1], mag_range[1]) + CROWD_DEPTH_MAG, progress_cb)
    comps = select_comps(rows, ra, dec, mag_range=mag_range, exclude=exclude)
    cache.write_text(json.dumps(comps, indent=1))
    return comps


# --------------------------------------------------------------------------- frames

def _aperture_mags(data, wcs, positions_radec: list, arcsec_per_px: float):
    """[(inst_mag, err_mag)] per sky position; (None, None) off-frame or no flux."""
    import numpy as np
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
    r = AP_R_ARCSEC / arcsec_per_px
    r_in, r_out = (a / arcsec_per_px for a in ANNULUS_ARCSEC)
    coords = SkyCoord([p[0] for p in positions_radec] * u.deg, [p[1] for p in positions_radec] * u.deg)
    x, y = wcs.world_to_pixel(coords)
    out = []
    h, w = data.shape
    for xi, yi in zip(np.atleast_1d(x), np.atleast_1d(y)):
        if not (r_out < xi < w - r_out and r_out < yi < h - r_out) or not (math.isfinite(xi) and math.isfinite(yi)):
            out.append((None, None)); continue
        ap = CircularAperture((xi, yi), r=r)
        an = CircularAnnulus((xi, yi), r_in=r_in, r_out=r_out)
        mask = an.to_mask(method="center")
        ann = mask.multiply(data)
        ann = ann[mask.data > 0]
        ann = ann[np.isfinite(ann)]
        if ann.size < 20:
            out.append((None, None)); continue
        sky_med = float(np.median(ann)); sky_std = float(1.4826 * np.median(np.abs(ann - sky_med)))
        tot = float(aperture_photometry(data, ap)["aperture_sum"][0])
        flux = tot - sky_med * ap.area
        noise = sky_std * math.sqrt(ap.area) * math.sqrt(1 + ap.area / ann.size)
        if not math.isfinite(flux) or flux <= 0 or noise <= 0:
            out.append((None, None)); continue
        out.append((-2.5 * math.log10(flux), 1.0857 * noise / flux))
    return out


def solve_frame(path: Path, near, astap_exe: str, progress_cb=None):
    """-> (data, header, wcs) for one frame, or None if it does not plate-solve."""
    import numpy as np
    from astropy.io import fits
    from photometry.cmd_diagram import _plate_solve_wcs
    with fits.open(path) as h:
        data = np.asarray(h[0].data, dtype=np.float32)
        hdr = h[0].header
    wcs = _plate_solve_wcs(data, near[0], near[1], FOV_HEIGHT_DEG, astap_exe, progress_cb)
    if wcs is None:
        return None
    return data, hdr, wcs


def measure_frame(path: Path, target, comps: list, arcsec_per_px: float, astap_exe: str, progress_cb=None):
    """-> {jd, mag, err, zp, zp_scatter, n_comps} for one frame, or None (unsolved / off-frame)."""
    solved = solve_frame(path, target, astap_exe, progress_cb)
    if solved is None:
        return None
    return measure_star(solved[0], solved[1], solved[2], path, target, comps, arcsec_per_px)


def measure_star(data, hdr, wcs, path: Path, target, comps: list, arcsec_per_px: float):
    """One star on an already-solved frame: the measure_frame record, or None."""
    from astropy.time import Time
    exp = float(hdr.get("EXPTIME", 0) or 0)
    jd = Time(hdr["DATE-OBS"], format="isot", scale="utc").jd + exp / 2 / 86400.0
    mags = _aperture_mags(data, wcs, [target] + [(c["ra"], c["dec"]) for c in comps], arcsec_per_px)
    t_mag, t_err = mags[0]
    if t_mag is None:
        return None
    zp, zs, n = zero_point([c["gmag"] for c in comps], [m for m, _e in mags[1:]])
    if zp is None or n < MIN_COMPS:
        return None
    return {"jd": jd, "mag": t_mag + zp, "err": math.sqrt(t_err ** 2 + (zs / math.sqrt(n)) ** 2),
            "zp": zp, "zp_scatter": zs, "n_comps": n, "file": path.name}


# --------------------------------------------------------------------------- run

def _target_for(dso: str):
    from control import instructions
    rec = instructions.get_instruction_by_dso(dso) or {}
    if rec.get("ra_deg") is not None and rec.get("dec_deg") is not None:
        return (float(rec["ra_deg"]), float(rec["dec_deg"])), rec
    t = instructions.resolve_target_by_name(dso)
    if t is None:
        raise ValueError("cannot resolve %s" % dso)
    return (float(t.coord.ra.deg), float(t.coord.dec.deg)), rec


def _frames(image_dir: Path, dso: str, filt: Optional[str]) -> dict:
    """{(night, filter): [paths]} of LIGHT frames for *dso*."""
    from astropy.io import fits
    key = dso.lower().replace(" ", "")
    root = next((d for d in image_dir.iterdir() if d.is_dir() and d.name.lower().replace(" ", "") == key), None)
    out = {}
    if root is None:
        return out
    for f in sorted(root.rglob("*.fits")):
        if f.parent.name.upper() != "LIGHT":
            continue
        try:
            fl = str(fits.getheader(f).get("FILTER", "")).strip() or "?"
        except Exception:
            continue
        if filt and fl.lower() != filt.lower():
            continue
        out.setdefault((night_of(f), fl), []).append(f)
    return out


def load(dso: str) -> dict:
    p = LC_DIR / ("%s.json" % dso)
    if p.exists():
        return json.loads(p.read_text())
    return {"dso": dso, "scale": SCALE, "records": []}


def save(dso: str, doc: dict) -> Path:
    LC_DIR.mkdir(parents=True, exist_ok=True)
    p = LC_DIR / ("%s.json" % dso)
    p.write_text(json.dumps(doc, indent=1))
    return p


def _slug(name: str) -> str:
    out = "".join(c if c.isalnum() else "_" for c in str(name).lower())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def stars_for(dso: str, target, rec: dict) -> list:
    """The stars to measure for a queue entry. Pure.

    [{key, name, ra, dec, period_days, epoch_jd}]: the entry's "stars" list,
    each keyed <dso>_<name>; without one, the entry itself keyed <dso> (the
    files hubblev1 has always had). A malformed star is skipped, not guessed.
    """
    listed = rec.get("stars") if isinstance(rec, dict) else None
    if not listed:
        return [{"key": dso, "name": dso, "ra": float(target[0]), "dec": float(target[1]),
                 "period_days": rec.get("period_days"), "epoch_jd": rec.get("epoch_jd")}]
    out = []
    for s in listed:
        try:
            name = str(s["name"])
            out.append({"key": "%s_%s" % (dso, _slug(name)), "name": name,
                        "ra": float(s["ra_deg"]), "dec": float(s["dec_deg"]),
                        "period_days": s.get("period_days"), "epoch_jd": s.get("epoch_jd")})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def run(dso: str, filt: Optional[str] = None, nights: Optional[list] = None, redo: bool = False,
        progress_cb: Optional[Callable[[str], None]] = None, cancel_cb: Optional[Callable[[], bool]] = None) -> dict:
    """Measure every (night, filter) of *dso* not yet in the file (or *nights*, or all with redo).

    Every star of the entry (stars_for) on every frame, the frame solved once.
    Returns {"stars": [{name, key, new: [records], doc, plot: path or None,
    comps: n}]}, one per star in the entry's order; comps is the most
    comparison stars any of its frames' exposures had.
    """
    from configs import config
    cfg = config.data()
    image_dir = Path(cfg["nina"]["image_dir"])
    aps = float(cfg["nina"]["arc_sec_per_pixel"])
    astap = (cfg.get("hardware") or {}).get("astap_exe") or "C:/Program Files/astap/astap.exe"
    target, rec = _target_for(dso)
    stars = stars_for(dso, target, rec)
    if not stars:
        raise ValueError("%s lists no usable stars" % dso)
    docs = {s["key"]: load(s["key"]) for s in stars}
    done = {s["key"]: {(r["night"], r["filter"]) for r in docs[s["key"]]["records"]} for s in stars}
    new = {s["key"]: [] for s in stars}
    n_comps = {s["key"]: 0 for s in stars}
    memo = {}

    def comps_at(star, exptime):
        rng = comp_range(exptime)
        k = (star["key"], rng)
        if k not in memo:
            others = [(o["ra"], o["dec"]) for o in stars if o is not star]
            memo[k] = comps_for(star["key"], star["ra"], star["dec"], progress_cb, rng, exclude=others)
            n_comps[star["key"]] = max(n_comps[star["key"]], len(memo[k]))
        return memo[k]

    for (night, fl), paths in sorted(_frames(image_dir, dso, filt).items()):
        if nights and night not in nights:
            continue
        need = [s for s in stars if redo or nights or (night, fl) not in done[s["key"]]]
        if not need:
            continue
        if progress_cb:
            progress_cb("%s %s %s: measuring %d frames, %d star%s…" % (
                dso, night, fl, len(paths), len(need), "" if len(need) == 1 else "s"))
        per = {s["key"]: [] for s in need}
        for p in paths:
            if cancel_cb and cancel_cb():
                raise RuntimeError("cancelled")
            solved = solve_frame(p, target, astap, None)
            if solved is None:
                continue
            data, hdr, wcs = solved
            for s in need:
                m = measure_star(data, hdr, wcs, p, (s["ra"], s["dec"]),
                                 comps_at(s, hdr.get("EXPTIME")), aps)
                if m:
                    per[s["key"]].append(m)
        for s in need:
            got = per[s["key"]]
            pt = nightly_point([(m["jd"], m["mag"], m["err"]) for m in got])
            if pt is None:
                if progress_cb:
                    progress_cb("%s %s %s: no usable frames for %s (unsolved, off frame, or too few comparisons)"
                                % (dso, night, fl, s["name"]))
                continue
            record = dict(pt, night=night, filter=fl, frames_total=len(paths),
                          zp_scatter=round(float(sum(m["zp_scatter"] for m in got) / len(got)), 4),
                          n_comps=int(min(m["n_comps"] for m in got)),
                          measured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            doc = docs[s["key"]]
            doc["records"] = merge_record(doc["records"], record)
            new[s["key"]].append(record)
            save(s["key"], doc)
    out = []
    for s in stars:
        doc = docs[s["key"]]
        out.append({"name": s["name"], "key": s["key"], "new": new[s["key"]], "doc": doc,
                    "plot": plot(s["key"], doc, s) if doc["records"] else None,
                    "comps": n_comps[s["key"]]})
    return {"stars": out}


# --------------------------------------------------------------------------- report

def plot(dso: str, doc: dict, rec: dict) -> Path:
    """The light curve as local/lightcurves/<dso>.png; *rec* carries the
    period/epoch (and, for one of several stars, its "name" for the title)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
    from astropy.time import Time
    SURF, INK, INK2, GRID = "#1a1a19", "#ffffff", "#c3c2b7", "#3a3a37"
    COL = {"L": "#f2c14e", "G": "#3fb950", "R": "#d95926", "B": "#3987e5", "V": "#3fb950"}
    period = rec.get("period_days"); epoch = rec.get("epoch_jd")
    recs = doc["records"]
    fig, axs = plt.subplots(1, 2 if period else 1, figsize=(16 if period else 11, 6), dpi=100, facecolor=SURF, squeeze=False)
    ax = axs[0][0]
    for fl in sorted({r["filter"] for r in recs}):
        rr = [r for r in recs if r["filter"] == fl]
        t = [Time(r["jd"], format="jd").to_datetime() for r in rr]
        ax.errorbar(t, [r["mag"] for r in rr], yerr=[r["err"] for r in rr], fmt="o", ms=6, color=COL.get(fl, INK2),
                    ecolor=COL.get(fl, INK2), capsize=3, label="%s (%d night%s)" % (fl, len(rr), "" if len(rr) == 1 else "s"))
    ax.invert_yaxis(); ax.set_ylabel("magnitude (%s)" % SCALE, color=INK2); ax.xaxis.set_major_formatter(DateFormatter("%b %d"))
    ax.set_title("%s: one point per night" % (rec.get("name") or dso), color=INK, fontsize=13, loc="left")
    if period:
        px = axs[0][1]
        for fl in sorted({r["filter"] for r in recs}):
            rr = [r for r in recs if r["filter"] == fl]
            ph = [phase(r["jd"], period, epoch or 0.0) for r in rr]
            for off in (0, 1):
                px.errorbar([p + off for p in ph], [r["mag"] for r in rr], yerr=[r["err"] for r in rr], fmt="o", ms=6,
                            color=COL.get(fl, INK2), ecolor=COL.get(fl, INK2), capsize=3, alpha=1 if off == 0 else 0.45)
        px.invert_yaxis(); px.set_xlim(0, 2); px.set_xlabel("phase (period %.2f d%s)" % (period, ", epoch JD %.1f" % epoch if epoch else ""), color=INK2)
        px.set_title("folded on the published period", color=INK, fontsize=13, loc="left")
    for a in axs[0]:
        a.set_facecolor(SURF); a.tick_params(colors=INK2); a.grid(color=GRID, lw=0.6)
        for sp in a.spines.values():
            sp.set_color("#5a5a55")
    ax.legend(loc="best", facecolor=SURF, labelcolor=INK, fontsize=10)
    fig.tight_layout()
    out = LC_DIR / ("%s.png" % dso)
    fig.savefig(out, dpi=100, facecolor=SURF); plt.close(fig)
    return out


def report_star(dso: str, star: dict) -> str:
    """One star's part of a run (an item of run()["stars"])."""
    title = dso if star["name"] == dso else "%s %s" % (dso, star["name"])
    lines = ["Light curve %s (%s; %d comparison stars)" % (title, SCALE, star["comps"])]
    if not star["new"]:
        lines.append("no new nights to measure")
    for r in star["new"]:
        lines.append("  %s %-5s %6.3f ± %.3f  (%d/%d frames, zp scatter %.3f)" % (
            r["night"], r["filter"], r["mag"], r["err"], r["n"], r["frames_total"], r["zp_scatter"]))
    recs = star["doc"]["records"]
    if recs:
        lines.append("%d point%s on file, %s to %s" % (len(recs), "" if len(recs) == 1 else "s", recs[0]["night"], recs[-1]["night"]))
    return "\n".join(lines)


def report(dso: str, result: dict) -> str:
    return "\n\n".join(report_star(dso, s) for s in result["stars"])
