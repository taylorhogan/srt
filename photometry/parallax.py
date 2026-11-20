"""A stellar parallax from this telescope: nearby stars against Gaia's frame.

    parallax <dso> [redo]            (web chat; also run at the end of a night
                                      for every imaged target whose queue entry
                                      has "parallax": true)
    parallax add <star>              queue a star from configs/parallax_stars.json
    parallax list                    the catalogue, with each star's season

Built 2026-11-20 (docs/PARALLAX_PLAN.md). The first two stars are Ross 248 and
Wolf 359, chosen by computing, for eighteen nearby stars, how many nights of the
next thirteen months this site's tree line and darkness allow and how well that
sampling separates a parallax from a proper motion; by the camera (both are
unsaturated at 10-15 s in R and have hundreds of Gaia references in the field);
and by the story (Ross 248 will be the Sun's nearest neighbour in 36,000 years;
Wolf 359 is the largest parallax this site can reach).

The measurement, per frame: plate-solve (ASTAP, lightcurve.solve_frame), put the
Gaia DR3 reference stars of the field where the catalogue says they are TONIGHT
(each reference's own proper motion and parallax applied from epoch 2016.0 to
the frame's time), centroid them and the target in a window, fit the pixel
grid to the references' tangent-plane positions (affine, or second order with
enough references) within REF_RADIUS_ARCMIN of the target, and read the
target's position off that fit. The tangent point is the target's own Gaia
position at 2016.0 with NO motion applied, so what comes out is the target's
displacement from where Gaia saw it: a straight line (proper motion) with a
yearly ellipse (parallax) on it. One point per night is the clipped mean of
the frames (nightly_epoch); the fit over nights (fit_motion) solves offset,
proper motion and parallax, optionally with one differential-refraction term
along the parallactic direction (DCR; the R filter keeps it small, which is why
R and not L). A monitor is done (coverage) when the nights span a year, there
are MIN_EPOCHS of them, and the parallax's formal error is under MAX_REL_ERR of
its value; auto-stop reads that and nothing else for a parallax row.

Reference catalogues: configs/parallax_refs/<key>.json (Gaia DR3 to G 17.5
within 15', pulled 2026-11-20 because the ESA archive closes 25 Nov to 2 Dec
for DR4), else the archive, else VizieR (no proper motions there: degraded,
and said so). Records: local/parallax/<key>.json; plot <key>.png.

What the numbers are compared with: the Gaia DR3 parallax on the catalogue
entry. Gaia's is 0.04 mas; ours will be a few mas, so the comparison is a test
of this telescope's astrometry, which is the point.
"""
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

if __package__ is None or __package__ == "":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

_logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
PLX_DIR = _ROOT / "local" / "parallax"
CATALOGUE = _ROOT / "configs" / "parallax_stars.json"
REFS_DIR = _ROOT / "configs" / "parallax_refs"

J2016 = 2457389.0                 # JD of 2016.0, Gaia DR3's reference epoch
REF_RADIUS_ARCMIN = 10.0          # references within this of the target make the local plate
REF_G_MAX = 17.0                  # faintest reference
REF_RUWE_MAX = 1.4                # Gaia's own astrometric quality cut
REF_SNR_MIN = 10.0                # a reference's flux over the window's sky noise, for it to count
REF_PEAK_MIN_SIG = 5.0            # ...and its peak above the sky, in sky sigmas (no noise blobs)
REF_WEIGHT_CAP = 50.0             # plate weights are the flux SNR, capped here
REF_MAX_MOVE_PX = 4.0             # a centroid farther than this from its predicted pixel is a blend or a miss
MIN_REFS = 8                      # fewer and the frame is not measured
SECOND_ORDER_REFS = 25            # from this many references the plate is second order
CENTROID_WIN_PX = 12              # half-size of the centroid window (seeing 2.5" is ~10 px FWHM)
ANNULUS_PX = (20, 28)             # local sky around each star
PLATE_CLIP = 3.0                  # sigma clip on the plate fit
EPOCH_FLOOR_MAS = 3.0             # no nightly point claims better than this (plate systematics)
MIN_FRAMES_PER_EPOCH = 3
MIN_SPAN_DAYS = 365               # coverage: the nights span a year...
MIN_EPOCHS = 30                   # ...with this many of them...
MAX_REL_ERR = 0.05                # ...and the parallax known to 5 %
DCR_MIN_EPOCHS = 15               # the refraction term is fitted only with this many nights
PC_LY = 3.26156                   # light years per parsec


# --------------------------------------------------------------------------- catalogue

def catalogue(path: Path = CATALOGUE) -> list:
    """The stars of configs/parallax_stars.json (empty when the file is missing)."""
    try:
        return list(json.loads(Path(path).read_text(encoding="utf-8")).get("stars") or [])
    except (OSError, ValueError):
        return []


def _slug(name: str) -> str:
    return "".join(c for c in str(name).lower() if c.isalnum())


def star(name: str, stars: Optional[list] = None) -> Optional[dict]:
    """The catalogue entry whose key, name or alias is *name* (case and spaces ignored)."""
    key = _slug(name)
    for s in (stars if stars is not None else catalogue()):
        names = [s.get("key"), s.get("name")] + list(s.get("aliases") or [])
        if key in {_slug(n) for n in names if n}:
            return s
    return None


def queue_fields(entry: dict) -> dict:
    """The fields `parallax add` writes on the queue row for *entry*. Pure.

    ra/dec are Gaia's 2016.0 position moved by the proper motion to now, so
    the scope points at the star and not at where it was ten years ago (Barnard's
    moves 1.7' in that time). The gaia block rides along for the report."""
    g = dict(entry.get("gaia") or {})
    plan = dict(entry.get("plan") or {})
    years = (_now_jd() - J2016) / 365.25
    dec = float(g["dec"]) + float(g.get("pmdec") or 0.0) / 3.6e6 * years
    ra = float(g["ra"]) + float(g.get("pmra") or 0.0) / 3.6e6 * years / max(1e-6, math.cos(math.radians(dec)))
    return {"ra_deg": round(ra % 360.0, 6), "dec_deg": round(dec, 6), "obj_type": "star",
            "parallax": True, "parallax_star": entry.get("key"),
            "filter_plan": dict(plan.get("filters") or {"R": 1}),
            "exposure_s": dict(plan.get("exposure_s") or {}),
            "max_hours": float(plan.get("max_hours") or 0.25),
            "priority": int(plan.get("priority") or 50),
            "gaia": {k: g.get(k) for k in ("source_id", "parallax", "parallax_error", "pmra", "pmdec", "g", "bp_rp")},
            "note": "Parallax monitor (docs/PARALLAX_PLAN.md): %s, Gaia DR3 parallax %.1f mas. %s"
                    % (entry.get("name"), float(g.get("parallax") or 0.0), entry.get("season") or "")}


def _now_jd() -> float:
    return datetime.now(timezone.utc).timestamp() / 86400.0 + 2440587.5


# --------------------------------------------------------------------------- geometry (pure)

def tangent(ra_deg: float, dec_deg: float, ra0_deg: float, dec0_deg: float) -> tuple:
    """Gnomonic (xi, eta) in arcsec of (ra, dec) about the tangent point; xi east, eta north."""
    a, d = math.radians(ra_deg), math.radians(dec_deg)
    a0, d0 = math.radians(ra0_deg), math.radians(dec0_deg)
    cosc = math.sin(d0) * math.sin(d) + math.cos(d0) * math.cos(d) * math.cos(a - a0)
    xi = math.cos(d) * math.sin(a - a0) / cosc
    eta = (math.cos(d0) * math.sin(d) - math.sin(d0) * math.cos(d) * math.cos(a - a0)) / cosc
    return math.degrees(xi) * 3600.0, math.degrees(eta) * 3600.0


def parallax_factors(jd: float, ra_deg: float, dec_deg: float) -> tuple:
    """(P_alpha, P_delta): the displacement in (RA cos dec, Dec) per unit parallax
    at *jd*, from the Earth's barycentric position (astropy's built-in ephemeris)."""
    from astropy.time import Time
    from astropy.coordinates import get_body_barycentric
    import astropy.units as u
    e = get_body_barycentric("earth", Time(jd, format="jd"))
    x, y, z = (e.x.to(u.au).value, e.y.to(u.au).value, e.z.to(u.au).value)
    a, d = math.radians(ra_deg), math.radians(dec_deg)
    pa = x * math.sin(a) - y * math.cos(a)
    pd = x * math.cos(a) * math.sin(d) + y * math.sin(a) * math.sin(d) - z * math.cos(d)
    return float(pa), float(pd)


def propagate(row: dict, jd: float, factors: Optional[tuple] = None) -> tuple:
    """A Gaia row's (ra, dec) at *jd*: its proper motion and, with *factors*
    (parallax_factors at jd), its own parallax applied from 2016.0."""
    ra, dec = float(row["ra"]), float(row["dec"])
    years = (jd - J2016) / 365.25
    dra = float(row.get("pmra") or 0.0) * years          # mas, already x cos dec (Gaia's pmra)
    ddec = float(row.get("pmdec") or 0.0) * years
    if factors is not None and row.get("plx"):
        dra += float(row["plx"]) * factors[0]
        ddec += float(row["plx"]) * factors[1]
    cosd = max(1e-9, math.cos(math.radians(dec)))
    return ra + dra / 3.6e6 / cosd, dec + ddec / 3.6e6


def zenith_terms(jd: float, ra_deg: float, dec_deg: float, lat_deg: float, lon_deg: float) -> tuple:
    """(tan z, parallactic angle in radians, airmass) for the star at *jd* from the site."""
    from astropy.time import Time
    import astropy.units as u
    lst = Time(jd, format="jd").sidereal_time("apparent", longitude=lon_deg * u.deg).deg
    h = math.radians((lst - ra_deg) % 360.0)
    phi, d = math.radians(lat_deg), math.radians(dec_deg)
    sin_alt = math.sin(phi) * math.sin(d) + math.cos(phi) * math.cos(d) * math.cos(h)
    alt = math.asin(max(-1.0, min(1.0, sin_alt)))
    q = math.atan2(math.sin(h), math.tan(phi) * math.cos(d) - math.sin(d) * math.cos(h))
    tanz = 1.0 / max(1e-6, math.tan(alt)) if alt > 0 else None
    sec = 1.0 / max(1e-6, sin_alt) if sin_alt > 0 else None
    am = sec - 0.0018167 * (sec - 1) - 0.002875 * (sec - 1) ** 2 - 0.0008083 * (sec - 1) ** 3 if sec else None
    return tanz, q, am


# --------------------------------------------------------------------------- centroids and the plate (pure, numpy)

def centroid(data, x0: float, y0: float, win: int = CENTROID_WIN_PX, annulus=ANNULUS_PX, iters: int = 4):
    """Windowed centroid about (x0, y0): -> (x, y, peak_above_sky, sky_sigma, snr)
    or None at the edge. The sky is the annulus median, the weights the signal
    above it; the window follows the centroid for *iters* rounds. *snr* is the
    window's flux over the sky noise summed over the window: a G 16 star in a
    10 s R frame is about 15, its centroid good to half a pixel, which is why
    the plate fit takes the SNR as a weight."""
    import numpy as np
    h, w = data.shape
    r_out = max(annulus[1], win) + 2
    x, y = float(x0), float(y0)
    if not (r_out <= x < w - r_out and r_out <= y < h - r_out):
        return None
    ix, iy = int(round(x)), int(round(y))
    patch = np.asarray(data[iy - r_out:iy + r_out + 1, ix - r_out:ix + r_out + 1], dtype=np.float64)
    yy, xx = np.mgrid[-r_out:r_out + 1, -r_out:r_out + 1]
    rr = np.hypot(xx, yy)
    ring = patch[(rr >= annulus[0]) & (rr < annulus[1])]
    sky = float(np.median(ring))
    sig = float(1.4826 * np.median(np.abs(ring - sky))) or 1.0
    for _ in range(iters):
        ix, iy = int(round(x)), int(round(y))
        if not (win + 1 <= ix < w - win - 1 and win + 1 <= iy < h - win - 1):
            return None
        box = np.asarray(data[iy - win:iy + win + 1, ix - win:ix + win + 1], dtype=np.float64) - sky
        box[box < 0] = 0.0
        tot = box.sum()
        if tot <= 0:
            return None
        gy, gx = np.mgrid[-win:win + 1, -win:win + 1]
        x, y = ix + float((box * gx).sum() / tot), iy + float((box * gy).sum() / tot)
    peak = float(box.max())
    snr = float(tot / (sig * box.size ** 0.5))
    return x, y, peak, sig, snr


def _design(x, y, order: int):
    import numpy as np
    cols = [np.ones_like(x), x, y]
    if order >= 2:
        cols += [x * x, x * y, y * y]
    return np.column_stack(cols)


def fit_plate(xy, xieta, order: int = 1, clip: float = PLATE_CLIP, weights=None) -> Optional[dict]:
    """Pixel (x, y) -> tangent plane (xi, eta) in arcsec, least squares with
    sigma clipping and optional per-star *weights* (a bright star's centroid
    is worth more). Pixels are centred and scaled by 1000 for conditioning.
    -> {"order", "x0", "y0", "cx", "cy", "rms", "n", "used"} or None."""
    import numpy as np
    xy = np.asarray(xy, dtype=np.float64)
    te = np.asarray(xieta, dtype=np.float64)
    wt = np.ones(len(xy)) if weights is None else np.clip(np.asarray(weights, dtype=np.float64), 1e-3, None)
    wt = wt / wt.max()
    n_par = 3 if order < 2 else 6
    if len(xy) < 2 * n_par:
        return None
    x0, y0 = float(xy[:, 0].mean()), float(xy[:, 1].mean())
    u, v = (xy[:, 0] - x0) / 1000.0, (xy[:, 1] - y0) / 1000.0
    keep = np.ones(len(xy), bool)
    cx = cy = None
    rms = 0.0
    for _ in range(4):
        if keep.sum() < 2 * n_par:
            return None
        A = _design(u[keep], v[keep], order) * wt[keep, None]
        cx, *_ = np.linalg.lstsq(A, te[keep, 0] * wt[keep], rcond=None)
        cy, *_ = np.linalg.lstsq(A, te[keep, 1] * wt[keep], rcond=None)
        Af = _design(u, v, order)
        res = np.hypot(Af @ cx - te[:, 0], Af @ cy - te[:, 1])
        rms = float(math.sqrt(np.mean(res[keep] ** 2)))
        new = res <= max(clip * rms, 0.05)
        if np.array_equal(new, keep):
            break
        keep = new
    return {"order": order, "x0": x0, "y0": y0, "cx": [float(c) for c in cx], "cy": [float(c) for c in cy],
            "rms": rms, "n": int(len(xy)), "used": int(keep.sum())}


def apply_plate(plate: dict, x: float, y: float) -> tuple:
    """(xi, eta) in arcsec for pixel (x, y) under *plate*."""
    import numpy as np
    u, v = np.array([(x - plate["x0"]) / 1000.0]), np.array([(y - plate["y0"]) / 1000.0])
    A = _design(u, v, plate["order"])
    return float((A @ np.asarray(plate["cx"]))[0]), float((A @ np.asarray(plate["cy"]))[0])


# --------------------------------------------------------------------------- the night and the year (pure)

def nightly_epoch(frames: list, floor_mas: float = EPOCH_FLOOR_MAS, min_frames: int = MIN_FRAMES_PER_EPOCH) -> Optional[dict]:
    """One point from a night's frame measurements [{jd, xi, eta, ...}]: the
    3-sigma-clipped mean of xi and eta (arcsec), the error the scatter over
    root n with *floor_mas* under it, the time the mean JD."""
    import numpy as np
    if len(frames) < min_frames:
        return None
    xi = np.array([f["xi"] for f in frames], dtype=float)
    eta = np.array([f["eta"] for f in frames], dtype=float)
    keep = np.ones(len(xi), bool)
    for _ in range(3):
        mx, my = xi[keep].mean(), eta[keep].mean()
        sx = max(float(xi[keep].std()), 1e-4)
        sy = max(float(eta[keep].std()), 1e-4)
        new = (np.abs(xi - mx) <= 3 * sx) & (np.abs(eta - my) <= 3 * sy)
        if new.sum() < min_frames or np.array_equal(new, keep):
            break
        keep = new
    n = int(keep.sum())
    out = {"jd": float(np.mean([f["jd"] for f, k in zip(frames, keep) if k])),
           "xi": float(xi[keep].mean()), "eta": float(eta[keep].mean()),
           "sig_xi": max(float(xi[keep].std(ddof=1) / math.sqrt(n)) if n > 1 else floor_mas / 1000.0, floor_mas / 1000.0),
           "sig_eta": max(float(eta[keep].std(ddof=1) / math.sqrt(n)) if n > 1 else floor_mas / 1000.0, floor_mas / 1000.0),
           "n": n, "n_clipped": int(len(xi) - n)}
    for k in ("tanz", "parallactic", "airmass", "n_refs", "plate_rms"):
        vals = [f.get(k) for f, kp in zip(frames, keep) if kp and f.get(k) is not None]
        if vals:
            out[k] = float(np.mean(vals)) if k != "n_refs" else int(min(vals))
    return out


def fit_motion(epochs: list, ra_deg: float, dec_deg: float, dcr: bool = False,
               factors_fn: Callable = None) -> Optional[dict]:
    """Offset, proper motion and parallax from the nightly points. Pure given
    *factors_fn* (jd -> (P_alpha, P_delta); default parallax_factors).

    xi  = xi0  + mu_a t + pi P_a [+ k tan z sin q]
    eta = eta0 + mu_d t + pi P_d [+ k tan z cos q]        (t in years from 2016.0)

    Weighted least squares on the points' own errors; the covariance is
    scaled up by the reduced chi-square when that exceeds one, so a fit the
    points disagree with does not report a precision they do not have.
    -> {parallax_mas, parallax_err_mas, pmra, pmdec (mas/yr) and errors,
        offset (arcsec), dcr_mas, chi2_red, n, span_days, distance_ly,
        distance_err_ly, model: per-epoch model values} or None (too few)."""
    import numpy as np
    pts = [e for e in epochs if e.get("jd") and e.get("xi") is not None and e.get("eta") is not None]
    n_par = 6 if dcr else 5
    if len(pts) * 2 <= n_par or len(pts) < 3:
        return None
    ff = factors_fn or (lambda jd: parallax_factors(jd, ra_deg, dec_deg))
    t = np.array([(p["jd"] - J2016) / 365.25 for p in pts])
    P = np.array([ff(p["jd"]) for p in pts])
    rows, rhs, w = [], [], []
    for i, p in enumerate(pts):
        dcr_x = dcr_y = 0.0
        if dcr and p.get("tanz") is not None and p.get("parallactic") is not None:
            dcr_x = p["tanz"] * math.sin(p["parallactic"])
            dcr_y = p["tanz"] * math.cos(p["parallactic"])
        rx = [1, 0, t[i], 0, P[i, 0]] + ([dcr_x] if dcr else [])
        ry = [0, 1, 0, t[i], P[i, 1]] + ([dcr_y] if dcr else [])
        rows += [rx, ry]
        rhs += [p["xi"] * 1000.0, p["eta"] * 1000.0]              # mas
        w += [1.0 / max(float(p.get("sig_xi") or 0.003) * 1000.0, 0.1),
              1.0 / max(float(p.get("sig_eta") or 0.003) * 1000.0, 0.1)]
    A, b, w = np.asarray(rows, float), np.asarray(rhs, float), np.asarray(w, float)
    Aw, bw = A * w[:, None], b * w
    sol, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
    try:
        cov = np.linalg.inv(Aw.T @ Aw)
    except np.linalg.LinAlgError:
        return None
    res = (A @ sol - b) * w
    dof = max(1, len(b) - n_par)
    chi2_red = float((res ** 2).sum() / dof)
    if chi2_red > 1:
        cov = cov * chi2_red
    err = np.sqrt(np.clip(np.diag(cov), 0, None))
    plx, plx_err = float(sol[4]), float(err[4])
    out = {"parallax_mas": plx, "parallax_err_mas": plx_err,
           "pmra_mas_yr": float(sol[2]), "pmra_err": float(err[2]),
           "pmdec_mas_yr": float(sol[3]), "pmdec_err": float(err[3]),
           "offset_arcsec": [float(sol[0]) / 1000.0, float(sol[1]) / 1000.0],
           "dcr_mas": float(sol[5]) if dcr else None, "dcr_err": float(err[5]) if dcr else None,
           "chi2_red": chi2_red, "n": len(pts),
           "span_days": float(max(p["jd"] for p in pts) - min(p["jd"] for p in pts)),
           "dcr": bool(dcr)}
    if plx > 0:
        out["distance_ly"] = 1000.0 / plx * PC_LY
        out["distance_err_ly"] = out["distance_ly"] * plx_err / plx
    model = A @ sol
    out["model"] = [{"jd": p["jd"], "xi": float(model[2 * i]) / 1000.0, "eta": float(model[2 * i + 1]) / 1000.0,
                     "P": [float(P[i, 0]), float(P[i, 1])]} for i, p in enumerate(pts)]
    return out


def coverage(records: list, ra_deg: float, dec_deg: float, min_span: int = MIN_SPAN_DAYS,
             min_epochs: int = MIN_EPOCHS, max_rel_err: float = MAX_REL_ERR, factors_fn: Callable = None) -> dict:
    """Is the parallax measured? The nights span *min_span* days, there are
    *min_epochs* of them, and the fit's error is under *max_rel_err* of the
    parallax. -> {complete, n, span_days, parallax_mas, err_mas, rel_err, why}."""
    pts = [r for r in records or [] if r.get("jd") is not None]
    n = len(pts)
    span = (max(p["jd"] for p in pts) - min(p["jd"] for p in pts)) if n > 1 else 0.0
    fit = fit_motion(pts, ra_deg, dec_deg, factors_fn=factors_fn) if n >= 5 else None
    plx = fit["parallax_mas"] if fit else None
    err = fit["parallax_err_mas"] if fit else None
    rel = (err / plx) if (plx and plx > 0 and err is not None) else None
    why = []
    if span < min_span:
        why.append("%d of %d days of baseline" % (int(span), min_span))
    if n < min_epochs:
        why.append("%d of %d nights" % (n, min_epochs))
    if rel is None:
        why.append("no parallax fit yet" if n < 5 else "the fit gives no positive parallax")
    elif rel > max_rel_err:
        why.append("parallax %.1f +/- %.1f mas (%.0f %%, needs %.0f %%)" % (plx, err, 100 * rel, 100 * max_rel_err))
    return {"complete": not why, "n": n, "span_days": span, "parallax_mas": plx, "err_mas": err,
            "rel_err": rel, "why": why}


def parallax_verdict(dso: str, cov: dict) -> dict:
    """auto_stop's {dso, done, why} for a parallax monitor from coverage(). Pure."""
    cov = cov or {}
    if cov.get("complete"):
        return {"dso": dso, "done": True, "why": [
            "parallax measured: %.1f +/- %.1f mas over %d nights spanning %d days"
            % (cov.get("parallax_mas") or 0.0, cov.get("err_mas") or 0.0, int(cov.get("n") or 0), int(cov.get("span_days") or 0))]}
    return {"dso": dso, "done": False, "why": ["parallax monitor: " + "; ".join(cov.get("why") or ["no records"])]}


def merge_record(records: list, new: dict) -> list:
    out = [r for r in records if (r.get("night"), r.get("filter")) != (new.get("night"), new.get("filter"))]
    out.append(new)
    return sorted(out, key=lambda r: (r.get("night") or "", r.get("filter") or ""))


# --------------------------------------------------------------------------- references

def refs_for(key: str, ra: float, dec: float, progress_cb=None) -> dict:
    """{"rows": [...], "source": str, "has_pm": bool}: the committed file for
    *key* when its centre is within 30' of the target, else the Gaia archive
    (cached under local/parallax), else VizieR through lightcurve.gaia_comps
    (no proper motions: the references then stand still, which is wrong by
    their motions, typically a few mas a year)."""
    p = REFS_DIR / ("%s.json" % key)
    if p.exists():
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
            c = doc.get("centre") or [None, None]
            if c[0] is not None and _sep_arcmin(ra, dec, float(c[0]), float(c[1])) < 30:
                return {"rows": doc["rows"], "source": "configs/parallax_refs/%s.json" % key, "has_pm": True}
        except (OSError, ValueError, KeyError):
            pass
    PLX_DIR.mkdir(parents=True, exist_ok=True)
    cache = PLX_DIR / ("%s_gaia.json" % key)
    if cache.exists():
        doc = json.loads(cache.read_text())
        return {"rows": doc["rows"], "source": str(cache), "has_pm": bool(doc.get("has_pm"))}
    rows, src, has_pm = [], "", False
    try:
        if progress_cb:
            progress_cb("fetching Gaia DR3 references for %s from the ESA archive…" % key)
        rows = gaia_archive_rows(ra, dec)
        src, has_pm = "ESA Gaia archive", True
    except Exception as exc:                                     # noqa: BLE001
        _logger.warning("Gaia archive failed for %s: %s", key, exc)
    if not rows:
        from photometry import lightcurve as lc
        for r in lc.gaia_comps(ra, dec, radius_arcmin=15.0, g_limit=REF_G_MAX + 0.5):
            rows.append({"id": r.get("source"), "ra": r["ra"], "dec": r["dec"], "g": r.get("gmag"),
                         "ruwe": r.get("ruwe"), "var": r.get("var") == "VARIABLE"})
        src, has_pm = "VizieR (no proper motions)", False
    cache.write_text(json.dumps({"ra": ra, "dec": dec, "rows": rows, "has_pm": has_pm, "source": src}))
    return {"rows": rows, "source": src, "has_pm": has_pm}


def gaia_archive_rows(ra: float, dec: float, radius_arcmin: float = 15.0, g_limit: float = REF_G_MAX + 0.5) -> list:
    """Gaia DR3 rows from the ESA archive (network), in the committed files' shape."""
    from astroquery.gaia import Gaia
    q = ("SELECT source_id, ra, dec, parallax, pmra, pmdec, phot_g_mean_mag, bp_rp, ruwe, phot_variable_flag "
         "FROM gaiadr3.gaia_source WHERE 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',%.6f,%.6f,%.5f)) "
         "AND phot_g_mean_mag < %.1f" % (ra, dec, radius_arcmin / 60.0, g_limit))
    out = []
    for r in Gaia.launch_job_async(q).get_results():
        def f(k):
            try:
                v = float(r[k]); return v if math.isfinite(v) else None
            except Exception:
                return None
        out.append({"id": str(r["source_id"]), "ra": f("ra"), "dec": f("dec"), "g": f("phot_g_mean_mag"),
                    "plx": f("parallax"), "pmra": f("pmra"), "pmdec": f("pmdec"), "bp_rp": f("bp_rp"),
                    "ruwe": f("ruwe"), "var": str(r["phot_variable_flag"]) == "VARIABLE"})
    return out


def _sep_arcmin(ra1, dec1, ra2, dec2) -> float:
    xi, eta = tangent(ra1, dec1, ra2, dec2)
    return math.hypot(xi, eta) / 60.0


def usable_refs(rows: list, target_id: Optional[str], ra: float, dec: float,
                radius_arcmin: float = REF_RADIUS_ARCMIN, g_max: float = REF_G_MAX) -> list:
    """The reference stars for the plate: within the radius, bright enough,
    good Gaia astrometry, not flagged variable, not the target itself. Pure."""
    out = []
    for r in rows or []:
        if r.get("ra") is None or r.get("dec") is None or r.get("g") is None:
            continue
        if target_id and str(r.get("id")) == str(target_id):
            continue
        if float(r["g"]) > g_max or r.get("var"):
            continue
        if r.get("ruwe") is not None and float(r["ruwe"]) > REF_RUWE_MAX:
            continue
        if r.get("plx") is not None and float(r["plx"]) > 20:          # a nearby star is not a fixed point
            continue
        if _sep_arcmin(ra, dec, float(r["ra"]), float(r["dec"])) <= radius_arcmin:
            out.append(r)
    return out


# --------------------------------------------------------------------------- one frame

def measure_frame(data, wcs, jd: float, target: dict, refs: list, raw=None,
                  saturation: float = 60000.0, site: Optional[tuple] = None) -> Optional[dict]:
    """The target's (xi, eta) on one solved frame, in arcsec from its Gaia 2016.0
    position. *target* is the catalogue's gaia block ({ra, dec, ...}); *refs*
    usable_refs(). -> {xi, eta, n_refs, plate_rms, order, peak, saturated,
    tanz, parallactic, airmass} or None (too few references / target lost)."""
    import numpy as np
    ra0, dec0 = float(target["ra"]), float(target["dec"])
    fac = parallax_factors(jd, ra0, dec0)
    # where the references are tonight, in the sky and on the chip
    sky = [propagate(r, jd, fac) for r in refs]
    px = wcs.world_to_pixel_values(np.array([s[0] for s in sky]), np.array([s[1] for s in sky]))
    xy, te, wts = [], [], []
    for (x, y), (ra, dec) in zip(zip(px[0], px[1]), sky):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        c = centroid(data, x, y)
        if c is None:
            continue
        cx, cy, peak, sig, snr = c
        if peak < REF_PEAK_MIN_SIG * sig or snr < REF_SNR_MIN or math.hypot(cx - x, cy - y) > REF_MAX_MOVE_PX:
            continue
        if raw is not None and saturation and _raw_peak(raw, cx, cy) >= saturation:
            continue
        xy.append((cx, cy))
        te.append(tangent(ra, dec, ra0, dec0))
        wts.append(min(snr, REF_WEIGHT_CAP))
    if len(xy) < MIN_REFS:
        return None
    order = 2 if len(xy) >= SECOND_ORDER_REFS else 1
    plate = fit_plate(xy, te, order=order, weights=wts)
    if plate is None:
        return None
    # the target, found near its catalogue motion (pm applied) but MEASURED, never assumed
    tra, tdec = propagate({"ra": ra0, "dec": dec0, "pmra": target.get("pmra"), "pmdec": target.get("pmdec")}, jd)
    tx, ty = wcs.world_to_pixel_values(tra, tdec)
    c = centroid(data, float(tx), float(ty))
    if c is None:
        return None
    cx, cy, peak, _sig, _snr = c
    sat = bool(raw is not None and saturation and _raw_peak(raw, cx, cy) >= saturation)
    xi, eta = apply_plate(plate, cx, cy)
    out = {"xi": xi, "eta": eta, "n_refs": plate["used"], "plate_rms": plate["rms"], "order": order,
           "peak": peak, "saturated": sat, "jd": jd, "x": cx, "y": cy}
    if site:
        tanz, q, am = zenith_terms(jd, ra0, dec0, site[0], site[1])
        out.update({"tanz": tanz, "parallactic": q, "airmass": am})
    return out


def _raw_peak(raw, x: float, y: float, r: int = 4) -> float:
    import numpy as np
    h, w = raw.shape
    ix, iy = int(round(x)), int(round(y))
    return float(np.max(raw[max(0, iy - r):min(h, iy + r + 1), max(0, ix - r):min(w, ix + r + 1)]))


# --------------------------------------------------------------------------- the run

def load(key: str) -> dict:
    p = PLX_DIR / ("%s.json" % key)
    if p.exists():
        return json.loads(p.read_text())
    return {"key": key, "records": []}


def save(key: str, doc: dict) -> Path:
    PLX_DIR.mkdir(parents=True, exist_ok=True)
    p = PLX_DIR / ("%s.json" % key)
    p.write_text(json.dumps(doc, indent=1))
    return p


def star_for_row(dso: str, rec: dict) -> Optional[dict]:
    """The catalogue entry a queue row means: its parallax_star, else its name."""
    return star(str(rec.get("parallax_star") or dso))


def run(dso: str, redo: bool = False, progress_cb: Optional[Callable[[str], None]] = None,
        cancel_cb: Optional[Callable[[], bool]] = None, nights: Optional[list] = None) -> dict:
    """Measure every night of *dso*'s frames not yet on file (all with redo),
    refit the year, draw the plot. -> {key, name, new, doc, fit, fit_dcr, plot, coverage, refs}."""
    from configs import config
    from photometry import lightcurve as lc
    cfg = config.data()
    image_dir = Path(cfg["nina"]["image_dir"])
    astap = (cfg.get("hardware") or {}).get("astap_exe") or "C:/Program Files/astap/astap.exe"
    site = lc.site_from(cfg)
    target, rec = lc._target_for(dso)
    entry = star_for_row(dso, rec)
    if entry is None:
        raise ValueError("%s is not in configs/parallax_stars.json (set parallax_star on the queue row)" % dso)
    g = entry["gaia"]
    key = entry["key"]
    doc = load(key)
    doc.update({"name": entry["name"], "gaia": g})
    done = {(r["night"], r["filter"]) for r in doc["records"]}
    refs_doc = refs_for(key, float(g["ra"]), float(g["dec"]), progress_cb)
    refs = usable_refs(refs_doc["rows"], g.get("source_id"), float(g["ra"]), float(g["dec"]))
    doc["refs"] = {"source": refs_doc["source"], "n": len(refs), "has_pm": refs_doc["has_pm"]}
    if progress_cb:
        progress_cb("%s: %d Gaia references within %.0f' (%s)" % (entry["name"], len(refs), REF_RADIUS_ARCMIN, refs_doc["source"]))
    new, cals = [], {}
    for (night, fl), paths in sorted(lc._frames(image_dir, dso, None).items()):
        if nights and night not in nights:
            continue
        if not redo and not nights and (night, fl) in done:
            continue
        if progress_cb:
            progress_cb("%s %s %s: measuring %d frames…" % (dso, night, fl, len(paths)))
        if fl not in cals:
            cals[fl] = lc.calibration_for(fl, paths[0], progress_cb)
        per, n_sat, n_unsolved = [], 0, 0
        for p in paths:
            if cancel_cb and cancel_cb():
                raise RuntimeError("cancelled")
            solved = lc.solve_frame(p, target, astap, None, cal=cals[fl])
            if solved is None:
                n_unsolved += 1
                continue
            data, hdr, wcs = solved
            from astropy.time import Time
            exp = float(hdr.get("EXPTIME", 0) or 0)
            jd = Time(hdr["DATE-OBS"], format="isot", scale="utc").jd + exp / 2 / 86400.0
            m = measure_frame(data, wcs, jd, g, refs, raw=getattr(solved, "raw", None), site=site)
            if m is None:
                continue
            if m["saturated"]:
                n_sat += 1
                continue
            per.append(m)
        pt = nightly_epoch(per)
        if pt is None:
            if progress_cb:
                progress_cb("%s %s %s: no point (%d frames, %d saturated, %d unsolved, %d measured)"
                            % (dso, night, fl, len(paths), n_sat, n_unsolved, len(per)))
            continue
        record = dict(pt, night=night, filter=fl, frames_total=len(paths), n_saturated=n_sat, n_unsolved=n_unsolved,
                      measured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        doc["records"] = merge_record(doc["records"], record)
        new.append(record)
        save(key, doc)
        if progress_cb:
            progress_cb("%s %s %s: %d frames -> (%+.3f, %+.3f)\" +/- %.0f mas, %d references"
                        % (dso, night, fl, pt["n"], pt["xi"], pt["eta"], 1000 * max(pt["sig_xi"], pt["sig_eta"]), pt.get("n_refs") or 0))
    ra0, dec0 = float(g["ra"]), float(g["dec"])
    fit = fit_motion(doc["records"], ra0, dec0)
    fit_dcr = fit_motion(doc["records"], ra0, dec0, dcr=True) if len(doc["records"]) >= DCR_MIN_EPOCHS else None
    cov = coverage(doc["records"], ra0, dec0)
    doc["fit"] = {k: v for k, v in (fit or {}).items() if k != "model"}
    doc["fit_dcr"] = {k: v for k, v in (fit_dcr or {}).items() if k != "model"}
    doc["coverage"] = cov
    save(key, doc)
    plot_path = plot(key, doc, entry, fit) if len(doc["records"]) >= 2 else None
    return {"key": key, "name": entry["name"], "new": new, "doc": doc, "fit": fit, "fit_dcr": fit_dcr,
            "plot": plot_path, "coverage": cov, "refs": len(refs), "star": entry}


# --------------------------------------------------------------------------- picture and words

def plot(key: str, doc: dict, entry: dict, fit: Optional[dict]) -> Path:
    """local/parallax/<key>.png: xi and eta against time with the fitted
    motion, and the yearly ellipse with the proper motion taken out."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.dates import DateFormatter
    from astropy.time import Time
    SURF, INK, INK2, GRID = "#1a1a19", "#ffffff", "#c3c2b7", "#3a3a37"
    recs = sorted(doc["records"], key=lambda r: r["jd"])
    g = entry["gaia"]
    ra0, dec0 = float(g["ra"]), float(g["dec"])
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), dpi=100, facecolor=SURF)
    ax, px = axs
    t = [Time(r["jd"], format="jd").to_datetime() for r in recs]
    ax.errorbar(t, [r["xi"] for r in recs], yerr=[r["sig_xi"] for r in recs], fmt="o", ms=5, color="#f2c14e", capsize=3, label="east (ξ)")
    ax.errorbar(t, [r["eta"] for r in recs], yerr=[r["sig_eta"] for r in recs], fmt="s", ms=5, color="#3987e5", capsize=3, label="north (η)")
    if fit:
        jds = np.linspace(recs[0]["jd"] - 10, recs[-1]["jd"] + 10, 200)
        mx, my, lx, ly = [], [], [], []
        for jd in jds:
            yr = (jd - J2016) / 365.25
            pa, pd = parallax_factors(jd, ra0, dec0)
            lx.append(fit["offset_arcsec"][0] + fit["pmra_mas_yr"] * yr / 1000.0)
            ly.append(fit["offset_arcsec"][1] + fit["pmdec_mas_yr"] * yr / 1000.0)
            mx.append(lx[-1] + fit["parallax_mas"] * pa / 1000.0)
            my.append(ly[-1] + fit["parallax_mas"] * pd / 1000.0)
        tt = [Time(j, format="jd").to_datetime() for j in jds]
        ax.plot(tt, mx, color="#f2c14e", lw=1.2); ax.plot(tt, my, color="#3987e5", lw=1.2)
        ax.plot(tt, lx, color="#f2c14e", lw=0.8, ls=":"); ax.plot(tt, ly, color="#3987e5", lw=0.8, ls=":")
        # the ellipse: points minus the fitted straight line, against the model ellipse
        ex = [r["xi"] - (fit["offset_arcsec"][0] + fit["pmra_mas_yr"] * (r["jd"] - J2016) / 365.25 / 1000.0) for r in recs]
        ey = [r["eta"] - (fit["offset_arcsec"][1] + fit["pmdec_mas_yr"] * (r["jd"] - J2016) / 365.25 / 1000.0) for r in recs]
        px.errorbar([1000 * v for v in ex], [1000 * v for v in ey], xerr=[1000 * r["sig_xi"] for r in recs],
                    yerr=[1000 * r["sig_eta"] for r in recs], fmt="o", ms=5, color=INK, capsize=2)
        yj = np.linspace(recs[0]["jd"], recs[0]["jd"] + 365.25, 300)
        ell = np.array([parallax_factors(j, ra0, dec0) for j in yj]) * fit["parallax_mas"]
        px.plot(ell[:, 0], ell[:, 1], color="#3fb950", lw=1.2, label="fitted ellipse, %.1f ± %.1f mas" % (fit["parallax_mas"], fit["parallax_err_mas"]))
        gp = float(g.get("parallax") or 0)
        if gp:
            px.plot(ell[:, 0] * gp / fit["parallax_mas"], ell[:, 1] * gp / fit["parallax_mas"], color=INK2, lw=0.8, ls="--", label="Gaia DR3, %.1f mas" % gp)
        px.set_aspect("equal"); px.invert_xaxis()
        px.legend(loc="best", facecolor=SURF, labelcolor=INK, fontsize=10)
    ax.set_ylabel("displacement from the Gaia 2016.0 position (arcsec)", color=INK2)
    ax.xaxis.set_major_formatter(DateFormatter("%b %d"))
    ax.set_title("%s: position by night (%d nights)" % (entry["name"], len(recs)), color=INK, fontsize=13, loc="left")
    ax.legend(loc="best", facecolor=SURF, labelcolor=INK, fontsize=10)
    px.set_xlabel("east (mas)", color=INK2); px.set_ylabel("north (mas)", color=INK2)
    px.set_title("the parallax ellipse, proper motion removed", color=INK, fontsize=13, loc="left")
    for a in axs:
        a.set_facecolor(SURF); a.tick_params(colors=INK2); a.grid(color=GRID, lw=0.6)
        for sp in a.spines.values():
            sp.set_color("#5a5a55")
    fig.tight_layout()
    out = PLX_DIR / ("%s.png" % key)
    fig.savefig(out, dpi=100, facecolor=SURF); plt.close(fig)
    return out


def report(result: dict) -> str:
    """The chat's answer to a run."""
    entry, fit, cov = result["star"], result.get("fit"), result.get("coverage") or {}
    g = entry["gaia"]
    lines = ["Parallax %s (%s)" % (entry["name"], result["key"])]
    recs = result["doc"]["records"]
    if result["new"]:
        for r in result["new"]:
            lines.append("  %s %s: %d frames -> (%+.3f, %+.3f)\" ± %.0f mas, %d refs, plate rms %.0f mas"
                         % (r["night"], r["filter"], r["n"], r["xi"], r["eta"], 1000 * max(r["sig_xi"], r["sig_eta"]),
                            r.get("n_refs") or 0, 1000 * (r.get("plate_rms") or 0)))
    else:
        lines.append("  nothing new to measure")
    lines.append("  %d night%s on file spanning %d days; %d references (%s)"
                 % (len(recs), "" if len(recs) == 1 else "s", int(cov.get("span_days") or 0), result.get("refs") or 0,
                    result["doc"].get("refs", {}).get("source", "?")))
    if fit:
        lines.append("  fit: parallax %.1f ± %.1f mas (Gaia DR3 %.1f), distance %.2f ± %.2f ly; pm %+.0f %+.0f mas/yr (Gaia %+.0f %+.0f); χ²/dof %.1f"
                     % (fit["parallax_mas"], fit["parallax_err_mas"], float(g.get("parallax") or 0),
                        fit.get("distance_ly") or 0, fit.get("distance_err_ly") or 0,
                        fit["pmra_mas_yr"], fit["pmdec_mas_yr"], float(g.get("pmra") or 0), float(g.get("pmdec") or 0), fit["chi2_red"]))
        fd = result.get("fit_dcr")
        if fd:
            lines.append("  with a refraction term: parallax %.1f ± %.1f mas, DCR %.0f ± %.0f mas per unit tan z"
                         % (fd["parallax_mas"], fd["parallax_err_mas"], fd.get("dcr_mas") or 0, fd.get("dcr_err") or 0))
    lines.append("  %s" % ("done: the parallax is measured" if cov.get("complete") else "still needs: " + "; ".join(cov.get("why") or [])))
    return "\n".join(lines)


def list_text(stars: Optional[list] = None, queued: Optional[set] = None) -> str:
    """`parallax list`: the catalogue in a few lines. Pure."""
    out = ["Parallax stars (configs/parallax_stars.json):"]
    for s in (stars if stars is not None else catalogue()):
        g = s.get("gaia") or {}
        mark = "queued" if queued and s["key"] in queued else ("chosen" if s.get("chosen") else ("from %s" % s["from"] if s.get("from") else "reserve"))
        out.append("  %-14s %-16s G %4.1f  parallax %5.1f mas  %-8s %s" % (
            s["key"], s["name"], float(g.get("g") or 0), float(g.get("parallax") or 0), mark, s.get("season") or ""))
    out.append("`parallax add <key>` queues one; `parallax <key>` measures its nights.")
    return "\n".join(out)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="measure a queued parallax star's nights")
    ap.add_argument("dso")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args(argv)
    res = run(a.dso, redo=a.redo, progress_cb=print)
    print(report(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
