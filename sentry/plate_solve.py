#!/usr/bin/env python3
"""Work out where the all-sky camera is pointing, from the stars alone.

Given time and site, the pattern of detected stars over-determines the camera's
orientation and focal length. Solving it turns the camera from a picture into an
instrument: every pixel maps to an altitude and azimuth, and "how faint can we
see tonight" becomes measurable. Limiting magnitude is a far better cloud reading
than a raw star count, because it is a property of the sky rather than of how
much of the frame happens to be trees.

Method is astrometry.net's, shrunk to fit. Orientation plus focal length is four
unknowns, so TWO correct correspondences between a detection and a catalogue
star pin all four. Try every pairing of the brightest detections against the
brightest stars that are up, and keep whichever hypothesis puts the REST of the
catalogue on top of the rest of the detections. No initial guess needed.

The camera is bolted down, so this runs once and the answer is cached. Later
frames verify against the stored solution and only re-solve if it stops fitting,
which is also how a knocked camera announces itself.

Sanity check that matters more than the residual: the named stars should come
out as neighbours. A correct solve names one contiguous patch of sky. A
coincidence names stars scattered across unrelated constellations.

Usage:
    python sentry/plate_solve.py <frame.jpg> [--save] [--report]
    python sentry/plate_solve.py <frame.jpg> --verify
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
from scipy.spatial import cKDTree

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from iris_astronomy import bright_stars
from sentry import star_count

SOLUTION = "local/sky_solution.json"

# Bright named stars, for the human-readable check on a solution. RA/Dec J2000.
NAMED = [
    ("Sirius", 101.287, -16.716), ("Arcturus", 213.915, 19.182),
    ("Vega", 279.234, 38.784), ("Capella", 79.172, 45.998),
    ("Rigel", 78.634, -8.202), ("Procyon", 114.825, 5.225),
    ("Betelgeuse", 88.793, 7.407), ("Altair", 297.696, 8.868),
    ("Aldebaran", 68.980, 16.509), ("Antares", 247.352, -26.432),
    ("Spica", 201.298, -11.161), ("Pollux", 116.329, 28.026),
    ("Fomalhaut", 344.413, -29.622), ("Deneb", 310.358, 45.280),
    ("Regulus", 152.093, 11.967), ("Castor", 113.650, 31.888),
    ("Bellatrix", 81.283, 6.350), ("Elnath", 81.573, 28.608),
    ("Alnilam", 84.053, -1.202), ("Alnitak", 85.190, -1.943),
    ("Mintaka", 83.002, -0.299), ("Dubhe", 165.932, 61.751),
    ("Alkaid", 206.885, 49.313), ("Merak", 165.460, 56.382),
    ("Phecda", 178.458, 53.695), ("Megrez", 183.857, 57.033),
    ("Alioth", 193.507, 55.960), ("Mizar", 200.981, 54.926),
    ("Polaris", 37.955, 89.264), ("Kochab", 222.676, 74.156),
    ("Alphecca", 233.672, 26.715), ("Rasalhague", 263.734, 12.560),
    ("Eltanin", 269.152, 51.489), ("Sadr", 305.557, 40.257),
    ("Albireo", 292.680, 27.960), ("Alderamin", 319.645, 62.586),
    ("Enif", 326.046, 9.875), ("Markab", 346.190, 15.205),
    ("Scheat", 345.944, 28.083), ("Alpheratz", 2.097, 29.090),
    ("Mirach", 17.433, 35.620), ("Almach", 30.975, 42.330),
    ("Hamal", 31.793, 23.462), ("Mirfak", 51.081, 49.861),
    ("Algol", 47.042, 40.956), ("Menkalinan", 89.882, 44.947),
    ("Alhena", 99.428, 16.399), ("Alphard", 141.897, -8.659),
    ("Denebola", 177.265, 14.572), ("Izar", 221.247, 27.074),
    ("Schedar", 10.127, 56.537), ("Caph", 2.294, 59.150),
    ("Ruchbah", 21.454, 60.235), ("Navi", 14.177, 60.717),
    ("Nunki", 283.816, -26.297), ("Tarazed", 296.565, 10.613),
]


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------- geometry

def unit_from_altaz(az_deg, alt_deg):
    """Horizontal unit vectors, (East, North, Up)."""
    az, alt = np.radians(az_deg), np.radians(alt_deg)
    return np.stack([np.cos(alt) * np.sin(az),
                     np.cos(alt) * np.cos(az),
                     np.sin(alt)], axis=-1)


def pix_to_cam(px, py, cx, cy, F, model="equi", flip=True):
    dx, dy = np.asarray(px) - cx, -(np.asarray(py) - cy)
    rho = np.hypot(dx, dy)
    theta = rho / F if model == "equi" else np.arctan(rho / F)
    phi = np.arctan2(dy, dx) * (-1.0 if flip else 1.0)
    return np.stack([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)], axis=-1)


def cam_to_pix(v, cx, cy, F, model="equi", flip=True):
    theta = np.arccos(np.clip(v[:, 2], -1, 1))
    rho = F * theta if model == "equi" else F * np.tan(np.clip(theta, 0, 1.45))
    phi = np.arctan2(v[:, 1], v[:, 0]) * (-1.0 if flip else 1.0)
    return cx + rho * np.cos(phi), cy - rho * np.sin(phi), theta


def _kabsch(Aw, Bc):
    U, _s, Vt = np.linalg.svd(Bc @ Aw.T)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def _rotvec_to_R(rv):
    th = np.linalg.norm(rv)
    if th < 1e-12:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def pixel_to_altaz(sol, px, py):
    """Where in the sky a pixel looks. Returns (alt_deg, az_deg)."""
    R = np.asarray(sol["R"])
    v = pix_to_cam(np.atleast_1d(px), np.atleast_1d(py), sol["cx"], sol["cy"],
                   sol["F"], sol["model"], sol["flip"])
    e, n, u = (v @ R).T
    return (np.degrees(np.arcsin(np.clip(u, -1, 1))),
            np.degrees(np.arctan2(e, n)) % 360)


def altaz_to_pixel(sol, alt_deg, az_deg):
    R = np.asarray(sol["R"])
    v = unit_from_altaz(np.atleast_1d(az_deg), np.atleast_1d(alt_deg)) @ R.T
    x, y, _th = cam_to_pix(v, sol["cx"], sol["cy"], sol["F"], sol["model"],
                           sol["flip"])
    return x, y


# ------------------------------------------------------------------- solve

def frame_time(path):
    """UTC instant a captured frame belongs to, from its name or its mtime."""
    p = Path(path)
    stem = p.stem
    if stem.startswith("sky_") and len(stem) >= 20:
        s = stem[4:]
        try:
            return datetime(int(s[0:4]), int(s[4:6]), int(s[6:8]), int(s[9:11]),
                            int(s[11:13]), int(s[13:15]), tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)


def _scorer(dx, dy, shape):
    """Distance-to-nearest-detection field. Lookup beats a tree in the inner loop."""
    H, W = shape
    occ = np.ones((H, W), bool)
    occ[np.clip(dy.astype(int), 0, H - 1), np.clip(dx.astype(int), 0, W - 1)] = False
    return ndimage.distance_transform_edt(occ)


def solve(frame, when=None, dets=None, verbose=False):
    """Blind solve. Returns a solution dict, or None if nothing verified."""
    frame = Path(frame)
    when = when or frame_time(frame)
    img = star_count._load(frame)
    H, W = img.shape
    cx, cy = W / 2.0, H / 2.0

    if dets is None:
        dets = star_count.count_stars(frame)["_stars"]
    dets = sorted(dets, key=lambda s: -s["peak"])
    dx = np.array([d["x"] for d in dets])
    dy = np.array([d["y"] for d in dets])

    alt, az, vmag, _ra, _dec = bright_stars.above_horizon(when)
    world = unit_from_altaz(az, alt)
    dist = _scorer(dx, dy, img.shape)

    def score(R, F, model, flip, sigma=10.0, mag_lim=4.6):
        sel = vmag < mag_lim
        x, y, th = cam_to_pix(world[sel] @ R.T, cx, cy, F, model, flip)
        ok = (x > 0) & (x < W) & (y > 0) & (y < H) & (th < 1.4)
        if ok.sum() < 8:
            return -1.0
        d = dist[np.clip(y[ok].astype(int), 0, H - 1),
                 np.clip(x[ok].astype(int), 0, W - 1)]
        return float(np.exp(-(d ** 2) / (2 * sigma ** 2)).sum())

    ND, NC = 14, 34
    bright_c = np.argsort(vmag)[:NC]
    best = None
    for model in ("equi", "rect"):
        for flip in (True, False):
            for i in range(min(ND, len(dets))):
                for j in range(i + 1, min(ND, len(dets))):
                    p1, p2 = (dx[i], dy[i]), (dx[j], dy[j])
                    s_pix = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
                    if s_pix < 150:
                        continue
                    for a in bright_c:
                        va = world[a]
                        for b in bright_c:
                            if b == a:
                                continue
                            ang = float(np.arccos(np.clip(va @ world[b], -1, 1)))
                            if ang < 0.12 or ang > 1.6:
                                continue
                            F = s_pix / (ang if model == "equi" else np.tan(ang))
                            if not (600 < F < 3200):
                                continue
                            c1 = pix_to_cam(p1[0], p1[1], cx, cy, F, model, flip)
                            c2 = pix_to_cam(p2[0], p2[1], cx, cy, F, model, flip)
                            Aw = np.stack([va, world[b], np.cross(va, world[b])], 1)
                            Bc = np.stack([c1, c2, np.cross(c1, c2)], 1)
                            try:
                                R = _kabsch(Aw, Bc)
                            except np.linalg.LinAlgError:
                                continue
                            s = score(R, F, model, flip)
                            if best is None or s > best[0]:
                                best = (s, R, F, model, flip)
    if best is None:
        return None
    s0, R, F, model, flip = best
    if verbose:
        print("hypothesis: score %.1f, F=%.0f, %s, flip=%s" % (s0, F, model, flip))

    # Refinement is caged on purpose. Left free it drives F towards zero, piling
    # the whole catalogue onto the image centre where everything is "within
    # sigma" of something: a verified 44.8 at F=1617 became 7.0 at F=227. So
    # sigma is fixed (letting it float is the same cheat by another route), F is
    # held to +/-15%, and a refinement that verifies worse is thrown away.
    SIG = 8.0

    def neg(p):
        if abs(p[3]) > 0.14:
            return 1e6
        return -score(_rotvec_to_R(p[:3]) @ R, F * np.exp(p[3]), model, flip, SIG)

    base = score(R, F, model, flip, SIG)
    out = minimize(neg, np.zeros(4), method="Nelder-Mead",
                   options=dict(maxiter=6000, xatol=1e-5, fatol=1e-4))
    Rf, Ff = _rotvec_to_R(out.x[:3]) @ R, F * np.exp(out.x[3])
    if score(Rf, Ff, model, flip, SIG) < base:
        Rf, Ff = R, F

    sol = {"R": np.asarray(Rf).tolist(), "F": float(Ff), "model": model,
           "flip": bool(flip), "cx": cx, "cy": cy,
           "solved_from": frame.name,
           "solved_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    sol.update(verify(sol, frame, when, dets))
    axis = np.asarray(Rf).T @ np.array([0.0, 0.0, 1.0])
    sol["axis_alt_deg"] = round(float(np.degrees(np.arcsin(np.clip(axis[2], -1, 1)))), 2)
    sol["axis_az_deg"] = round(float(np.degrees(np.arctan2(axis[0], axis[1])) % 360), 2)
    sol["fov_diag_deg"] = round(float(2 * np.degrees(np.hypot(W, H) / 2.0 / Ff)), 1)
    sol["deg_per_px"] = round(float(np.degrees(1.0 / Ff)), 5)
    return sol


def verify(sol, frame, when=None, dets=None, tol_px=8.0):
    """How well a stored solution still fits a frame. Cheap; safe on a timer."""
    frame = Path(frame)
    when = when or frame_time(frame)
    if dets is None:
        dets = star_count.count_stars(frame)["_stars"]
    dx = np.array([d["x"] for d in dets])
    dy = np.array([d["y"] for d in dets])
    if len(dx) == 0:
        return {"matches": 0, "residual_px": None}
    img = star_count._load(frame)
    H, W = img.shape
    alt, az, vmag, _ra, _dec = bright_stars.above_horizon(when)
    x, y, th = cam_to_pix(unit_from_altaz(az, alt) @ np.asarray(sol["R"]).T,
                          sol["cx"], sol["cy"], sol["F"], sol["model"], sol["flip"])
    inframe = (x > 20) & (x < W - 20) & (y > 20) & (y < H - 20) & (th < 1.45)
    d, _i = cKDTree(np.stack([dx, dy], 1)).query(np.stack([x, y], 1))
    good = inframe & (d < tol_px)
    return {"matches": int(good.sum()),
            "residual_px": round(float(np.sqrt((d[good] ** 2).mean())), 2)
            if good.sum() else None}


# ------------------------------------------------------- limiting magnitude

def limiting_magnitude(sol, frame, when=None, dets=None, mask=None,
                       tol_px=8.0, step=0.5):
    """Completeness against V magnitude, and the 50% crossing.

    Stars landing on masked foliage are dropped from the DENOMINATOR, not
    counted as misses. Otherwise the reading would track how much of the frame
    is trees -- which never changes -- instead of how clear the sky is, which is
    the entire point.
    """
    frame = Path(frame)
    when = when or frame_time(frame)
    img = star_count._load(frame)
    H, W = img.shape
    if dets is None or mask is None:
        res = star_count.count_stars(frame)
        dets = dets or res["_stars"]
        mask = res["_mask"] if mask is None else mask
    dx = np.array([d["x"] for d in dets])
    dy = np.array([d["y"] for d in dets])
    if len(dx) == 0:
        return {"limiting_mag": None, "completeness": []}

    alt, az, vmag, _ra, _dec = bright_stars.above_horizon(when)
    x, y, th = cam_to_pix(unit_from_altaz(az, alt) @ np.asarray(sol["R"]).T,
                          sol["cx"], sol["cy"], sol["F"], sol["model"], sol["flip"])
    inframe = (x > 20) & (x < W - 20) & (y > 20) & (y < H - 20) & (th < 1.45)
    xi = np.clip(x.astype(int), 0, W - 1)
    yi = np.clip(y.astype(int), 0, H - 1)
    visible = inframe & (~mask[yi, xi])
    d, _i = cKDTree(np.stack([dx, dy], 1)).query(np.stack([x, y], 1))
    found = visible & (d < tol_px)

    table, lim = [], None
    prev = None
    for lo in np.arange(0.0, 6.0, step):
        sel = visible & (vmag >= lo) & (vmag < lo + step)
        n = int(sel.sum())
        if n < 4:                      # too few to mean anything
            continue
        frac = float((found & sel).sum()) / n
        mid = lo + step / 2
        table.append({"vmag": round(mid, 2), "n": n, "found": int((found & sel).sum()),
                      "fraction": round(frac, 3)})
        if lim is None and prev is not None and prev[1] >= 0.5 > frac:
            # linear interpolation across the 50% crossing
            lim = prev[0] + (prev[1] - 0.5) / (prev[1] - frac) * (mid - prev[0])
        prev = (mid, frac)
    return {"limiting_mag": round(float(lim), 2) if lim else None,
            "stars_visible_area": int(visible.sum()),
            "stars_matched": int(found.sum()),
            "completeness": table}


# ---------------------------------------------------------------- storage

def save(sol, root=None):
    p = (Path(root) if root else _root()) / SOLUTION
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sol, indent=1))
    return p


def load(root=None):
    p = (Path(root) if root else _root()) / SOLUTION
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (ValueError, OSError):
        return None


def named_identifications(sol, when, dets, tol_px=8.0):
    """Which named stars a solution puts on a detection. The human check."""
    alt, az, vmag, ra, dec = bright_stars.above_horizon(when)
    x, y, _th = cam_to_pix(unit_from_altaz(az, alt) @ np.asarray(sol["R"]).T,
                           sol["cx"], sol["cy"], sol["F"], sol["model"], sol["flip"])
    dx = np.array([d["x"] for d in dets])
    dy = np.array([d["y"] for d in dets])
    d, _i = cKDTree(np.stack([dx, dy], 1)).query(np.stack([x, y], 1))
    out = []
    for nm, nra, ndec in NAMED:
        sep = np.hypot((ra - nra) * np.cos(np.radians(ndec)), dec - ndec)
        k = int(np.argmin(sep))
        if sep[k] < 0.35 and d[k] < tol_px:
            out.append((nm, float(vmag[k]), float(alt[k]), float(az[k]),
                        float(x[k]), float(y[k]), float(d[k])))
    out.sort(key=lambda r: r[1])
    return out


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    if not args:
        print(__doc__)
        return 2
    frame = Path(args[0])
    when = frame_time(frame)
    res = star_count.count_stars(frame)
    dets, mask = res["_stars"], res["_mask"]
    print("frame %s at %s: %d detections, purity %.3f"
          % (frame.name, when.isoformat(timespec="seconds"), res["stars"],
             res["purity"]))

    if "--verify" in argv:
        sol = load()
        if sol is None:
            print("no stored solution")
            return 1
        v = verify(sol, frame, when, dets)
        print("stored solution: %d matches, residual %s px"
              % (v["matches"], v["residual_px"]))
    else:
        sol = solve(frame, when, dets, verbose=True)
        if sol is None:
            print("no solution found")
            return 1
        print("solved: %d matches, residual %s px" % (sol["matches"],
                                                      sol["residual_px"]))
        print("  axis alt %.1f deg, az %.1f deg | FOV %.0f deg diagonal | %.4f deg/px"
              % (sol["axis_alt_deg"], sol["axis_az_deg"], sol["fov_diag_deg"],
                 sol["deg_per_px"]))
        if "--save" in argv:
            print("saved ->", save(sol))

    if "--report" in argv:
        lm = limiting_magnitude(sol, frame, when, dets, mask)
        print("\nlimiting magnitude: %s  (%d of %d unmasked catalogue stars found)"
              % (lm["limiting_mag"], lm["stars_matched"], lm["stars_visible_area"]))
        print("  Vmag   n  found  frac")
        for row in lm["completeness"]:
            print("  %4.2f %4d %6d  %4.0f%%"
                  % (row["vmag"], row["n"], row["found"], 100 * row["fraction"]))
        named = named_identifications(sol, when, dets)
        print("\nnamed stars identified (%d):" % len(named))
        for nm, vm, al, azd, x, y, dd in named:
            print("  %-12s V=%4.2f  alt %5.1f  az %5.1f  at (%4.0f,%4.0f)  %.1f px"
                  % (nm, vm, al, azd, x, y, dd))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
