"""Measure the reflection ghost from the ghost-test frames (scripts/ghost_test_sequence.py).

    python scripts/ghost_test_measure.py <LIGHT folder> [more folders] [--pattern '*_L_*'] [--min-move 200]

Every frame holds one very bright star at a known place on the sensor. Field stars move
with the pointing, one for one, between frames; a hot pixel does not move at all; a
reflection ghost moves by a different rule. This finds, in each frame, the bright star
(the biggest saturated blob) and the compact sources, pairs the frames, drops every source
that moved like the field or not at all, and keeps the ones that moved like nothing else
and agree across all the pairs. For each ghost it fits

    ghost = A + M . star        (detector pixels; M a 2x2 matrix, A a vector)

over all frames, which reads directly: M = -m I is an inversion through the fixed point
C = (I - M)^-1 A with magnification m; a one-to-one M with an offset is a tilted flat;
anything else names a tilted or decentred element. The ghost's counts per second are
printed beside the star's so the reflection fraction follows from the star's magnitude.

Reads FITS only; writes nothing but the report (and --json).
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)


def sources(img, star_xy, star_radius=400, sigma=8.0, smooth=1.0, binf=4):
    """Compact sources (x, y, peak) on the binned frame, the bright star's halo masked."""
    from scipy import ndimage as ndi
    h, w = img.shape[0] // binf * binf, img.shape[1] // binf * binf
    b = img[:h, :w].reshape(h // binf, binf, w // binf, binf).mean(axis=(1, 3))
    b = b - np.median(b)
    noise = np.median(np.abs(b)) * 1.4826
    sm = ndi.gaussian_filter(b, smooth)
    yy, xx = np.mgrid[0:b.shape[0], 0:b.shape[1]]
    halo = np.hypot(xx * binf - star_xy[0], yy * binf - star_xy[1]) < star_radius
    sm[halo] = 0
    pk = (sm == ndi.maximum_filter(sm, 7)) & (sm > sigma * noise / (smooth * np.sqrt(4 * np.pi)))
    ys, xs = np.nonzero(pk)
    return np.array([(x * binf + binf / 2, y * binf + binf / 2, sm[y, x]) for y, x in zip(ys, xs)]), noise


def bright_star(img, sat=60000):
    """Centroid of the largest saturated blob: the test star."""
    from scipy import ndimage as ndi
    lab, n = ndi.label(img >= sat * 0.98)
    if n == 0:
        return None
    sizes = ndi.sum(np.ones_like(lab), lab, index=range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    cy, cx = ndi.center_of_mass(lab == k)
    return np.array([cx, cy]), int(sizes[k - 1])


def refine_shift(a, b, shift, tol):
    """The median displacement of the sources of a that land near a source of b."""
    if len(a) == 0 or len(b) == 0:
        return shift
    moved = a[:, :2] + shift
    d = np.hypot(moved[:, 0, None] - b[None, :, 0], moved[:, 1, None] - b[None, :, 1])
    k = d.argmin(axis=1); ok = d.min(axis=1) < tol
    if ok.sum() < 5:
        return shift
    return shift + np.median(b[k[ok], :2] - moved[ok], axis=0)


def match(a, b, shift, tol):
    """Indices of a whose shifted position has a source in b within tol."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros(len(a), bool)
    moved = a[:, :2] + shift
    d = np.hypot(moved[:, 0, None] - b[None, :, 0], moved[:, 1, None] - b[None, :, 1])
    return d.min(axis=1) < tol


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--pattern", default="*.fits")
    ap.add_argument("--min-move", type=float, default=200, help="pairs need the star moved this far (px)")
    ap.add_argument("--tol", type=float, default=12, help="match tolerance (px)")
    ap.add_argument("--json", help="write the fit here")
    args = ap.parse_args(argv)
    from astropy.io import fits
    import cv2

    frames = []
    for folder in args.folders:
        frames += sorted(glob.glob(os.path.join(folder, args.pattern)))
    print("%d frames" % len(frames))
    data = []
    for p in frames:
        with fits.open(p, memmap=False) as h:
            img = h[0].data.astype(np.float32)
            hdr = h[0].header
        img = cv2.medianBlur(img, 3)
        star = bright_star(img)
        if star is None:
            print("  %s: no saturated star, skipped" % os.path.basename(p))
            continue
        (sxy, npx) = star
        src, noise = sources(img, sxy)
        exp = float(hdr.get("EXPTIME", hdr.get("EXPOSURE", 0)) or 0)
        data.append({"path": p, "name": os.path.basename(p), "filter": hdr.get("FILTER", "?"),
                     "exptime": exp, "star": sxy, "sat_px": npx, "src": src, "noise": noise})
        print("  %-45s %-4s star at (%6.0f,%6.0f) %5d saturated px, %3d sources, noise %.0f"
              % (os.path.basename(p)[:45], hdr.get("FILTER", "?"), sxy[0], sxy[1], npx, len(src), noise))
    if len(data) < 2:
        raise SystemExit("need at least two frames with the star")

    # Field alignment between frames: the star's own displacement (the pointing), refined
    # by the median residual of the matched field sources.
    cand = {i: np.ones(len(d["src"]), bool) for i, d in enumerate(data)}
    pairs = 0
    for i in range(len(data)):
        for j in range(len(data)):
            if i == j:
                continue
            dS = data[j]["star"] - data[i]["star"]
            if np.hypot(*dS) < args.min_move:
                continue
            pairs += 1
            a, b = data[i]["src"], data[j]["src"]
            # The star's centroid is a saturated blob and the pair may carry a little
            # rotation, so refine the field shift on the sources themselves first.
            shift = refine_shift(a, b, dS, 3 * args.tol)
            like_field = match(a, b, shift, args.tol)
            still = match(a, b, np.zeros(2), args.tol)      # hot pixels and the like
            cand[i] &= ~like_field & ~still
    for i, d in enumerate(data):
        cand[i] &= d["src"][:, 2] < 30000                 # a ghost is never saturated
    print("%d frame pairs with the star moved > %.0f px" % (pairs, args.min_move))
    # Candidates per frame; link them across frames by fitting ghost = A + M star on the
    # brightest consistent set: try every candidate in the first frame as the seed.
    first = next(i for i in cand if cand[i].any()) if any(c.any() for c in cand.values()) else None
    if first is None:
        print("no source moves unlike the field: no ghost above the threshold")
        return 0
    seeds = data[first]["src"][cand[first]]
    print("%d candidate sources in the first frame move unlike the field" % len(seeds))
    results = []
    for seed in seeds[np.argsort(-seeds[:, 2])][:10]:
        # In each other frame, the ghost candidate nearest to the prediction of a pure
        # inversion through the sensor centre with m=1 is a poor guess; instead take, per
        # frame, the brightest candidate within 1500 px of the seed, then refine.
        pts = []
        for i, d in enumerate(data):
            c = d["src"][cand[i]]
            if len(c) == 0:
                continue
            dd = np.hypot(c[:, 0] - seed[0], c[:, 1] - seed[1])
            near = c[dd < 1500]
            if len(near) == 0:
                continue
            g = near[np.argmax(near[:, 2])]
            pts.append((d["star"], g[:2], g[2], i))
        if len(pts) < 3:
            continue
        S = np.array([p[0] for p in pts]); G = np.array([p[1] for p in pts])
        X = np.c_[S, np.ones(len(S))]
        coef, *_ = np.linalg.lstsq(X, G, rcond=None)       # G = S M^T + A
        M = coef[:2].T; A = coef[2]
        resid = G - X @ coef
        rms = float(np.sqrt((resid ** 2).sum(axis=1).mean()))
        # drop the worst point once if it helps
        if len(pts) > 4 and rms > 3 * args.tol:
            keep = np.argsort(-np.hypot(*resid.T))[1:]
            coef, *_ = np.linalg.lstsq(X[keep], G[keep], rcond=None)
            M = coef[:2].T; A = coef[2]; resid = G[keep] - X[keep] @ coef
            rms = float(np.sqrt((resid ** 2).sum(axis=1).mean()))
        try:
            C = np.linalg.solve(np.eye(2) - M, A)
        except np.linalg.LinAlgError:
            C = np.array([np.nan, np.nan])
        ev = np.linalg.eigvals(M)
        results.append({"n": len(pts), "rms": rms, "M": M.tolist(), "A": A.tolist(), "centre": C.tolist(),
                        "eig": [complex(e).real for e in ev], "peak": float(np.mean([p[2] for p in pts])),
                        "frames": [data[p[3]]["name"] for p in pts]})
    results.sort(key=lambda r: (r["rms"] / max(r["n"], 1)))
    seen = []
    print("\nghost fits (ghost = A + M.star, detector pixels):")
    for r in results:
        key = tuple(np.round(r["A"], -2))
        if key in seen:
            continue
        seen.append(key)
        M = np.array(r["M"])
        print("  %d frames, rms %.1f px, mean peak %.0f ADU/binned px" % (r["n"], r["rms"], r["peak"]))
        print("    M = [[%+.3f %+.3f] [%+.3f %+.3f]]  eigenvalues %+.2f %+.2f  ->  %s"
              % (M[0, 0], M[0, 1], M[1, 0], M[1, 1], r["eig"][0], r["eig"][1],
                 "inversion (point reflection)" if max(r["eig"]) < -0.3 else
                 "follows the star (flat plate)" if min(r["eig"]) > 0.7 else "anisotropic"))
        print("    fixed point C = (%.0f, %.0f)  [sensor centre (4788, 3194)]" % tuple(r["centre"]))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"frames": [{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in d.items() if k != "src"}
                                  for d in data], "fits": results}, f, indent=1, default=float)
        print("wrote", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
