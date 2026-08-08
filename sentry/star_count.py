#!/usr/bin/env python3
"""Count stars in an all-sky frame from the Kasa camera.

Why a star count at all: the camera has no manual exposure (see
scripts/probe_kasa_camera.report_exposure_verdict), so auto-exposure
renormalises every frame and absolute sky brightness is not comparable between
frames. How many stars are visible is scale-invariant, so it survives the
renormalisation -- and it is the quantity a cloud detector actually wants.

Two things had to be got right, and both were got wrong first:

  * The threshold is set by a NEGATIVE-IMAGE CONTROL, not by a sigma
    multiplier. Run the identical detector on the negated residual and every
    hit is by construction a false positive, because sky noise is close to
    symmetric. On a clear frame the default 12 ADU gave 185 real against 1
    false. A plain 5-sigma cut gave 347 "stars", mostly noise.

  * Foliage is masked on LEVEL, not texture. At night exposure the trees are
    too dim to be rough -- high-pass RMS runs 2.6-3.4 over foliage against
    1.5-2.5 over open sky, which does not separate them -- but they sit
    10-22 ADU above a ~2 ADU sky. So the skyglow is fitted with a robust 2-D
    cubic and the residual is thresholded. A texture mask masked 57% of the
    frame including the three brightest stars.

The returned false_positives is worth logging: it is a live check that the
threshold still means what it meant, and it climbs when the frame goes strange.

Usage:  python sentry/star_count.py frame.jpg [--annotate out.png] [--sweep]
"""
import os
import sys

import numpy as np
from scipy import ndimage

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Bands the camera burns into every frame; never sky.
TIMESTAMP_BOX = (slice(0, 130), slice(0, 900))
WATERMARK_BOX = (slice(0, 130), slice(2000, None))


def _load(image):
    if isinstance(image, np.ndarray):
        return image.astype(float)
    from PIL import Image
    return np.array(Image.open(str(image)).convert("L")).astype(float)


def _design(x, y, w, h):
    x, y = x / w, y / h
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y,
                     x ** 3, x * x * y, x * y * y, y ** 3], 1)


def skyglow_fit(a, step=8):
    """Robust 2-D cubic fit to the sky background. Returns (model, residual_sigma).

    Iterated with an asymmetric clip: anything well ABOVE the current fit is
    dropped (trees, and the bright left-edge glow lobe), anything below is
    kept. A symmetric clip lets the trees pull the model up into themselves.
    """
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    lvl = ndimage.median_filter(a, 31)
    xs, ys, zs = (xx[::step, ::step].ravel(), yy[::step, ::step].ravel(),
                  lvl[::step, ::step].ravel())
    A = _design(xs, ys, w, h)
    ok = ys >= TIMESTAMP_BOX[0].stop            # the burnt-in text is not sky
    sigma = 1.0
    coef = None
    for _ in range(6):
        coef, *_ = np.linalg.lstsq(A[ok], zs[ok], rcond=None)
        r = zs - A @ coef
        sigma = 1.4826 * np.median(np.abs(r[ok] - np.median(r[ok]))) or 1.0
        ok = (r < 2.0 * sigma) & (r > -5.0 * sigma) & (ys >= TIMESTAMP_BOX[0].stop)
    model = (_design(xx.ravel(), yy.ravel(), w, h) @ coef).reshape(h, w)
    return model, lvl, float(sigma)


def foliage_mask(a, resid_adu=2.5):
    """True where the frame is trees, burnt-in overlay, or otherwise not sky."""
    model, lvl, sigma = skyglow_fit(a)
    m = (lvl - model) > max(resid_adu, 3.0 * sigma)
    m = ndimage.binary_closing(m, np.ones((25, 25)))
    m = ndimage.binary_opening(m, np.ones((17, 17)))
    m = ndimage.binary_dilation(m, np.ones((21, 21)))
    m[TIMESTAMP_BOX] = True
    m[WATERMARK_BOX] = True
    return m, model, sigma


def _detect(resid, mask, threshold, fwhm_range, max_elong):
    lo, hi = fwhm_range
    lab, _ = ndimage.label((resid > threshold) & (~mask), structure=np.ones((3, 3)))
    found = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        ys, xs = np.where(lab[sl] == i)
        if len(ys) < 6:
            continue
        Y, X = ys + sl[0].start, xs + sl[1].start
        w = resid[Y, X]
        cx, cy = (X * w).sum() / w.sum(), (Y * w).sum() / w.sum()
        ev, _e = np.linalg.eigh(np.cov(np.stack([X - cx, Y - cy]), aweights=w))
        ev = np.clip(ev, 1e-6, None)
        fwhm, elong = 2.355 * ev[-1] ** 0.5, (ev[-1] / ev[0]) ** 0.5
        if not (lo <= fwhm <= hi) or elong > max_elong:
            continue
        found.append(dict(x=float(cx), y=float(cy), peak=float(w.max()),
                          flux=float(w.sum()), fwhm=float(fwhm),
                          elong=float(elong), npx=int(len(ys))))
    return found


def count_stars(image, threshold=None, fwhm_range=None, max_elong=None,
                foliage_resid=None, annotate=None):
    """Count star-like point sources. Returns a dict of the measurement."""
    from configs import config
    cam = config.data().get("sky camera", {})
    threshold = float(threshold if threshold is not None
                      else cam.get("star_threshold_adu", 12.0))
    fwhm_range = tuple(fwhm_range if fwhm_range is not None
                       else cam.get("star_fwhm_px", (2.0, 9.0)))
    max_elong = float(max_elong if max_elong is not None
                      else cam.get("star_max_elongation", 2.2))
    foliage_resid = float(foliage_resid if foliage_resid is not None
                          else cam.get("foliage_resid_adu", 2.5))

    a = _load(image)
    mask, model, glow_sigma = foliage_mask(a, foliage_resid)
    resid = a - ndimage.median_filter(a, 25)
    open_sky = ~mask
    noise = float(1.4826 * np.median(np.abs(resid[open_sky]))) or 1.0

    stars = _detect(resid, mask, threshold, fwhm_range, max_elong)
    # Same detector on the negated residual: every hit there is a false
    # positive, so this is a live purity measurement rather than an assumption.
    false_pos = len(_detect(-resid, mask, threshold, fwhm_range, max_elong))

    # Purity is the whole reason the negative control is computed every frame
    # rather than once at tuning time. On a clear frame it sits at ~99%. On the
    # 2026-08-07 rain frame the same detector returned 521 "stars" against 286
    # false positives -- 45% -- because rain fills the frame with defocused
    # streak fragments that pass a point-source cut. Publishing that count
    # unqualified would have reported the rainiest frame of the night as the
    # clearest sky of the week, so a low-purity frame is marked untrustworthy
    # and its count must not be read as a sky-condition measurement.
    purity = (len(stars) - false_pos) / max(len(stars), 1)
    out = {
        "stars": len(stars),
        "false_positives": false_pos,
        "purity": round(float(purity), 3),
        "trustworthy": bool(purity >= 0.85 and len(stars) >= 0),
        "threshold_adu": threshold,
        "noise_adu": round(noise, 3),
        "threshold_sigma": round(threshold / noise, 2),
        "sky_median_adu": round(float(np.median(a[open_sky])), 2),
        "masked_fraction": round(float(mask.mean()), 4),
        "median_fwhm_px": round(float(np.median([s["fwhm"] for s in stars])), 2)
                          if stars else None,
        "brightest_peak_adu": round(max((s["peak"] for s in stars), default=0.0), 1),
    }
    if annotate:
        _annotate(a, model, mask, stars, out, annotate)
    # Underscore keys are working data, not measurements: callers that publish
    # this dict strip them. The mask is handed back so the plate solver can
    # reuse it instead of paying for a second median filter over 3.7 Mpx.
    out["_stars"] = stars
    out["_mask"] = mask
    return out


def _annotate(a, model, mask, stars, meta, path):
    from PIL import Image, ImageDraw
    h, w = a.shape
    disp = np.clip((a - model * 0.15) / 60.0, 0, 1) ** 0.75 * 255
    rgb = Image.fromarray(disp.astype("uint8")).convert("RGBA")
    shade = np.zeros((h, w, 4), "uint8")
    shade[mask] = (255, 70, 70, 30)
    rgb = Image.alpha_composite(rgb, Image.fromarray(shade)).convert("RGB")
    dr = ImageDraw.Draw(rgb)
    for s in stars:
        r = max(11, 3.0 * s["fwhm"])
        col = (255, 210, 60) if s["peak"] > 40 else (90, 230, 255)
        dr.ellipse([s["x"] - r, s["y"] - r, s["x"] + r, s["y"] + r],
                   outline=col, width=3)
    dr.text((30, h - 40), "%d stars  ( >%.0f ADU = %.1f sigma; %d false by "
            "negative control )" % (meta["stars"], meta["threshold_adu"],
                                    meta["threshold_sigma"],
                                    meta["false_positives"]),
            fill=(255, 255, 0))
    rgb.save(str(path))


def sweep(image):
    """Threshold vs purity, the way the default was chosen. For re-tuning."""
    a = _load(image)
    mask, _model, _s = foliage_mask(a)
    resid = a - ndimage.median_filter(a, 25)
    print(" thr(ADU)   stars   false   purity")
    for t in (8, 10, 12, 15, 20, 30, 50):
        p = len(_detect(resid, mask, t, (2.0, 9.0), 2.2))
        q = len(_detect(-resid, mask, t, (2.0, 9.0), 2.2))
        print("   %5d   %5d   %5d   %5.1f%%"
              % (t, p, q, 100.0 * (p - q) / max(p, 1)))


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    if not args:
        print(__doc__)
        sys.exit(2)
    if "--sweep" in sys.argv:
        sweep(args[0])
        sys.exit(0)
    ann = None
    if "--annotate" in sys.argv:
        i = sys.argv.index("--annotate")
        ann = sys.argv[i + 1] if len(sys.argv) > i + 1 else "annotated.png"
    res = count_stars(args[0], annotate=ann)
    for k, v in res.items():
        if not k.startswith("_"):
            print("%-20s %s" % (k, v))
