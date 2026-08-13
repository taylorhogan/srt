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

Everything tunable is read from a named config profile, so a second camera gets
its own thresholds instead of inheriting numbers measured on this one. The
default profile is the Kasa camera the detector was written for.

Usage:  python sentry/star_count.py frame.jpg [--annotate out.png] [--sweep]
                                              [--profile NAME]
"""
import os
import sys

import numpy as np
from scipy import ndimage

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

DEFAULT_PROFILE = "sky camera"

# Bands the Kasa camera burns into every frame; never sky. Other cameras burn in
# nothing, so this is a property of that camera rather than of the detector --
# a profile overrides it with "overlay_boxes", and an empty list means the whole
# frame is real pixels.
TIMESTAMP_BOX = (slice(0, 130), slice(0, 900))
WATERMARK_BOX = (slice(0, 130), slice(2000, None))
DEFAULT_BOXES = (TIMESTAMP_BOX, WATERMARK_BOX)


def _boxes(spec):
    """Config ``[[[y0, y1], [x0, x1]], ...]`` as slices. None keeps the Kasa bands."""
    if spec is None:
        return DEFAULT_BOXES
    return tuple((slice(y0, y1), slice(x0, x1)) for (y0, y1), (x0, x1) in spec)


def _sky_y0(boxes):
    """First row that can be sky: below any overlay band along the top edge.

    Only bands that start at row 0 count. A box floating in the middle of the
    frame excludes its own pixels but says nothing about where sky begins, and
    treating it as a top margin would throw away everything above it.
    """
    return max([b[0].stop or 0 for b in boxes if not b[0].start], default=0)


def _load(image):
    """A frame as a 2-D float array, from an array, a FITS file or a picture.

    FITS is here because the ASI all-sky camera stores its 16-bit frames that
    way: a JPEG would throw away the depth the faint end of the completeness
    table is measured in.
    """
    if isinstance(image, np.ndarray):
        return image.astype(float)
    if str(image).lower().endswith((".fits", ".fit", ".fts")):
        from astropy.io import fits
        with fits.open(str(image)) as hdul:
            return np.asarray(hdul[0].data, dtype=float)
    from PIL import Image
    return np.array(Image.open(str(image)).convert("L")).astype(float)


def _design(x, y, w, h):
    x, y = x / w, y / h
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y,
                     x ** 3, x * x * y, x * y * y, y ** 3], 1)


def skyglow_fit(a, step=8, boxes=DEFAULT_BOXES):
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
    y0 = _sky_y0(boxes)                         # burnt-in text is not sky
    ok = ys >= y0
    sigma = 1.0
    coef = None
    for _ in range(6):
        coef, *_ = np.linalg.lstsq(A[ok], zs[ok], rcond=None)
        r = zs - A @ coef
        sigma = 1.4826 * np.median(np.abs(r[ok] - np.median(r[ok]))) or 1.0
        ok = (r < 2.0 * sigma) & (r > -5.0 * sigma) & (ys >= y0)
    model = (_design(xx.ravel(), yy.ravel(), w, h) @ coef).reshape(h, w)
    return model, lvl, float(sigma)


def foliage_mask(a, resid_adu=2.5, boxes=DEFAULT_BOXES):
    """True where the frame is trees, burnt-in overlay, or otherwise not sky."""
    model, lvl, sigma = skyglow_fit(a, boxes=boxes)
    m = (lvl - model) > max(resid_adu, 3.0 * sigma)
    m = ndimage.binary_closing(m, np.ones((25, 25)))
    m = ndimage.binary_opening(m, np.ones((17, 17)))
    m = ndimage.binary_dilation(m, np.ones((21, 21)))
    for b in boxes:
        m[b] = True
    return m, model, sigma


def _detect(resid, mask, threshold, fwhm_range, max_elong, min_px=6):
    """Point sources above threshold. *min_px* is the smallest blob allowed.

    That floor is the only thing standing between the count and the sensor's hot
    pixels, which are single-pixel spikes far brighter than any star. It is a
    profile setting because it trades directly against how well the camera
    samples a star: at 6 px the Kasa keeps its stars and loses every hot pixel,
    but a sharply focused lens on a small sensor puts a star into fewer pixels
    than that, and the same floor would then reject the sky along with the
    spikes. Lowering it means the hot pixels have to be removed some other way
    -- which is what the all-sky camera's master dark is for.
    """
    lo, hi = fwhm_range
    lab, _ = ndimage.label((resid > threshold) & (~mask), structure=np.ones((3, 3)))
    found = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        ys, xs = np.where(lab[sl] == i)
        if len(ys) < min_px:
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
                foliage_resid=None, annotate=None, profile=DEFAULT_PROFILE):
    """Count star-like point sources. Returns a dict of the measurement.

    *profile* names the config section the thresholds come from, so each camera
    is measured against its own tuning rather than the one this was written for.
    """
    from configs import config
    cam = config.data().get(profile, {})
    threshold = float(threshold if threshold is not None
                      else cam.get("star_threshold_adu", 12.0))
    fwhm_range = tuple(fwhm_range if fwhm_range is not None
                       else cam.get("star_fwhm_px", (2.0, 9.0)))
    max_elong = float(max_elong if max_elong is not None
                      else cam.get("star_max_elongation", 2.2))
    foliage_resid = float(foliage_resid if foliage_resid is not None
                          else cam.get("foliage_resid_adu", 2.5))
    boxes = _boxes(cam.get("overlay_boxes"))
    min_px = int(cam.get("star_min_pixels", 6))

    a = _load(image)
    mask, model, glow_sigma = foliage_mask(a, foliage_resid, boxes)
    resid = a - ndimage.median_filter(a, 25)
    open_sky = ~mask
    noise = float(1.4826 * np.median(np.abs(resid[open_sky]))) or 1.0

    stars = _detect(resid, mask, threshold, fwhm_range, max_elong, min_px)
    # Same detector on the negated residual: every hit there is a false
    # positive, so this is a live purity measurement rather than an assumption.
    false_pos = len(_detect(-resid, mask, threshold, fwhm_range, max_elong, min_px))

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


def sweep(image, profile=DEFAULT_PROFILE, thresholds=None):
    """Threshold vs purity, the way the default was chosen. For re-tuning.

    This is how a new camera's threshold gets set: the negative-image control
    makes the false column a measurement rather than a guess, so the threshold
    can be read off the purity a given camera actually achieves.
    """
    from configs import config
    cam = config.data().get(profile, {})
    fwhm_range = tuple(cam.get("star_fwhm_px", (2.0, 9.0)))
    max_elong = float(cam.get("star_max_elongation", 2.2))
    boxes = _boxes(cam.get("overlay_boxes"))
    min_px = int(cam.get("star_min_pixels", 6))
    a = _load(image)
    mask, _model, _s = foliage_mask(a, float(cam.get("foliage_resid_adu", 2.5)),
                                    boxes)
    resid = a - ndimage.median_filter(a, 25)
    noise = float(1.4826 * np.median(np.abs(resid[~mask]))) or 1.0
    if thresholds is None:
        # Scaled off the frame's own noise, so this is usable on a 16-bit camera
        # whose ADU are nothing like the 8-bit ladder the Kasa was tuned on.
        thresholds = [round(k * noise, 1) for k in (2, 3, 4, 5, 6, 8, 12, 20)]
    print("noise %.2f ADU in the unmasked frame" % noise)
    print(" thr(ADU)  sigma   stars   false   purity")
    for t in thresholds:
        p = len(_detect(resid, mask, t, fwhm_range, max_elong, min_px))
        q = len(_detect(-resid, mask, t, fwhm_range, max_elong, min_px))
        print("  %7.1f  %5.1f   %5d   %5d   %5.1f%%"
              % (t, t / noise, p, q, 100.0 * (p - q) / max(p, 1)))


def pop_arg(argv, flag, default=None):
    """Value of ``--flag``, REMOVED from argv along with the flag.

    Removing it is the point: what is left is the positional arguments. Reading
    the value in place and then filtering on a leading ``--`` leaves the value
    itself sitting in the positional list, so ``--profile "allsky camera"``
    would be taken for the frame to measure.
    """
    if flag not in argv:
        return default
    i = argv.index(flag)
    val = argv[i + 1] if len(argv) > i + 1 else None
    if val is None or val.startswith("--"):
        del argv[i:i + 1]
        return default
    del argv[i:i + 2]
    return val


if __name__ == "__main__":
    argv = list(sys.argv[1:])
    prof = pop_arg(argv, "--profile", DEFAULT_PROFILE)
    ann = (pop_arg(argv, "--annotate", "annotated.png")
           if "--annotate" in argv else None)
    args = [x for x in argv if not x.startswith("--")]
    if not args:
        print(__doc__)
        sys.exit(2)
    if "--sweep" in argv:
        sweep(args[0], prof)
        sys.exit(0)
    res = count_stars(args[0], annotate=ann, profile=prof)
    for k, v in res.items():
        if not k.startswith("_"):
            print("%-20s %s" % (k, v))
