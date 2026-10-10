"""Richardson-Lucy deconvolution of one linear channel, ring-free by construction.

`process <dso> <recipe> deblur=8` runs this on the luminance stack before the
stretch. Everything below is a rule that was paid for on the M33 L stack on
2026-10-09, each measured on crops before it went in:

* RL must see a SKY-FREE image. On a synthetic star with the exactly correct
  kernel, 15 iterations with the sky left in dug a ring 36 sigma below sky
  around it; with the sky taken out the ring was gone (-0.2 sigma) and the
  sharpening was better. Positivity is the only brake RL has, and a sky
  pedestal gives it 350 ADU of room to dig.
* The floor has to follow the galaxy, not the empty sky, or stars inside the
  disc still ring - but a smoothed percentile model sits ABOVE the dust lanes,
  and clipping the image to it erased them (8.6% of pixels clipped, the core
  turned to haze). The floor is therefore min(smoothed minimum-filter model,
  the pixel itself) - 2 sigma: nothing is ever clipped.
* Even so RL rings at finite iterations, so the same dark deringing clamp
  PixInsight applies is used: no pixel ends below its input value minus one
  sky sigma. Stars keep a soft halo under a brighter core and their flux is
  not conserved, which is cosmetic.
* Faint sky texture curdles into worms under RL, so the result is blended in
  only where the signal above the local model is well clear of the noise.
* The kernel needs the WING. A kernel measured with a local annulus at 28-32
  px has no profile beyond it, and RL cannot remove a wing it does not know
  about: on ngc2146 the bright-star profile matched the kernel to 10 px, was
  1.4x the kernel at 20 px, 2.3x at 25, and carried on to 70 px where the
  kernel was zero. 18% of a star's flux sits beyond 12 px. Deconvolving with
  that kernel tightened the core, drained the inner wing to the sky and left
  the outer wing standing as a ring of glow around every bright star (the
  operator saw them at once). The wing is now measured from the brightest
  clean stars out to 72 px and grafted onto the median core at 8-14 px.
* Only clipped pixels are protected. The earlier rule (anything above 30% of
  the saturation level) kept whole bright stars with their halos; a star is
  PSF-shaped right up to the pixels that are actually flat.
* Shrinking the kernel 15% was a hedge against ringing that the sky-free
  floor made unnecessary, and it under-sharpened; it is off by default.

* The clamp's local-sky reference follows the galaxy's slopes (a 60 px
  median, binned), not the minimum filter, which read tens of ADU low on the
  disc's edge: RL drained every star's wing down to it and left a dark moat
  around each star on the galaxy (the operator saw it at once, 2026-10-10).
* split=True (off by default) takes the stars off and treats them apart, the
  operator's suggestion for the moat. Three forms were tried that day and
  none beat the clamp fix: whole discs off a fill (the galaxy light under a
  bright star went into the star layer and RL swept it into the core, a dark
  oval beside each one); a full kernel fit (real stars have diffraction
  spikes and lopsided cores, and the symmetric fit's residual was a black
  four-lobed flower around every bright star); and the hybrid star_layer
  implements now, the fitted wing off beyond 10-16 px and the core lifted
  off a fill, which still showed the lobes because the asymmetry lives at
  10-20 px too. Kept for the record and for a field with cleaner stars.

Returns the input untouched, with a reason, when the field has too few
isolated stars to measure a kernel from or the stars are already at the
pixel scale.
"""
import logging
from typing import Optional

import numpy as np

_logger = logging.getLogger(__name__)

DEFAULT_ITERS = 8
PSF_SCALE = 1.0           # kernel shrink; 1.0 = as measured (see module doc)
PSF_RADIUS = 32           # half-size of the median-stacked core in px
PSF_TAPER = 8             # cosine taper width at the core's edge (no-wing case)
WING_RADIUS = 72          # the wing profile is measured out to here
WING_BLEND = (8.0, 14.0)  # core to wing crossfade, px
WING_MIN_STARS = 8        # fewer clean bright stars and the core alone is used
WING_MIN_PEAK = 3000.0    # ADU above sky: bright enough to show the wing
SPLIT_STARS = False       # take the stars off and treat them apart: see the module doc
CLAMP_BIN, CLAMP_MEDIAN = 4, 25   # the clamp's local-sky reference: median over 100 px, binned
STAR_SPLIT_MIN_SIGMA = 20.0  # candidates this far above the sky mean (or 1e-3 of saturation)
STAR_COMPACT = 0.6        # of a point source's peak/matched ratio: broader is not a star
STAR_CORE_PX = 14         # a star's core disc, lifted off a fill into its own layer
STAR_MASK_MAX_FRAC = 0.25 # more of the frame than this in the layer and the split is skipped
MIN_STARS = 15            # fewer isolated stars than this and nothing happens
MAX_STARS = 400           # brightest isolated stars used for the kernel
SAT_FRAC = 0.98           # of the saturation level: protected as clipped
SAT_MIN_PIXELS = 4        # pixels at the ceiling before it counts as saturation
SAT_DILATE = 6            # px of protection around the clipped pixels
SAT_FEATHER = 3.0         # and the feather on its edge
FLOOR_MIN_SIZE = 25       # minimum-filter window for the local floor model
FLOOR_SMOOTH = 8.0        # and its smoothing
CLAMP_SIGMA = 1.0         # dark deringing: never below input - this many sigma
BLEND_LO_SIGMA = 3.0      # signal above the local model where deblur starts
BLEND_HI_SIGMA = 10.0     # and where it is fully in


def sky_sigma(chan: np.ndarray, model: Optional[np.ndarray] = None) -> float:
    """Sky noise: the scatter of sky pixels about the local floor model.

    Not neighbour differences: a denoised stack has almost no pixel-to-pixel
    noise (0.05 ADU where the sky wobbles by 0.43) and that estimate turned
    every noise crest into a star. The residual about the minimum-filter
    model, over the darker half of the frame, measures the wobble itself,
    correlated or not; the model's constant offset drops out of the MAD.
    """
    from scipy.ndimage import gaussian_filter, minimum_filter
    x = np.asarray(chan, dtype=np.float32)[::2, ::2]
    if model is None:
        model = gaussian_filter(minimum_filter(x, FLOOR_MIN_SIZE), FLOOR_SMOOTH)
    else:
        model = np.asarray(model, dtype=np.float32)[::2, ::2]
    r = x - model
    # Sky pixels are chosen by BRIGHTNESS (the smoothed image's darker 60%),
    # never by the residual itself: cutting the residual's upper tail would
    # bias the MAD low (measured 2.1 for a true 3.0).
    sky = gaussian_filter(x, 5.0) <= np.nanpercentile(gaussian_filter(x, 5.0), 60)
    r = r[sky & np.isfinite(r)]
    if r.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def saturation_level(chan: np.ndarray) -> Optional[float]:
    """The ceiling if stars are clipped against it, else None.

    A saturated star is a flat top: several pixels at the same maximum. A
    field whose brightest star simply has one highest pixel has no saturation,
    and treating a fraction of ITS peak as "saturated" throws away every
    bright star in the field (28 of 30 on the synthetic test field).
    """
    ceiling = float(np.nanmax(chan))
    if not np.isfinite(ceiling) or ceiling <= 0:
        return None
    at_top = int(np.count_nonzero(chan >= 0.98 * ceiling))
    return ceiling if at_top >= SAT_MIN_PIXELS else None


def local_sky(x: np.ndarray, binning: int = 4, size: int = 25) -> np.ndarray:
    """Local sky that follows the galaxy's slopes: a median filter over
    size x size blocks of binning x binning px, back at full size."""
    from scipy.ndimage import median_filter, zoom
    h, w = x.shape
    hb, wb = h // binning, w // binning
    b = x[:hb * binning, :wb * binning].reshape(hb, binning, wb, binning).mean(axis=(1, 3))
    m = median_filter(b, size=size, mode="nearest")
    up = zoom(m, binning, order=1)
    out = np.empty(x.shape, np.float32); out[:] = float(np.median(m))
    out[:up.shape[0], :up.shape[1]] = up
    if up.shape[0] < h: out[up.shape[0]:, :up.shape[1]] = up[-1:, :]
    if up.shape[1] < w: out[:, up.shape[1]:] = out[:, up.shape[1] - 1:up.shape[1]]
    return out


def _gauss2(xy, amp, x0, y0, sig, base):
    x, y = xy
    return amp * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sig * sig)) + base


def measure_psf(chan: np.ndarray, radius: int = PSF_RADIUS,
                taper: int = PSF_TAPER) -> tuple[Optional[np.ndarray], dict]:
    """Median-stack the isolated, unsaturated stars of *chan* into a kernel.

    Each star is centred by a Gaussian fit to its core, its own sky taken
    from a local annulus (so the halo is kept — a global background mesh
    absorbs it), shifted to the centre pixel, peak-normalised; the median
    across stars rejects neighbours. A cosine taper takes the edge to zero.
    """
    from scipy.ndimage import maximum_filter, shift as ndshift
    from scipy.optimize import curve_fit

    a = np.asarray(chan, dtype=np.float64)
    h, w = a.shape
    sat = saturation_level(a)
    sig = sky_sigma(a)
    sky = float(np.nanmedian(a))
    finite = np.nan_to_num(a, nan=-np.inf)
    # Noise-relative, not ceiling-relative: a sky at 1% of the brightest star
    # (every field with a saturated star has one) would make every noise
    # peak a candidate and leave nothing isolated.
    peaks = (finite == maximum_filter(finite, 25)) & (finite > sky + 20.0 * sig)
    if sat is not None:
        peaks &= finite < SAT_FRAC * sat
    ys, xs = np.nonzero(peaks)
    stars = []
    yy7, xx7 = np.mgrid[:15, :15]
    for y, x in zip(ys, xs):
        if not (radius + 2 < y < h - radius - 2 and radius + 2 < x < w - radius - 2):
            continue
        c = a[y - 7:y + 8, x - 7:x + 8]
        if not np.isfinite(c).all():
            continue
        try:
            p, _ = curve_fit(_gauss2, (xx7.ravel(), yy7.ravel()), c.ravel(),
                             p0=(c.max() - np.median(c), 7, 7, 2.5, np.median(c)),
                             maxfev=2000)
        except Exception:
            continue
        if 0.5 < abs(p[3]) < 8 and 0 <= p[1] < 15 and 0 <= p[2] < 15 and p[0] > 10.0 * sig:
            stars.append((x - 7 + p[1], y - 7 + p[2], 2.3548 * abs(p[3]), float(c.max())))
    if not stars:
        return None, {"stars": 0, "why": "no fittable stars"}
    stars = np.array(stars)
    fwhm_px = float(np.median(stars[:, 2]))
    order = np.argsort(stars[:, 3])[::-1]
    cuts = []
    yy, xx = np.mgrid[:2 * radius + 1, :2 * radius + 1]
    rr = np.hypot(yy - radius, xx - radius)
    ann = (rr > radius - 4) & (rr <= radius)
    for i in order:
        xc, yc = stars[i, 0], stars[i, 1]
        d = np.hypot(stars[:, 0] - xc, stars[:, 1] - yc)
        d[i] = np.inf
        if d.min() < 2.5 * radius:
            continue
        xi, yi = int(round(xc)), int(round(yc))
        c = a[yi - radius:yi + radius + 1, xi - radius:xi + radius + 1].copy()
        if c.shape != (2 * radius + 1, 2 * radius + 1) or not np.isfinite(c).all():
            continue
        c -= np.median(c[ann])
        c = ndshift(c, (yi - yc, xi - xc), order=3, mode="nearest")
        if c.max() > 0:
            cuts.append(c / c.max())
        if len(cuts) >= MAX_STARS:
            break
    if len(cuts) < MIN_STARS:
        return None, {"stars": len(cuts), "fwhm_px": fwhm_px,
                      "why": f"only {len(cuts)} isolated stars (need {MIN_STARS})"}
    core = np.clip(np.median(np.stack(cuts), axis=0), 0, None)
    info = {"stars": len(cuts), "fwhm_px": fwhm_px}
    wing, nwing = _wing_profile(a, stars, sat, sig)
    info["wing_stars"] = nwing
    if wing is None:
        # No bright clean star to read a wing from: the core's own azimuthal
        # profile stands in, so the kernel is still smooth beyond the
        # crossfade. The median stack's outskirts carry ~1e-3 of the peak in
        # noise, which on a 3000 ADU star is 3 ADU of wrong wing.
        rb = np.round(rr).astype(int)
        wing = np.bincount(rb.ravel(), weights=core.ravel(), minlength=radius + 1)[:radius] \
            / np.bincount(rb.ravel(), minlength=radius + 1)[:radius]
        wing = np.clip(wing / max(float(core.max()), 1e-12), 0, None)
        wing[0] = 1.0
        t = np.clip((np.arange(radius) - (radius - taper)) / taper, 0, 1)
        wing = wing * 0.5 * (1 + np.cos(np.pi * t))
        R = radius
    else:
        R = WING_RADIUS
    if True:
        wy, wx = np.mgrid[-R:R + 1, -R:R + 1]; wr = np.hypot(wy, wx)
        psf = np.interp(wr, np.arange(len(wing)), wing, right=0.0) * core.max()
        c = np.zeros_like(psf); c[R - radius:R + radius + 1, R - radius:R + radius + 1] = core
        f = np.clip((wr - WING_BLEND[0]) / (WING_BLEND[1] - WING_BLEND[0]), 0, 1)
        f = f * f * (3 - 2 * f)
        psf = c * (1 - f) + psf * f
        t = np.clip((wr - (R - 10)) / 10.0, 0, 1)
        psf *= 0.5 * (1 + np.cos(np.pi * t)); psf[wr > R] = 0.0
        info["wing_flux_beyond_24px"] = float(psf[wr > 24].sum() / max(psf.sum(), 1e-12))
        info["wing_profile"] = wing
    s = float(psf.sum())
    if not np.isfinite(s) or s <= 0:
        return None, {"stars": len(cuts), "why": "kernel summed to nothing"}
    return psf / s, info


def _wing_profile(a: np.ndarray, stars: np.ndarray, sat: Optional[float],
                  sig: float) -> tuple[Optional[np.ndarray], int]:
    """Azimuthal profile of the brightest clean stars out to WING_RADIUS,
    peak-normalised, median across stars. The core stack cannot show the wing:
    its stars are fainter and its sky annulus sits inside the wing."""
    R = WING_RADIUS
    h, w = a.shape
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]; rr = np.hypot(yy, xx)
    sky_ann = (rr > R - 8) & (rr <= R)
    rbin = np.round(rr).astype(int)
    peak = stars[:, 3] - np.median(a)
    cand = np.nonzero(peak > WING_MIN_PEAK)[0]
    if sat is not None:
        cand = cand[stars[cand, 3] < SAT_FRAC * sat]
    profs = []
    for i in cand[np.argsort(peak[cand])[::-1]]:
        xc, yc = int(round(stars[i, 0])), int(round(stars[i, 1]))
        if not (R < yc < h - R and R < xc < w - R):
            continue
        c = a[yc - R:yc + R + 1, xc - R:xc + R + 1]
        if not np.isfinite(c).all():
            continue
        c = c - np.median(c[sky_ann])
        pk = float(c[R, R])
        if pk <= 0 or (c[rr > 20] > 0.05 * pk).any():      # a neighbour in the wing
            continue
        prof = np.bincount(rbin.ravel(), weights=c.ravel(), minlength=R + 1)[:R] \
            / np.bincount(rbin.ravel(), minlength=R + 1)[:R]
        profs.append(prof / pk)
        if len(profs) >= 60:
            break
    if len(profs) < WING_MIN_STARS:
        return None, len(profs)
    med = np.clip(np.median(np.stack(profs), axis=0), 0, None)
    med[0] = 1.0
    return med, len(profs)


def _shrink(psf: np.ndarray, factor: float) -> np.ndarray:
    from scipy.ndimage import zoom
    p = np.clip(zoom(psf, factor, order=3), 0, None)
    if p.shape[0] % 2 == 0:
        p = p[:-1, :-1]
    return p / p.sum()


def _fill_holes(x: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """x with *hole* pixels replaced by a smooth estimate from outside them:
    normalised convolution at growing scales, each pixel taking the smallest
    scale that sees enough un-masked support."""
    from scipy.ndimage import gaussian_filter
    keep = (~hole).astype(np.float32)
    out = x.astype(np.float32).copy()
    todo = hole.copy()
    for sg in (3.0, 6.0, 12.0, 24.0):
        if not todo.any():
            break
        num = gaussian_filter(x.astype(np.float32) * keep, sg)
        den = gaussian_filter(keep, sg)
        ok = todo & (den > 0.25)
        out[ok] = num[ok] / den[ok]
        todo &= ~ok
    if todo.any():
        out[todo] = float(np.median(x[~hole])) if (~hole).any() else float(np.median(x))
    return out


def star_layer(x: np.ndarray, model: np.ndarray, sig: float,
               kernel: np.ndarray, level: Optional[float] = None,
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """(starless, inner, mask, count, fits): the stars taken off the image.

    Candidates are local maxima STAR_SPLIT_MIN_SIGMA (or 1e-3 of the
    saturation level) above the sky mean that are as COMPACT as the kernel:
    peak over kernel-matched value at least STAR_COMPACT of a point source's
    (a galaxy knot is broader and fails). Brightest first, each star's
    sub-pixel centre is a Gaussian fit to its core and its core and wing
    amplitudes are least-squares fits of the shifted kernel, the fit being
    subtracted before the next star is measured.

    What comes off is in two parts, because a real star is not the kernel:
    it has diffraction spikes and a lopsided core, and subtracting the
    symmetric fit there left a four-lobed residual RL turned into a black
    flower around every bright star. So the FIT removes only the wing,
    beyond the 10-16 px crossfade, which is smooth and symmetric and fits to
    a few ADU; the CORE (r <= STAR_CORE_PX) is lifted off a smooth fill of
    its surroundings into *inner*, a layer on a zero background with the
    spikes and asymmetry intact, for positivity-only RL. The galaxy light
    under a core is the fill, lost to the deconvolution but not to the
    picture, since the core goes back on top of it. Outside the footprint
    the starless image is the input bit for bit.
    """
    from scipy.ndimage import maximum_filter, shift as ndshift
    from scipy.optimize import curve_fit
    from scipy.signal import fftconvolve
    h, w = x.shape
    ref = model + float(np.median(x - model))
    above = (x - ref).astype(np.float32)
    thresh = STAR_SPLIT_MIN_SIGMA * sig
    if level is not None:
        thresh = max(thresh, 1e-3 * level)
    peaks = (x == maximum_filter(x, 5)) & (above > thresh)
    R = kernel.shape[0] // 2
    core = kernel[R - 8:R + 9, R - 8:R + 9].astype(np.float32); core = core / core.sum()
    matched = fftconvolve(above, core, mode="same")
    point = float(kernel.max() / fftconvolve(kernel, core, mode="same").max())
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(matched > 0, above / np.maximum(matched, 1e-6), 0.0)
    peaks &= ratio >= STAR_COMPACT * point
    peaks[:R + 1] = peaks[-R - 1:] = peaks[:, :R + 1] = peaks[:, -R - 1:] = False
    ys, xs = np.nonzero(peaks)
    empty = (x.astype(np.float32), np.zeros_like(x), np.zeros(x.shape, bool), 0, np.zeros((0, 5)))
    if len(ys) == 0:
        return empty
    order = np.argsort(above[ys, xs])[::-1]
    work = above.copy()
    wing_part = np.zeros(x.shape, np.float32)
    inner_mask = np.zeros(x.shape, bool)
    clipped = (x >= 0.98 * level) if level is not None else np.zeros(x.shape, bool)
    ky, kx = np.mgrid[-R:R + 1, -R:R + 1]; kr = np.hypot(ky, kx)
    core_fit = (kr <= 6)
    wing_fit = (kr > 10) & (kr <= 36)
    k64 = kernel.astype(np.float64)
    fw = np.clip((kr - WING_BLEND[0]) / (WING_BLEND[1] - WING_BLEND[0]), 0, 1); fw = fw * fw * (3 - 2 * fw)
    disc = kr <= STAR_CORE_PX
    n = 0
    fits = []
    g7y, g7x = np.mgrid[:7, :7]
    for i in order:
        y, xx_ = int(ys[i]), int(xs[i])
        c = work[y - 3:y + 4, xx_ - 3:xx_ + 4]
        if c.max() <= 0:
            continue
        try:
            pf, _ = curve_fit(_gauss2, (g7x.ravel(), g7y.ravel()), c.ravel().astype(np.float64),
                              p0=(float(c.max()), 3.0, 3.0, 2.0, 0.0), maxfev=400)
            dy, dx = float(pf[2] - 3.0), float(pf[1] - 3.0)
            if not (abs(dy) < 1.5 and abs(dx) < 1.5):
                raise ValueError
        except Exception:
            cw = np.clip(c[1:6, 1:6], 0, None)
            dy = float((cw * np.arange(-2, 3)[:, None]).sum() / cw.sum())
            dx = float((cw * np.arange(-2, 3)[None, :]).sum() / cw.sum())
        k = ndshift(k64, (dy, dx), order=1, mode="constant", cval=0.0)
        sl = (slice(y - R, y + R + 1), slice(xx_ - R, xx_ + R + 1))
        d = work[sl]
        unclipped = ~clipped[sl]
        ok = core_fit & unclipped; kk = k[ok]
        amp = float((d[ok] * kk).sum() / max((kk * kk).sum(), 1e-12))
        if amp <= 0:
            continue
        ok = wing_fit & unclipped; kk = k[ok]
        amp_w = float(np.clip((d[ok] * kk).sum() / max((kk * kk).sum(), 1e-12), 0.0, 1.5 * amp))
        sub = ((amp * (1 - fw) + amp_w * fw) * k).astype(np.float32)
        work[sl] -= sub                       # the whole fit, so the next star is measured clean
        wing_part[sl] += sub * fw             # but only the wing comes off the picture
        inner_mask[sl] |= disc
        fits.append((y, xx_, dy, dx, float(sub.sum())))
        n += 1
    if n == 0:
        return empty
    base = (x - wing_part).astype(np.float32)
    fill = _fill_holes(base, inner_mask)
    starless = np.where(inner_mask, fill, base).astype(np.float32)
    inner = np.where(inner_mask, np.clip(base - fill, 0.0, None), 0.0).astype(np.float32)
    mask = inner_mask | (wing_part > max(0.5 * sig, 0.05))
    starless = np.where(mask, starless, x.astype(np.float32))
    return starless, inner, mask, n, np.array(fits, dtype=np.float64).reshape(-1, 5)


def deblur(chan: np.ndarray, iters: int = DEFAULT_ITERS,
           psf: Optional[np.ndarray] = None,
           psf_scale: float = PSF_SCALE,
           clamp_to: str = "input",
           split: bool = SPLIT_STARS) -> tuple[np.ndarray, dict]:
    """Return (deblurred channel, info). The input is linear ADU with its sky.

    clamp_to="input": no pixel ends below its input minus one sigma (star
    halos stay under a brighter core). "sky": a pixel may fall as far as the
    local sky mean minus one sigma, so halos are allowed to drain into the
    core, but nothing goes below the sky, which is the ring.

    info carries what was measured; info["why"] is set when the channel was
    returned as given.
    """
    from scipy.ndimage import binary_dilation, gaussian_filter, minimum_filter
    from skimage.restoration import richardson_lucy

    iters = int(iters)
    if iters <= 0:
        return chan, {"iters": 0, "why": "deblur=0"}
    if clamp_to not in ("input", "sky"):
        raise ValueError(f"clamp_to must be 'input' or 'sky', not {clamp_to!r}")
    info: dict = {"iters": iters, "psf_scale": psf_scale, "clamp_to": clamp_to, "split": bool(split)}
    wing = None
    if psf is None:
        psf, pinfo = measure_psf(chan)
        wing = pinfo.pop("wing_profile", None)
        info.update(pinfo)
        if psf is None:
            _logger.warning("deblur skipped: %s", pinfo.get("why"))
            return chan, info
        if pinfo["fwhm_px"] < 1.5:
            info["why"] = f"stars already at the pixel scale ({pinfo['fwhm_px']:.2f} px)"
            _logger.warning("deblur skipped: %s", info["why"])
            return chan, info
    kernel = _shrink(psf, psf_scale) if psf_scale != 1.0 else psf
    info["kernel_px"] = int(kernel.shape[0])

    x = np.nan_to_num(np.asarray(chan, dtype=np.float32), nan=0.0)
    model0 = gaussian_filter(minimum_filter(x, FLOOR_MIN_SIZE), FLOOR_SMOOTH)
    sig = sky_sigma(x, model0)
    info["sky_sigma"] = sig
    level = saturation_level(x)
    sat = x > SAT_FRAC * level if level is not None else np.zeros(x.shape, bool)
    full = x
    stars = None
    if split:
        starless, inner, smask, nstars, fits = star_layer(x, model0, sig, kernel, level)
        info["split_stars"] = nstars
        info["star_mask_frac"] = float(smask.mean())
        if smask.mean() > STAR_MASK_MAX_FRAC or nstars == 0:
            info["split_skipped"] = f"{nstars} stars masked {100*smask.mean():.1f}% of the frame"
            _logger.warning("deblur: star split skipped — %s", info["split_skipped"])
            split = False
    if split:
        x = starless                                          # from here on
        # The core layer on its zero background: positivity is the whole
        # constraint, so no floor, no clamp, no blend. float64 on a real
        # pedestal: in float32 the FFT round-off of a 145 px kernel leaves
        # near-zero model values under a 1e-3 pedestal and the ratio image
        # explodes (measured: flux x70e9). A flat pedestal is invariant under
        # RL and comes straight back off. Clipped cores keep their input.
        ped = max(1.0, 3.0 * sig)
        srl = richardson_lucy(inner.astype(np.float64) + ped, kernel.astype(np.float64),
                              num_iter=iters, clip=False)
        srl = np.clip(srl - ped, 0.0, None)
        if sat.any():
            m = gaussian_filter(binary_dilation(sat, iterations=SAT_DILATE).astype(np.float32), SAT_FEATHER)
            m = np.clip(m / max(float(m.max()), 1e-6), 0.0, 1.0)
            srl = srl * (1.0 - m) + inner * m
        srl[~binary_dilation(smask, iterations=2)] = 0.0
        stars_out = srl.astype(np.float32)
        del srl
    # A floor that follows the galaxy and never clips a real pixel.
    model = gaussian_filter(minimum_filter(x, FLOOR_MIN_SIZE), FLOOR_SMOOTH) if split else model0
    floor = np.minimum(model, x) - 2.0 * sig
    rl = richardson_lucy(np.clip(x - floor, 1e-3, None).astype(np.float32),
                         kernel.astype(np.float32), num_iter=iters, clip=False) + floor
    # The minimum-filter model sits ~3 sigma under the sky mean (the minimum
    # of 625 noisy pixels); ref is the model lifted to the local sky mean.
    ref = model + float(np.median(x - model))
    # The clamp needs the local sky where the galaxy SLOPES too: the minimum
    # filter reads 12 px downhill of every pixel, so on the disc's edge it
    # was tens of ADU low and RL drained each star's wing down to it, a dark
    # moat around every star on the galaxy. A median over 100 px follows the
    # slope and ignores the stars; bright-star wings lift it a little (17 ADU
    # under the brightest star here against 91 for a 60 px window, while the
    # slope error grows from 3 to 7 ADU), which leaves a trace of halo
    # rather than a moat.
    sky_ref = local_sky(x, CLAMP_BIN, CLAMP_MEDIAN)
    bound = x if clamp_to == "input" else np.minimum(x, sky_ref)
    rl = np.maximum(rl, bound - CLAMP_SIGMA * sig)             # dark deringing
    # Only where there is signal to sharpen, measured from ref, or the blend
    # would start at the sky itself and RL texture would leak into half of it.
    lo, hi = BLEND_LO_SIGMA * sig, BLEND_HI_SIGMA * sig
    t = np.clip((gaussian_filter(x, 1.5) - ref - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    blend = t * t * (3 - 2 * t)
    if sat.any() and not split:
        m = gaussian_filter(binary_dilation(sat, iterations=SAT_DILATE).astype(np.float32), SAT_FEATHER)
        blend = blend * (1.0 - np.clip(m / max(float(m.max()), 1e-6), 0.0, 1.0))
    out = (x + blend * (rl - x)).astype(np.float32)
    if split:
        out = (out + stars_out).astype(np.float32)
    info["deblurred_frac"] = float(np.mean(blend > 0.5))
    info["saturated_frac"] = float(sat.mean())
    _logger.info("deblur: RL%d from %d stars (FWHM %.2f px), %.2f%% of pixels, sky sigma %.3g",
                 iters, info.get("stars", 0), info.get("fwhm_px", float("nan")),
                 100 * info["deblurred_frac"], sig)
    return out, info


def deblur_channels(channels: dict, iters: int, clamp_to: str = "input") -> tuple[dict, dict]:
    """Deblur the luminance of a channel set, or every plane if there is no L.

    LRGB/HALRGB take their brightness from L, so L is the plane whose
    sharpness is seen; the colour planes only set hue and would cost a kernel
    measurement each for nothing. HOO/SHO have no L and show their planes
    directly, so each distinct array is done once (G and B share O-III).
    """
    if int(iters) <= 0:
        return channels, {}
    out = dict(channels)
    infos = {}
    if "L" in channels:
        out["L"], infos["L"] = deblur(channels["L"], iters, clamp_to=clamp_to)
        return out, infos
    done: dict[int, tuple] = {}
    for name in ("R", "G", "B"):
        if name not in channels:
            continue
        key = id(channels[name])
        if key not in done:
            done[key] = deblur(channels[name], iters, clamp_to=clamp_to)
        out[name], infos[name] = done[key]
    return out, infos
