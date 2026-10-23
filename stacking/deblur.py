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
  And it is measured with the fitted star wings taken off, or a bright
  star's own wing lifts it and the clamp holds that much halo.
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

2026-10-17, ngc7320 (Stephan's Quintet), the denoised L at sky sigma 0.055
ADU, where a 0.1 ADU error around a star is a two-sigma mark:

* A DENOISED stack does not have one PSF. On the raw stack the wing and
  spike flux per unit peak is the same at every brightness (4.2e-4 of the
  peak along the arms, a linear PSF); on the denoised stack it falls to
  1e-4 for faint stars, because the denoiser erodes what sits near its
  noise. A kernel measured from the bright stars therefore oversubtracts
  the faint ones: dark lines along the arms, and a moat. Two things follow.
  The spikes are no longer in the RL kernel at all: each star's spikes are
  fitted on their own pixels (star_layer, a template out to ARM_RADIUS)
  and taken off the picture, and RL runs with the azimuthal kernel. And the
  dark clamp sits AT the local sky reference (CLAMP_SIGMA 0): with it one
  sigma below, RL drained every wing to exactly that bound and the flat
  ring at -1 sigma was the moat (measured at -0.055 at every radius inside
  the blend, +0.001 with the clamp at 0). The blend starts at 1.5 sigma so
  the faint stars' outer wings drain too (0.10 -> 0.04 ADU left at 20-30 px).
* Deblurring the raw stack instead (one true kernel) and denoising after,
  or carrying its delta onto the denoised stack, was tried and is worse:
  the blend set in raw-noise units stops at 6 ADU, so every mid star keeps
  a bullseye of halo outside a tight core, and the bright stars moat by
  1 ADU, 20 sigma of the denoised sky. The order is stack, denoise, deblur.
* 40 iterations is the working count (DEFAULT_ITERS), not 8 or 15. RL
  converges fast on points and slowly on extended light, and the galaxies
  had barely moved at 15: the three nuclei went 12.4/12.0/15.8 px FWHM to
  8.3/8.6/10.4 at 15 and 6.7/7.5/8.6 at 40 (stars 2.00" -> 1.04"). The
  split-half test (each half stack deblurred on its own kernel, stars
  masked) says what RL adds on the galaxies is shared by both halves:
  correlation 0.96 (disc), 0.97 (nuclei), 0.91 (arms) at 40 iterations,
  0.95/0.93/0.88 at 80, so 80 is mostly real too but visibly grainier.
  Total-variation RL (tv > 0) was tried for the posterising and rejected:
  it brakes the stars as hard as the texture (1.35" at tv 0.05 against
  1.03" plain) and gains nothing the split-half test could see.
* The kernel stays at 72 px. With WING_RADIUS 100 the kernel-shaped wing
  fit overshot the brightest star by 10% at every radius (1368 against
  1236 ADU at 10-20 px), the clamp's sky reference, measured with that fit
  off, went 1-5 ADU below the sky, and RL dug a black blob 100 px across
  down to it. Replacing the fit with each star's own ring-median profile
  for the reference (bounded by the fit or not) held every halo instead:
  bright rings at 10-20 px on stars of every brightness (+2.1 ADU on the
  brightest, +0.13 on mid stars, against -0.02 and -0.08 with the fit).
  The 72 px kernel leaves ~0.3 ADU of the brightest stars' wings standing
  past 72 px; that is the trade.
* RL is now the module's own FFT loop (_rl), torch on the GPU when there
  is one (3 s per 5 full-frame iterations on the Spark), numpy otherwise,
  starting from the data rather than skimage's flat 0.5.

Returns the input untouched, with a reason, when the field has too few
isolated stars to measure a kernel from or the stars are already at the
pixel scale.
"""
import logging
from typing import Optional

import numpy as np

_logger = logging.getLogger(__name__)

DEFAULT_ITERS = 40
PSF_SCALE = 1.0           # kernel shrink; 1.0 = as measured (see module doc)
PSF_RADIUS = 32           # half-size of the median-stacked core in px
PSF_TAPER = 8             # cosine taper width at the core's edge (no-wing case)
WING_RADIUS = 72          # the wing profile is measured out to here (not 100: see the module doc)
ARM_RADIUS = 100          # and the diffraction arms, for the per-star spike template
ARM_SIGMA = 4.0           # the arms' Gaussian cross-section (px) when it cannot be measured
WING_BLEND = (8.0, 14.0)  # core to wing crossfade, px
WING_MIN_STARS = 8        # fewer clean bright stars and the core alone is used
WING_MIN_PEAK = 3000.0    # ADU above sky: bright enough to show the wing
SPLIT_STARS = False       # take the stars off and treat them apart: see the module doc
CLAMP_BIN, CLAMP_MEDIAN = 4, 15   # the clamp's local-sky reference: median over 60 px, binned
STAR_SPLIT_MIN_SIGMA = 20.0  # candidates this far above the sky mean (or 1e-3 of saturation)
STAR_COMPACT = 0.6        # of a point source's peak/matched ratio: broader is not a star
STAR_CORE_PX = 14         # a star's core disc, lifted off a fill into its own layer
STAR_MASK_MAX_FRAC = 0.25 # more of the frame than this in the layer and the split is skipped
MIN_STARS = 15            # fewer isolated stars than this and nothing happens
MAX_STARS = 400           # brightest isolated stars used for the kernel
SAT_FRAC = 0.98           # of the saturation level: protected as clipped
SAT_MIN_PIXELS = 4        # pixels at the ceiling before it counts as saturation
SAT_PLATEAU_SIGMA = 50.0  # local maxima this far above the sky are checked for a flat top
SAT_PLATEAU_MIN = 12      # px within 10% of the peak before a top can count as flat
SAT_FLATNESS = 0.7        # of those within 5%: a Gaussian of any width has 0.49
SAT_PLATEAU_EDGE = 0.02   # of the peak: a clipped star is kept as imaged out to here
SAT_PLATEAU_MAX = 2500    # px: a kept disc bigger than this (r ~28 px) is not a star
SAT_PLATEAU_MIN_AMP = 0.2 # of the ceiling above the local base: a clipped star is bright
SAT_DILATE = 6            # px of protection around the clipped pixels
SAT_FEATHER = 3.0         # and the feather on its edge
FLOOR_MIN_SIZE = 25       # minimum-filter window for the local floor model
FLOOR_SMOOTH = 8.0        # and its smoothing
CLAMP_SIGMA = 0.0         # dark deringing: never below the bound minus this many sigma (see the module doc)
TV_LAMBDA = 0.0           # total-variation weight of the RL update, 0 = plain RL (see _rl)
BLEND_LO_SIGMA = 1.5      # signal above the local model where deblur starts
BLEND_HI_SIGMA = 5.0      # and where it is fully in


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


def flat_tops(chan: np.ndarray, sig: Optional[float] = None,
              ceiling: Optional[float] = None) -> np.ndarray:
    """Pixels on a star's flat top, whether or not the top is at the ceiling.

    A saturated star in a STACK is a plateau below the per-frame ceiling,
    not at it: sub-pixel registration smears the clipped disc's edge, the
    flat divides it by a different gain at every star, and the sigma clip
    mixes frames where the star did and did not clip. On ngc6946's L the
    twelve brightest stars sat at 0.94-0.97 of the ceiling with 20-140
    pixels within 5% of their peak each, and SAT_FRAC protected none of
    them; RL then read each plateau as a broad real core, dug a hole in its
    middle and raised a ring round it (2026-10-18). A star clipped in only
    some frames is a plateau at a fraction of the ceiling and does the same.

    Flat, not merely broad: of the pixels within 10% of the peak, a Gaussian
    of any width has ln(0.95)/ln(0.9) = 49% within 5%, a plateau 80-90%. A
    galaxy nucleus is broad but not flat and is left to RL. Every local
    maximum SAT_PLATEAU_SIGMA above the sky is checked; the top must hold
    SAT_PLATEAU_MIN pixels and SAT_FLATNESS of them within 5%. The protected
    region is the connected top down to SAT_PLATEAU_EDGE of the peak, holes
    filled (a stacked plateau is jagged, and RL dug to the sky through a gap
    in the 90% mask).
    """
    from scipy.ndimage import binary_fill_holes, label, maximum_filter, minimum_filter

    a = np.nan_to_num(np.asarray(chan, dtype=np.float32), nan=-np.inf)
    out = np.zeros(a.shape, bool)
    if ceiling is None:
        ceiling = float(np.nanmax(chan))
    if not np.isfinite(ceiling) or ceiling <= 0:
        return out
    sky = float(np.nanmedian(chan))
    if sig is None:
        sig = sky_sigma(np.nan_to_num(np.asarray(chan, dtype=np.float32), nan=sky))
    # Against a LOCAL base, not the sky: on a denoised stack the galaxy's
    # disc is smooth enough that, measured from the sky, its gentle local
    # maxima read as flat tops (ngc6946: 150k px of disc flagged). The base
    # is the window's minimum over 25 px, the top must be compact -- inside
    # the window, at most SAT_PLATEAU_MAX px -- and the whole top above the
    # base must be well out of the noise.
    base = minimum_filter(a, 25)
    peaks = ((a == maximum_filter(a, 25)) & (a - base > SAT_PLATEAU_SIGMA * sig)
             & (a - base > SAT_PLATEAU_MIN_AMP * ceiling))
    h, w = a.shape
    r = 40
    for y, x in zip(*np.nonzero(peaks)):
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        win = a[y0:y1, x0:x1]
        lo = float(base[y, x])
        amp = float(a[y, x]) - lo
        lab, _ = label(win >= lo + 0.9 * amp)
        top90 = lab == lab[y - y0, x - x0]
        n90 = int(top90.sum())
        if n90 < SAT_PLATEAU_MIN:
            continue
        n95 = int((top90 & (win >= lo + 0.95 * amp)).sum())
        if n95 < SAT_FLATNESS * n90:
            continue
        lab, _ = label(win >= lo + SAT_PLATEAU_EDGE * amp)
        disc = lab == lab[y - y0, x - x0]
        if disc.sum() > SAT_PLATEAU_MAX or disc[0].any() or disc[-1].any() or disc[:, 0].any() or disc[:, -1].any():
            continue
        out[y0:y1, x0:x1] |= binary_fill_holes(disc)
    return out


def saturated_mask(chan: np.ndarray, level: Optional[float],
                   sig: Optional[float] = None) -> np.ndarray:
    """Clipped pixels: at the ceiling (SAT_FRAC of *level*) or on a flat top."""
    x = np.asarray(chan)
    sat = x > SAT_FRAC * level if level is not None else np.zeros(x.shape, bool)
    return sat | flat_tops(x, sig, level)


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
    peaks &= ~flat_tops(a, sig, sat)           # a plateau below the ceiling is clipped too
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
    wing, nwing, wing2d = _wing_profile(a, stars, sat, sig)
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
    spikes = None
    if True:
        wy, wx = np.mgrid[-R:R + 1, -R:R + 1]; wr = np.hypot(wy, wx)
        psf = np.interp(wr, np.arange(len(wing)), wing, right=0.0)
        if wing2d is not None and wing2d.shape == psf.shape:
            # The profile is the floor; only the 2-D median's EXCESS over it
            # (the spikes) is added, and only where it clears the median's
            # own noise, read from the outer annulus. Adding the whole 2-D
            # median put the positive half of its noise into the far wing
            # (38% too much flux at 24-60 px on a Moffat test field).
            excess = wing2d - psf
            noise = 1.4826 * float(np.median(np.abs(excess[wr > 45] - np.median(excess[wr > 45]))))
            keep = np.where(excess > 4.0 * max(noise, 1e-12), excess, 0.0)
            w2 = np.clip((wr - 40.0) / 15.0, 0, 1); w2 = 1 - w2 * w2 * (3 - 2 * w2)   # spikes to 40 px, fading by 55
            spikes = keep * w2
            psf = np.clip(psf + spikes, 0, None)
            # The spike TEMPLATE goes on past the kernel: the 2-D median
            # fades out over 40-55 px and a measured arm profile with a
            # Gaussian cross-section fades in, out to ARM_RADIUS. Only the
            # per-star fit uses it (star_layer), never RL, so it can be
            # parametric and smooth where the median would be noise. On
            # ngc7320 a 32000 ADU star's arms were still 0.85 ADU at 50 px
            # and 0.25 at 60, 15 and 5 sigma of the denoised sky, and stood
            # as streaks where the template had ended.
            arm = _arm_template(a, stars, sat, keep, wr, w2)
            if arm is not None:
                spikes, info["spike_axes"], info["arm_sigma"] = arm
        psf = psf * core.max()
        spikes = spikes * core.max() if spikes is not None else None
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
    # The spike component on the kernel's own scale, so that psf - spikes is
    # the azimuthal kernel: see deblur() for why RL never sees the spikes.
    info["spikes"] = (spikes / s if spikes is not None else np.zeros_like(psf)).astype(np.float64)
    return psf / s, info


def _centre(t: np.ndarray, size: int) -> np.ndarray:
    """The central size x size crop of an odd square array (or the array)."""
    if t.shape[0] <= size:
        return t
    c = t.shape[0] // 2; r = size // 2
    return t[c - r:c + r + 1, c - r:c + r + 1]


def _spike_axes(keep: np.ndarray, wr: np.ndarray) -> list[float]:
    """The two diffraction-arm axes (radians, mod pi) from the kernel's
    spike excess at 15-35 px: the angular profile's two best-separated
    peaks."""
    R = keep.shape[0] // 2
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    ang = np.mod(np.arctan2(yy, xx), np.pi)
    band = (wr >= 15) & (wr <= 35)
    bins = np.linspace(0, np.pi, 91)
    idx = np.clip(np.digitize(ang[band], bins) - 1, 0, 89)
    prof = np.bincount(idx, weights=keep[band], minlength=90)
    if prof.max() <= 0:
        return []
    centres = 0.5 * (bins[:-1] + bins[1:])
    order = np.argsort(prof)[::-1]
    axes = [float(centres[order[0]])]
    for i in order[1:]:
        d = abs(((centres[i] - axes[0]) + np.pi / 2) % np.pi - np.pi / 2)
        if d > np.radians(30) and prof[i] > 0.2 * prof[order[0]]:
            axes.append(float(centres[i])); break
    return axes


def _arm_template(a: np.ndarray, stars: np.ndarray, sat: Optional[float],
                  keep: np.ndarray, wr: np.ndarray, w2: np.ndarray) -> Optional[tuple]:
    """(template, axes, arm sigma), or None when the arms cannot be measured.
    The spike template out to ARM_RADIUS, peak-normalised like the kernel:
    the 2-D median's excess inside (faded by w2) and, outside, the median
    along-arm profile of the brightest clean stars times a Gaussian cross
    section, scaled to meet the inner part at 38-42 px."""
    from scipy.ndimage import map_coordinates
    axes = _spike_axes(keep, wr)
    if not axes:
        return None
    h, w = a.shape
    RA = ARM_RADIUS
    peak = stars[:, 3] - np.median(a)
    cand = np.nonzero(peak > WING_MIN_PEAK)[0]
    if sat is not None:
        cand = cand[stars[cand, 3] < SAT_FRAC * sat]
    cand = cand[np.argsort(peak[cand])[::-1]][:20]
    rs = np.arange(24.0, RA + 1.0, 1.0)
    profs, widths = [], []
    per = np.arange(-8, 8.5, 0.5)
    for i in cand:
        xc, yc, pk = float(stars[i, 0]), float(stars[i, 1]), float(peak[i])
        for t in axes:
            for sgn in (1, -1):
                yy = yc + sgn * rs * np.sin(t); xx = xc + sgn * rs * np.cos(t)
                if yy.min() < 12 or xx.min() < 12 or yy.max() > h - 12 or xx.max() > w - 12:
                    continue
                on = map_coordinates(a, [yy, xx], order=1)
                off = np.median([map_coordinates(a, [yy + d * np.cos(t), xx - d * np.sin(t)], order=1)
                                 for d in (-10, -7, 7, 10)], axis=0)
                profs.append((on - off) / pk)
                if sgn == 1:
                    cs = np.median([map_coordinates(a, [yc + r * np.sin(t) + per * np.cos(t),
                                                        xc + r * np.cos(t) - per * np.sin(t)], order=1)
                                    for r in np.arange(40, 60, 2.0)], axis=0)
                    cs = np.clip(cs - np.median(cs[[0, 1, -2, -1]]), 0, None)
                    if cs.sum() > 0:
                        widths.append(float(np.sqrt((cs * per ** 2).sum() / cs.sum())))
    if len(profs) < 8 or not widths:
        return None
    P = np.clip(np.median(np.stack(profs), axis=0), 0, None)
    sig_arm = float(np.clip(np.median(widths), 2.0, 8.0))
    info_axes = [float(t) for t in axes]
    # continuity: the inner template along the arms at 36-40 px against P there
    R = keep.shape[0] // 2
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    inner_vals = []
    for t in axes:
        for sgn in (1, -1):
            r = np.arange(36.0, 40.5, 1.0)
            inner_vals.append(map_coordinates(keep, [R + sgn * r * np.sin(t), R + sgn * r * np.cos(t)], order=1))
    inner = float(np.median(np.concatenate(inner_vals)))
    pj = float(np.median(np.interp(np.arange(36.0, 40.5, 1.0), rs, P)))
    scale = float(np.clip(inner / pj, 0.5, 2.0)) if pj > 0 else 1.0
    T = np.zeros((2 * RA + 1, 2 * RA + 1))
    T[RA - R:RA + R + 1, RA - R:RA + R + 1] = keep * w2
    ty, tx = np.mgrid[-RA:RA + 1, -RA:RA + 1]
    tr = np.hypot(ty, tx)
    fade = np.clip((tr - 40.0) / 15.0, 0, 1); fade = fade * fade * (3 - 2 * fade)     # 1 - w2, out to RA
    outer = np.zeros_like(T)
    for t in axes:
        u = np.abs(tx * np.cos(t) + ty * np.sin(t))          # along the axis, either direction
        v = np.abs(-tx * np.sin(t) + ty * np.cos(t))         # across it
        outer += np.interp(u, rs, P, left=0.0, right=0.0) * np.exp(-0.5 * (v / sig_arm) ** 2)
    taper = np.clip((tr - (RA - 10)) / 10.0, 0, 1)
    T += scale * outer * fade * 0.5 * (1 + np.cos(np.pi * taper))
    T[tr > RA] = 0.0
    return np.clip(T, 0, None), info_axes, sig_arm


def _wing_profile(a: np.ndarray, stars: np.ndarray, sat: Optional[float],
                  sig: float) -> tuple[Optional[np.ndarray], int, Optional[np.ndarray]]:
    """(azimuthal profile, count, 2-D wing) of the brightest clean stars out
    to WING_RADIUS, peak-normalised, median across stars. The core stack
    cannot show the wing: its stars are fainter and its sky annulus sits
    inside the wing. The 2-D median keeps the diffraction spikes, whose
    orientation the spider fixes: with an azimuthal wing the spikes were
    the one part of a star RL could not drain, and they stood as detached
    bright dashes 20-40 px from every bright star once the halo was gone.
    Beyond 30 px the 2-D median fades into the profile, where per-star noise
    is all it would add."""
    from scipy.ndimage import shift as ndshift
    R = WING_RADIUS
    h, w = a.shape
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]; rr = np.hypot(yy, xx)
    sky_ann = (rr > R - 8) & (rr <= R)
    rbin = np.round(rr).astype(int)
    peak = stars[:, 3] - np.median(a)
    cand = np.nonzero(peak > WING_MIN_PEAK)[0]
    if sat is not None:
        cand = cand[stars[cand, 3] < SAT_FRAC * sat]
    profs, cuts = [], []
    for i in cand[np.argsort(peak[cand])[::-1]]:
        xc, yc = float(stars[i, 0]), float(stars[i, 1])
        xi, yi = int(round(xc)), int(round(yc))
        if not (R < yi < h - R and R < xi < w - R):
            continue
        c = a[yi - R:yi + R + 1, xi - R:xi + R + 1].astype(np.float64)
        if not np.isfinite(c).all():
            continue
        c = c - np.median(c[sky_ann])
        pk = float(c[R, R])
        if pk <= 0 or (c[rr > 20] > 0.05 * pk).any():      # a neighbour in the wing
            continue
        c = ndshift(c, (yi - yc, xi - xc), order=1, mode="nearest") / pk
        prof = np.bincount(rbin.ravel(), weights=c.ravel(), minlength=R + 1)[:R] \
            / np.bincount(rbin.ravel(), minlength=R + 1)[:R]
        profs.append(prof)
        cuts.append(c)
        if len(profs) >= 60:
            break
    if len(profs) < WING_MIN_STARS:
        return None, len(profs), None
    med = np.clip(np.median(np.stack(profs), axis=0), 0, None)
    med[0] = 1.0
    wing2d = np.median(np.stack(cuts), axis=0)                 # signed: its noise is measured later
    wing2d[R, R] = 1.0
    return med, len(profs), wing2d


def _rl(y: np.ndarray, k: np.ndarray, n: int, tv: float = 0.0) -> np.ndarray:
    """Richardson-Lucy by FFT, starting from the data, reflect-padded by the
    kernel radius so the frame edge does not wrap. tv > 0 is Dey et al.
    (2006) total-variation RL: the multiplicative update is divided by
    1 - tv * div(grad x / |grad x|), which holds flat regions flat and lets
    the iteration count climb without the result curdling.

    torch when it is installed (the Spark: 3 s per 5 full-frame iterations),
    numpy otherwise (the observatory PC has no torch).
    """
    pad = k.shape[0] // 2
    y = np.pad(np.asarray(y, dtype=np.float32), pad, mode="reflect")
    H, W = y.shape
    kh, kw = k.shape
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        yt = torch.as_tensor(y, device=dev)
        kp = torch.zeros((H, W), dtype=torch.float32, device=dev)
        kp[:kh, :kw] = torch.as_tensor(np.asarray(k, dtype=np.float32), device=dev)
        otf = torch.fft.rfft2(torch.roll(kp, shifts=(-(kh // 2), -(kw // 2)), dims=(0, 1)))
        otf_c = torch.conj(otf)
        conv = lambda a, o: torch.fft.irfft2(torch.fft.rfft2(a) * o, s=(H, W))
        x = yt.clone()
        eps = float(yt.std()) * 1e-3 + 1e-6
        for _ in range(int(n)):
            corr = conv(yt / torch.clamp(conv(x, otf), min=1e-6), otf_c)
            if tv > 0:
                gx = torch.zeros_like(x); gy = torch.zeros_like(x)
                gx[:, :-1] = x[:, 1:] - x[:, :-1]; gy[:-1, :] = x[1:, :] - x[:-1, :]
                nrm = torch.sqrt(gx * gx + gy * gy + eps * eps); px, py = gx / nrm, gy / nrm
                div = torch.zeros_like(x)
                div[:, 1:] += px[:, 1:] - px[:, :-1]; div[:, 0] += px[:, 0]
                div[1:, :] += py[1:, :] - py[:-1, :]; div[0, :] += py[0, :]
                corr = corr / torch.clamp(1.0 - tv * div, min=0.2)
            x = torch.clamp(x * corr, min=0.0)
        out = x.cpu().numpy()
    except ImportError:
        from scipy import fft as sfft
        kp = np.zeros((H, W), np.float32); kp[:kh, :kw] = k
        otf = sfft.rfft2(np.roll(kp, (-(kh // 2), -(kw // 2)), axis=(0, 1)))
        otf_c = np.conj(otf)
        conv = lambda a, o: sfft.irfft2(sfft.rfft2(a) * o, s=(H, W)).astype(np.float32)
        x = y.copy()
        eps = float(y.std()) * 1e-3 + 1e-6
        for _ in range(int(n)):
            corr = conv(y / np.maximum(conv(x, otf), 1e-6), otf_c)
            if tv > 0:
                gx = np.zeros_like(x); gy = np.zeros_like(x)
                gx[:, :-1] = x[:, 1:] - x[:, :-1]; gy[:-1, :] = x[1:, :] - x[:-1, :]
                nrm = np.sqrt(gx * gx + gy * gy + eps * eps); px, py = gx / nrm, gy / nrm
                div = np.zeros_like(x)
                div[:, 1:] += px[:, 1:] - px[:, :-1]; div[:, 0] += px[:, 0]
                div[1:, :] += py[1:, :] - py[:-1, :]; div[0, :] += py[0, :]
                corr = corr / np.maximum(1.0 - tv * div, 0.2)
            x = np.maximum(x * corr, 0.0)
        out = x
    return out[pad:-pad, pad:-pad]


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
               spikes: Optional[np.ndarray] = None,
               axes: Optional[list] = None, arm_sigma: float = ARM_SIGMA,
               
               clipped: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]:
    """(starless, inner, mask, count, fits, wing, spikes): the stars taken off the image.

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

    *spikes* is the kernel's diffraction-spike component (measure_psf's
    info["spikes"]); the kernel less it is the azimuthal kernel the wing is
    fitted with. Each star's spike amplitude is then its own least-squares
    fit on the spike pixels, and the last array returned is the sum of those
    fits: the spikes as they actually are, star by star, which deblur()
    takes off the picture before RL sees it (see there for why). For the
    stars bright enough to show arms past the template (WING_MIN_PEAK), what
    the template left along each arm (*axes*, both directions, 30 px to
    ARM_RADIUS) is then measured on the star itself, on-axis minus the
    ribbon's own edges, and taken off as a Gaussian ribbon of *arm_sigma*:
    the arms of the brightest stars decline more slowly than the median
    template, which the denoiser eroded on the fainter stars it was built
    from (a 61000 ADU star kept 19 ADU of arm at 50 px against 7 in the
    template; RL then SHARPENED what was left into a streak brighter than
    the input, 28 ADU at 55 px).
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
    empty = (x.astype(np.float32), np.zeros_like(x), np.zeros(x.shape, bool), 0, np.zeros((0, 5)),
             np.zeros(x.shape, np.float32), np.zeros(x.shape, np.float32))
    if len(ys) == 0:
        return empty
    order = np.argsort(above[ys, xs])[::-1]
    work = above.copy()
    wing_part = np.zeros(x.shape, np.float32)
    spike_part = np.zeros(x.shape, np.float32)

    inner_mask = np.zeros(x.shape, bool)
    # *clipped* (deblur passes its flat-top mask) says which stars are not
    # PSF-shaped. They are not fitted at all: fitting a clipped core's
    # smeared edge overshoots the wing, and the clamp's sky reference then
    # read below the sky round the brightest stars -- a dark moat (ngc6946,
    # 2026-10-18). Left unfitted, their halo lifts the reference and the
    # clamp holds it: a clipped star keeps its halo, which is the trade.
    if clipped is None:
        clipped = (x >= 0.98 * level) if level is not None else np.zeros(x.shape, bool)
    ky, kx = np.mgrid[-R:R + 1, -R:R + 1]; kr = np.hypot(ky, kx)
    core_fit = (kr <= 6)
    wing_fit = (kr > 10) & (kr <= 36)
    sp64 = spikes.astype(np.float64) if spikes is not None and spikes.shape[0] >= kernel.shape[0] else np.zeros(kernel.shape)
    k64 = kernel.astype(np.float64) - _centre(sp64, kernel.shape[0])          # the azimuthal kernel
    RS = sp64.shape[0] // 2
    fw = np.clip((kr - WING_BLEND[0]) / (WING_BLEND[1] - WING_BLEND[0]), 0, 1); fw = fw * fw * (3 - 2 * fw)
    disc = kr <= STAR_CORE_PX
    n = 0
    fits = []
    g7y, g7x = np.mgrid[:7, :7]
    for i in order:
        y, xx_ = int(ys[i]), int(xs[i])
        if clipped[y, xx_]:
            continue
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
        if sp64.any():
            # The spikes, fitted on their own pixels with the azimuthal fit
            # taken off, bounded by the core's amplitude: a star that has
            # none (the denoiser erodes them on the fainter stars) fits ~0.
            spk = ndshift(sp64, (dy, dx), order=1, mode="constant", cval=0.0)
            # the template's footprint, cropped at the frame edge
            y0, y1 = max(y - RS, 0), min(y + RS + 1, h); x0, x1 = max(xx_ - RS, 0), min(xx_ + RS + 1, w)
            sls = (slice(y0, y1), slice(x0, x1))
            spk = spk[y0 - (y - RS):y1 - (y - RS), x0 - (xx_ - RS):x1 - (xx_ - RS)]
            ds = work[sls]
            ok = (spk > 0) & ~clipped[sls]; kk = spk[ok]
            amp_s = float(np.clip((ds[ok] * kk).sum() / max((kk * kk).sum(), 1e-12), 0.0, 1.2 * amp))
            ssub = (amp_s * spk).astype(np.float32)
            work[sls] -= ssub
            spike_part[sls] += ssub
            if axes and above[y, xx_] > WING_MIN_PEAK:
                _take_arms(work, spike_part, y, xx_, axes, arm_sigma, sig)
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
    return starless, inner, mask, n, np.array(fits, dtype=np.float64).reshape(-1, 5), wing_part, spike_part


def _take_arms(work: np.ndarray, spike_part: np.ndarray, y: int, x: int,
               axes: list, sig_arm: float, sig: float) -> None:
    """Measure what is left of one star's diffraction arms on *work* and
    take it off (into *spike_part*): see star_layer."""
    from scipy.ndimage import map_coordinates, median_filter
    h, w = work.shape
    RA = ARM_RADIUS
    rs = np.arange(30.0, RA + 1.0, 1.0)
    half = int(np.ceil(3.0 * sig_arm))
    for t in axes:
        ct, st = float(np.cos(t)), float(np.sin(t))
        for sgn in (1, -1):
            yy = y + sgn * rs * st; xx = x + sgn * rs * ct
            if yy.min() < half + 1 or xx.min() < half + 1 or yy.max() > h - half - 2 or xx.max() > w - half - 2:
                continue
            on = map_coordinates(work, [yy, xx], order=1)
            off = np.median([map_coordinates(work, [yy + d * ct, xx - d * st], order=1)
                             for d in (-3.0 * sig_arm, -2.5 * sig_arm, 2.5 * sig_arm, 3.0 * sig_arm)], axis=0)
            prof = median_filter(np.clip(on - off, 0.0, None), size=5, mode="nearest")
            prof[prof < 2.0 * sig] = 0.0
            if prof.max() <= 0:
                continue
            # the ribbon's bounding box
            ys = [y + sgn * r * st for r in (rs[0], rs[-1])]; xs = [x + sgn * r * ct for r in (rs[0], rs[-1])]
            y0, y1 = max(int(min(ys)) - half - 1, 0), min(int(max(ys)) + half + 2, h)
            x0, x1 = max(int(min(xs)) - half - 1, 0), min(int(max(xs)) + half + 2, w)
            gy, gx = np.mgrid[y0:y1, x0:x1]
            u = sgn * ((gx - x) * ct + (gy - y) * st)
            v = -(gx - x) * st + (gy - y) * ct
            rib = (u >= rs[0]) & (u <= RA) & (np.abs(v) <= 3.0 * sig_arm)
            if not rib.any():
                continue
            fade = np.clip((u - rs[0]) / 10.0, 0, 1)
            val = np.where(rib, np.interp(u, rs, prof, left=0.0, right=0.0)
                           * np.exp(-0.5 * (v / sig_arm) ** 2) * fade, 0.0).astype(np.float32)
            work[y0:y1, x0:x1] -= val
            spike_part[y0:y1, x0:x1] += val


def deblur(chan: np.ndarray, iters: int = DEFAULT_ITERS,
           psf: Optional[np.ndarray] = None,
           psf_scale: float = PSF_SCALE,
           clamp_to: str = "input",
           split: bool = SPLIT_STARS,
           spikes: Optional[np.ndarray] = None,
           tv: float = TV_LAMBDA,
           axes: Optional[list] = None, arm_sigma: float = ARM_SIGMA) -> tuple[np.ndarray, dict]:
    """Return (deblurred channel, info). The input is linear ADU with its sky.

    A caller passing its own *psf* passes measure_psf's info["spikes"] as
    *spikes*, or the kernel is used whole. tv is the total-variation weight
    of the RL update (see _rl), 0 for plain RL.

    clamp_to="input": no pixel ends below its input minus one sigma (star
    halos stay under a brighter core). "sky": a pixel may fall as far as the
    local sky mean minus one sigma, so halos are allowed to drain into the
    core, but nothing goes below the sky, which is the ring.

    info carries what was measured; info["why"] is set when the channel was
    returned as given.
    """
    from scipy.ndimage import binary_dilation, gaussian_filter, minimum_filter

    iters = int(iters)
    if iters <= 0:
        return chan, {"iters": 0, "why": "deblur=0"}
    if clamp_to not in ("input", "sky"):
        raise ValueError(f"clamp_to must be 'input' or 'sky', not {clamp_to!r}")
    info: dict = {"iters": iters, "psf_scale": psf_scale, "clamp_to": clamp_to, "split": bool(split), "tv": tv}
    wing = None
    if psf is None:
        psf, pinfo = measure_psf(chan)
        wing = pinfo.pop("wing_profile", None)
        spikes = pinfo.pop("spikes", None)
        axes, arm_sigma = pinfo.get("spike_axes"), pinfo.get("arm_sigma", ARM_SIGMA)
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
    if spikes is not None and (psf_scale != 1.0 or spikes.shape[0] < kernel.shape[0]):
        spikes = None
    # RL runs with the AZIMUTHAL kernel on an image whose spikes have been
    # fitted off star by star. A kernel that carries the spikes assumes every
    # star has them in the same proportion, which a linear stack does (the
    # excess along the arms is 4.2e-4 of the peak at every brightness on the
    # ngc7320 raw L) but a DENOISED one does not: the denoiser erodes what is
    # near its noise, so there the faint stars keep 1e-4 of the peak and the
    # bright ones 4.2e-4. RL with the spiked kernel then subtracted spikes
    # the fainter stars did not have and left a dark line along each arm
    # (2026-10-17). With the azimuthal kernel alone the bright stars' spikes
    # stood as detached dashes (2026-10-10), so they come off by fit instead.
    kernel_az = kernel
    if spikes is not None and spikes.any():
        kernel_az = np.clip(kernel - _centre(spikes, kernel.shape[0]), 0, None); kernel_az = kernel_az / kernel_az.sum()

    x = np.nan_to_num(np.asarray(chan, dtype=np.float32), nan=0.0)
    model0 = gaussian_filter(minimum_filter(x, FLOOR_MIN_SIZE), FLOOR_SMOOTH)
    sig = sky_sigma(x, model0)
    info["sky_sigma"] = sig
    level = saturation_level(x)
    sat = saturated_mask(x, level, sig)
    full = x
    stars = None
    # The star fit runs either way: its wing model is what the clamp's sky
    # reference is measured without (below).
    starless, inner, smask, nstars, fits, wing_part, spike_part = star_layer(
        x, model0, sig, kernel, level, spikes, axes if spikes is not None else None, arm_sigma,
        clipped=sat)
    info["fitted_stars"] = nstars
    # Never more than the pixel holds above the floor: the template's
    # inner pixels times a clipped star's amplitude exceed the plateau's
    # edge, and the subtraction cut a notch into the core (ngc6946).
    spike_part = np.minimum(spike_part, np.clip(x - model0, 0.0, None)).astype(np.float32)
    info["spike_flux_frac"] = float(spike_part.sum() / max(float(np.abs(x - model0).sum()), 1e-12))
    x = full = (x - spike_part).astype(np.float32)          # the picture without its spikes
    if split:
        info["split_stars"] = nstars
        info["star_mask_frac"] = float(smask.mean())
        if smask.mean() > STAR_MASK_MAX_FRAC or nstars == 0:
            info["split_skipped"] = f"{nstars} stars masked {100*smask.mean():.1f}% of the frame"
            _logger.warning("deblur: star split skipped — %s", info["split_skipped"])
            split = False
    if split:
        x = (starless - spike_part).astype(np.float32)       # from here on
        # The core layer on its zero background: positivity is the whole
        # constraint, so no floor, no clamp, no blend. float64 on a real
        # pedestal: in float32 the FFT round-off of a 145 px kernel leaves
        # near-zero model values under a 1e-3 pedestal and the ratio image
        # explodes (measured: flux x70e9). A flat pedestal is invariant under
        # RL and comes straight back off. Clipped cores keep their input.
        ped = max(1.0, 3.0 * sig)
        srl = _rl(inner.astype(np.float32) + ped, kernel_az.astype(np.float32), iters, tv)
        srl = np.clip(srl - ped, 0.0, None)
        if sat.any():
            m = gaussian_filter(binary_dilation(sat, iterations=SAT_DILATE).astype(np.float32), SAT_FEATHER)
            m = np.clip(m / max(float(m.max()), 1e-6), 0.0, 1.0)
            srl = srl * (1.0 - m) + inner * m
        srl[~binary_dilation(smask, iterations=2)] = 0.0
        stars_out = srl.astype(np.float32)
        del srl
    # A floor that follows the galaxy and never clips a real pixel. Measured
    # with the fitted wings off: inside a bright star's wing the minimum
    # over 25 px is still wing, and a floor there is a floor RL cannot drain
    # the wing below (69 ADU left at 12-16 px with the model on the raw image,
    # whatever the clamp's reference was).
    model = gaussian_filter(minimum_filter(x if split else full - wing_part, FLOOR_MIN_SIZE), FLOOR_SMOOTH)
    floor = np.minimum(model, x) - 2.0 * sig
    rl = _rl(np.clip(x - floor, 1e-3, None).astype(np.float32), kernel_az.astype(np.float32), iters, tv) + floor
    # The minimum-filter model sits ~3 sigma under the sky mean (the minimum
    # of 625 noisy pixels); ref is the model lifted to the local sky mean.
    ref = model + float(np.median(x - model))
    # The clamp needs the local sky where the galaxy SLOPES too: the minimum
    # filter reads 12 px downhill of every pixel, so on the disc's edge it
    # was tens of ADU low and RL drained each star's wing down to it, a dark
    # moat around every star on the galaxy. A median over 60 px follows the
    # slope to ~3 ADU and ignores the stars' cores, but a bright star's WING
    # lifted it by up to 91 ADU and the clamp then held that much halo (the
    # operator: "still significant halos"). So it is measured on the image
    # with the fitted wings taken off: the symmetric fit is good to a few
    # ADU in the wing, which is all a 60 px median can feel, even though it
    # cannot take the wings off the picture itself (see star_layer).
    sky_ref = local_sky(full - wing_part, CLAMP_BIN, CLAMP_MEDIAN)
    bound = x if clamp_to == "input" else np.minimum(x, sky_ref)
    rl = np.maximum(rl, bound - CLAMP_SIGMA * sig)             # dark deringing
    # Only where there is signal to sharpen, measured from ref, or the blend
    # would start at the sky itself and RL texture would leak into half of it.
    lo, hi = BLEND_LO_SIGMA * sig, BLEND_HI_SIGMA * sig
    t = np.clip((gaussian_filter(x, 1.5) - ref - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    blend = t * t * (3 - 2 * t)
    keep = None
    if sat.any() and not split:
        m = gaussian_filter(binary_dilation(sat, iterations=SAT_DILATE).astype(np.float32), SAT_FEATHER)
        keep = np.clip(m / max(float(m.max()), 1e-6), 0.0, 1.0)
        blend = blend * (1.0 - keep)
    out = (x + blend * (rl - x)).astype(np.float32)
    if keep is not None:
        # The clipped disc is given back AS INPUT, spikes included: the spike
        # fit runs through the core, and a clipped core minus a fitted spike
        # is a notch, not a star (ngc6946, 2026-10-18).
        out = (out * (1.0 - keep) + np.nan_to_num(np.asarray(chan, dtype=np.float32), nan=0.0) * keep).astype(np.float32)
    if split:
        out = (out + stars_out).astype(np.float32)
    info["deblurred_frac"] = float(np.mean(blend > 0.5))
    info["saturated_frac"] = float(sat.mean())
    _logger.info("deblur: RL%d from %d stars (FWHM %.2f px), %.2f%% of pixels, sky sigma %.3g",
                 iters, info.get("stars", 0), info.get("fwhm_px", float("nan")),
                 100 * info["deblurred_frac"], sig)
    return out, info


# How a luminance is made for a set that has none (HOO/SHO) before the learned
# deconvolution: the palette's own luminance weights on the planes as mapped
# to R, G and B. compose() lights the picture by L through the ratio of the
# stretched L to the 0.299/0.587/0.114 luminance of the stretched RGB, so an L
# built with those same weights is, before deconvolution, the luminance the
# compose will divide by: the substitution ratio is then pure sharpening gain
# and no region is relit. An inverse-variance sum would have more signal but
# would weight O-III-only structure down and light the picture by an L that
# disagrees with its colour. Renormalised over the planes present.
LUM_WEIGHTS = {"R": 0.299, "G": 0.587, "B": 0.114}
# The deconvolved synthetic luminance and the one it was made from, returned
# side by side. compose() lights the picture by their RATIO, not by the
# deconvolved one alone as it does a real L: the stretch clips every plane at
# its own black before the RGB luminance is taken, so where one plane is above
# black and the others below, the RGB keeps the positive part and a summed L
# carries the deficit -- measured on trunk SHO as the faintest levels coming
# out 27 % darker with no deconvolution involved. The ratio of two frames
# built, subtracted and stretched the same way is 1 wherever the network did
# nothing, by construction.
SYNTHETIC_LUM = "LSYN"
SYNTHETIC_LUM_REF = "LSYN_REF"
# What the learned method does with a set that has no luminance (HOO/SHO).
# "starless": stacking/starless on every plane, the product written as
# "deblurred" (the rule from the nebula sweep of 2026-10-21 to the evening of
# 2026-10-22). "synthetic": make a luminance and sharpen that (above), the
# rule since 2026-10-22 evening. Tried that morning on trunk, ic1396, ngc7380
# and squid: no mesh, no specks, the rims sharpen, the colour is the planes'
# own, the picture is not relit -- but every bright star sat in a dark
# square inside its own halo. That was the model removing the measured
# kernel's far wing (the 145 px stamp's footprint) from the star while the
# halo beyond the stamp stayed: 63 % of the 32-60 px wing taken, the sky
# with it. The model is now trained to remove only the kernel's core
# (nn/deblur_data.taper_kernel, 2-4 FWHM since 2026-10-23, stamped on the
# checkpoint): 7 % of that wing taken, no moat on ngc1961, trunk or ic1396,
# stars 3.85 px, the galaxy's 6-12 px sharpening 85 % of the whole-kernel
# model's (the 1.5-3 taper of the day before: 5 %, 4 px, 65 %)
# (docs/LEARNED_DEBLUR_PLAN.md, "The bright-star moat is the kernel's wing").
NO_L_RULE = "synthetic"


def plateau_union(channels: dict) -> np.ndarray:
    """The planes' clipped cores and flat tops, united: what a sum of the
    planes must keep as given, since the sum's own detector cannot see them
    (nn/deblur_model.deblur_frame). Each plane is measured as the sidecar
    measures it, NaN filled with its median first."""
    seen: set = set()
    union = None
    for a in channels.values():
        if id(a) in seen:
            continue
        seen.add(id(a))
        work = np.asarray(a, dtype=np.float32)
        nan = ~np.isfinite(work)
        if nan.any():
            work = np.where(nan, float(np.nanmedian(work)), work)
        sig = sky_sigma(work)
        level = saturation_level(work)
        m = saturated_mask(work, level, sig) if level else flat_tops(work, sig)
        union = m if union is None else (union | m)
    return union


def synthetic_luminance(channels: dict) -> tuple:
    """(L, weights used) from the R/G/B planes of *channels* (sky in), by
    LUM_WEIGHTS; (None, {}) when none of them is there. NaN borders propagate,
    as the planes share one grid."""
    present = {c: float(LUM_WEIGHTS[c]) for c in ("R", "G", "B") if c in channels}
    if not present:
        return None, {}
    tot = sum(present.values())
    weights = {c: w / tot for c, w in present.items()}
    L = np.zeros(np.shape(channels[next(iter(present))]), dtype=np.float32)
    for c, w in weights.items():
        L += np.float32(w) * np.asarray(channels[c], dtype=np.float32)
    return L, weights


def deblur_channels(channels: dict, iters: int, clamp_to: str = "input",
                    method: str = "starless", log=None,
                    stars: str = "imaged", model=None, planes=None,
                    no_l: Optional[str] = None) -> tuple[dict, dict]:
    """Deblur a channel set. Returns (channels, info per channel touched).

    method="learned" (the default since 2026-10-21 afternoon): the luminance
    through nn/deblur_model, the flux-moving network trained on re-blurred
    stacks of this telescope, by the field's own measured kernel; every
    other plane comes back as given (info["kept"]). Deconvolving the colour
    planes too was measured on ngc6946 to sharpen each plane's own noise
    into coloured specks and to turn a saturated star's core cyan, so the
    colour stays where the blur put it under a sharpened L. Without an L
    (HOO/SHO) *no_l* decides (None: NO_L_RULE): "starless" sends the set
    to STARLESS, info["fallback"] saying so; "synthetic" makes one
    (synthetic_luminance: the palette's luminance weights on the mapped
    planes, 2026-10-22), deconvolves it, and
    returned beside its undeconvolved self as the SYNTHETIC_LUM /
    SYNTHETIC_LUM_REF planes; compose() scales the picture by the ratio of
    the two, info[SYNTHETIC_LUM]["synthetic"] saying how it was built and
    the planes kept as given. Per-plane deconvolution of narrowband sets was
    tried first (nebula sweep 2026-10-21) and made coloured single-pixel
    specks and mottled star halos for the same reason it did on broadband
    colour planes; the mesh it also showed was the old narrowband denoiser's
    checkerboard (lab manual step 39), not the network. A synthetic L whose
    field gives no kernel sends the set to STARLESS instead, info["fallback"]
    saying so. *iters* is starless's RL
    count on that path and only on/off on the learned one. *model* is a
    loaded checkpoint (tests); None loads deblur_model.DEFAULT_MODEL.
    *planes* names planes to sharpen explicitly (the render's
    --learned-planes); None is the rule above, and a selection that names
    nothing in the set falls back to the rule rather than refusing.
    Without torch or the checkpoint the set falls back to starless, with
    info["fallback"] saying why: this path exists on the Spark.

    method="starless" (the default 2026-10-19 to 2026-10-21): stacking/starless --
    RL on the picture with its stars taken off, the stars put back
    afterwards. Every channel is touched (its stars are lifted and put back)
    and RL runs on the luminance, or on every distinct plane when there is
    no L. *stars* is starless's star mode: "imaged" put back as lifted off
    (default since 2026-10-19) or "points" rendered back at their flux.

    method="rl": this module's deblur() on the luminance alone, or on every
    distinct plane when there is no L. LRGB/HALRGB take their brightness
    from L, so L is the plane whose sharpness is seen; HOO/SHO show their
    planes directly, so each distinct array is done once (G and B share
    O-III).
    """
    if int(iters) <= 0:
        return channels, {}
    # Distinct arrays once: HOO's G and B are one O-III stack.
    distinct: dict[int, str] = {}
    for name, a in channels.items():
        distinct.setdefault(id(a), name)
    uniq = {distinct[id(a)]: a for a in channels.values()}
    if method == "learned":
        try:
            import torch
            from nn import deblur_model as _dm
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            if model is None:
                model, _meta = _dm.load(_dm.DEFAULT_MODEL, dev)
            dev = str(next(model.parameters()).device)     # a model handed in says where it lives
        except Exception as e:  # noqa: BLE001 -- no torch, no checkpoint: not the Spark
            if log:
                log(f"learned deconvolution unavailable ({e!r}) -- starless RL{int(iters)} instead")
            out, infos = deblur_channels(channels, iters, clamp_to=clamp_to, method="starless",
                                         log=log, stars=stars)
            for i in infos.values():
                i["fallback"] = f"starless: {e!r}"
            return out, infos
        chosen = None if planes is None else tuple(p for p in planes if p in uniq)
        rule = (no_l or NO_L_RULE).lower()
        if rule not in ("starless", "synthetic"):
            raise ValueError(f"no_l must be 'starless' or 'synthetic', not {no_l!r}")
        if not chosen and "L" not in uniq and rule == "starless":
            why = ("starless: no luminance plane -- the learned deconvolution is luminance-only "
                   "and no_l='starless' was asked for (a synthetic luminance is the default)")
            if log:
                log(f"learned: {why}; RL{int(iters)} on every distinct plane instead")
            out, infos = deblur_channels(channels, iters, clamp_to=clamp_to, method="starless",
                                         log=log, stars=stars)
            for i in infos.values():
                i["fallback"] = why
            return out, infos
        if not chosen and "L" not in uniq:
            L, weights = synthetic_luminance(channels)      # the full set: B weighs the shared O-III too
            recipe = " + ".join(f"{w:.3f} {c}" for c, w in weights.items())
            why = None
            if L is None:
                why = "starless: no luminance plane and no R/G/B plane to make one from"
            else:
                keep = plateau_union(uniq)
                if log:
                    log(f"learned: no luminance plane -- made one, L = {recipe}; "
                        f"{int(keep.sum())} plateau px inherited from the planes")
                done, infos_u = _dm.deblur_channels({"L": L}, model, device=dev, planes=("L",),
                                                    log=log or (lambda s: None),
                                                    plateau={"L": keep})
                if infos_u["L"].get("why"):
                    why = (f"starless: the synthetic luminance ({recipe}) gave "
                           f"{infos_u['L']['why']}")
            if why:
                if log:
                    log(f"learned: {why}; RL{int(iters)} on every distinct plane instead")
                out, infos = deblur_channels(channels, iters, clamp_to=clamp_to, method="starless",
                                             log=log, stars=stars)
                for i in infos.values():
                    i["fallback"] = why
                return out, infos
            out = dict(channels)
            out[SYNTHETIC_LUM] = done["L"]
            out[SYNTHETIC_LUM_REF] = L
            infos = {name: {"why": None, "kept": True, "method": "learned", "iters": 0}
                     for name in channels}
            infos[SYNTHETIC_LUM] = {**infos_u["L"], "method": "learned", "iters": 0,
                                    "synthetic": recipe}
            return out, infos
        if not chosen:
            if planes is not None and log:
                log(f"learned: none of {planes} in the set -- the luminance instead")
            chosen = ("L",)
        done, infos_u = _dm.deblur_channels(uniq, model, device=dev, planes=chosen,
                                            log=log or (lambda s: None))
        out = {name: done[distinct[id(a)]] for name, a in channels.items()}
        infos = {}
        for name, a in channels.items():
            i = dict(infos_u[distinct[id(a)]])
            i.update(method="learned", iters=0)
            if i.get("why") == "not selected":
                i.update(why=None, kept=True)
            infos[name] = i
        return out, infos
    if method == "starless":
        from stacking import starless
        ref = "L" if "L" in uniq else None
        rl_on = {"L"} if "L" in uniq else set(uniq)
        done, infos_u = starless.deblur_set(uniq, iters, ref=ref, deconvolve=rl_on,
                                            clamp_to=clamp_to if clamp_to in ("input", "sky") else "sky",
                                            log=log, stars=stars)
        out = {name: done[distinct[id(a)]] for name, a in channels.items()}
        infos = {name: infos_u[distinct[id(a)]] for name, a in channels.items()
                 if distinct[id(a)] in infos_u}
        return out, infos
    if method != "rl":
        raise ValueError(f"method must be 'learned', 'starless' or 'rl', not {method!r}")
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
