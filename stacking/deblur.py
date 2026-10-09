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
* The median-stacked star profile comes out ~10% broader than the stars it
  was built from (centroid error and the shift interpolation), and a kernel
  broader than the stars is the other way to ring. It is shrunk 15%: a kernel
  that is too narrow under-sharpens, which is the safe side.
* Saturated stars are clipped profiles, not PSF-shaped; they keep their input.

Returns the input untouched, with a reason, when the field has too few
isolated stars to measure a kernel from or the stars are already at the
pixel scale.
"""
import logging
from typing import Optional

import numpy as np

_logger = logging.getLogger(__name__)

DEFAULT_ITERS = 8
PSF_SCALE = 0.85          # shrink of the stacked star profile (see module doc)
PSF_RADIUS = 32           # half-size of the kernel in px
PSF_TAPER = 8             # cosine taper width at the kernel edge
MIN_STARS = 15            # fewer isolated stars than this and nothing happens
MAX_STARS = 400           # brightest isolated stars used for the kernel
SAT_FRAC = 0.30           # of the saturation level: protected as saturated
SAT_MIN_PIXELS = 4        # pixels at the ceiling before it counts as saturation
FLOOR_MIN_SIZE = 25       # minimum-filter window for the local floor model
FLOOR_SMOOTH = 8.0        # and its smoothing
CLAMP_SIGMA = 1.0         # dark deringing: never below input - this many sigma
BLEND_LO_SIGMA = 3.0      # signal above the local model where deblur starts
BLEND_HI_SIGMA = 10.0     # and where it is fully in


def sky_sigma(chan: np.ndarray) -> float:
    """Pixel noise from neighbour differences: immune to gradients and stars."""
    d = np.diff(chan[::4, ::4].astype(np.float64), axis=1)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0))


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
    psf = np.clip(np.median(np.stack(cuts), axis=0), 0, None)
    wgt = np.ones_like(psf)
    t = (rr > radius - taper) & (rr <= radius)
    wgt[t] = 0.5 * (1 + np.cos(np.pi * (rr[t] - (radius - taper)) / taper))
    wgt[rr > radius] = 0.0
    psf *= wgt
    s = float(psf.sum())
    if not np.isfinite(s) or s <= 0:
        return None, {"stars": len(cuts), "why": "kernel summed to nothing"}
    return psf / s, {"stars": len(cuts), "fwhm_px": fwhm_px}


def _shrink(psf: np.ndarray, factor: float) -> np.ndarray:
    from scipy.ndimage import zoom
    p = np.clip(zoom(psf, factor, order=3), 0, None)
    if p.shape[0] % 2 == 0:
        p = p[:-1, :-1]
    return p / p.sum()


def deblur(chan: np.ndarray, iters: int = DEFAULT_ITERS,
           psf: Optional[np.ndarray] = None,
           psf_scale: float = PSF_SCALE) -> tuple[np.ndarray, dict]:
    """Return (deblurred channel, info). The input is linear ADU with its sky.

    info carries what was measured; info["why"] is set when the channel was
    returned as given.
    """
    from scipy.ndimage import binary_dilation, gaussian_filter, minimum_filter
    from skimage.restoration import richardson_lucy

    iters = int(iters)
    if iters <= 0:
        return chan, {"iters": 0, "why": "deblur=0"}
    info: dict = {"iters": iters}
    if psf is None:
        psf, pinfo = measure_psf(chan)
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
    sig = sky_sigma(x)
    info["sky_sigma"] = sig
    # A floor that follows the galaxy and never clips a real pixel.
    model = gaussian_filter(minimum_filter(x, FLOOR_MIN_SIZE), FLOOR_SMOOTH)
    floor = np.minimum(model, x) - 2.0 * sig
    rl = richardson_lucy(np.clip(x - floor, 1e-3, None).astype(np.float32),
                         kernel.astype(np.float32), num_iter=iters, clip=False) + floor
    rl = np.maximum(rl, x - CLAMP_SIGMA * sig)                 # dark deringing
    # Only where there is signal to sharpen. The minimum-filter model sits
    # ~3 sigma under the sky mean (the minimum of 625 noisy pixels), so the
    # signal is measured from the model plus that offset, or the blend would
    # start at the sky itself and RL texture would leak into half of it.
    ref = model + float(np.median(x - model))
    lo, hi = BLEND_LO_SIGMA * sig, BLEND_HI_SIGMA * sig
    t = np.clip((gaussian_filter(x, 1.5) - ref - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    blend = t * t * (3 - 2 * t)
    level = saturation_level(x)
    sat = x > SAT_FRAC * level if level is not None else np.zeros(x.shape, bool)
    if sat.any():
        m = gaussian_filter(binary_dilation(sat, iterations=14).astype(np.float32), 6.0)
        blend = blend * (1.0 - np.clip(m / max(float(m.max()), 1e-6), 0.0, 1.0))
    out = (x + blend * (rl - x)).astype(np.float32)
    info["deblurred_frac"] = float(np.mean(blend > 0.5))
    info["saturated_frac"] = float(sat.mean())
    _logger.info("deblur: RL%d from %d stars (FWHM %.2f px), %.2f%% of pixels, sky sigma %.3g",
                 iters, info.get("stars", 0), info.get("fwhm_px", float("nan")),
                 100 * info["deblurred_frac"], sig)
    return out, info


def deblur_channels(channels: dict, iters: int) -> tuple[dict, dict]:
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
        out["L"], infos["L"] = deblur(channels["L"], iters)
        return out, infos
    done: dict[int, tuple] = {}
    for name in ("R", "G", "B"):
        if name not in channels:
            continue
        key = id(channels[name])
        if key not in done:
            done[key] = deblur(channels[name], iters)
        out[name], infos[name] = done[key]
    return out, infos
