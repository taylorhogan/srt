"""Starless deconvolution: RL on the picture with its stars taken off, the
stars rendered back as points at each channel's own flux.

This is `deblur=` since 2026-10-19, and the third image class of the morning
render beside raw and denoised. It exists because every attempt to make
Richardson-Lucy behave on the STARS of a denoised stack (stacking/deblur.py,
the record of 2026-10-09 to 2026-10-18) traded one artefact for the next:
halos, moats, rings, flowers, notches. RL on a star is a fight against an
object whose only information is its flux and position, and the measured
kernel is never exactly that star. So the stars are not deconvolved at all.

The flow, per channel set (L first, it supplies the star list):

1. Measure the kernel (deblur.measure_psf) and take the stars off with
   deblur.star_layer: the fitted wing beyond 10-16 px, the core lifted off a
   fill, each star's spikes fitted on its own pixels. What is left is the
   galaxy with holes where the stars were.
2. Classify every fitted peak against the kernel. A core broader than RING_X
   times the kernel at 4-6 px (a nucleus, an HII knot, NGC 7317's cusp at
   1.8x) is NOT a star and goes back as imaged within KNOT_RADIUS, except
   inside a real star's disc or deep mask (restoring those put a patch of
   the neighbour's halo back). A clipped star (deblur.flat_tops) is exempt,
   since a plateau is broad by definition. There is deliberately no test on
   the envelope at 12-20 px: on a denoised stack the brightest stars carry
   4-7x the kernel's wing there (the kernel is a median over stars the
   denoiser eroded) and a star in a neighbour's wing reads the same, so
   that test took 10 kADU stars for knots, restored them with their spikes
   and let the neighbour's deep fill erase them (ngc6946, 2026-10-19).
   A BRIGHT knot (BRIGHT_AMP above the floor) is put back as imaged AFTER
   the RL pass, not before: a close double star fails the ring test the
   same way a knot does (NGC 1961's pair at 4 px, 1.4x the kernel at 4-6
   px in every channel), and RL on a 7 kADU unresolved pair made the
   ringed, moated core this module exists to prevent (2026-10-19). A knot
   that bright is either a pair, a compact galaxy or a nucleus, none of
   which rings visibly when left alone; a faint HII region stays in the
   deconvolution, where it never rings visibly either.
3. Bright stars (amplitude BRIGHT_AMP or clipped) get a DEEP mask: the disc
   out to where the wing drops under DEEP_SIGMA sigma, which the
   star_layer's 10-16 px core fill does not cover. Twenty sigma, because at
   one sigma the discs were 200 px across and the fills under them replaced
   whole stretches of ngc6946's arms with grey (2026-10-18). A CLIPPED star
   is the exception at DEEP_SIGMA_CLIPPED: star_layer does not fit a clipped
   core, so nothing has taken its wing off and its spike arms run past the
   template, and at 20 sigma the wing stood as a ring round the rendered
   point with the arms as streaks beyond it (2026-10-19). The masked discs are
   filled by normalised convolution at the smallest scale that reaches each
   pixel, so a fill on a galaxy's envelope follows the envelope's gradient
   (a box pyramid averaged sky into those holes: dark discs round every star
   on NGC 7317's envelope).
4. RL the starless image with the measured kernel (deblur.deblur, clamp at
   the sky). Nothing in it is a point source, so the artefacts RL makes of
   point sources do not happen.
5. Render the stars back. One common core shape -- the azimuthal kernel
   deconvolved against itself for CORE_ITERS, which is what RL WOULD have
   converged a perfect star to (FWHM 5.2 px from 8.0 on ngc6946) -- in a
   49 px cosine-tapered stamp at each star's sub-pixel position, scaled to
   its flux. L's flux is the least-squares amplitude of the kernel on the
   star's core (annulus 10-20 px for a clipped one). A colour channel's flux
   is L's times the star's colour ratio, measured as the ratio of fixed
   r <= 8 px apertures on the two channels' star layers (sky from a 12-16 px
   annulus), identically on both so PSF-shape differences and the galaxy
   pedestal cancel; the ratio leans on the field's median below an aperture
   SNR of RATIO_SNR. Per-channel kernel fits were 20-35% off and gave the
   stars false colours. Every L star is masked and rendered in every colour
   channel, at a radius from its L flux: the companion of a double that B's
   own fitter missed stayed as a broad blob under a tight L core and came out
   as a pure-blue dot.

stars="imaged" (the DEFAULT since 2026-10-19 evening; "points" is the
rendering above, kept as a switch) skips step 5: the star layer
star_layer lifted off (core, fitted wing, spikes) is added back on the
deconvolved field exactly as imaged, every star at its denoised width.
Nothing is fitted, rendered or coloured, so a close double, a missed
companion or a mask edge cannot show -- at the cost that the stars stay at
the kernel's 7-8 px while the galaxy sharpens to 5. Chosen on NGC 1961 at
full resolution: the 4 px pair that the ring test calls a knot matched its
neighbours only this way, and nothing else in the field paid for it. Open
question from that comparison: the imaged stars carry a cyan-white cast at
2x saturation where the rendered points were neutral, and which of the two
star colours is right has not been run down.

Only the reference channel is deconvolved unless the caller says otherwise
(LRGB/HALRGB show L's sharpness; HOO/SHO show every plane). The colour
channels get the same star treatment without RL, so their stars sit under
L's rendered points instead of smeared around them.

Approved on ngc6946 HALRGB (2026-10-18, "the best rendition yet") and run on
ngc7320 LRGB: nuclei, tidal arcs and dust lanes deconvolved, stars clean
coloured points without halos. Known weak case: a bright star superposed on
a smooth galaxy, where the per-channel mask edges and the fill under the
star leave coloured rings (NGC 7317's star). Untested: crowded fields, pure
nebulae, stars within 20 px of the frame edge (left as imaged).

Returns the channel set untouched, with a reason in the reference's info,
when the field cannot give a kernel or has too few stars.
"""
import logging
import time
from typing import Callable, Optional

import numpy as np

from stacking import deblur as db

_logger = logging.getLogger(__name__)

ITERS = db.DEFAULT_ITERS   # RL iterations on the starless image (40: galaxies need them)
CORE_ITERS = 15            # the kernel against itself, for the rendered star shape
BRIGHT_AMP = 2000.0        # ADU above the floor: a star this bright gets a deep mask
DEEP_SIGMA = 20.0          # the deep mask ends where the fitted wing drops under this many sigma
DEEP_SIGMA_CLIPPED = 1.0   # ...for a clipped star, whose wing is not fitted off, under this many
DEEP_RADIUS = (16, 200)    # px: bounds on a deep mask's radius (10 for a colour channel's)
RING_X = 1.4               # core at 4-6 px more than this times the kernel's: not a star
KNOT_RADIUS = 36           # a non-stellar peak goes back as imaged within this
KNOT_FEATHER = 6           # px over which a bright knot's untouched disc blends into the RL'd field
STAR_DISC = 18             # ...but never inside a star's own disc
STAMP = 24                 # half-size of the rendered star stamp, 49 px across
STAMP_TAPER = 8            # cosine taper on its edge
APERTURE = 8               # colour ratio: star flux inside this radius
ANNULUS = (12, 16)         # ...less the median in this annulus
RATIO_SNR = 10.0           # aperture SNR at which a star's own ratio and the field's weigh equally
RATIO_MAX = 5.0            # of the field median: a ratio past this is a blend, clipped
RATIO_MIN_STARS = 20       # fewer well-measured stars than this and the field ratio is 1
EDGE = 20                  # px: a star closer than this to the edge is left as imaged
MIN_STARS = db.MIN_STARS   # fewer stars than this in the reference and nothing happens


def normalised_fill(img: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Fill *hole* from its surroundings at the smallest scale that reaches each pixel.

    Normalised convolution: the known pixels and a mask are both blurred at
    sigma 3, 6, 12 ... 192 px and each unknown pixel takes the first ratio
    whose support covers it. A hole's rim follows the local gradient, only
    its centre is coarse. A box pyramid instead averaged sky into holes on
    a galaxy's envelope (dark discs round every star on NGC 7317, 2026-10-18).
    """
    from scipy.ndimage import gaussian_filter

    known = (~hole).astype(np.float32)
    a = (np.asarray(img, dtype=np.float32) * known).astype(np.float32)
    out = np.asarray(img, dtype=np.float32).copy()
    todo = hole.copy()
    for sg in (3, 6, 12, 24, 48, 96, 192):
        if not todo.any():
            break
        num = gaussian_filter(a, sg)
        den = gaussian_filter(known, sg)
        ok = todo & (den > 0.02)
        out[ok] = (num / np.maximum(den, 1e-6))[ok]
        todo &= ~ok
    if todo.any():
        out[todo] = float(np.median(out[~hole])) if (~hole).any() else 0.0
    return out


def _disc(mask: np.ndarray, y: int, x: int, r: int) -> None:
    """OR a disc of radius r at (y, x) into *mask*, clipped to the frame."""
    h, w = mask.shape
    y0, y1, x0, x1 = max(0, y - r), min(h, y + r + 1), max(0, x - r), min(w, x + r + 1)
    gy, gx = np.mgrid[y0 - y:y1 - y, x0 - x:x1 - x]
    mask[y0:y1, x0:x1] |= (gy * gy + gx * gx) <= r * r


def _analyse(chan: np.ndarray) -> Optional[dict]:
    """Kernel, azimuthal kernel and profile, floor, sigma and star_layer products of one channel."""
    from scipy.ndimage import gaussian_filter, minimum_filter

    kernel, pinfo = db.measure_psf(chan)
    if kernel is None:
        return {"why": pinfo.get("why", "no kernel"), "pinfo": pinfo}
    if pinfo["fwhm_px"] < 1.5:
        return {"why": f"stars already at the pixel scale ({pinfo['fwhm_px']:.2f} px)", "pinfo": pinfo}
    spikes = pinfo.get("spikes")
    R = kernel.shape[0] // 2
    kaz = kernel if spikes is None else np.clip(kernel - db._centre(spikes, kernel.shape[0]), 0, None)
    kaz = kaz / kaz.sum()
    ky, kx = np.mgrid[-R:R + 1, -R:R + 1]
    kr = np.hypot(ky, kx)
    kpeak = float(kaz.max())
    profile = np.array([kaz[(kr >= r - 0.5) & (kr < r + 0.5)].mean() for r in range(R + 1)]) / kpeak
    model0 = gaussian_filter(minimum_filter(chan, db.FLOOR_MIN_SIZE), db.FLOOR_SMOOTH)
    sig = db.sky_sigma(chan, model0)
    level = db.saturation_level(chan)
    sl = db.star_layer(chan, model0, sig, kernel, level, spikes, pinfo.get("spike_axes"),
                       pinfo.get("arm_sigma", db.ARM_SIGMA))
    return dict(kernel=kernel, pinfo=pinfo, spikes=spikes, R=R, kaz=kaz, kr=kr, kpeak=kpeak,
                profile=profile, model0=model0, sig=sig, level=level, starless=sl[0], fits=sl[4],
                spike_part=sl[6], flat=db.flat_tops(chan, sig, level),
                above=chan - db.local_sky(chan, 4, 15))


def _classify(chan: np.ndarray, A: dict) -> tuple[list, list]:
    """(stars, knots): the fitted peaks split by the ring test in the module doc."""
    h, w = chan.shape
    kring = A["kaz"][(A["kr"] >= 4) & (A["kr"] < 6)].mean() / A["kpeak"]
    rr = np.hypot(*np.mgrid[-20:21, -20:21])
    ring = (rr >= 4) & (rr < 6)
    stars, knots = [], []
    for (y, x, dy, dx, _flux) in A["fits"]:
        y, x = int(y), int(x)
        if not (EDGE <= y < h - EDGE and EDGE <= x < w - EDGE):
            stars.append((y, x, dy, dx))
            continue
        a0 = float(chan[y, x] - A["model0"][y, x])
        st = chan[y - 20:y + 21, x - 20:x + 21] - A["model0"][y - 20:y + 21, x - 20:x + 21]
        broad = a0 > 0 and float(st[ring].mean()) / a0 > RING_X * kring
        if broad and not A["flat"][y, x]:
            knots.append((y, x))
        else:
            stars.append((y, x, dy, dx))
    return stars, knots


def _core_stamp(A: dict) -> np.ndarray:
    """The rendered star: the azimuthal kernel RL'd against itself, tapered, unit sum."""
    k = A["kaz"].astype(np.float32)
    core = np.clip(db._rl(k, k, CORE_ITERS), 0, None)
    R, cs = A["R"], STAMP
    core = core[R - cs:R + cs + 1, R - cs:R + cs + 1]
    tr = np.hypot(*np.mgrid[-cs:cs + 1, -cs:cs + 1])
    tp = np.clip((cs - tr) / float(STAMP_TAPER), 0, 1)
    core = core * (0.5 - 0.5 * np.cos(np.pi * tp))
    return (core / core.sum()).astype(np.float32)


def _wing_at(A: dict) -> Callable[[np.ndarray], np.ndarray]:
    """The azimuthal profile (peak = 1) as a function of radius, power-law past the kernel."""
    R, pn = A["R"], A["profile"]
    f0, f1 = min(30, R // 2), min(55, R - 5)
    beta = -np.polyfit(np.log(np.arange(f0, f1 + 1)),
                       np.log(np.clip(pn[f0:f1 + 1], 1e-12, None)), 1)[0]

    def at(r):
        r = np.asarray(r, float)
        out = np.interp(r, np.arange(R + 1), pn)
        far = r > f1
        out[far] = pn[f1] * (r[far] / f1) ** (-beta)
        return out
    return at


def _channel(name: str, chan: np.ndarray, A: dict, A_ref: dict, stars: list, knots: list,
             core: np.ndarray, iters: int, deconvolve: bool, clamp_to: str,
             ref: Optional[tuple], log: Callable[[str], None],
             stars_mode: str = "imaged") -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """One channel through steps 3-5 of the module doc. Returns (image, flux, apertures, info)."""
    from scipy.ndimage import shift as ndshift

    h, w = chan.shape
    R, kaz, kpeak, sig, flat, above = A["R"], A["kaz"], A["kpeak"], A["sig"], A["flat"], A["above"]
    wing_at = _wing_at(A)
    radii = np.arange(0, 400)
    kc = kaz[R - 20:R + 21, R - 20:R + 21]
    rr = np.hypot(*np.mgrid[-20:21, -20:21])
    corefit, ann = rr <= 6, (rr >= 10) & (rr < 20)
    deep = np.zeros((h, w), bool)
    n_deep = 0

    def deep_mask(y, x, amp, rmin):
        nonlocal n_deep
        nsig = DEEP_SIGMA_CLIPPED if A_ref["flat"][y, x] else DEEP_SIGMA
        ok = np.nonzero(wing_at(radii) < nsig * sig / max(amp, 1e-6))[0]
        rm = int(np.clip(ok[0] if len(ok) else radii[-1], rmin, DEEP_RADIUS[1]))
        _disc(deep, y, x, rm)
        n_deep += 1

    if ref is None:
        # The reference: its own star_layer, and each star's amplitude from a
        # least-squares fit of the kernel core (annulus for a clipped one).
        sl1 = (A["starless"] - A["spike_part"]).astype(np.float32)
        amps = np.zeros(len(stars))
        for i, (y, x, dy, dx) in enumerate(stars):
            if not (EDGE <= y < h - EDGE and EDGE <= x < w - EDGE):
                continue
            st = above[y - 20:y + 21, x - 20:x + 21]
            k2 = ndshift(kc, (dy, dx), order=1, mode="nearest")
            # A clipped core (a flat top, or within a factor two of the
            # ceiling) is fitted on the 10-20 px annulus instead, where the
            # wing is still PSF-shaped -- but only if the kernel has a wing
            # there: on Gaussian stars the annulus is noise over nothing.
            clipped = flat[y, x] or (A["level"] is not None and chan[y, x] > 0.5 * A["level"])
            sel = ann if clipped and (k2[ann] ** 2).sum() > 1e-3 * (k2[corefit] ** 2).sum() else corefit
            amp = max(float((st[sel] * k2[sel]).sum() / (k2[sel] ** 2).sum()) * kpeak, 0.0)
            amps[i] = amp
            if amp >= BRIGHT_AMP or flat[y, x]:
                deep_mask(y, x, amp, DEEP_RADIUS[0])
        flux = amps / kpeak
    else:
        # A colour channel: every reference star masked at a radius from its
        # reference flux scaled by the field's peak ratio, whether or not this
        # channel's own fitter found it.
        flux_ref, _ = ref
        pk0 = np.array([above[y, x] for (y, x, dy, dx) in stars])
        pk_ref = np.array([A_ref["above"][y, x] for (y, x, dy, dx) in stars])
        g0 = pk_ref > 500
        r0 = float(np.median(pk0[g0] / pk_ref[g0])) if g0.sum() >= RATIO_MIN_STARS else 1.0
        for (y, x, dy, dx), f_ in zip(stars, flux_ref):
            deep_mask(y, x, f_ * r0 * kpeak, 10)
        sl1 = (chan - A["spike_part"]).astype(np.float32)
        flux = None

    # Non-stellar peaks back as imaged, except inside a star's disc or deep mask.
    star_fp = np.zeros((h, w), bool)
    for (y, x, dy, dx) in stars:
        _disc(star_fp, y, x, STAR_DISC)
    protect = np.zeros((h, w), bool)
    for (y, x) in knots:
        _disc(protect, y, x, KNOT_RADIUS)
    protect &= ~(star_fp | deep)
    sl1 = np.where(protect, (chan - A["spike_part"]).astype(np.float32), sl1)
    sl2 = normalised_fill(sl1, deep)

    info = {"method": "starless", "iters": int(iters) if deconvolve else 0, "stars": len(stars),
            "non_stellar": len(knots), "fwhm_px": float(A["pinfo"]["fwhm_px"]),
            "deep_masks": n_deep, "deep_frac": float(deep.mean()), "sky_sigma": float(sig)}
    # Bright knots are classified on the reference, so every channel protects
    # the same ones; the disc is feathered over KNOT_FEATHER px so the join
    # between the RL'd field and the untouched core does not show.
    bright = [(y, x) for (y, x) in knots if A_ref["above"][y, x] >= BRIGHT_AMP]
    info["bright_knots"] = len(bright)
    if deconvolve:
        base, dinfo = db.deblur(sl2, iters, psf=A["kernel"], spikes=A["spikes"],
                                axes=A["pinfo"].get("spike_axes"),
                                arm_sigma=A["pinfo"].get("arm_sigma", db.ARM_SIGMA), clamp_to=clamp_to)
        info["deblurred_frac"] = dinfo.get("deblurred_frac", 0.0)
        if dinfo.get("why"):
            info["why_rl"] = dinfo["why"]
        if bright:
            from scipy.ndimage import distance_transform_edt
            keep = np.zeros((h, w), bool)
            for (y, x) in bright:
                _disc(keep, y, x, KNOT_RADIUS)
            keep &= ~(star_fp | deep)
            wgt = np.clip(distance_transform_edt(keep) / float(KNOT_FEATHER), 0.0, 1.0).astype(np.float32)
            base = (wgt * sl2 + (1.0 - wgt) * base).astype(np.float32)
    else:
        base = sl2

    # Colour by aperture on this channel's star layer, identically on the reference.
    layer = chan - sl2
    rr16 = np.hypot(*np.mgrid[-16:17, -16:17])
    ap, ann16 = rr16 <= APERTURE, (rr16 >= ANNULUS[0]) & (rr16 <= ANNULUS[1])
    n_ap = float(ap.sum()) ** 0.5
    aps = np.zeros(len(stars))
    for i, (y, x, dy, dx) in enumerate(stars):
        if 16 <= y < h - 16 and 16 <= x < w - 16:
            win = layer[y - 16:y + 17, x - 16:x + 17]
            aps[i] = float((win[ap] - np.median(win[ann16])).sum())
    if ref is not None:
        flux_ref, aps_ref = ref
        good = (aps_ref > 50 * sig * n_ap) & (aps > 0)
        med = float(np.median(aps[good] / aps_ref[good])) if good.sum() >= RATIO_MIN_STARS else 1.0
        snr = aps / (max(sig, 1e-6) * n_ap)
        wgt = snr * snr / (snr * snr + RATIO_SNR ** 2)
        ratio = np.where(aps_ref > 0, aps / np.maximum(aps_ref, 1e-6), med)
        ratio = wgt * np.clip(ratio, 0.0, RATIO_MAX * med) + (1.0 - wgt) * med
        flux = ratio * flux_ref
        info["ratio_to_ref"] = med
        info["ratio_from_field"] = int((wgt < 0.5).sum())

    info["stars_mode"] = stars_mode
    if stars_mode == "imaged":
        # The layer as lifted off, back on the deconvolved field: every star
        # at its imaged width and flux, nothing rendered.
        star_img = layer.astype(np.float32)
    else:
        star_img = np.zeros((h, w), np.float32)
        cs = STAMP
        for (y, x, dy, dx), f_ in zip(stars, flux):
            if f_ <= 0:
                continue
            stamp = ndshift(core, (dy, dx), order=1, mode="constant") * f_
            y0, y1, x0, x1 = max(0, y - cs), min(h, y + cs + 1), max(0, x - cs), min(w, x + cs + 1)
            star_img[y0:y1, x0:x1] += stamp[cs - (y - y0):cs + (y1 - y), cs - (x - x0):cs + (x1 - x)]
    info["star_flux"] = float(star_img.sum())
    out = (base + star_img).astype(np.float32)
    log(f"{name}: FWHM {info['fwhm_px']:.2f} px, {len(stars)} stars ({stars_mode}), {len(knots)} non-stellar "
        f"({len(bright)} bright, kept as imaged), "
        f"{n_deep} deep masks ({100 * deep.mean():.1f}% of the frame)"
        + (f", RL{int(iters)}" if deconvolve else "")
        + (f", colour ratio to reference {info['ratio_to_ref']:.3f}" if ref is not None else ""))
    return out, flux, aps, info


def deblur_set(channels: dict[str, np.ndarray], iters: int = ITERS, ref: Optional[str] = None,
               deconvolve: Optional[set] = None, clamp_to: str = "sky",
               log: Optional[Callable[[str], None]] = None,
               stars: str = "imaged") -> tuple[dict, dict]:
    """Starless-deconvolve a channel set. Returns (channels, info per channel).

    *stars* is "imaged" (the lifted-off layer put back as it was, the
    default) or "points" (rendered back as points at their flux; module doc).

    *ref* names the channel the star list comes from ("L" if present, else the
    first of R, G, B); *deconvolve* the channels RL runs on (the reference
    alone by default). Every channel gets its stars taken off and rendered
    back. All channels must share one pixel grid. NaN (the registration
    border) is filled for the work and put back.

    On refusal (no kernel, too few stars) every channel comes back as given
    and info[ref]["why"] says why.
    """
    iters = int(iters)
    if iters <= 0 or not channels:
        return channels, {}
    if stars not in ("points", "imaged"):
        raise ValueError(f"stars must be 'points' or 'imaged', not {stars!r}")
    say = log or (lambda m: _logger.info("starless %s", m))
    if ref is None:
        ref = "L" if "L" in channels else next((c for c in ("R", "G", "B") if c in channels), None)
    if ref is None or ref not in channels:
        raise ValueError(f"no reference channel in {sorted(channels)}")
    deconvolve = {ref} if deconvolve is None else set(deconvolve)
    shapes = {tuple(np.shape(a)) for a in channels.values()}
    if len(shapes) != 1:
        raise ValueError(f"channels on different grids: {shapes}")

    t0 = time.time()
    filled, nans = {}, {}
    for k, v in channels.items():
        a = np.asarray(v, dtype=np.float32)
        nans[k] = ~np.isfinite(a)
        filled[k] = np.nan_to_num(a, nan=float(np.nanmedian(a)), posinf=0.0, neginf=0.0) if nans[k].any() else a

    A_ref = _analyse(filled[ref])
    if "why" in A_ref:
        say(f"{ref}: left as given — {A_ref['why']}")
        return channels, {ref: {"method": "starless", "iters": 0, "why": A_ref["why"]}}
    stars_mode = stars
    stars, knots = _classify(filled[ref], A_ref)
    if len(stars) < MIN_STARS:
        why = f"only {len(stars)} stars in {ref}"
        say(f"{ref}: left as given — {why}")
        return channels, {ref: {"method": "starless", "iters": 0, "why": why}}
    core = _core_stamp(A_ref)

    out, infos = dict(channels), {}
    out[ref], flux, aps, infos[ref] = _channel(ref, filled[ref], A_ref, A_ref, stars, knots, core,
                                               iters, ref in deconvolve, clamp_to, None, say,
                                               stars_mode=stars_mode)
    for name in channels:
        if name == ref:
            continue
        A = _analyse(filled[name])
        if "why" in A:
            say(f"{name}: left as given — {A['why']}")
            infos[name] = {"method": "starless", "iters": 0, "why": A["why"]}
            continue
        out[name], _f, _a, infos[name] = _channel(name, filled[name], A, A_ref, stars, knots, core,
                                                  iters, name in deconvolve, clamp_to, (flux, aps), say,
                                                  stars_mode=stars_mode)
    for k in out:
        if k in infos and "why" not in infos[k] and nans[k].any():
            out[k] = out[k].copy()
            out[k][nans[k]] = np.nan
    say(f"starless deblur of {', '.join(channels)} in {time.time() - t0:.0f}s")
    return out, infos
