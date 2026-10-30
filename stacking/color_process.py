"""Multi-filter colour processing: stack each channel, combine, stretch.

Backs the webchat ``process <dso> <recipe>`` command. Recipes:

    LRGB    R->R  G->G  B->B, with L substituted as luminance
    HALRGB  LRGB plus the Ha EXCESS (Ha minus its continuum share of R) added
            into R and L, so HII regions get redder AND brighter (see HA_GAIN)
    HOO     Ha->R, O-III->G and B      (H-alpha region palette)
    SHO     S-II->R, Ha->G, O-III->B   (the "Hubble" palette)

Two things here are not obvious and were both learned the hard way on sh2-92:

1. Every filter registers to ONE shared reference frame. Stacking each filter
   against its own reference leaves the channels offset by however far the mount
   drifted between them, and nothing downstream recovers it.

2. Channels are stretched on a SHARED ADU scale, not normalised individually.
   The whole point of a palette is the ratio between channels; scaling each to
   its own range destroys exactly that. Normalising by each channel's *noise* is
   the same mistake in a subtler form — on sh2-92 it handed O-III a 1.6x boost
   (its noise is 0.68 ADU against Ha's 1.10) and turned every continuum star cyan.

   The one deliberate exception is the palette gain (`gain=`, since
   2026-10-28): an explicit, logged scaling of a palette's filters, asked for
   by name, because the "classic" Hubble palette IS each channel stretched to
   its own range and a true-ratio SHO of an HII region is simply green (Ha is
   3-5x S-II and O-III). See PALETTE_GAIN.
"""

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

_logger = logging.getLogger(__name__)

# channel -> filter, per recipe. "L" is luminance, applied after the RGB blend.
RECIPES: dict[str, dict[str, str]] = {
    "LRGB": {"R": "R", "G": "G", "B": "B", "L": "L"},
    "HOO":  {"R": "Ha", "G": "O-III", "B": "O-III"},
    "SHO":  {"R": "S-II", "G": "Ha", "B": "O-III"},
    # HSO is SHO with R and G swapped. Which reads better is set by the
    # relative channel strengths, not by taste: on NGC 6888 the nebula measures
    # Ha 4.20 ADU, S-II 2.62 (0.62 of Ha), O-III 0.36 (0.09). SHO therefore puts
    # the strongest channel in green and renders yellow-green; HSO puts it in
    # red and renders warm. Both are honest — compose holds one brightness scale
    # across channels so their ratio survives — so the choice is which channel
    # you want carrying the image.
    "HSO":  {"R": "Ha", "G": "S-II", "B": "O-III"},
    # HA is a fifth ROLE, not a primary: _prepare turns it into an emission
    # map and folds that into R and L before anything is stretched. The raw
    # Ha stack and the excess map are still exported beside the primaries.
    "HALRGB": {"R": "R", "G": "G", "B": "B", "L": "L", "HA": "Ha"},
}

# FITS FILTER values vary by capture software; match on a squashed form.
_ALIASES: dict[str, tuple[str, ...]] = {
    "Ha":    ("HA", "HALPHA", "H-ALPHA", "HALFA"),
    # "O2" is what the 2024-era capture software wrote for O-III, and the whole
    # archive on /media/taylor/cdk17 uses it. Identified 2026-08-21 from focus
    # position, which tracks wavelength: O2 focuses at 71909, between B (72053,
    # 445nm) and G (71888, 530nm), exactly where a 500nm filter belongs — while
    # Ha and S-II sit far off at 71432 and 70682. Before this entry those frames
    # were invisible to every path that resolves a filter, so an HOO or SHO
    # process on that data rendered a one-channel image and looked like it
    # worked. 122 O-III frames on NGC 6888 alone.
    "O-III": ("OIII", "O3", "O-III", "OXYGEN3", "O2"),
    "S-II":  ("SII", "S2", "S-II", "SULFUR2"),
    "L":     ("L", "LUM", "LUMINANCE", "LUMA", "CLEAR"),
    "R":     ("R", "RED"),
    "G":     ("G", "GREEN"),
    "B":     ("B", "BLUE"),
}

# The black point is a PERCENTILE of the channel, not a fixed multiple of sky
# noise. A constant "median + k sigma" cannot suit both kinds of target: it
# clips a fixed fraction of a *Gaussian*, so on a field that is mostly empty sky
# it crushes most of the frame. At the old 0.8 sigma, abell2151 rendered with
# over half its pixels at exactly zero — the intracluster light was thrown away,
# and the hard-stretched grain left around the bright parts was convincing
# enough to be mistaken for satellite trails. A percentile clips the same share
# of pixels whatever the target, so a sparse cluster and a nebula that fills the
# frame both keep their faint end.
BLACK_PCT = 65.0     # share of pixels that go to black
WHITE_PCT = 99.0
SOFTENING = 0.025
EDGE_CROP = 0.02     # dithered border where not every frame contributed
# Ceiling on the LRGB luminance rescale. The blend divides by the RGB
# luminance, which is near zero in the background — with the old high black
# point those pixels were already clipped to nothing, but a lower black point
# lets them through and the division amplifies their noise into colour speckle.
MAX_LUM_BOOST = 3.0
# Large-scale background model, subtracted per channel before stretching.
# Without flats the frame carries a vignetting dome comparable in size to the
# signal itself; the old crushing black point hid it, and lowering the black
# point simply revealed it as a bright blob covering most of abell2151. Sky
# gradients do the same on flat-corrected data. The mesh is coarse on purpose —
# big enough that cluster galaxies and nebulosity are not absorbed into it.
SUBTRACT_BACKGROUND = True
BG_MESH_FRACTION = 4      # mesh boxes across the short axis
# Measured trade-off on synthetic fields (dome 40x sky noise, nebula filling the
# middle third).  Mesh 8 flattens the dome best but absorbs 30-45% of a large
# nebula; mesh 3 leaves the nebula untouched but barely dents the dome.  Mesh 4
# with black at p65: cluster saturation 3.4%, nebula centre retained 94%.

# Average-neutral SCNR strength, 0..1. Off by default: it is the right move on
# LRGB and a matter of taste on SHO, but on HOO it actively destroys the palette
# (see _scnr), so it has to be asked for rather than assumed.
SCNR_AMOUNT = 0.0

# Deconvolution before the stretch: 0 = off. ON by default since 2026-10-19.
# Since the afternoon of 2026-10-21 the method is `learned` (nn/deblur_model
# via stacking/deblur.deblur_channels): the luminance through the network
# trained on re-blurred stacks of this telescope, by the field's own kernel,
# flux conserved by construction; the colour planes are left as stacked.
# Measured on the three holdout galaxies (docs/LEARNED_DEBLUR_PLAN.md): stars
# 7.5 -> ~4 px with no rings, 96-98 % of the flux kept, the galaxy improved
# at every scale; chosen by the owner by eye on ngc6946 and ngc1961. The
# iteration count only matters to the RL methods: method=starless (the
# default 2026-10-19 to 10-21) takes the stars off, RL40s what is left and
# puts the stars back as imaged -- 40 because the stars converge in 8-15 but
# the galaxies need 40 (ngc7320's nuclei 12.4 -> 6.7 px FWHM). On a denoised
# stack the order is stack, denoise, deblur; the other way round was
# measured worse (deblur module doc), and the learned model sharpens noise on
# a raw stack, so it is not optional for it.
DEBLUR_ITERS = 40
# method=rl is the earliest form, RL on the whole luminance with the stars in
# (stacking/deblur: sky out, non-clipping floor, dark clamp at the local sky,
# signal-only blend, spikes fitted off star by star, azimuthal kernel). It
# sharpens the stars too but on a denoised stack every bright star paid for
# it with a halo, moat, ring or notch in turn (2026-10-09 to 10-18); kept for
# comparison, as is starless.
DEBLUR_METHOD = "learned"
# `dering=sky` (default) lets a star's halo drain down to the local sky mean
# and no further, which conserves flux and is what a kernel with a measured
# wing needs: the wing cannot drain under `dering=input`, which holds every
# pixel at or above its input and leaves the halo standing under a brighter
# core. ngc2146 L 2.26" -> 1.44" at deblur=15 with no halos or rings.
DEBLUR_DERING = "sky"

# Star white balance, "none" or "stars". The channels are stretched on a
# shared scale so their MEASURED ratios survive into the picture -- which is
# right, but those ratios are the instrument's (filter widths, QE), not the
# sky's, and on m33 they came out yellow-green where the arms are blue-white.
# "stars" scales R and B so the median star in the field is neutral: the
# standard "average field star = white" reference, measured with aperture
# photometry and a local sky annulus so stars on the galaxy do not carry its
# light. G and L are never touched -- this is hue, not brightness. Off by
# default because HOO and SHO are palettes, not colour, and must not be
# balanced.
WHITE_BALANCE = "none"
STAR_WB_MIN_STARS = 20
STAR_WB_MAX_STARS = 300
STAR_WB_THRESH_SIGMA = 10.0
STAR_WB_MAX_NPIX = 600
STAR_WB_EDGE_PX = 40

# --- HALRGB: how much Ha excess goes into R and L ---------------------------
#
# An Ha filter still passes continuum: a star is (bandwidth ratio) as bright
# in Ha as in R, roughly 5%. Adding the Ha stack straight into R therefore
# reddens every star and the whole galaxy disk, not the HII regions. So first
# the continuum share is measured — the median Ha/R ratio over the brightest
# R pixels, which are stars and galaxy body, with HII regions a minority that
# the median ignores — and only the EXCESS, max(Ha - ratio*R, 0), is blended.
# The excess is line emission in Ha-frame ADU. A line photon lands in R at
# (almost) the same ADU it lands in Ha, so gain 1.0 counts the emission twice
# in R: the contrast of an HII region against the disk doubles. It goes into
# L at the same gain, because LRGB takes brightness from L and an excess in
# R alone would only change hue — HII regions would turn red without getting
# brighter, the classic complaint about HaRGB done as a hue-only blend.
# `ha=` on the process/movie command sets the gain; 0 makes HALRGB = LRGB.
HA_GAIN = 1.0
HA_CONTINUUM_TOP_PCT = 98.0   # R pixels above this percentile define the ratio
HA_CONTINUUM_MIN_PIX = 500    # fewer than this and the top slice is widened

# --- palette gain: toning a narrowband filter up or down, on purpose ---------
#
# compose holds one brightness scale across the channels so a palette shows
# the real ratio between the lines. On an HII region that ratio is Ha >> S-II,
# O-III (IC 410, 2026-10-28: S-II 0.2-0.3 of Ha, O-III 0.3-0.5), so a
# true-ratio SHO is green with a little blue, and the gold-and-blue "Hubble"
# look people expect is only reachable by scaling Ha down. `gain=` does that
# and nothing else, on the background-subtracted planes before the shared
# white point is found, so the per-channel exports show the scaled planes:
#   gain=Ha:0.4            Ha at 0.4x; several as Ha:0.4/O-III:1.5 (`/`, since
#                          `,` is the sweep separator on the process command)
#   gain=hubble            every palette filter scaled so its nebula level
#                          matches the strongest filter's: the classic
#                          palette, each channel on its own range. The level
#                          is PALETTE_LEVEL_PCT (90) of the plane's block
#                          medians (PALETTE_LEVEL_BLOCK px squares), NOT of
#                          the pixels: a high pixel percentile is the stars,
#                          which are continuum and the same in every
#                          narrowband filter, and equalising them does
#                          nothing (IC 410, first try: 1.00/1.02/1.04). A
#                          32 px block median is blind to a 4-8 px star. And
#                          p90 of the blocks, not p99.5: the top half percent
#                          is the brightest knot, which on IC 410 is the
#                          O-III-strong rim, so O-III was lifted to match Ha
#                          there (x1.75) and S-II only x2.4, and the tadpoles
#                          stayed green; p90 is the body of the nebula
#                          (S-II x3.6, O-III x1.7: tadpoles gold, rim
#                          cyan-white, the look this palette is asked for).
#                          A factor is capped at PALETTE_GAIN_MAX so a filter
#                          with no nebula in it (level at the noise) cannot
#                          be scaled to noise. The factors are measured on
#                          the raw set, logged, and pinned on the denoised
#                          and learned composites (_prepare's info). The
#                          shared white is taken before the gain (see
#                          _prepare), so the gain never darkens the picture.
# Palettes only (HOO/SHO/HSO): LRGB and HALRGB are colour, and a per-filter
# gain there is a white balance done wrong. The stars do NOT take the gain
# (since the afternoon of 2026-10-28): they are continuum, the same in every
# filter, and scaling R and B lifted every one of IC 410's stars to magenta.
# Each scaled plane's stars are lifted off a smooth fill (star_mask discs
# over the block-median background, PALETTE_FILL_BLOCK) before the gain and
# put back as imaged, so a star keeps the colour the plain palette gives it
# and the nebula under its disc takes the gain with the rest.
PALETTE_GAIN = None
PALETTE_FILL_BLOCK = 16
PALETTE_PROTECT_STARS = True
# What a star gets under the gain. "neutral" (default since the afternoon
# of 2026-10-28): its light above the fill averaged over the planes and put
# back the same in every channel, so it is white at the palette's
# brightness. "imaged" puts each plane's own star light back: in SHO that
# is magenta even without a gain, because a star is FAINTER through the Ha
# filter (its own Ha absorption line sits in the band) than through S-II
# and O-III, and Ha is the green channel; the owner's "stars remain
# magenta" after the first fix was this. "none" lets the stars take the gain.
PALETTE_STAR_MODE = "neutral"
# The gain's star mask is wider and deeper than stars=rgb's: a 300 s
# narrowband star's halo reaches ~20 px (IC 410, Ha: 34 ADU above the fill
# at 10 px, gone by 20-30) and the faint stars sit at 3-5 sigma, and every
# one left outside the mask comes out magenta. 3 sigma, 5x the isophotal
# size, 90 px at most.
PALETTE_STAR_THRESH_SIGMA = 3.0
PALETTE_STAR_RADIUS_A = 5.0
PALETTE_STAR_MAX_RADIUS_PX = 160.0   # the brightest star's halo (Melotte 15: 108 px at 1 ADU)
PALETTE_STAR_MAX_NPIX = 20000
PALETTE_STAR_MIN_PEAK_FLUX = 0.008   # compactness: a point, not a knot (see star_mask); stars 0.011+
# The faint stars are found on the DENOISED planes when the render has them
# (apply_palette_gain(quiet=)): at 3 sigma of the raw sum Melotte 15 gave
# 2614 stars and every fainter one took the x4 S-II gain and came out as a
# red speck over the whole field ("many red stars", 2026-10-28); the
# denoised sum, a tenth of the noise, gives 7360. The two lists are joined,
# not swapped: 8-14 of the raw's 60 brightest fall out of the denoised pass
# (the quiet isophote outgrows the size limit, see below), and the discs
# of the denoised pass are sized from the RAW floor, or every star is a
# decade brighter and the discs cover 40 % of the field (10 % joined).
PALETTE_STAR_MATCH_PX = 4.0          # a denoised find this close to a raw find is the same star
PALETTE_STAR_FEATHER_MAX_PX = 24.0   # the skirt is 0.6 r, which on a 145 px halo disc is 87 px
# The stars are found ONCE, on the raw set, and the same discs are pinned on
# the denoised and learned sets (info["palette_stars"]): found again on a
# denoised plane, whose noise is a fraction of the raw's, a bright star's
# 3-sigma isophote outgrew the size limit, the star fell out of the mask,
# the fill under it was its own halo, and the gain painted a magenta blob
# the size of the halo (IC 410, 2026-10-28). Same rule as the factors and
# the white balance: a series shares one measurement.
PALETTE_LEVEL_PCT = 90.0
PALETTE_LEVEL_BLOCK = 32
PALETTE_GAIN_MAX = 6.0


def nebula_level(plane: np.ndarray, pct: float = PALETTE_LEVEL_PCT,
                 block: int = PALETTE_LEVEL_BLOCK) -> float:
    """A bright-nebula level of a subtracted plane that the stars cannot set:
    *pct* of the medians of block x block squares."""
    a = np.asarray(plane, dtype=np.float32)
    h, w = a.shape[0] // block * block, a.shape[1] // block * block
    if h == 0 or w == 0:
        return float(np.nanpercentile(a, pct))
    blocks = a[:h, :w].reshape(h // block, block, w // block, block)
    med = np.nanmedian(blocks, axis=(1, 3))
    return float(np.nanpercentile(med, pct))
_FILTER_NAMES = {"ha": "Ha", "h-alpha": "Ha", "o-iii": "O-III", "oiii": "O-III",
                 "o3": "O-III", "s-ii": "S-II", "sii": "S-II", "s2": "S-II"}


def parse_palette_gain(spec) -> Optional[object]:
    """`gain=` as typed -> None, "hubble" or {filter: factor}.

    Filters by their FITS names, case-insensitive (ha, o-iii/oiii/o3,
    s-ii/sii/s2); pairs separated by `/`. A dict or "hubble" passes through.
    """
    if spec is None or spec == "" or spec is False:
        return None
    if isinstance(spec, dict):
        return {_FILTER_NAMES.get(str(k).lower(), k): float(v) for k, v in spec.items()}
    text = str(spec).strip()
    if text.lower() in ("hubble", "classic"):
        return "hubble"
    out: dict[str, float] = {}
    for part in text.split("/"):
        part = part.strip()
        if not part:
            continue
        key, sep, val = part.partition(":")
        if not sep:
            key, sep, val = part.partition("=")
        name = _FILTER_NAMES.get(key.strip().lower())
        if name is None or not sep:
            raise ValueError(f"gain: expected hubble or Ha:0.4/O-III:1.5, got {spec!r}")
        try:
            f = float(val)
        except ValueError:
            raise ValueError(f"gain: {part!r} is not filter:number") from None
        if not f > 0:
            raise ValueError(f"gain: {name} must be > 0, got {f:g}")
        out[name] = f
    return out or None


def palette_gain_roles(recipe: str, gain) -> Optional[object]:
    """The parsed gain keyed by the recipe's ROLES (R/G/B), for _prepare.

    HOO points G and B at one filter, so a gain on O-III scales both. A
    recipe with a luminance (LRGB/HALRGB) refuses: it is colour, not a palette.
    """
    gain = parse_palette_gain(gain)
    if gain is None:
        return None
    mapping = RECIPES.get(str(recipe).upper())
    if mapping is None:
        raise ValueError(f"gain: unknown recipe {recipe!r}")
    if "L" in mapping:
        raise ValueError(f"gain: {recipe} is colour, not a palette; gain= is for HOO/SHO/HSO")
    if gain == "hubble":
        return "hubble"
    roles: dict[str, float] = {}
    for filt, f in gain.items():
        hit = [role for role in ("R", "G", "B") if mapping.get(role) == filt]
        if not hit:
            raise ValueError(f"gain: {recipe} has no {filt} channel")
        for role in hit:
            roles[role] = f
    return roles


def _block_median(a: np.ndarray, block: int) -> np.ndarray:
    h, w = a.shape
    ph, pw = (-h) % block, (-w) % block
    if ph or pw:
        a = np.pad(a, ((0, ph), (0, pw)), mode="edge")
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN blocks
            return np.nanmedian(a.reshape(a.shape[0] // block, block, a.shape[1] // block, block),
                                axis=(1, 3)).astype(np.float32)


def smooth_fill(plane: np.ndarray, block: int = PALETTE_FILL_BLOCK,
                mask: Optional[np.ndarray] = None) -> np.ndarray:
    """The plane without its stars: block medians, bilinearly back to size.

    A block median is blind to a star smaller than the block; what it keeps
    is the nebula and the sky at block resolution. With *mask* (0..1, the
    stars) the masked pixels are left out of the medians, because a bright
    star's halo fills a 16 px block and pulled the fill under it up (IC 410:
    the Ha profile went 5 ADU NEGATIVE against the fill at 20 px). A block
    the mask covers entirely is interpolated from its unmasked neighbours
    by normalised Gaussian smoothing at block resolution, narrow first
    (half a block) and wider only where nothing narrower reaches: a 4x
    block median pasted in instead carried the halo it was meant to
    exclude and showed as a red square around every bright star.
    """
    from scipy.ndimage import zoom, gaussian_filter
    a = np.asarray(plane, dtype=np.float32)
    h, w = a.shape
    if mask is not None:
        a = np.where(np.asarray(mask) > 0.5, np.nan, a).astype(np.float32)
    med = _block_median(a, block)
    good = np.isfinite(med)
    if not good.all():
        filled = np.where(good, med, 0.0).astype(np.float32)
        wgt = good.astype(np.float32)
        out = med.copy()
        need = ~good
        for sigma in (0.5, 1.0, 2.0, 4.0, 8.0):
            if not need.any():
                break
            num = gaussian_filter(filled, sigma, mode="nearest")
            den = gaussian_filter(wgt, sigma, mode="nearest")
            ok = need & (den > 0.05)
            out[ok] = num[ok] / den[ok]
            need &= ~ok
        if need.any():
            rest = float(np.nanmedian(a)) if np.isfinite(a).any() else 0.0
            out[need] = np.float32(rest)
        med = out
    up = zoom(med, block, order=1, mode="nearest", grid_mode=True)
    return np.ascontiguousarray(up[:h, :w])


def apply_palette_gain(subbed: dict[str, np.ndarray], gain,
                       level_pct: float = PALETTE_LEVEL_PCT,
                       protect_stars: bool = PALETTE_PROTECT_STARS,
                       star_discs=None, info: Optional[dict] = None,
                       star_mode: str = PALETTE_STAR_MODE, quiet=None, star_arms=None
                       ) -> tuple[dict[str, np.ndarray], Optional[dict[str, float]]]:
    """Scale the palette planes; returns (channels, {role: factor} used).

    *gain* is {role: factor} (applied as given, which is how a series pins
    one measurement) or "hubble" (measured here: each of R/G/B scaled so its
    nebula_level matches the highest). None leaves everything alone.

    With *protect_stars* a plane is P*g + M*(S - (P - F)*g): outside the
    star mask M the plane is simply scaled; inside, the smooth fill F under
    the star takes the gain and the star light S goes back on top. S is the
    star's own light above the fill (P - F) with star_mode "imaged", or
    with "neutral" (PALETTE_STAR_MODE) that light averaged over the
    distinct planes, the same in every channel, so the star is white at the
    palette's brightness; "none" is protect_stars=False. One mask for every
    plane, found on their sum, so a star is treated the same in each.
    *star_discs* (x, y, r) pins the stars a previous call found (returned
    in *info*["star_discs"]) instead of finding them again; see
    PALETTE_STAR_MAX_NPIX for why a series must. *quiet* is the same
    planes denoised, when the caller has them: the faint stars are found
    there and joined to the raw's (PALETTE_STAR_MATCH_PX). The bright stars'
    diffraction spikes get arms (spike_axis / spike_arms, returned in
    *info*["star_arms"]); *star_arms* pins a previous call's with *star_discs*.
    """
    if star_mode == "none":
        protect_stars = False
    if gain is None:
        return subbed, None
    roles = [c for c in ("R", "G", "B") if c in subbed]
    if gain == "hubble":
        levels = {c: nebula_level(subbed[c], level_pct) for c in roles}
        top = max(levels.values(), default=0.0)
        factors = {}
        for c, lv in levels.items():
            f = top / lv if lv > 0 and top > 0 else 1.0
            if f > PALETTE_GAIN_MAX:
                _logger.warning("Palette gain: %s level %.3g is noise against %.3g; "
                                "capped at x%g", c, lv, top, PALETTE_GAIN_MAX)
                f = PALETTE_GAIN_MAX
            factors[c] = f
    else:
        factors = {c: float(f) for c, f in dict(gain).items() if c in roles}
    out = dict(subbed)
    scaled = [c for c, f in factors.items() if f != 1.0]
    mask = None
    if scaled and protect_stars:
        touched = roles if star_mode == "neutral" else scaled
        shape = np.asarray(subbed[roles[0]]).shape
        arms = list(star_arms or [])
        if star_discs is not None:
            discs = list(star_discs)
            mask = (mask_from_discs(shape, discs, arms=arms, feather_max=PALETTE_STAR_FEATHER_MAX_PX)
                    if discs else None)
        else:
            distinct = {id(subbed[c]): subbed[c] for c in roles}
            total = np.sum([np.nan_to_num(np.asarray(p, dtype=np.float32))
                            for p in distinct.values()], axis=0)
            mask, minfo = star_mask(total, thresh_sigma=PALETTE_STAR_THRESH_SIGMA,
                                    max_npix=PALETTE_STAR_MAX_NPIX,
                                    radius_a=PALETTE_STAR_RADIUS_A,
                                    max_radius=PALETTE_STAR_MAX_RADIUS_PX,
                                    min_peak_flux=PALETTE_STAR_MIN_PEAK_FLUX,
                                    radius_from="peak")
            discs = minfo.get("discs", [])
            n_raw = len(discs)
            if quiet is not None and discs and "rms" in minfo:
                discs = join_quiet_stars(discs, quiet, roles, minfo["rms"])
            if discs:
                # the spikes' axis is read off the quiet planes when there are any
                axis_src = total
                if quiet is not None:
                    qd = {id(quiet[c]): quiet[c] for c in roles if c in quiet}
                    if qd:
                        axis_src = np.sum([np.nan_to_num(np.asarray(p, dtype=np.float32))
                                           for p in qd.values()], axis=0)
                axis = spike_axis(axis_src, discs)
                arms = spike_arms(discs, axis)
                mask = mask_from_discs(shape, discs, arms=arms, feather_max=PALETTE_STAR_FEATHER_MAX_PX)
                _logger.info("Palette gain: spikes %s", f"on {len(arms)} stars, axis {axis:.0f} deg"
                             if arms else "none found")
            _logger.info("Palette gain: %d stars %s (%d on the raw planes; %.2f%% of the field)",
                         len(discs), "made neutral" if star_mode == "neutral" else "kept as imaged",
                         n_raw, 100.0 * float(mask.mean()))
            if not discs:
                mask = None
        if info is not None:
            info["star_discs"] = discs
            info["star_arms"] = arms
    fills: dict[int, np.ndarray] = {}
    def fill_of(c):
        k = id(subbed[c])
        if k not in fills:
            fills[k] = smooth_fill(np.asarray(subbed[c], dtype=np.float32), mask=mask)
        return fills[k]
    star_common = None
    if mask is not None and star_mode == "neutral":
        # The star light above the fill, averaged over the distinct planes.
        planes = {id(subbed[c]): c for c in roles}
        star_common = np.mean([np.asarray(subbed[c], dtype=np.float32) - fill_of(c)
                               for c in planes.values()], axis=0).astype(np.float32)
    done: dict[tuple, np.ndarray] = {}          # HOO: G and B are one plane, scale it once
    for c in (touched if mask is not None else scaled):
        f = factors.get(c, 1.0)
        key = (id(subbed[c]), f)
        if key in done:
            out[c] = done[key]
            continue
        p = np.asarray(subbed[c], dtype=np.float32)
        res = p * np.float32(f)
        if mask is not None:
            fill = fill_of(c)
            star = star_common if star_common is not None else (p - fill)
            res += mask * (star - (p - fill) * np.float32(f))
        done[key] = res
        out[c] = res
    return out, factors


def join_quiet_stars(discs: list, quiet: dict, roles, raw_rms: float) -> list:
    """*discs* (x, y, r) found on the raw planes plus the stars found on the
    *quiet* (denoised) planes' sum that are not already among them: the
    quiet pass sees the faint stars (PALETTE_STAR_MATCH_PX comment), and
    its discs are sized from the raw's detection floor, not its own."""
    distinct = {id(quiet[c]): quiet[c] for c in roles if c in quiet}
    if not distinct:
        return list(discs)
    total = np.sum([np.nan_to_num(np.asarray(p, dtype=np.float32))
                    for p in distinct.values()], axis=0)
    _, qinfo = star_mask(total, thresh_sigma=PALETTE_STAR_THRESH_SIGMA,
                         max_npix=PALETTE_STAR_MAX_NPIX, radius_a=PALETTE_STAR_RADIUS_A,
                         max_radius=PALETTE_STAR_MAX_RADIUS_PX,
                         min_peak_flux=PALETTE_STAR_MIN_PEAK_FLUX, radius_from="peak",
                         peak_floor=PALETTE_STAR_THRESH_SIGMA * raw_rms)
    extra = qinfo.get("discs", [])
    if not extra:
        if qinfo.get("why"):
            _logger.warning("Palette gain: faint stars not found on the denoised planes: %s",
                            qinfo["why"])
        return list(discs)
    from scipy.spatial import cKDTree
    have = cKDTree(np.array([(x, y) for x, y, _ in discs]))
    d, _ = have.query(np.array([(x, y) for x, y, _ in extra]), distance_upper_bound=PALETTE_STAR_MATCH_PX)
    new = [disc for disc, dist in zip(extra, d) if not np.isfinite(dist)]
    _logger.info("Palette gain: %d faint stars from the denoised planes joined to %d",
                 len(new), len(discs))
    return list(discs) + new


def describe_palette_gain(gain) -> str:
    if gain is None:
        return ""
    if isinstance(gain, str):
        return gain
    return "/".join(f"{k}:{float(v):g}" for k, v in gain.items())
# The difference map Ha - ratio*R is noise almost everywhere, and clipping it
# at zero keeps the positive half of that noise: measured on m33 channels
# with a 1 ADU Ha noise, the "excess" was non-zero on 49% of the field and
# put a pedestal plus speckle into R and L across the whole sky. The noise
# sigma is read from the NEGATIVE half of the map (pure noise: emission only
# adds on the positive side) and the excess starts this many sigma above it.
HA_NOISE_FLOOR_SIGMA = 2.0
# Planes _prepare derives rather than stacks. They have no sky: the excess
# map is zero-floored, so a sky/noise model fitted to it is meaningless and
# its "black point" is 0 by construction. The black-point pickers skip them.
DERIVED_PLANES = frozenset({"HA_EXCESS"})


def synthetic_gain(lum: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """The brightness factor a deconvolved synthetic luminance applies, from
    the stretched deconvolved frame and the stretched frame it was made from
    (stacking/deblur.SYNTHETIC_LUM / _REF, same black, same white, same
    curve). 1 where the reference is black (no information, nothing moved),
    capped at MAX_LUM_BOOST like the LRGB substitution. A set whose two
    frames are identical composes exactly as if neither were there."""
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(ref > 1e-4, lum / np.maximum(ref, 1e-4), 1.0)
    return np.clip(np.nan_to_num(g, nan=1.0), 0.0, MAX_LUM_BOOST)

# Per-channel inspection JPEGs are capped at the preview's size; full resolution
# is what the per-channel FITS is for.
CHANNEL_JPG_MAX_PX = 2200

# What `process <dso> <recipe> auto` sweeps. Chosen from the sh2-92 HOO study in
# Iris/sh2-92/analysis: white above 99.5 guts the nebula whatever else is set,
# and softening past 0.1 costs more than it returns, so the grid brackets the
# useful range rather than exploring past its edges. 27 variants renders in
# under a minute from cached channels.
AUTO_SWEEP: dict[str, list] = {
    "black_pct": [45.0, 55.0, 65.0],
    "white_pct": [99.0, 99.5, 99.9],
    "softening": [0.025, 0.05, 0.1],
}


def _squash(name: str) -> str:
    return "".join(ch for ch in str(name).upper() if ch.isalnum() or ch == "-")


def canonical_filter(name: str) -> Optional[str]:
    """Map any spelling of a filter onto the canonical key, or None.

    `resolve_filter` answers "which of these available names is my filter";
    this answers "what filter is this frame", which is what anything reading a
    FITS FILTER card actually needs. Matching that card literally is a silent
    data-loss bug whenever the capture software's spelling changes — see the
    `O2` entry in _ALIASES.
    """
    sq = _squash(name)
    for canon, spellings in _ALIASES.items():
        if sq == _squash(canon) or sq in {_squash(a) for a in spellings}:
            return canon
    return None


def resolve_filter(wanted: str, available: list[str]) -> Optional[str]:
    """Map a recipe's filter name onto whatever this DSO's frames actually say."""
    targets = {_squash(a) for a in _ALIASES.get(wanted, ())} | {_squash(wanted)}
    for name in available:
        if _squash(name) in targets:
            return name
    return None


def _stretch(chan: np.ndarray, black_pct: float, white: float,
             softening: float = SOFTENING,
             label: Optional[str] = None) -> np.ndarray:
    """asinh stretch onto 0..1 with a per-channel black and a SHARED white.

    black_pct is a percentile of this channel, so the same share of pixels goes
    to black on any target — see BLACK_PCT.

    With a *label*, logs the ADU the percentile actually resolved to. The
    percentile is a relative instruction and lands somewhere different on every
    channel and every target; the absolute value is the only way to see whether
    it crushed one channel's faint end while leaving another's intact. The white
    point is already reported in ADU for the same reason.
    """
    black = float(np.nanpercentile(chan, black_pct))
    if label:
        _logger.info("Stretch %s: black %.2f ADU (p%.1f), white %.2f, soft %.3f",
                     label, black, black_pct, white, softening)
    y = np.clip((chan - black) / max(white - black, 1e-6), 0.0, 1.0)
    return np.arcsinh(y / softening) / np.arcsinh(1.0 / softening)


PLANE_CELLS = 16          # sky-surface fit: cells across the short axis


def sky_plane(chan: np.ndarray, cells: int = PLANE_CELLS) -> np.ndarray:
    """The smooth surface through the sky: a sigma-clipped least-squares
    quadratic (1, x, y, x^2, xy, y^2) through the medians of *cells* x
    (cells * aspect) cells. Galaxies, nebulae and bright stars are outliers
    to the fit and are clipped out of it; a sky slope (moonlight, haze, a
    light dome towards one horizon) and the residual vignetting dome are
    what it carries, out to the frame's edges. Returned at full size."""
    h, w = chan.shape
    c = max(16, min(h, w) // max(1, cells))
    ny, nx = max(3, h // c), max(3, w // c)
    sub = np.asarray(chan[: ny * c, : nx * c], dtype=np.float32).reshape(ny, c, nx, c)
    med = np.nanmedian(np.nanmedian(sub, axis=3), axis=1)
    yy, xx = np.mgrid[:ny, :nx]
    cx, cy = ((xx + 0.5) * c) / w, ((yy + 0.5) * c) / h          # 0..1, keeps the fit conditioned
    terms = lambda X, Y: np.stack([np.ones_like(X), X, Y, X * X, X * Y, Y * Y], axis=-1)
    keep = np.isfinite(med)
    coef = np.zeros(6)
    for _ in range(6):
        coef, *_ = np.linalg.lstsq(terms(cx[keep], cy[keep]), med[keep], rcond=None)
        res = med - terms(cx, cy) @ coef
        s = 1.4826 * np.nanmedian(np.abs(res[keep] - np.nanmedian(res[keep]))) + 1e-9
        new = np.isfinite(med) & (np.abs(res) < 2.5 * s)
        if new.sum() < 12 or np.array_equal(new, keep):
            break
        keep = new
    Y, X = np.mgrid[:h, :w]
    return (terms(X / w, Y / h) @ coef).astype(np.float32)


EDGE_BAND = 0.10          # edge roll-off: the outer tenth of each axis is corrected
EDGE_SMOOTH = 50          # ...with the column/row medians smoothed to 1/50 of the axis


def _edge_rolloff(resid: np.ndarray, band: float = EDGE_BAND, smooth: int = EDGE_SMOOTH) -> np.ndarray:
    """Take the sky's residual roll-off out of the outer *band* of each axis.

    The master flat falls 1.5-3 % in the outer few hundred pixels of a side,
    and the sky in a light does not always fall with it (stray light does
    not pass through the same vignetting), so after the flat division the
    sky at that edge rises by the difference -- on ngc2276 (2026-10-22) +5 ADU,
    a sigma on a 16-frame stack, in the last 2 % of the width, a feature too
    sharp for the sky surface and too close to the edge for the mesh, whose
    model is held constant in its outer half-box. This takes the clipped
    median of every column (then every row), smooths it to 1/*smooth* of the
    axis, and subtracts it under a cosine window that is 1 at the edge and 0
    one *band* in. The interior is untouched: a nebula spanning the frame
    loses nothing inside the band, and a galaxy is clipped out of the medians.
    """
    out = np.asarray(resid, dtype=np.float32).copy()
    for axis in (0, 1):
        n = out.shape[1 - axis]
        med = np.nanmedian(out, axis=axis)
        dev = out - (med if axis == 0 else med[:, None])
        sig = 1.4826 * float(np.nanmedian(np.abs(dev))) + 1e-6
        med = np.nanmedian(np.where(np.abs(dev) < 3 * sig, out, np.nan), axis=axis)
        med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
        sm = gaussian_filter1d(med, max(2.0, n / smooth), mode="nearest")
        t = np.arange(n, dtype=np.float32)
        edge = np.minimum(t, n - 1 - t) / max(1.0, band * n)             # 0 at the edge, 1 one band in
        win = np.where(edge < 1, 0.5 + 0.5 * np.cos(np.pi * np.clip(edge, 0, 1)), 0.0).astype(np.float32)
        corr = sm * win
        out -= corr if axis == 0 else corr[:, None]
    return out


def _remove_gradient(chan: np.ndarray, mesh: int = BG_MESH_FRACTION) -> np.ndarray:
    """Subtract a coarse 2-D background model — vignetting and sky gradient.

    The sky's smooth surface goes first (sky_plane: a clipped quadratic), the
    mesh model second, on what it leaves, and the edge roll-off last
    (_edge_rolloff: the flat's edge, which neither of the first two reach). The mesh alone cannot reach a slope at the frame's edges:
    its model is interpolated between the box centres and held constant in
    the outer half-box, so a 30 ADU left-to-right slope (ngc2276, a moonlit
    night, 2026-10-22) came out flat in the middle and still +/-5 ADU in the
    outer 800 px -- a sigma on a 16-frame stack, one edge clipped to black and
    the other a bright blue band under the stretch. The surface carries the
    slope and the dome to the edges; the mesh then has only the local sky to
    model, which it does as before.

    Falls back to a flat median subtraction if photutils is unavailable, which
    leaves the gradient in but never makes things worse.
    """
    try:
        from astropy.stats import SigmaClip
        from photutils.background import Background2D, SExtractorBackground
        flat = chan - sky_plane(chan)
        box = max(64, min(chan.shape) // max(1, mesh))
        bkg = Background2D(
            flat, box_size=box, filter_size=3,
            sigma_clip=SigmaClip(sigma=3.0), bkg_estimator=SExtractorBackground(),
        )
        return _edge_rolloff(flat - bkg.background)
    except Exception:
        _logger.warning("Background2D unavailable — leaving the gradient in",
                        exc_info=True)
        return chan - float(np.nanmedian(chan))


def ha_continuum_ratio(ha: np.ndarray, r: np.ndarray,
                       top_pct: float = HA_CONTINUUM_TOP_PCT,
                       min_pix: int = HA_CONTINUUM_MIN_PIX) -> Optional[float]:
    """Ha/R for continuum sources — the share of R an Ha frame also sees.

    Both inputs must be background-subtracted and on the same grid. Measured
    as the median Ha/R over the brightest R pixels (stars and galaxy body);
    HII regions push individual ratios up, but they are a minority of bright
    pixels and the median steps over them. Returns None when there is nothing
    bright enough to measure — the caller then blends the raw Ha, which is
    wrong in a visible way (red stars) rather than a silent one.
    """
    ok = np.isfinite(ha) & np.isfinite(r) & (r > 0)
    if not ok.any():
        return None
    pct = float(top_pct)
    while True:
        cut = float(np.nanpercentile(r[ok], pct))
        sel = ok & (r > cut)
        if int(sel.sum()) >= min_pix or pct <= 50.0:
            break
        pct -= 10.0
    if int(sel.sum()) < 10:
        return None
    ratio = float(np.nanmedian(ha[sel] / r[sel]))
    if not np.isfinite(ratio) or ratio <= 0:
        return None
    _logger.info("Ha continuum ratio %.4f from %d R pixels above p%.0f (%.2f ADU)",
                 ratio, int(sel.sum()), pct, cut)
    return ratio


def apply_ha_blend(subbed: dict[str, np.ndarray], gain: float = HA_GAIN,
                   ratio: Optional[float] = None
                   ) -> tuple[dict[str, np.ndarray], Optional[float]]:
    """Fold the Ha excess into R and L; returns (channels, ratio used).

    Expects background-subtracted channels with an "HA" plane. R and L are
    replaced by R + gain*excess and L + gain*excess; "HA" is left as the raw
    subtracted plane and "HA_EXCESS" is added, so the exporters can show what
    was blended. Without "HA" the channels come back untouched, and with gain
    0 only the excess map is added — HALRGB with ha=0 is exactly LRGB.
    *ratio* pins the continuum share (a movie measures it once on the full
    stack and reuses it for every depth, so the frames share one blend as
    they share one stretch); None measures it here.
    """
    if "HA" not in subbed or "R" not in subbed:
        return subbed, None
    out = dict(subbed)
    ha = out["HA"]
    if ratio is None:
        ratio = ha_continuum_ratio(ha, out["R"])
    if ratio is None:
        _logger.warning("Ha continuum ratio not measurable — blending the raw Ha "
                        "(stars will redden); check the Ha stack")
        diff = np.asarray(ha, dtype=np.float32)
    else:
        diff = (ha - ratio * out["R"]).astype(np.float32)
    # Noise floor from the negative half of the map, MAD-scaled to sigma.
    neg = diff[np.isfinite(diff) & (diff < 0)]
    sigma = float(1.4826 * np.median(-neg)) if neg.size else 0.0
    floor = HA_NOISE_FLOOR_SIGMA * sigma
    excess = np.clip(diff - floor, 0.0, None)
    out["HA_EXCESS"] = excess
    gain = float(gain)
    lit = float(np.mean(excess > 0)) * 100.0
    _logger.info("Ha excess: noise %.2f ADU, floor %.2f ADU, above it on %.1f%% "
                 "of pixels, p99.9 %.2f ADU", sigma, floor, lit,
                 float(np.nanpercentile(excess, 99.9)))
    if gain != 0.0:
        out["R"] = out["R"] + gain * excess
        if "L" in out:
            out["L"] = out["L"] + gain * excess
        _logger.info("Ha blend: gain %.2f -> R%s", gain, " and L" if "L" in out else "")
    return out, ratio


def star_white_balance(subbed: dict[str, np.ndarray],
                       max_stars: int = STAR_WB_MAX_STARS,
                       min_stars: int = STAR_WB_MIN_STARS) -> tuple[dict, dict]:
    """({R, G, B: factor}, info): the scaling that makes the median star neutral.

    Stars are found on L (or G) with sep, kept if compact, round, unsaturated
    and away from the edge, and the brightest *max_stars* are measured in R, G
    and B through one aperture with a local annulus. The factors are the
    inverse of the median R/G and B/G. Fewer than *min_stars* usable stars
    means no change at all -- an identity with a reason -- never a guess.
    """
    identity = {"R": 1.0, "G": 1.0, "B": 1.0}
    if not all(c in subbed for c in ("R", "G", "B")):
        return identity, {"why": "needs R, G and B"}
    try:
        import sep
    except ImportError:
        return identity, {"why": "sep not installed"}
    ref_name = "L" if "L" in subbed else "G"
    ref = np.ascontiguousarray(np.nan_to_num(subbed[ref_name]), dtype=np.float32)
    try:
        bkg = sep.Background(ref)
        data = ref - bkg.back()
        rms = float(bkg.globalrms) or 1.0
        objs = sep.extract(data, STAR_WB_THRESH_SIGMA, err=rms, minarea=7)
    except Exception as exc:  # noqa: BLE001
        return identity, {"why": "star detection failed: %s" % exc}
    h, w = ref.shape
    e = STAR_WB_EDGE_PX
    keep = ((objs["b"] / np.maximum(objs["a"], 1e-6)) > 0.6)
    keep &= (objs["npix"] >= 7) & (objs["npix"] <= STAR_WB_MAX_NPIX)
    keep &= (objs["x"] > e) & (objs["x"] < w - e) & (objs["y"] > e) & (objs["y"] < h - e)
    keep &= objs["peak"] < float(np.nanpercentile(ref, 99.995))     # saturated cores
    objs = objs[keep]
    if len(objs) < min_stars:
        return identity, {"why": "only %d usable stars (need %d)" % (len(objs), min_stars),
                          "stars": int(len(objs))}
    objs = objs[np.argsort(objs["flux"])[::-1][:max_stars]]
    # Aperture from the stars themselves, so this works at any binning.
    a_med = float(np.median(objs["a"]))
    r_ap = float(np.clip(4.0 * a_med, 4.0, 16.0))
    ann = (r_ap * 1.8, r_ap * 3.0)
    flux = {}
    for c in ("R", "G", "B"):
        plane = np.ascontiguousarray(np.nan_to_num(subbed[c]), dtype=np.float32)
        f, _err, _flag = sep.sum_circle(plane, objs["x"], objs["y"], r_ap, bkgann=ann, subpix=5)
        flux[c] = np.asarray(f, dtype=np.float64)
    good = (flux["G"] > 0) & (flux["R"] > 0) & (flux["B"] > 0)
    if int(good.sum()) < min_stars:
        return identity, {"why": "only %d stars with positive flux in all three channels"
                                 % int(good.sum()), "stars": int(good.sum())}
    rr = flux["R"][good] / flux["G"][good]
    bb = flux["B"][good] / flux["G"][good]
    med_r, med_b = float(np.median(rr)), float(np.median(bb))
    info = {"stars": int(good.sum()), "reference": ref_name,
            "aperture_px": round(r_ap, 1),
            "median_R_over_G": round(med_r, 4), "median_B_over_G": round(med_b, 4),
            "mad_R": round(float(np.median(np.abs(rr - med_r)) / med_r), 4),
            "mad_B": round(float(np.median(np.abs(bb - med_b)) / med_b), 4)}
    return {"R": 1.0 / med_r, "G": 1.0, "B": 1.0 / med_b}, info


def apply_white_balance(subbed: dict[str, np.ndarray], mode,
                        factors: Optional[dict] = None) -> tuple[dict, dict]:
    """Scale the colour channels; *factors* applies given ones instead of
    measuring. See _prepare for why a series must share one set."""
    if factors:
        out = dict(subbed)
        for c, f in factors.items():
            if c in out and f != 1.0:
                out[c] = out[c] * float(f)
        return out, {"mode": "given", "factors": dict(factors)}
    return _measure_white_balance(subbed, mode)


def _measure_white_balance(subbed: dict[str, np.ndarray], mode) -> tuple[dict, dict]:
    """(channels, info) with R and B scaled per *mode*; "none" returns the input."""
    if not mode or str(mode).lower() in ("none", "off", "0", "false"):
        return subbed, {"mode": "none", "factors": {"R": 1.0, "G": 1.0, "B": 1.0}}
    if str(mode).lower() != "stars":
        raise ValueError("wb must be 'none' or 'stars', not %r" % (mode,))
    factors, info = star_white_balance(subbed)
    out = dict(subbed)
    for c in ("R", "B"):
        if c in out and factors[c] != 1.0:
            out[c] = out[c] * np.float32(factors[c])
    if info.get("why"):
        _logger.warning("White balance: left alone -- %s", info["why"])
    else:
        _logger.info("White balance from %d stars on %s: R x%.3f  B x%.3f "
                     "(median R/G %.3f, B/G %.3f; scatter %.1f%% / %.1f%%)",
                     info["stars"], info["reference"], factors["R"], factors["B"],
                     info["median_R_over_G"], info["median_B_over_G"],
                     100 * info["mad_R"], 100 * info["mad_B"])
    return out, {"mode": "stars", "factors": factors, **info}


# --- RGB stars on a narrowband palette ---------------------------------------
#
# A narrowband palette gives every star the palette's colour: in HSO a G2
# star is whatever Ha:S-II:O-III its continuum happens to be, which is not a
# colour any star has. `stars=rgb` keeps the palette's brightness inside a
# star mask and takes the hue there from an RGB stack of the same field —
# the LRGB rule (luminance from one image, chroma from another) applied only
# where the mask is on. Outside the mask the palette is untouched, so the
# nebula never sees the RGB data.
#
# The mask is built from stars detected on the reference stack, kept only if
# compact and round (a nebula knot or a galaxy core fails the size cut and
# stays palette), each drawn as a flat disc a few times its own size with a
# Gaussian skirt so the hue change has no edge.
STAR_MASK_THRESH_SIGMA = 5.0
STAR_MASK_MIN_AXIS_RATIO = 0.5
STAR_MASK_MAX_NPIX = 2500       # bigger is a knot or a core, not a star
STAR_MASK_RADIUS_A = 3.0        # disc radius in units of sep's semi-major axis a
STAR_MASK_MIN_RADIUS_PX = 3.0
STAR_MASK_MAX_RADIUS_PX = 60.0
STAR_MASK_FEATHER = 0.6         # skirt sigma as a fraction of the disc radius
STAR_PEAK_R0 = 5.0              # radius_from="peak": px at the detection threshold ...
STAR_PEAK_R_PER_DEX = 12.0      # ... plus this per decade of peak above it ...
# ... plus a halo term for the bright ones: the halo grows faster than the
# log. Melotte 15 on the denoised sum (1 ADU isophote, sky and spikes out):
# 108 px at 4.2 dex above the floor, 84-116 at 4.0, 80 at 3.8, 70 at 3.5,
# 52-68 at 3.2; 5 + 12 d gives 56 at 4.2 and the halo outside the disc took
# the palette gain (magenta). 30 (d - 2.5)^2 beyond 2.5 dex: 146 at 4.2,
# 120 at 4.0, 101 at 3.8, 77 at 3.5, 58 at 3.2.
STAR_PEAK_HALO_FROM_DEX = 2.5
STAR_PEAK_HALO_PER_DEX2 = 30.0
# Diffraction spikes, same stars: four arms on two axes 90 deg apart, the
# axes' angle measured on the field (spike_axis: the spider's orientation
# on the sensor, 67 deg on Melotte 15, could change with a camera
# rotation), length 40 + 100 (d - 2.5)^2 px (Melotte 15 at 1 ADU: 200-400 px
# at 4.0-4.2 dex, 150-370 at 3.8, 90-180 at 3.4, 50-100 at 3.1), half-width
# 4 px plus a 3 px feather. Outside the mask a spike took the gain and was
# red ("difraction spike are red", 2026-10-28).
STAR_SPIKE_FROM_DEX = 2.5
STAR_SPIKE_L0 = 40.0
STAR_SPIKE_L_PER_DEX2 = 100.0
STAR_SPIKE_MAX_LEN_PX = 400.0
STAR_SPIKE_HALF_WIDTH_PX = 4.0
STAR_SPIKE_FEATHER_PX = 3.0
STAR_SPIKE_MIN_CONTRAST = 3.0   # folded azimuthal profile peak / median, or the field has no spikes
STAR_CHROMA_FLOOR = 0.04        # RGB luminance (0..1) below which the palette keeps its colour


def star_mask(chan: np.ndarray, thresh_sigma: float = STAR_MASK_THRESH_SIGMA,
              max_npix: int = STAR_MASK_MAX_NPIX,
              radius_a: float = STAR_MASK_RADIUS_A,
              max_radius: float = STAR_MASK_MAX_RADIUS_PX,
              feather: float = STAR_MASK_FEATHER,
              min_peak_flux: float = 0.0,
              radius_from: str = "isophote",
              peak_floor: Optional[float] = None,
              feather_max: Optional[float] = None) -> tuple[np.ndarray, dict]:
    """(mask 0..1, info) — feathered discs on the stars of *chan*.

    *chan* is a linear stack (background-subtracted or not; sep models the
    background itself). Sources are kept if round, not too large and not on
    the edge; the disc radius follows sep's isophotal size, so a bright star
    with a wide profile gets a wider disc than a faint one. *min_peak_flux*
    is a compactness cut, peak over isophotal flux: a star is a point, so
    its peak is a large share of its flux (IC 410 at 3 sigma: 0.011 for the
    brightest, halo included, 0.018 median), an HII knot or a nebula blob
    is not (0.0004-0.002). 0 turns it off. *radius_from* "peak" sizes the
    disc from the star's peak instead of its isophote: STAR_PEAK_R0 px at
    the detection threshold plus STAR_PEAK_R_PER_DEX per decade above it
    (IC 410: a 1756 ADU star's halo reaches 30-45 px, 5 + 12 * 2.5 = 35), so
    a star on bright nebula, whose isophote is the nebula's, gets the disc
    its brightness earns and not one the size of the knot it sits on.
    *peak_floor* (ADU) is the floor those radii are counted from when it is
    not this image's own threshold: a denoised plane's noise is a tenth of
    the raw's, so counted from its threshold every star is a decade
    brighter and the discs cover the field (Melotte 15: 40 %).
    """
    import sep
    a = np.ascontiguousarray(np.nan_to_num(chan), dtype=np.float32)
    h, w = a.shape
    mask = np.zeros((h, w), np.float32)
    # sep's default 300k-pixel stack fills on a quiet (denoised) 60 Mpx frame
    sep.set_extract_pixstack(max(300_000, h * w // 20))
    try:
        bkg = sep.Background(a)
        data = a - bkg.back()
        rms = float(bkg.globalrms) or 1.0
        objs = sep.extract(data, thresh_sigma, err=rms, minarea=5)
    except Exception as exc:  # noqa: BLE001
        return mask, {"stars": 0, "why": "star detection failed: %s" % exc}
    keep = (objs["b"] / np.maximum(objs["a"], 1e-6)) >= STAR_MASK_MIN_AXIS_RATIO
    keep &= objs["npix"] <= max_npix
    if min_peak_flux > 0:
        keep &= (objs["peak"] / np.maximum(objs["flux"], 1e-6)) >= min_peak_flux
    objs = objs[keep]
    if radius_from == "peak":
        floor = max(peak_floor if peak_floor is not None else thresh_sigma * rms, 1e-6)
        discs = [(float(o["x"]), float(o["y"]),
                  float(np.clip(radius_from_dex(np.log10(max(float(o["peak"]) / floor, 1.0))),
                                STAR_MASK_MIN_RADIUS_PX, max_radius)))
                 for o in objs]
    else:
        discs = [(float(o["x"]), float(o["y"]),
                  float(np.clip(radius_a * float(o["a"]), STAR_MASK_MIN_RADIUS_PX, max_radius)))
                 for o in objs]
    n = paint_discs(mask, discs, feather, feather_max)
    return mask, {"stars": n, "detected": int(len(keep)), "kept": int(keep.sum()),
                  "coverage": float(mask.mean()), "discs": discs, "rms": rms}


def radius_from_dex(d: float) -> float:
    """Disc radius (px, uncapped) of a star *d* decades above the detection
    floor: STAR_PEAK_R0 + STAR_PEAK_R_PER_DEX d, plus the halo term."""
    return (STAR_PEAK_R0 + STAR_PEAK_R_PER_DEX * d
            + STAR_PEAK_HALO_PER_DEX2 * max(d - STAR_PEAK_HALO_FROM_DEX, 0.0) ** 2)


def dex_from_radius(r: float) -> float:
    """radius_from_dex inverted (it is monotonic); a capped radius reads as
    the cap's dex, which is the brightest the formula knows."""
    d = (r - STAR_PEAK_R0) / STAR_PEAK_R_PER_DEX
    if d <= STAR_PEAK_HALO_FROM_DEX:
        return max(d, 0.0)
    lo, hi = STAR_PEAK_HALO_FROM_DEX, STAR_PEAK_HALO_FROM_DEX + 10.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if radius_from_dex(mid) < r:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def spike_axis(chan: np.ndarray, discs, n: int = 4) -> Optional[float]:
    """The diffraction spikes' axis angle (deg, 0..90, from +x towards +y on
    the stored grid) measured on the *n* biggest *discs* of *chan*: the
    azimuthal median profile of an annulus 0.5-1.5 r around each, folded modulo
    90 (four arms on two axes), summed over the stars. None when the folded
    profile's peak is under STAR_SPIKE_MIN_CONTRAST times its median: no
    spikes, or none the mask needs."""
    if not discs:
        return None
    big = sorted(discs, key=lambda t: -t[2])[:n]
    h, w = chan.shape
    prof = np.zeros(180, np.float64)
    for cx, cy, r in big:
        # inside the halo, where the spikes stand above it; outside the disc the
        # nebula wins (Melotte 15, 1.5-4 r: no axis; 0.5-1.5 r: 67 deg, clear)
        r_in, r_out = 0.5 * r, max(1.5 * r, 0.5 * r + 40.0)
        R = int(np.ceil(r_out))
        x0, x1 = max(0, int(cx) - R), min(w, int(cx) + R + 1)
        y0, y1 = max(0, int(cy) - R), min(h, int(cy) + R + 1)
        if x1 - x0 < 10 or y1 - y0 < 10:
            continue
        sub = np.nan_to_num(np.asarray(chan[y0:y1, x0:x1], dtype=np.float32))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        rr = np.hypot(yy - cy, xx - cx)
        ann = (rr >= r_in) & (rr < r_out)
        if ann.sum() < 100:
            continue
        sky = float(np.median(sub[ann]))
        th = (np.degrees(np.arctan2(yy - cy, xx - cx)) % 180.0).astype(int)
        vals = sub[ann] - sky
        bins = th[ann]
        p = np.zeros(180, np.float64)
        for a in range(180):
            v = vals[bins == a]
            if v.size:
                p[a] = np.median(v)
        scale = float(np.max(np.abs(p))) or 1.0
        prof += p / scale
    folded = prof[:90] + prof[90:]
    if not np.any(folded):
        return None
    med = float(np.median(folded))
    pk = int(np.argmax(folded))
    spread = float(np.std(folded)) or 1.0
    if (folded[pk] - med) / spread < STAR_SPIKE_MIN_CONTRAST:
        return None
    return float(pk)


def spike_arms(discs, axis_deg: Optional[float]) -> list:
    """(x, y, axis_deg, length) for every disc bright enough to spike
    (STAR_SPIKE_FROM_DEX, dex read back from the disc radius); empty with no axis."""
    if axis_deg is None:
        return []
    arms = []
    for x, y, r in discs:
        d = dex_from_radius(r)
        if d < STAR_SPIKE_FROM_DEX:
            continue
        length = min(STAR_SPIKE_L0 + STAR_SPIKE_L_PER_DEX2 * (d - STAR_SPIKE_FROM_DEX) ** 2,
                     STAR_SPIKE_MAX_LEN_PX)
        arms.append((float(x), float(y), float(axis_deg), float(length)))
    return arms


def paint_arms(mask: np.ndarray, arms, half_width: float = STAR_SPIKE_HALF_WIDTH_PX,
               feather: float = STAR_SPIKE_FEATHER_PX) -> int:
    """Four feathered arms per (x, y, axis_deg, length) painted into *mask*
    in place: two full-length bars through the centre, on the axis and at
    90 deg to it. The count painted."""
    h, w = mask.shape
    n = 0
    for cx, cy, ang, length in arms:
        reach = int(np.ceil(length + half_width + 3.0 * feather))
        x0, x1 = max(0, int(cx) - reach), min(w, int(cx) + reach + 1)
        y0, y1 = max(0, int(cy) - reach), min(h, int(cy) + reach + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dy, dx = (yy - cy).astype(np.float32), (xx - cx).astype(np.float32)
        a = np.radians(ang)
        ux, uy = np.cos(a), np.sin(a)
        bar = np.zeros((y1 - y0, x1 - x0), np.float32)
        for ex, ey in ((ux, uy), (-uy, ux)):
            t = dx * ex + dy * ey                       # along the bar
            pdist = np.abs(dx * -ey + dy * ex)          # across it
            across = np.where(pdist <= half_width, 1.0,
                              np.exp(-0.5 * ((pdist - half_width) / feather) ** 2))
            along = np.where(np.abs(t) <= length, 1.0,
                             np.exp(-0.5 * ((np.abs(t) - length) / feather) ** 2))
            np.maximum(bar, (across * along).astype(np.float32), out=bar)
        np.maximum(mask[y0:y1, x0:x1], bar, out=mask[y0:y1, x0:x1])
        n += 1
    return n


def paint_discs(mask: np.ndarray, discs, feather: float = STAR_MASK_FEATHER,
                feather_max: Optional[float] = None) -> int:
    """Feathered discs (x, y, r) painted into *mask* in place; the count.
    The skirt's sigma is *feather* r, at most *feather_max* px if given."""
    h, w = mask.shape
    n = 0
    for cx, cy, r in discs:
        sig = feather * r
        if feather_max is not None:
            sig = min(sig, feather_max)
        reach = int(np.ceil(r + 3.0 * sig))
        x0, x1 = max(0, int(cx) - reach), min(w, int(cx) + reach + 1)
        y0, y1 = max(0, int(cy) - reach), min(h, int(cy) + reach + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.hypot(yy - cy, xx - cx)
        disc = np.where(d <= r, 1.0, np.exp(-0.5 * ((d - r) / sig) ** 2)).astype(np.float32)
        np.maximum(mask[y0:y1, x0:x1], disc, out=mask[y0:y1, x0:x1])
        n += 1
    return n


def mask_from_discs(shape, discs, feather: float = STAR_MASK_FEATHER, arms=None,
                    feather_max: Optional[float] = None) -> np.ndarray:
    """The star mask a list of (x, y, r) discs describes, on a grid of
    *shape*, with the spike *arms* (spike_arms) if given."""
    mask = np.zeros(shape, np.float32)
    paint_discs(mask, discs, feather, feather_max)
    if arms:
        paint_arms(mask, arms)
    return mask


def rgb_stars(palette: np.ndarray, stars: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Palette brightness, RGB hue, inside *mask*; the palette elsewhere.

    Both images are stretched 0..1 on the same grid. Inside the mask the
    output pixel is the palette's luminance times the RGB image's chroma
    (its colour with the brightness divided out), so the star keeps the
    profile the narrowband stack gave it and takes the colour the RGB stack
    measured. A pixel where the RGB image is black has no chroma and keeps
    the palette.
    """
    lw = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lum_p = palette @ lw
    lum_s = stars @ lw
    with np.errstate(divide="ignore", invalid="ignore"):
        chroma = np.where(lum_s[:, :, None] > 1e-4, stars / np.maximum(lum_s, 1e-4)[:, :, None], 1.0)
    recol = np.clip(lum_p[:, :, None] * chroma, 0.0, 1.0)
    # Only star light sets a hue. In a disc's feathered skirt the RGB image
    # is sky, whose "colour" is per-channel offset noise; on the first M33
    # dry run that painted every skirt pink. The gate ramps in over the
    # first few percent of RGB brightness so the hue arrives with the star.
    # The floor is the larger of a fixed few percent and 5x the star image's
    # own sky noise (robust sigma of its luminance), so a noisy RGB stack
    # does not hand its grain a colour inside the discs.
    med = float(np.nanmedian(lum_s))
    sig = 1.4826 * float(np.nanmedian(np.abs(lum_s - med)))
    floor = max(STAR_CHROMA_FLOOR, med + 5.0 * sig)
    # zero at the floor, full at twice it: nothing at noise level gets a hue
    conf = np.clip((lum_s - floor) / max(floor, 1e-6), 0.0, 1.0)
    m = (np.clip(mask, 0.0, 1.0) * conf)[:, :, None]
    return np.clip(palette * (1.0 - m) + recol * m, 0.0, 1.0)


def stars_image(rgb_stacks: dict[str, np.ndarray], **compose_kw) -> np.ndarray:
    """The RGB image the star colours come from.

    R, G and B background-subtracted, star white balance on so the median
    star is neutral, and black at each channel's sky *exactly* — not a
    percentile — so sky is zero in all three and carries no colour into the
    skirts. Stretched with the palette's white percentile and softening.
    """
    opts = effective_options(**{k: v for k, v in compose_kw.items() if k in effective_options()})
    # deblur is not applied here: these planes only lend the stars a hue.
    subbed, white = _prepare({c: rgb_stacks[c] for c in ("R", "G", "B")},
                             opts["subtract_background"], opts["mesh"], opts["white_pct"],
                             ha_gain=None, white_balance="stars")
    blacks = {c: float(np.nanmedian(subbed[c])) for c in subbed}
    return compose_prepared(subbed, blacks, white, softening=opts["softening"], scnr=0.0)


def shared_white(subbed: dict[str, np.ndarray], white_pct: float) -> float:
    """The one white point every channel is stretched against: a percentile
    of the brightest primary at each pixel. Separate so a caller that blends
    after _prepare (the movie) can re-derive it from the blended planes."""
    colour = [subbed[c] for c in ("R", "G", "B") if c in subbed]
    return float(np.nanpercentile(np.maximum.reduce(colour), white_pct))


def _prepare(channels: dict[str, np.ndarray], subtract_background: bool,
             mesh: int, white_pct: float,
             ha_gain: Optional[float] = HA_GAIN,
             white_balance: str = WHITE_BALANCE,
             wb_factors: Optional[dict] = None,
             ha_ratio: Optional[float] = None,
             info: Optional[dict] = None,
             palette_gain=PALETTE_GAIN, palette_stars=None,
             palette_quiet: Optional[dict] = None, palette_arms=None) -> tuple[dict, float]:
    """Background-subtract every channel and find the shared white point.

    Factored out of compose() because the per-channel exports have to be the
    exact same pixels on the exact same scale — a channel JPEG rendered against
    its own white point would look nothing like its contribution to the
    composite, which defeats the point of being able to inspect it.

    *info*, if given, receives what was measured here: "wb" (the white-balance
    info, factors included), "ha_ratio" and "palette_gain" (the {role: factor}
    applied, measured when gain=hubble). A raw/denoised pair passes the raw
    set's back in as wb_factors / ha_ratio / palette_gain so both share one
    balance, blend and gain. *palette_quiet* is the denoised set, when the
    caller has it, for the gain's faint-star search (join_quiet_stars).
    """
    if subtract_background:
        subbed = {k: _remove_gradient(v, mesh) for k, v in channels.items()}
    else:
        subbed = {k: v - float(np.nanmedian(v)) for k, v in channels.items()}
    if "LSYN" in channels and "LSYN_REF" in channels:
        # The deconvolved synthetic luminance takes its REFERENCE's background,
        # not its own: a background model fitted to the sharpened frame differs
        # from one fitted to the unsharpened frame wherever the network moved
        # flux, and that difference would ride into the gain as brightness the
        # network never produced. Same background, same black: the ratio is
        # the deconvolution and nothing else.
        bg = channels["LSYN_REF"] - subbed["LSYN_REF"]
        subbed["LSYN"] = channels["LSYN"] - bg
    # Balance BEFORE the Ha blend: the excess is measured against R and added
    # to R, so scaling R first changes the ratio and the subtraction together
    # and the excess itself is unchanged; balancing afterwards would rescale it.
    # The white balance is MEASURED FROM THE DATA, so it is one more thing a
    # series has to share rather than recompute. Pass wb_factors to pin it.
    # Found 2026-10-09: a raw/denoised pair rendered with wb=stars measured
    # the balance separately on each, and the denoiser moved the stars' colours
    # enough to swing the factors by a third (ngc7320 R 1.189 -> 0.822, B 1.323
    # -> 1.769). The denoised frame came out strongly blue and it read exactly
    # like the model wrecking the colour. It was the renderer. On ngc2146 it
    # was worse: star_white_balance found too few usable stars on the denoised
    # frame, returned the identity, and that frame got no balance at all while
    # its raw counterpart got R 1.176 / B 1.403. Same class of error the black
    # point comment above describes, one layer up.
    subbed, wb_info = apply_white_balance(subbed, white_balance, wb_factors)
    # Palette gain (PALETTE_GAIN): after the balance, which it never shares a
    # recipe with. The white point is taken BEFORE the gain: the p99.95 white
    # is the stars, and a filter scaled up carries its stars up with it, so a
    # white taken after would rise with the gain and darken the whole picture
    # (IC 410 at gain=hubble: white 124 -> 266 ADU, the nebula at half its
    # brightness). The gain is a tone, not an exposure; stars in a scaled-up
    # plane clip harder, which they did anyway.
    white_before = shared_white(subbed, white_pct)
    gain_info: dict = {}
    subbed, gain_used = apply_palette_gain(subbed, palette_gain, star_discs=palette_stars,
                                           info=gain_info, quiet=palette_quiet,
                                           star_arms=palette_arms)
    if gain_used:
        _logger.info("Palette gain: %s", describe_palette_gain(gain_used))
    # HALRGB: the blend happens here, on subtracted linear data, so the
    # shared white point and every per-channel export see the blended R.
    # ha_gain=None leaves it to the caller (the movie pins one ratio for
    # every depth); 0 adds only the excess map, for inspection.
    ratio = None
    if ha_gain is not None:
        subbed, ratio = apply_ha_blend(subbed, ha_gain, ha_ratio)
    if info is not None:
        info.update({"wb": wb_info, "ha_ratio": ratio, "palette_gain": gain_used,
                     "palette_stars": gain_info.get("star_discs"),
                     "palette_arms": gain_info.get("star_arms")})
    return subbed, (white_before if gain_used else shared_white(subbed, white_pct))



# --- auto stretch -----------------------------------------------------------
#
# The by-hand NGC 7380 HOO stretch of 2026-09-12 came down to three decisions,
# two of which were measurements: (1) a channel with faint signal just above
# its sky needs black placed a margin BELOW sky or that signal is clipped, (2) a
# channel with nothing near its sky must have black AT sky or the harder curve
# lifts its noise into a colour cast (O-III made the whole background teal),
# (3) the softening sets how hard the faint end is pushed. auto_stretch()
# measures (1) and (2) from each channel against a pure-noise sky model and
# turns (3) into a brightness target for the faint end.
#
# The measurement is made on a BINNED image, not per pixel. Faint nebulosity
# sits at a fraction of a sigma per pixel — invisible in a per-pixel histogram,
# where the first version of this rule measured 0.0% faint signal on IC 1396,
# whose whole field is 1-5 sigma emission — but a viewer sees the picture at
# thumbnail scale, where 16x16 pixels average and that fraction of a sigma
# becomes several. So the field is block-averaged AUTO_BIN on a side, and
# "faint" means what stands above the noise of that image.
#
# Noise model: sky pixels are symmetric about the sky level and signal only adds
# on the positive side, so the negative half of the histogram is pure noise. Its
# MAD gives sigma without nebulosity inflating it, and its count, doubled, is
# the number of sky pixels; the excess over that model on the positive side is
# the fraction of the field carrying signal.
AUTO_BIN = 16                  # block size the field is averaged over
AUTO_FAINT_BAND = (1.0, 4.0)   # binned sigma above sky that "faint signal" is
AUTO_BRIGHT_SIGMA = 4.0        # ...and above which it is "bright"
AUTO_MARGIN_SIGMA = 0.5        # full black margin below sky, in PER-PIXEL sigma
AUTO_MARGIN_GATE = 0.03        # faint fraction below which margin is 0
AUTO_MARGIN_FULL = 0.25        # ...and above which it is the full margin
AUTO_FAINT_TARGET = 0.15       # output (of white) for sky + 2 binned sigma
AUTO_BRIGHT_CAP = 0.90         # p90 of the bright signal may not exceed this
AUTO_SOFT_RANGE = (0.002, 0.2)


def _asinh_out(y, soft: float):
    return np.arcsinh(y / soft) / np.arcsinh(1.0 / soft)


def _solve_soft(y: float, target: float, lo: float, hi: float) -> float:
    """Softening at which the asinh curve maps y to target; monotone, bisect.

    Output falls as softening rises, so if even the hardest curve is too dim
    the answer is lo, and if the softest is still too bright it is hi.
    """
    if _asinh_out(y, lo) <= target:
        return lo
    if _asinh_out(y, hi) >= target:
        return hi
    for _ in range(60):
        mid = (lo * hi) ** 0.5
        if _asinh_out(y, mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo * hi) ** 0.5


def _block_mean(a: np.ndarray, b: int) -> np.ndarray:
    h, w = (a.shape[0] // b) * b, (a.shape[1] // b) * b
    return np.nanmean(a[:h, :w].reshape(h // b, b, w // b, b), axis=(1, 3))


AUTO_SKY_TILES = 8             # tiles per side for the clean-sky search
AUTO_SKY_QUARTILE = 0.25       # sky = median of the darkest quarter of tile medians
AUTO_CLEAN_MIN = 0.15          # fewer tiles than this at sky = "no clean sky"


def clean_sky(chan: np.ndarray, bin_factor: int = AUTO_BIN) -> dict:
    """Sky level from the darkest part of the field, and how much of the field is sky.

    sep's global background is the level of the whole field, which on a target
    that fills it (IC 1396, squid, NGC 5907: auto_stretch's three failures,
    652f8c9) is nebula, not sky, so black landed at the binned median. Here
    the binned field is cut into AUTO_SKY_TILES^2 tiles and sky is the median
    of the darkest quarter of tile medians. "clean_fraction" is the share of
    tiles whose median sits within the noise of that sky; below
    AUTO_CLEAN_MIN the answer is flagged rather than trusted, so an automatic
    publish can refuse instead of guessing.
    """
    a = np.ascontiguousarray(chan.astype(np.float32))
    binned = _block_mean(a, bin_factor)
    h, w = binned.shape
    t = AUTO_SKY_TILES
    th, tw = max(1, h // t), max(1, w // t)
    meds = np.array([np.nanmedian(binned[i * th:(i + 1) * th, j * tw:(j + 1) * tw])
                     for i in range(t) for j in range(t)
                     if binned[i * th:(i + 1) * th, j * tw:(j + 1) * tw].size])
    meds = meds[np.isfinite(meds)]
    k = max(1, int(round(AUTO_SKY_QUARTILE * meds.size)))
    sky = float(np.median(np.sort(meds)[:k]))
    # The tolerance for "this tile is sky" comes from the NOISE, measured on
    # neighbouring-pixel differences, which structure does not inflate. A sigma
    # from the histogram around the darkest tiles includes the nebula's own
    # ripples on a field it fills, and then every tile passes as sky.
    sub = a[::2, ::2]
    dif = np.diff(sub, axis=1).ravel()
    dif = dif[np.isfinite(dif)]
    sigma_px = max(1.4826 * float(np.median(np.abs(dif - np.median(dif)))) / np.sqrt(2.0), 1e-6) \
        if dif.size else 1e-6
    tile_px = th * tw * bin_factor * bin_factor
    tol = 3.0 * sigma_px / np.sqrt(max(tile_px, 1)) + 1.0 * sigma_px / bin_factor
    clean = float(np.mean(meds <= sky + tol)) if meds.size else 0.0
    return {"sky": sky, "sigma_px_noise": sigma_px, "clean_fraction": clean,
            "confident": clean >= AUTO_CLEAN_MIN}


def sky_noise_model(chan: np.ndarray, bin_factor: int = AUTO_BIN,
                    sky: Optional[float] = None) -> dict:
    """Sky level, per-pixel and binned noise, and the faint/bright signal
    fractions of one background-subtracted channel. *sky* overrides sep's
    whole-field estimate (clean_sky passes its darkest-tiles level)."""
    import sep
    from math import erf, sqrt

    def phi(v):
        return 0.5 * (1.0 + erf(v / sqrt(2.0)))

    a = np.ascontiguousarray(chan.astype(np.float32))
    binned = np.ascontiguousarray(_block_mean(a, bin_factor).astype(np.float32))
    if sky is None:
        box = max(8, min(64, min(binned.shape) // 8))
        sky = float(sep.Background(binned, bw=box, bh=box).globalback)

    # Per-pixel sigma, negative side, on a strided subsample of the full field.
    px = a[::4, ::4].ravel()
    px = px[np.isfinite(px)] - sky
    neg = -px[px < 0]
    sigma_px = max(1.4826 * float(np.median(neg)) if neg.size else float(np.std(px)), 1e-6)

    zb = binned.ravel()
    zb = zb[np.isfinite(zb)] - sky
    negb = -zb[zb < 0]
    sigma_b = max(1.4826 * float(np.median(negb)) if negb.size else float(np.std(zb)), 1e-6)
    zb = zb / sigma_b
    n_total = zb.size
    n_sky = 2 * negb.size
    lo, hi = AUTO_FAINT_BAND
    faint = max(0.0, int(np.count_nonzero((zb >= lo) & (zb < hi))) - n_sky * (phi(hi) - phi(lo))) / n_total
    bright_px = zb[zb >= AUTO_BRIGHT_SIGMA]
    bright = max(0.0, bright_px.size - n_sky * (1.0 - phi(AUTO_BRIGHT_SIGMA))) / n_total
    bright_p90 = float(np.percentile(bright_px, 90)) * sigma_b + sky if bright_px.size else None
    return {"sky": sky, "sigma": sigma_px, "sigma_binned": sigma_b,
            "faint_fraction": faint, "bright_fraction": bright, "bright_p90": bright_p90}


def auto_stretch(subbed: dict[str, np.ndarray], white: float,
                 lum: Optional[str] = None, log=None) -> dict:
    """Per-channel black points and one softening, chosen from the data.

    Returns {"blacks": {chan: ADU}, "softening": float, "channels": {chan:
    diagnostics}, "dominant": chan, "anchors": {...}}. The caller applies the
    same values to the raw and the denoised set, exactly as it would with
    hand-chosen ones.
    """
    diag = {c: sky_noise_model(a) for c, a in subbed.items()
            if c not in DERIVED_PLANES}
    blacks = {}
    for c, d in diag.items():
        t = (d["faint_fraction"] - AUTO_MARGIN_GATE) / (AUTO_MARGIN_FULL - AUTO_MARGIN_GATE)
        k = AUTO_MARGIN_SIGMA * float(np.clip(t, 0.0, 1.0))
        d["margin_sigma"] = k
        blacks[c] = d["sky"] - k * d["sigma"]
        d["black"] = blacks[c]

    # The softening is one number for every channel, so it is set from the
    # channel that carries the picture: the luminance when there is one, else
    # the channel with the most signal.
    if lum and lum in diag:
        dom = lum
    else:
        dom = max(diag, key=lambda c: diag[c]["faint_fraction"] + diag[c]["bright_fraction"])
    d = diag[dom]
    span = max(white - blacks[dom], 1e-6)
    faint_adu = d["sky"] + 2.0 * d["sigma_binned"]
    y_faint = (faint_adu - blacks[dom]) / span
    lo, hi = AUTO_SOFT_RANGE
    soft = _solve_soft(y_faint, AUTO_FAINT_TARGET, lo, hi)
    anchors = {"faint_adu": faint_adu, "soft_from_faint": soft}
    if d["bright_p90"] is not None and d["bright_fraction"] > 1e-3:
        y_bright = min((d["bright_p90"] - blacks[dom]) / span, 1.0)
        if _asinh_out(y_bright, soft) > AUTO_BRIGHT_CAP:
            soft = max(soft, _solve_soft(y_bright, AUTO_BRIGHT_CAP, lo, hi))
        anchors.update({"bright_adu": d["bright_p90"],
                        "bright_out": float(_asinh_out(y_bright, soft))})
    anchors["faint_out"] = float(_asinh_out(y_faint, soft))
    if log:
        for c in sorted(diag):
            x = diag[c]
            log(f"    {c}: sky {x['sky']:+.3f} sigma {x['sigma']:.3f} (binned {x['sigma_binned']:.3f}) "
                f"faint {100*x['faint_fraction']:.1f}% bright {100*x['bright_fraction']:.1f}% -> "
                f"margin {x['margin_sigma']:.2f} sigma, black {x['black']:+.3f} ADU")
        b = anchors.get("bright_out")
        log(f"    softening {soft:.4f} from {dom}: sky+2 binned sigma ({faint_adu:.2f} ADU) -> "
            f"{anchors['faint_out']:.2f} of white" + (f", bright p90 -> {b:.2f}" if b is not None else ""))
    return {"blacks": blacks, "softening": float(soft), "channels": diag,
            "dominant": dom, "anchors": anchors}


def measure_stretch(subbed: dict[str, np.ndarray], blacks: dict[str, float],
                    white: float, softening: float, lum: Optional[str] = None,
                    fm: Optional[dict] = None) -> dict:
    """What a stretch DOES to this field, in units that carry across targets.

    black_pct=60 on a galaxy and on a field-filling nebula are unrelated
    stretches, so a pick cannot be stored as its percentiles
    (docs/AUTO_STRETCH_PLAN.md, phase 0). Measured instead:
      margin_sigma   per channel, how far black sits below sky in per-pixel sigma
                     (negative = above sky, clipping noise and faint signal)
      faint_out      output level (0..1) of sky + 2 binned sigma, dominant channel
      bright_out     output level of the bright signal's p90, dominant channel
      sky_out        output level of the sky itself, dominant channel
      ext_p10_out / ext_p50_out
                     output level of the faint / typical extended-signal pixel
      ext_clip      fraction of extended (non-star) signal at or above white
      star_clip      fraction of star-mask pixels at or above white
    *subbed* is what _prepare returns (background-subtracted, balanced,
    Ha-blended); *blacks*/*white*/*softening* are the stretch being measured.
    *fm* is a field_model() of the same *subbed*, to skip recomputing it.
    """
    fm = fm if fm is not None else field_model(subbed, lum)
    diag, dom, colour, is_star, ext = fm["diag"], fm["dominant"], fm["colour"], fm["is_star"], fm["ext"]
    d = diag[dom]
    span = max(white - blacks[dom], 1e-6)

    def out(adu):
        return float(_asinh_out(float(np.clip((adu - blacks[dom]) / span, 0.0, 1.0)), softening))

    clipped = colour >= white
    n_ext, n_star = int(ext.sum()), int(is_star.sum())
    # How bright the object itself is drawn: the faint and typical pixel of the
    # extended signal in the dominant channel. Picks put black ABOVE sky
    # (ngc6946: +0.5 sigma), where sky + 2 binned sigma reads 0 for every
    # candidate and stops discriminating.
    ext_vals = subbed[dom][ext] if n_ext else np.empty(0, np.float32)
    ext_q = {q: float(np.percentile(ext_vals, q)) for q in (10, 50)} if ext_vals.size else {}
    return {
        "dominant": dom,
        "margin_sigma": {c: float((x["sky"] - blacks[c]) / x["sigma"]) for c, x in diag.items()},
        "sky_out": out(d["sky"]),
        "faint_out": out(d["sky"] + 2.0 * d["sigma_binned"]),
        "bright_out": out(d["bright_p90"]) if d["bright_p90"] is not None else None,
        "ext_p10_out": out(ext_q[10]) if ext_q else None,
        "ext_p50_out": out(ext_q[50]) if ext_q else None,
        "ext_fraction": n_ext / float(colour.size),
        "ext_clip": float((clipped & ext).sum() / n_ext) if n_ext else None,
        "star_clip": float((clipped & is_star).sum() / n_star) if n_star else None,
        "softening": float(softening),
        "faint_fraction": d["faint_fraction"], "bright_fraction": d["bright_fraction"],
        "stars": fm["stars"], "clean_fraction": d["clean_fraction"],
    }


def field_model(subbed: dict[str, np.ndarray], lum: Optional[str] = None) -> dict:
    """The field as the stretch sees it: per-channel sky and noise, the
    dominant channel, the brightest-primary plane, the star mask and the
    extended-signal mask. Shared by measure_stretch and auto_stretch_v2 so a
    stretch is chosen and scored against the same masks."""
    diag = {}
    for c, a in subbed.items():
        if c in DERIVED_PLANES:
            continue
        cs = clean_sky(a)
        diag[c] = {**sky_noise_model(a, sky=cs["sky"]), "clean_fraction": cs["clean_fraction"],
                   "confident": cs["confident"]}
    if lum and lum in diag:
        dom = lum
    else:
        dom = max(diag, key=lambda c: diag[c]["faint_fraction"] + diag[c]["bright_fraction"])
    d = diag[dom]
    colour = np.maximum.reduce([subbed[c] for c in ("R", "G", "B") if c in subbed])
    stars, sinfo = star_mask(subbed[dom])
    is_star = stars > 0.5
    # Extended signal: off the stars, in blocks of the DOMINANT channel that
    # stand 3 binned sigma above sky. Binned, not per pixel: on a faint nebula
    # a per-pixel cut keeps only the brightest ~1% of the object, which is the
    # part that clips (bubble read 97% "clipped"). Same channel the output
    # levels are read from, or the mask and the levels disagree.
    b = AUTO_BIN
    sig = _block_mean(subbed[dom], b) > d["sky"] + 3.0 * d["sigma_binned"]
    blocks = np.zeros(colour.shape, bool)
    bh, bw = sig.shape
    blocks[:bh * b, :bw * b] = np.repeat(np.repeat(sig, b, axis=0), b, axis=1)
    return {"diag": diag, "dominant": dom, "colour": colour, "is_star": is_star,
            "ext": (~is_star) & blocks, "stars": sinfo.get("stars", 0)}


# --- auto stretch v2 (docs/AUTO_STRETCH_PLAN.md, phase 1) -------------------
#
# The stretch is set by three TARGETS in output units, the same units the
# labels are measured in (measure_stretch), so a pick on one target can be
# applied to another: where black sits against sky (margin, per-pixel sigma;
# negative = above sky), what share of the object may clip (ext_clip, stars
# excluded -- they may clip), and how bright the object's bright p90 is drawn
# (bright_out). Defaults per class are the medians of the owner's picks as of
# 2026-10-16 (local/stretch_labels.json); scripts/stretch_benchmark.py scores
# them leave-one-out and prints the values to put here.
STRETCH_TARGETS = {
    # m33, ngc7320 (owner): margins -0.3..-0.8, object clip 0.4-0.7%, bright 0.58-0.76
    "broadband": {"margin": -0.45, "ext_clip": 0.005, "bright_out": 0.65},
    # ngc7380, sh2-92, bubble, ic1396 (owner): margins +0.5..-0.5, bright 0.58-1.0
    "narrowband": {"margin": -0.2, "ext_clip": 0.006, "bright_out": 0.85},
}
NARROWBAND_RECIPES = frozenset({"HOO", "SHO", "HSO"})


def stretch_class(recipe: str) -> str:
    return "narrowband" if str(recipe).upper().replace("_NOFLAT", "") in NARROWBAND_RECIPES \
        else "broadband"


def auto_stretch_v2(subbed: dict[str, np.ndarray], cls: str = "broadband",
                    targets: Optional[dict] = None, lum: Optional[str] = None,
                    fm: Optional[dict] = None) -> dict:
    """Blacks, white and softening that hit the class's targets on this field.

    -> {"blacks", "white", "softening", "confident", "why", "targets", "fm"}.
    "confident" is False when the dominant channel has no clean sky or the
    field has no extended signal to set the white from; the stretch is still
    returned (sky-anchored, white at a percentile) but should not be published
    unattended.
    """
    t = dict(STRETCH_TARGETS[cls], **(targets or {}))
    fm = fm if fm is not None else field_model(subbed, lum)
    diag, dom, colour, ext = fm["diag"], fm["dominant"], fm["colour"], fm["ext"]
    why = []
    blacks = {c: d["sky"] - t["margin"] * d["sigma"] for c, d in diag.items()}
    if not diag[dom]["confident"]:
        why.append("no clean sky in %s (%.0f%% of tiles at sky)"
                   % (dom, 100 * diag[dom]["clean_fraction"]))

    # White: the object may clip by ext_clip, stars by whatever it takes.
    vals = colour[ext]
    if vals.size >= 1000:
        white = float(np.percentile(vals, 100.0 * (1.0 - t["ext_clip"])))
    else:
        why.append("no extended signal to set white from")
        white = shared_white(subbed, 99.9)
    white = max(white, max(blacks[c] for c in ("R", "G", "B") if c in blacks)
                + 10.0 * diag[dom]["sigma"])

    # Softening: draw the dominant channel's bright p90 at bright_out.
    d = diag[dom]
    lo, hi = AUTO_SOFT_RANGE
    if d["bright_p90"] is not None and d["bright_fraction"] > 1e-4:
        y = float(np.clip((d["bright_p90"] - blacks[dom]) / max(white - blacks[dom], 1e-6), 0, 1))
        soft = _solve_soft(y, t["bright_out"], lo, hi)
        if soft in (lo, hi):
            why.append("softening at its %s limit: bright p90 cannot reach %.2f"
                       % ("hard" if soft == lo else "soft", t["bright_out"]))
    else:
        soft = SOFTENING
        why.append("no bright signal; default softening")
    return {"blacks": blacks, "white": white, "softening": float(soft),
            "confident": not why or all(w.startswith("softening") for w in why),
            "why": why, "targets": t, "dominant": dom, "fm": fm}


# ------------------------------------------------------------------ render QC (D1)
# docs/AUTOMATION_PLAN.md D1 (2026-10-30): is this composite publishable
# unattended, or does it need a hand? Every check is on the finished 0..1 RGB
# and the stretch that made it; the numbers are the bands the owner's
# hand-approved renders sat in (STRETCH_TARGETS) plus the failure classes
# seen this autumn: a black-crushed sky (IC 1396 at black_pct 65), a lit
# channel over the sky (the brown M1 sky, 2026-10-25), a red bar from an
# uncropped footprint (IC 410, 2026-10-28), magenta stars under a palette gain.
QC_SKY_RANGE = (0.01, 0.20)      # the sky's grey level, as a fraction of white
QC_BLACK_CLIP_MAX = 0.01         # share of the field at pure black
QC_WHITE_CLIP_MAX = 0.005        # share of the field at pure white (stars saturate; the field must not)
QC_CHANNEL_MEDIAN_MAX = 0.5      # a channel whose median is this bright is lit over the sky
QC_STAR_NEUTRAL_MAX = 0.15       # palettes: mean |channel difference| on the star discs


def render_qc(rgb: np.ndarray, stretch: Optional[dict] = None, cropped: bool = True,
              star_mask: Optional[np.ndarray] = None, cls: str = "broadband") -> dict:
    """-> {"pass": bool, "reasons": [str], "measures": {...}}. Pure.

    *rgb* is the finished composite (H, W, 3) in 0..1; *stretch* is
    auto_stretch_v2's result (its "confident" and "why" are the first check);
    *cropped* says the common footprint was applied (n2n_lrgb_render
    crop_common had a coverage box for every channel); *star_mask* (bool,
    H x W) lets a palette's stars be checked for neutrality.
    """
    a = np.clip(np.nan_to_num(np.asarray(rgb, dtype=np.float32)), 0.0, 1.0)
    reasons, m = [], {}
    if stretch is not None and not stretch.get("confident", False):
        why = "; ".join(str(w) for w in stretch.get("why") or []) or "no reason given"
        reasons.append("stretch not confident: " + why)
    y = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    sky = float(np.median(y))
    m["sky_grey"] = round(sky, 4)
    lo, hi = QC_SKY_RANGE
    if sky < lo:
        reasons.append("sky at %.1f%% grey: black-crushed (below %.0f%%)" % (100 * sky, 100 * lo))
    elif sky > hi:
        reasons.append("sky at %.1f%% grey: too bright (above %.0f%%)" % (100 * sky, 100 * hi))
    black = float(np.mean(np.all(a <= 0.0, axis=2)))
    white = float(np.mean(np.all(a >= 0.995, axis=2)))
    m["black_clip"], m["white_clip"] = round(black, 5), round(white, 5)
    if black > QC_BLACK_CLIP_MAX:
        reasons.append("%.1f%% of the field clipped to black" % (100 * black))
    if white > QC_WHITE_CLIP_MAX:
        reasons.append("%.2f%% of the field saturated white" % (100 * white))
    meds = {c: float(np.median(a[:, :, i])) for i, c in enumerate("RGB")}
    m["channel_medians"] = {c: round(v, 4) for c, v in meds.items()}
    lit = [c for c, v in meds.items() if v > QC_CHANNEL_MEDIAN_MAX]
    if lit:
        reasons.append("channel %s lit over the sky (median %s)" % (
            ", ".join(lit), ", ".join("%.2f" % meds[c] for c in lit)))
    if not cropped:
        reasons.append("common footprint not applied: a stack predates the coverage box "
                       "(--force rebuilds it)")
    if star_mask is not None and cls == "narrowband":
        sm = np.asarray(star_mask, dtype=bool)
        if sm.shape == a.shape[:2] and sm.any():
            px = a[sm]
            diff = np.maximum.reduce([np.abs(px[:, 0] - px[:, 1]), np.abs(px[:, 1] - px[:, 2]),
                                      np.abs(px[:, 0] - px[:, 2])])
            tint = float(np.mean(diff))
            m["star_tint"] = round(tint, 4)
            if tint > QC_STAR_NEUTRAL_MAX:
                reasons.append("stars not neutral on the palette (mean channel difference %.2f)" % tint)
    return {"pass": not reasons, "reasons": reasons, "measures": m}


def effective_options(**overrides) -> dict:
    """Every compose() display knob with its default filled in.

    The command parser only carries the options the user actually typed, so a
    record built from it says nothing about the settings that did most of the
    work. Reproducing a render means knowing all of them.
    """
    opts = {"black_pct": BLACK_PCT, "white_pct": WHITE_PCT,
            "softening": SOFTENING, "mesh": BG_MESH_FRACTION,
            "subtract_background": SUBTRACT_BACKGROUND, "scnr": SCNR_AMOUNT,
            "ha_gain": HA_GAIN, "white_balance": WHITE_BALANCE,
            "deblur": DEBLUR_ITERS, "dering": DEBLUR_DERING, "method": DEBLUR_METHOD,
            "palette_gain": PALETTE_GAIN}
    opts.update({k: v for k, v in overrides.items() if k in opts})
    return opts


def describe_options(opts: dict, scale: int = 1) -> str:
    """One line of compose settings, in the spelling the command accepts."""
    parts = [f"black={opts['black_pct']:g}", f"white={opts['white_pct']:g}",
             f"soft={opts['softening']:g}"]
    if str(opts.get("white_balance", "none")).lower() not in ("none", ""):
        parts.append(f"wb={opts['white_balance']}")
    if opts.get("subtract_background", True):
        parts.append(f"mesh={opts['mesh']:g}")
    else:
        parts.append("nobg")
    if opts.get("scnr"):
        parts.append(f"scnr={opts['scnr']:g}")
    if opts.get("ha_gain", HA_GAIN) != HA_GAIN:
        parts.append(f"ha={opts['ha_gain']:g}")
    if opts.get("palette_gain"):
        parts.append(f"gain={describe_palette_gain(opts['palette_gain'])}")
    if opts.get("deblur"):
        parts.append(f"deblur={int(opts['deblur'])}")
        if str(opts.get("dering", DEBLUR_DERING)).lower() != DEBLUR_DERING:
            parts.append(f"dering={opts['dering']}")
        if str(opts.get("method", DEBLUR_METHOD)).lower() != DEBLUR_METHOD:
            parts.append(f"method={opts['method']}")
    if scale and scale > 1:
        parts.append(f"scale={scale:g}")
    return "  ".join(parts)


def _scnr(rgb: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Average-neutral SCNR: clip green at the mean of red and blue.

        g' = min(g, (r + b) / 2)

    Almost nothing in space is genuinely green — hydrogen is red, reflection
    nebulae and hot stars are blue, and there is no strong broadband green
    emitter — so a green-dominant pixel is nearly always sky glow, a residual
    gradient, or per-channel noise. That is what makes this better than a global
    colour balance: it is one-sided. Pixels that were never green-dominant are
    returned untouched, so the operation costs nothing where there was nothing
    wrong, and it cannot introduce a magenta cast of its own the way scaling the
    whole green channel down would.

    *amount* blends toward the clipped result (PixInsight's amount slider), so
    partial strengths are available for palettes where green carries real signal.
    """
    g = rgb[:, :, 1]
    neutral = 0.5 * (rgb[:, :, 0] + rgb[:, :, 2])
    limited = np.minimum(g, neutral)
    if amount < 1.0:
        limited = g * (1.0 - amount) + limited * amount

    touched = np.isfinite(g) & np.isfinite(limited) & (limited < g)
    n = int(touched.sum())
    if n:
        _logger.info("SCNR (amount %.2f): green reduced on %.1f%% of pixels, "
                     "median cut %.3f", amount, 100.0 * n / g.size,
                     float(np.median((g - limited)[touched])))
    else:
        _logger.info("SCNR (amount %.2f): no green-dominant pixels; no change",
                     amount)

    out = rgb.copy()
    out[:, :, 1] = limited
    return out


def _deblurred(channels: dict[str, np.ndarray], iters,
               dering: str = DEBLUR_DERING,
               method: str = DEBLUR_METHOD) -> dict[str, np.ndarray]:
    """The channel set deconvolved (see DEBLUR_ITERS), or itself for deblur=0."""
    if not iters or int(iters) <= 0:
        return channels
    from stacking import deblur as _deblur
    out, infos = _deblur.deblur_channels(channels, int(iters), clamp_to=str(dering).lower(),
                                         method=str(method).lower())
    for c, info in infos.items():
        if info.get("fallback"):
            _logger.warning("deblur %s: %s", c, info["fallback"])
        if info.get("why"):
            _logger.warning("deblur %s: left as stacked — %s", c, info["why"])
        elif info.get("method") == "learned":
            if info.get("kept"):
                _logger.info("deblur %s: kept as stacked (learned deconvolution is luminance-only)", c)
            elif info.get("synthetic"):
                _logger.info("deblur %s: synthetic luminance (%s), learned deconvolution, "
                             "kernel %.2f px from %d stars", c, info["synthetic"],
                             info.get("fwhm_px", 0.0), info.get("stars", 0))
            else:
                _logger.info("deblur %s: learned deconvolution, kernel %.2f px from %d stars",
                             c, info.get("fwhm_px", 0.0), info.get("stars", 0))
        elif info.get("method") == "starless":
            _logger.info("deblur %s: starless RL%d, %d stars re-rendered, %d non-stellar, "
                         "FWHM %.2f px, %.1f%% of the frame filled",
                         c, info["iters"], info.get("stars", 0), info.get("non_stellar", 0),
                         info.get("fwhm_px", 0.0), 100 * info.get("deep_frac", 0.0))
        else:
            _logger.info("deblur %s: RL%d, %d stars, FWHM %.2f px, %.2f%% of pixels",
                         c, info["iters"], info.get("stars", 0), info.get("fwhm_px", 0.0),
                         100 * info.get("deblurred_frac", 0.0))
    return out


def compose(channels: dict[str, np.ndarray], black_pct: float = BLACK_PCT,
            white_pct: float = WHITE_PCT,
            subtract_background: bool = SUBTRACT_BACKGROUND,
            softening: float = SOFTENING,
            mesh: int = BG_MESH_FRACTION,
            scnr: float = SCNR_AMOUNT,
            ha_gain: float = HA_GAIN,
            white_balance: str = WHITE_BALANCE,
            deblur: int = DEBLUR_ITERS,
            dering: str = DEBLUR_DERING,
            method: str = DEBLUR_METHOD,
            palette_gain=PALETTE_GAIN) -> np.ndarray:
    """Combine channel stacks into an RGB image in 0..1.

    channels holds any of R/G/B plus an optional L, and for HALRGB an HA plane
    that _prepare folds into R and L. Every channel must already be on the
    same pixel grid — that is what the shared reference guarantees.

    deblur > 0 (the default, DEBLUR_ITERS) deconvolves first, on the linear
    stacks with their sky, which is what the kernel measurement and the
    floor model need: method=learned (default since 2026-10-21) puts the
    luminance through nn/deblur_model and leaves the colour planes as
    stacked; method=starless takes the stars off every channel, RL's the
    luminance (every plane when there is no L) and puts the stars back as
    imaged; method=rl is RL on the whole luminance. See stacking/deblur.
    """
    channels = _deblurred(channels, deblur, dering, method)
    subbed, white = _prepare(channels, subtract_background, mesh, white_pct, ha_gain,
                             white_balance, palette_gain=palette_gain)
    _logger.info("Compose: shared white point %.2f ADU (p%.1f)", white, white_pct)

    rgb = np.dstack([_stretch(subbed[c], black_pct, white, softening, label=c)
                     for c in ("R", "G", "B")])

    if scnr > 0.0:
        # Before the luminance substitution, not after: L then re-establishes
        # brightness, so SCNR acts purely on hue instead of darkening wherever
        # it pulled green down. Same order PixInsight uses (SCNR, then
        # LRGBCombination).
        rgb = _scnr(rgb, float(np.clip(scnr, 0.0, 1.0)))

    if "L" in subbed:
        # Classic LRGB: keep the colour from RGB, take the brightness from L.
        # Scaling by the ratio preserves hue instead of washing it out, which is
        # what simply averaging L into each channel would do.
        lum = _stretch(subbed["L"], black_pct, white, softening, label="L")
        rgb_lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(rgb_lum > 1e-4, lum / np.maximum(rgb_lum, 1e-4), 0.0)
        rgb = rgb * np.clip(scale, 0.0, MAX_LUM_BOOST)[:, :, None]
    elif "LSYN" in subbed and "LSYN_REF" in subbed:
        # A set with no L whose luminance the deconvolution made (see
        # stacking/deblur.SYNTHETIC_LUM): one black for the pair, from the
        # reference, so the ratio carries only what the network moved.
        black = float(np.nanpercentile(subbed["LSYN_REF"], black_pct))
        def st(chan):
            y = np.clip((chan - black) / max(white - black, 1e-6), 0.0, 1.0)
            return np.arcsinh(y / softening) / np.arcsinh(1.0 / softening)
        _logger.info("Stretch LSYN: black %.2f ADU (p%.1f, from the reference)", black, black_pct)
        rgb = rgb * synthetic_gain(st(subbed["LSYN"]), st(subbed["LSYN_REF"]))[:, :, None]

    return np.clip(np.nan_to_num(rgb), 0.0, 1.0)


def sky_anchored_blacks(subbed: dict[str, np.ndarray], black_sigma: float,
                        black_pct: float = BLACK_PCT, log=None,
                        gains: Optional[dict] = None) -> dict[str, float]:
    """Black point per channel, anchored to the sky rather than a percentile.

    A percentile is a *relative* rule, and anything that changes the pixel
    distribution — denoising, or simply stacking deeper — moves it: on the
    2026-08-24 trunk render p65 on Ha landed 0.36 sigma above sky, inside the
    noise, so raw sky pixels read as a veil while denoised ones went black and
    the model looked as though it had eaten the nebulosity. Anchoring to
    `sky_median - black_sigma * sigma` puts black below the sky by a known
    amount in every frame of a series, which is what keeps them comparable.
    sigma comes from sep.Background, which sigma-clips, so nebulosity does not
    inflate it. black_sigma <= 0 restores the percentile rule.

    ONE offset for every channel (since 2026-10-25): black_sigma times the
    LARGEST channel's sigma, below each channel's own sky. With each channel
    offset by its own sigma the sky landed at a different output level per
    channel, and the shallowest channel's sky sat highest: m1 SHO on 5 S-II
    subs against 10 Ha and 10 O-III put the S-II sky 1.33 ADU above black
    against 1.03 and 0.95, which the saturation boost turned into a brown
    sky (RGB 54/32/25). The sky is the same grey in every channel now, and
    the noisier channel just scatters more about it; on a balanced set the
    offsets were already nearly equal and nothing changes.

    *gains* is the palette gain each channel carries ({role: factor}, from
    _prepare's info): the offset is sized by the sigma each channel had
    BEFORE its gain. A channel scaled up carries its noise up, and an offset
    sized to that put IC 410's sky at 30 % grey instead of 9 (S-II x3.9:
    offset 1.3 -> 4.8 ADU, 2026-10-28). The gain is a tone; the sky stays
    where it was, and the scaled-up channel's raw noise clips harder below
    it, which the denoised and learned composites pinned to this black do
    not show.
    """
    if black_sigma is None or black_sigma <= 0:
        return blacks_from(subbed, black_pct)
    import sep
    stats: dict[str, tuple[float, float]] = {}
    out = {}
    for c, arr in subbed.items():
        if c in DERIVED_PLANES:
            continue
        a = np.ascontiguousarray(arr.astype(np.float32))
        try:
            stats[c] = (float(np.nanmedian(a)), float(sep.Background(a).globalrms))
        except Exception:
            out[c] = float(np.nanpercentile(arr, black_pct))
    gains = gains or {}
    offset = black_sigma * max((sig / (float(gains.get(c, 1.0)) or 1.0)
                                for c, (_, sig) in stats.items()), default=0.0)
    for c, (med, sig) in stats.items():
        out[c] = med - offset
        if log:
            a = subbed[c]
            pct = float(np.nanmean(a <= out[c]) * 100.0)
            log(f"    {c}: sky {med:+.3f} sigma {sig:.3f} -> black {out[c]:+.3f} ADU "
                f"(shared offset {offset:.3f}; {pct:.0f}% of pixels, vs {black_pct:.0f}% "
                f"under the old rule)")
    return out


def blacks_from(subbed: dict[str, np.ndarray], black_pct: float) -> dict[str, float]:
    """The per-channel black points compose() would pick, as numbers.

    compose() decides black by percentile *per call*, which is right for one
    image and wrong for a series that has to be compared — a stacking movie, a
    raw/denoised pair — where the darkest 65% of a noisier frame is a different
    ADU. Take the blacks from the frame that should set the scale and hand
    them to compose_prepared() for every other one.
    """
    return {c: float(np.nanpercentile(subbed[c], black_pct)) for c in subbed}


def compose_prepared(subbed: dict[str, np.ndarray], blacks: dict[str, float],
                     white: float, softening: float = SOFTENING,
                     scnr: float = SCNR_AMOUNT) -> np.ndarray:
    """compose() from background-subtracted channels with black and white given.

    Same curve, same SCNR order, same luminance substitution as compose(); the
    only difference is that nothing is measured from the input, so a series
    rendered through this shares one scale exactly.
    """
    def st(chan, black):
        y = np.clip((chan - black) / max(white - black, 1e-6), 0.0, 1.0)
        return np.arcsinh(y / softening) / np.arcsinh(1.0 / softening)

    rgb = np.dstack([st(subbed[c], blacks[c]) for c in ("R", "G", "B")])
    if scnr > 0.0:
        rgb = _scnr(rgb, float(np.clip(scnr, 0.0, 1.0)))
    if "L" in subbed:
        lum = st(subbed["L"], blacks["L"])
        rgb_lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(rgb_lum > 1e-4, lum / np.maximum(rgb_lum, 1e-4), 0.0)
        rgb = rgb * np.clip(scale, 0.0, MAX_LUM_BOOST)[:, :, None]
    elif "LSYN" in subbed and "LSYN_REF" in subbed:
        black = blacks.get("LSYN_REF", blacks.get("LSYN", 0.0))
        rgb = rgb * synthetic_gain(st(subbed["LSYN"], black), st(subbed["LSYN_REF"], black))[:, :, None]
    return np.clip(np.nan_to_num(rgb), 0.0, 1.0)


def cache_tag(recipe: str, use_flats: bool) -> str:
    """Cache key for a stacking run.

    Includes the flat state: flat-corrected and uncorrected channels are
    different data, and a cache that ignored the difference would let a sweep
    silently re-render the wrong ones while reporting the settings you asked
    for. Matches the naming of the rendered files.
    """
    return recipe.upper() if use_flats else f"{recipe.upper()}_noflat"


def channel_cache_path(cache_dir: Path, dso: str, tag: str, chan: str) -> Path:
    return cache_dir / f"channels_{dso}_{tag}_{chan}.npy"


def load_cached_channels(cache_dir: Path, dso: str, tag: str,
                         roles: Optional[tuple] = None) -> Optional[dict]:
    """Return the stacked channels from a previous run, or None if incomplete.

    *roles* names the cached planes to load instead of the recipe's channels
    — ("STARS_R", "STARS_G", "STARS_B") for the star-colour stacks, which
    come back keyed R/G/B.

    Stacking is ~17 minutes and the stretch is a second; caching the channels is
    what makes the display parameters worth exposing at all, because otherwise
    every tweak costs a re-stack.
    """
    if roles:
        out = {}
        for role in roles:
            f = channel_cache_path(cache_dir, dso, tag, role)
            if not f.exists():
                return None
            out[role.replace("STARS_", "")] = np.load(f)
        return out or None
    mapping = RECIPES.get(tag.upper().replace("_NOFLAT", ""))
    if mapping is None:
        return None
    out = {}
    for chan in mapping:
        f = channel_cache_path(cache_dir, dso, tag, chan)
        if not f.exists():
            if chan == "L":          # optional
                continue
            return None
        out[chan] = np.load(f)
    return out or None


def apply_rgb_stars(rgb: np.ndarray, stacks: dict[str, np.ndarray],
                    star_stacks: dict[str, np.ndarray], compose_kw: dict) -> tuple[np.ndarray, dict]:
    """Recolour the stars of a composed palette *rgb* from R/G/B stacks.

    The mask is detected on the sum of the RGB stacks: those are the stars
    that have a colour to give, and on a shallow narrowband stack (M33's
    7-sub Ha found 300 stars where the field has thousands) the palette
    plane misses most of them.
    """
    ref = star_stacks["R"] + star_stacks["G"] + star_stacks["B"]
    mask, info = star_mask(ref)
    stars = stars_image(star_stacks, **compose_kw)
    return rgb_stars(rgb, stars, mask), info


def process_dso(
    dso_dir: Path,
    recipe: str,
    progress_cb: Optional[Callable[[str], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
    scale: int = 1,
    use_flats: bool = True,
    cache_dir: Optional[Path] = None,
    reuse: bool = False,
    products_dir: Optional[Path] = None,
    star_source: str = "none",
    **compose_kw,
) -> tuple[np.ndarray, dict]:
    """Stack every filter a recipe needs and return (rgb 0..1, info).

    star_source="rgb" also stacks R, G and B onto the same reference and
    recolours the stars from them (see rgb_stars); the palette is untouched
    everywhere else. The extra stacks are cached beside the channels as
    STARS_R/G/B, so `reuse` re-renders them too.

    scale > 1 bins the output, trading resolution for speed and SNR — useful for
    a quick look, but the command defaults to 1 (full resolution).

    use_flats=False drops flat correction and keeps bias+dark. Worth having
    because flats from the wrong epoch can be worse than none: dust moves and
    focus shifts, so a flat shot months after the lights may stamp in a mote the
    data never had. Measured on abell2151 (May data, July flats) the flat was
    near-neutral, +0.21% over the region in question — but that was luck, not a
    guarantee, and it is cheap to check both ways.
    """
    from stacking import stacker

    recipe = recipe.upper()
    tag = cache_tag(recipe, use_flats)
    if compose_kw.get("palette_gain") is not None:
        # Typed by filter name; compose wants it by role, which needs the recipe.
        compose_kw["palette_gain"] = palette_gain_roles(recipe, compose_kw["palette_gain"])
    if reuse and cache_dir is not None:
        cached = load_cached_channels(cache_dir, dso_dir.name, tag)
        if cached is not None:
            if progress_cb:
                progress_cb(f"reusing cached {tag} channels "
                            f"({', '.join(sorted(cached))}) — no re-stack")
            # Binning lives on the stacking path below, which this branch skips.
            # Without this, `reuse scale=2` returned a full-resolution image
            # while the options line still said scale=2 -- the render and its
            # own description disagreeing, which is the trap the WCS block
            # further down already refuses to fall into.
            if scale > 1:
                cached = {c: stacker._downsample_mean(a, scale)
                          for c, a in cached.items()}
            star_cached = {}
            if str(star_source).lower() == "rgb":
                star_cached = load_cached_channels(cache_dir, dso_dir.name, tag,
                                                   roles=("STARS_R", "STARS_G", "STARS_B")) or {}
                if scale > 1:
                    star_cached = {c: stacker._downsample_mean(a, scale)
                                   for c, a in star_cached.items()}
            rgb = compose(cached, **compose_kw)
            star_info = None
            if str(star_source).lower() == "rgb":
                if len(star_cached) == 3:
                    rgb, star_info = apply_rgb_stars(rgb, cached, star_cached, compose_kw)
                    if progress_cb:
                        progress_cb(f"stars from RGB: {star_info['stars']} stars, "
                                    f"{100 * star_info['coverage']:.2f}% of the field")
                elif progress_cb:
                    progress_cb("stars=rgb: no cached STARS_R/G/B — run without reuse once")
            out = {"recipe": recipe, "reused": True, "stars": star_info,
                   "channels": {c: c for c in cached}, "frames": {},
                   "reference": "(cached)", "shape": rgb.shape[:2],
                   "flats": use_flats}
            if products_dir is not None:
                # The FITS are unchanged — same pixels — but the channel JPEGs
                # are rendered with the stretch, so they follow the new options.
                try:
                    out["channel_jpgs"] = save_channel_jpgs(
                        cached, products_dir, dso_dir.name, tag,
                        **{k: v for k, v in compose_kw.items() if k != "scnr"})
                except Exception:
                    _logger.exception("channel JPEG export failed")
            return rgb, out
        if progress_cb:
            progress_cb("no cached channels found — stacking")

    if recipe not in RECIPES:
        raise ValueError(f"Unknown recipe '{recipe}'. Choose one of "
                         f"{', '.join(sorted(RECIPES))}.")

    lights = sorted((f for f in dso_dir.rglob("*.fits")
                     if f.parent.name.upper() == "LIGHT"),
                    key=lambda f: f.stat().st_mtime)
    if not lights:
        raise ValueError(f"No LIGHT frames under {dso_dir}")
    by_filter = stacker.group_by_filter(lights)

    mapping = RECIPES[recipe]
    resolved: dict[str, str] = {}
    for chan, wanted in mapping.items():
        found = resolve_filter(wanted, list(by_filter))
        if found:
            resolved[chan] = found
    missing = [f"{c}={mapping[c]}" for c in mapping if c not in resolved]
    if not {"R", "G", "B"} <= set(resolved):
        raise ValueError(
            f"{recipe} needs {', '.join(mapping[c] for c in ('R','G','B'))}; "
            f"this target has {', '.join(sorted(by_filter))}. Missing: "
            f"{', '.join(missing)}")
    if "HA" in mapping and "HA" not in resolved:
        # Without Ha this recipe is LRGB under another name; say so instead
        # of rendering something that looks finished and is not what was asked.
        raise ValueError(
            f"{recipe} needs {mapping['HA']} frames; this target has "
            f"{', '.join(sorted(by_filter))}. Use LRGB, or shoot Ha first.")
    if missing and progress_cb:
        progress_cb(f"missing {', '.join(missing)} — continuing without")

    # One reference for every filter. Prefer the channel with the most frames,
    # since more frames means a better chance of a genuinely sharp one.
    ref_filter = max(set(resolved.values()), key=lambda f: len(by_filter[f]))
    ref_paths = by_filter[ref_filter]
    arcsec = stacker._get_arcsec_per_pixel()
    from fits_processing.frame_cache import load_precomputed_fwhm_stars
    ref_pre = load_precomputed_fwhm_stars(dso_dir, ref_paths, arcsec)
    ref_fwhm = {p: v[0] for p, v in ref_pre.items()}
    ref_idx = stacker._reference_index_by_fwhm(
        [ref_fwhm.get(p, 0.0) for p in ref_paths]) or 0
    ref_path = ref_paths[ref_idx]
    if progress_cb:
        progress_cb(f"shared reference: {ref_path.name} ({ref_filter})")

    ref_cal = stacker.calibration_from_config(ref_filter if use_flats else None)
    reference = stacker._load_calibrated(ref_path, ref_cal)
    ref_shape = reference.shape
    det_target = stacker._reference_control_points(stacker._despike(reference))
    if det_target is None:
        raise ValueError("Could not extract control points from the reference frame")
    reference = None

    stacks: dict[str, np.ndarray] = {}
    used: dict[str, int] = {}
    for chan in ("R", "G", "B", "L", "HA"):
        if chan not in resolved:
            continue
        filt = resolved[chan]
        if filt in used:                       # HOO points G and B at one filter
            stacks[chan] = stacks[[c for c in stacks if resolved[c] == filt][0]]
            continue
        paths = by_filter[filt]
        if progress_cb:
            progress_cb(f"{filt}: stacking {len(paths)} frames…")
        bias, dark, flat = stacker.calibration_paths_from_config(filt)
        if not use_flats:
            flat = []
        pre = load_precomputed_fwhm_stars(dso_dir, paths, arcsec)
        data, info = stacker.stack(
            paths, method=stacker.StackMethod.SIGMA_CLIP_FWHM,
            bias_paths=bias, dark_paths=dark, flat_paths=flat,
            precomputed_fwhm_stars=pre, shared_reference=(det_target, ref_shape),
            progress_cb=(lambda m, _f=filt: progress_cb(f"{_f}: {m}")) if progress_cb else None,
            cancel_cb=cancel_cb,
        )
        stacks[chan] = data if scale <= 1 else stacker._downsample_mean(data, scale)
        used[filt] = info.get("n_frames", len(paths))
        if progress_cb:
            progress_cb(f"{filt}: {used[filt]} frames stacked")

    # stars=rgb: R, G and B onto the same reference, kept apart from the
    # palette channels so a narrowband recipe never composes them.
    star_stacks: dict[str, np.ndarray] = {}
    if str(star_source).lower() == "rgb":
        for chan in ("R", "G", "B"):
            filt = resolve_filter(chan, list(by_filter))
            if not filt:
                if progress_cb:
                    progress_cb(f"stars=rgb: no {chan} frames — stars stay palette")
                star_stacks = {}
                break
            if filt in used and chan in stacks and resolved.get(chan) == filt:
                star_stacks[chan] = stacks[chan]          # LRGB already stacked it
                continue
            paths = by_filter[filt]
            if progress_cb:
                progress_cb(f"{filt}: stacking {len(paths)} frames for star colour…")
            bias, dark, flat = stacker.calibration_paths_from_config(filt)
            if not use_flats:
                flat = []
            pre = load_precomputed_fwhm_stars(dso_dir, paths, arcsec)
            data, info = stacker.stack(
                paths, method=stacker.StackMethod.SIGMA_CLIP_FWHM,
                bias_paths=bias, dark_paths=dark, flat_paths=flat,
                precomputed_fwhm_stars=pre, shared_reference=(det_target, ref_shape),
                progress_cb=(lambda m, _f=filt: progress_cb(f"{_f}: {m}")) if progress_cb else None,
                cancel_cb=cancel_cb,
            )
            star_stacks[chan] = data if scale <= 1 else stacker._downsample_mean(data, scale)
            used[filt] = info.get("n_frames", len(paths))

    # Under a shared reference stack() skips its per-filter coverage crop, so
    # every channel is on the identical grid. Trimming to a common size from the
    # corner would silently mis-align them if that ever stopped being true.
    shapes = {v.shape for v in list(stacks.values()) + list(star_stacks.values())}
    if len(shapes) != 1:
        raise ValueError(f"channels are on different grids: {shapes} — "
                         "they cannot be combined without re-registration")
    h, w = shapes.pop()
    m = int(EDGE_CROP * min(h, w))
    if m > 0:
        stacks = {k: v[m:-m, m:-m] for k, v in stacks.items()}
        star_stacks = {k: v[m:-m, m:-m] for k, v in star_stacks.items()}

    if cache_dir is not None:
        for c, v in stacks.items():
            try:
                np.save(channel_cache_path(cache_dir, dso_dir.name, tag, c),
                        v.astype(np.float32))
            except Exception:
                _logger.warning("could not cache channel %s", c, exc_info=True)
        for c, v in star_stacks.items():
            try:
                np.save(channel_cache_path(cache_dir, dso_dir.name, tag, f"STARS_{c}"),
                        v.astype(np.float32))
            except Exception:
                _logger.warning("could not cache star channel %s", c, exc_info=True)

    rgb = compose(stacks, **compose_kw)
    star_info = None
    if len(star_stacks) == 3:
        rgb, star_info = apply_rgb_stars(rgb, stacks, star_stacks, compose_kw)
        if progress_cb:
            progress_cb(f"stars from RGB: {star_info['stars']} stars, "
                        f"{100 * star_info['coverage']:.2f}% of the field recoloured")
    info = {
        "stars": star_info,
        "recipe": recipe,
        "flats": use_flats,
        "channels": {c: resolved[c] for c in resolved},
        "frames": used,
        "reference": ref_path.name,
        "shape": rgb.shape[:2],
    }

    if products_dir is not None:
        try:
            from astropy.io import fits as _fits
            ref_header = _fits.getheader(ref_path)
        except Exception:
            ref_header = None
            _logger.warning("could not read reference header for WCS", exc_info=True)
        try:
            info["fits"] = save_channel_fits(
                stacks, products_dir, dso_dir.name, tag, info,
                ref_header=ref_header, crop_margin=m, scale=scale)
            info["channel_jpgs"] = save_channel_jpgs(
                stacks, products_dir, dso_dir.name, tag,
                **{k: v for k, v in compose_kw.items() if k != "scnr"})
            if progress_cb:
                progress_cb(f"wrote {len(info['fits'])} channel FITS + "
                            f"{len(info['channel_jpgs'])} channel JPEGs")
        except Exception:
            # The colour image is the deliverable; losing the per-channel
            # exports must not lose a 20-minute stack.
            _logger.exception("channel product export failed")

    return rgb, info


@lru_cache(maxsize=8)
def _label_font(size: int):
    """A TrueType font at *size*, or PIL's bitmap default if none can be loaded.

    PIL's default font is a ~11px bitmap that does not scale, which is why the
    sweep labels were unreadable on a 760px panel. matplotlib is already a
    dependency and ships DejaVuSans, so it is the one scalable font this machine
    is guaranteed to have; the rest are fallbacks, and the bitmap default is the
    last resort so a missing font can never cost someone their sweep sheet.
    """
    from PIL import ImageFont

    candidates = []
    try:
        from matplotlib import font_manager
        candidates.append(font_manager.findfont("DejaVu Sans"))
    except Exception:
        _logger.debug("matplotlib font lookup failed", exc_info=True)
    candidates += ["DejaVuSans.ttf", "arial.ttf", "segoeui.ttf"]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    _logger.warning("no scalable font found — sweep labels will be tiny")
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _wrap_tokens(draw, tokens: list[str], font, avail: float) -> tuple[list[str], bool]:
    """Greedily pack *tokens* into lines no wider than *avail*.

    Returns (lines, fits) — *fits* is False when some single token is wider than
    *avail* even alone, which is the caller's signal to try a smaller font.
    """
    lines: list[str] = []
    current = ""
    fits = True
    for token in tokens:
        trial = f"{current}  {token}" if current else token
        if current and draw.textlength(trial, font=font) > avail:
            lines.append(current)
            current = token
        else:
            current = trial
        if draw.textlength(current, font=font) > avail:
            fits = False
    if current:
        lines.append(current)
    return lines, fits


def _draw_panel_label(draw, width: int, panel_id: int, label: str) -> None:
    """Stamp *panel_id* and its settings across the top of one sweep panel.

    The id is drawn far larger than the settings on purpose: it is the one thing
    a human reads off the sheet — pick a panel, then type its number back — and
    it has to survive being viewed on a phone.

    The settings wrap onto extra lines rather than shrinking without limit. A
    narrow panel (a tall target thumbnailed to well under 760px) or a sweep over
    more keys than the usual three would otherwise drive the font down until the
    text was both unreadable AND clipped at the panel edge — which is the bug
    this label was rewritten to fix, so it is worth keeping wrapped.
    """
    pad = max(6, width // 90)
    id_font = _label_font(max(34, width // 13))
    id_text = f"#{panel_id}"
    id_box = draw.textbbox((0, 0), id_text, font=id_font)
    id_h = id_box[3] - id_box[1]

    text_x = pad + (id_box[2] - id_box[0]) + 2 * pad
    avail = max(1, width - text_x - pad)
    tokens = [t for t in label.split("  ") if t]

    size = max(22, width // 26)
    while True:
        font = _label_font(size)
        lines, fits = _wrap_tokens(draw, tokens, font, avail)
        if fits or size <= 10:
            break
        size -= 2

    line_h = draw.textbbox((0, 0), "Ag", font=font)[3] if lines else 0
    text_h = line_h * len(lines)
    bar_h = max(id_h, text_h) + 2 * pad

    draw.rectangle([0, 0, width, bar_h], fill=(0, 0, 0))
    draw.text((pad, (bar_h - id_h) // 2 - id_box[1]), id_text,
              font=id_font, fill=(255, 214, 0))
    y = (bar_h - text_h) // 2
    for line in lines:
        draw.text((text_x, y), line, font=font, fill=(255, 255, 255))
        y += line_h


def sweep(channels: dict, grid: dict, bin_factor: int = 4,
          progress_cb=None) -> tuple[np.ndarray, list[dict]]:
    """Render every combination in *grid* and tile them into one labelled sheet.

    grid maps a compose() keyword to a list of values; the product is rendered.
    Channels are binned first — the point is comparing the look, not the pixels,
    and binning keeps a nine-panel sweep to a few seconds instead of minutes.

    Returns (sheet, combos) where combos[i] is the settings for panel i, so the
    caller can tell the user which `reuse` invocation reproduces each one.
    """
    import itertools
    from PIL import Image, ImageDraw
    from stacking import stacker

    binned = {k: (v if bin_factor <= 1 else
                  v[:v.shape[0] // bin_factor * bin_factor,
                    :v.shape[1] // bin_factor * bin_factor]
                  .reshape(v.shape[0] // bin_factor, bin_factor,
                           v.shape[1] // bin_factor, bin_factor).mean((1, 3)))
              for k, v in channels.items()}

    keys = sorted(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
    panels = []
    for i, combo in enumerate(combos, 1):
        if progress_cb:
            progress_cb(f"sweep {i}/{len(combos)}: "
                        + "  ".join(f"{k}={v}" for k, v in combo.items()))
        rgb = compose(binned, **combo)
        img = Image.fromarray(stacker.sky_parity((np.clip(rgb, 0, 1) * 255).astype(np.uint8)))
        img.thumbnail((760, 760))
        d = ImageDraw.Draw(img)
        label = "  ".join(f"{k.replace('_pct','').replace('softening','soft')}={v}"
                          for k, v in combo.items())
        # i is 1-based and matches the numbering the chat listing prints, so the
        # number on the panel is the number the user quotes back.
        _draw_panel_label(d, img.width, i, label)
        panels.append(np.asarray(img))

    cols = min(3, len(panels))
    rows = (len(panels) + cols - 1) // cols
    ph, pw = max(p.shape[0] for p in panels), max(p.shape[1] for p in panels)
    sheet = np.zeros((rows * (ph + 4), cols * (pw + 4), 3), np.uint8)
    for i, pnl in enumerate(panels):
        r, c = divmod(i, cols)
        sheet[r*(ph+4):r*(ph+4)+pnl.shape[0], c*(pw+4):c*(pw+4)+pnl.shape[1]] = pnl
    return sheet, combos


def save_channel_jpgs(channels: dict[str, np.ndarray], out_dir: Path, dso: str,
                      tag: str, black_pct: float = BLACK_PCT,
                      white_pct: float = WHITE_PCT,
                      subtract_background: bool = SUBTRACT_BACKGROUND,
                      softening: float = SOFTENING,
                      mesh: int = BG_MESH_FRACTION,
                      max_px: int = CHANNEL_JPG_MAX_PX,
                      ha_gain: float = HA_GAIN,
                      white_balance: str = WHITE_BALANCE,
                      deblur: int = DEBLUR_ITERS,
                      dering: str = DEBLUR_DERING,
                      method: str = DEBLUR_METHOD,
                      palette_gain=PALETTE_GAIN) -> list[Path]:
    """Write one mono JPEG per channel, on the composite's shared scale.

    Deliberately not per-channel autostretch: these are meant to explain the
    colour image, so a channel that is genuinely faint has to *look* faint here.
    Stretched with the same black/white/softening the composite used, so each
    one is literally that channel's plane before SCNR and the L substitution.

    Downscaled to preview size by default. At full resolution these are ~41 MB
    each — 164 MB of mono JPEG per render, more than the colour image itself —
    to answer a question ("what is this channel contributing?") that a screen
    cannot ask at full resolution anyway. The FITS is the archival copy.
    """
    from PIL import Image
    from stacking import stacker
    channels = _deblurred(channels, deblur, dering, method)
    subbed, white = _prepare(channels, subtract_background, mesh, white_pct, ha_gain,
                             white_balance, palette_gain=palette_gain)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    # For HALRGB, R and L here are the BLENDED planes (what the composite
    # used), HA is the raw subtracted Ha stack and HA_EXCESS the emission map
    # that was added — the three together explain where the red came from.
    for chan in ("R", "G", "B", "L", "HA", "HA_EXCESS"):
        if chan not in subbed:
            continue
        mono = _stretch(subbed[chan], black_pct, white, softening)
        arr = stacker.sky_parity((np.clip(np.nan_to_num(mono), 0.0, 1.0) * 255).astype(np.uint8))
        img = Image.fromarray(arr, mode="L")
        if max_px and max(img.size) > max_px:
            ratio = max_px / max(img.size)
            img = img.resize((max(1, int(img.width * ratio)),
                              max(1, int(img.height * ratio))), Image.LANCZOS)
        path = out_dir / f"process_{dso}_{tag}_{chan}.jpg"
        img.save(path, quality=92, optimize=True)
        written.append(path)
    return written


def save_channel_fits(channels: dict[str, np.ndarray], out_dir: Path, dso: str,
                      tag: str, info: dict, ref_header=None,
                      crop_margin: int = 0, scale: int = 1) -> list[Path]:
    """Write each stacked channel as a linear float32 FITS.

    This is the scientific product: calibrated, registered, sigma-clip combined
    ADU with the sky level restored — the thing worth handing to PixInsight or
    re-measuring later. The JPEGs are a rendering of it and throw most of it
    away.

    IMAGETYP is 'STACK', never 'LIGHT'. Nothing here can be collected as data
    anyway (every light-gathering path requires a LIGHT parent directory, and
    these land under Iris/<dso>/), but a stack that announces itself as a light
    frame is an accident waiting for the one path that forgets to check.
    """
    from astropy.io import fits

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for chan, data in channels.items():
        hdu = fits.PrimaryHDU(np.asarray(data, dtype=np.float32))
        h = hdu.header
        h["OBJECT"] = (dso, "target")
        h["IMAGETYP"] = ("STACK", "combined result, not a light frame")
        h["FILTER"] = (info.get("channels", {}).get(chan, "?"), "source filter")
        h["CHANNEL"] = (chan, "role in the colour recipe")
        h["RECIPE"] = (info.get("recipe", "?"), "colour recipe")
        h["NFRAMES"] = (int(info.get("frames", {}).get(
            info.get("channels", {}).get(chan, ""), 0)), "frames combined")
        h["FLATCOR"] = (bool(info.get("flats", True)), "flat correction applied")
        h["STACKMTH"] = ("SIGMA_CLIP_FWHM", "combine method")
        h["REFFRAME"] = (str(info.get("reference", "?"))[:68], "registration reference")
        h["BUNIT"] = ("ADU", "sky level restored after levelling")
        h["DATE"] = (datetime.now(timezone.utc).isoformat(timespec="seconds"), "file written (UTC)")

        # Pointing and optics, always. These are the mount's estimate rather
        # than a solve, so they are not a WCS and must not be dressed up as one
        # — but they are exactly the hint a plate solver wants, which makes the
        # difference between a blind solve and an instant one.
        if ref_header is not None:
            for k in ("OBJCTRA", "OBJCTDEC", "RA", "DEC", "FOCALLEN", "XPIXSZ",
                      "YPIXSZ", "INSTRUME", "TELESCOP", "SITELAT", "SITELONG"):
                if k in ref_header:
                    h[k] = ref_header[k]

        # Carry the reference frame's plate solution, but only when it is still
        # true of these pixels. The edge crop shifts the reference pixel, and
        # binning changes the plate scale — a silently wrong WCS is worse than
        # none, so anything unusual means we simply omit it.
        if ref_header is not None and scale <= 1:
            wcs_keys = ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
                        "CD1_1", "CD1_2", "CD2_1", "CD2_2", "CDELT1", "CDELT2",
                        "CROTA2", "EQUINOX", "RADESYS")
            if all(k in ref_header for k in ("CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2")):
                for k in wcs_keys:
                    if k in ref_header:
                        h[k] = ref_header[k]
                h["CRPIX1"] = float(ref_header["CRPIX1"]) - crop_margin
                h["CRPIX2"] = float(ref_header["CRPIX2"]) - crop_margin
                h["HISTORY"] = f"WCS from {info.get('reference','?')}, CRPIX-{crop_margin}"
        elif ref_header is not None:
            h["HISTORY"] = f"WCS omitted: output binned {scale}x"

        path = out_dir / f"process_{dso}_{tag}_{chan}.fits"
        hdu.writeto(path, overwrite=True)
        written.append(path)
    return written


RENDER_LOG_NAME = "process_log.json"
RENDER_LOG_MAX = 200


def latest_sweep(out_dir: Path) -> Optional[dict]:
    """The most recent sweep entry from the render log, or None.

    Read back rather than re-derived from AUTO_SWEEP on purpose: a sweep that
    pinned an axis (`auto black=55`) does not use the standard grid, and the
    grid itself may change. The numbers printed on the sheet only mean anything
    against the list that sheet was built from.
    """
    import json
    path = out_dir / RENDER_LOG_NAME
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(entries, dict):
        entries = entries.get("entries", [])
    sweeps = [e for e in entries
              if isinstance(e, dict) and e.get("kind") == "sweep" and e.get("variants")]
    if not sweeps:
        return None
    return max(sweeps, key=lambda e: str(e.get("time", "")))


def record_render(out_dir: Path, entry: dict,
                  max_entries: int = RENDER_LOG_MAX) -> Optional[Path]:
    """Append one render to Iris/<dso>/process_log.json.

    Output filenames are keyed on dso + recipe + flat state only, so every
    re-render at a different stretch overwrites the last one. Without a record,
    a picture on disk cannot tell you what made it — and since the interesting
    parameters are exactly the ones that leave no trace (black, white, soft,
    mesh, scnr), that is the difference between an experiment and a guess.

    Read-append-atomic-replace. Two renders finishing in the same instant on the
    same target could lose one entry; that is worth accepting to avoid a lock
    file in a directory the user browses. Never raises — a lost log entry must
    not fail a render.
    """
    import json
    import os
    path = out_dir / RENDER_LOG_NAME
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except FileNotFoundError:
            existing = []
        except ValueError:
            # Truncated by a crash mid-write, say. Keep it rather than silently
            # dropping the render history on the floor.
            existing = []
            try:
                path.replace(path.with_suffix(".json.bad"))
                _logger.warning("unreadable %s kept as %s.bad", path.name, path.name)
            except OSError:
                pass
        existing.append(entry)
        if len(existing) > max_entries:
            existing = existing[-max_entries:]
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(existing, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, path)
        return path
    except Exception:
        _logger.warning("could not write %s", path, exc_info=True)
        return None


def sweep_link(out_dir: Path, tag: str, opts: dict) -> Optional[dict]:
    """Which panel of the latest sweep these display options are, if any.

    -> {"from_sweep": {"time", "file"}, "panel": n} or None. A render picked
    off a sweep sheet is the owner's choice between alternatives, which is
    exactly the label an automatic stretch is scored against
    (docs/AUTO_STRETCH_PLAN.md). Before this, the link existed only as
    timestamp order in the log.
    """
    entry = latest_sweep(out_dir)
    if entry is None or (entry.get("tag") or entry.get("recipe")) != tag:
        return None

    def same(a, b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)))
        return str(a).lower() == str(b).lower()

    for i, combo in enumerate(entry.get("variants") or [], 1):
        if combo and all(k in opts and same(v, opts[k]) for k, v in combo.items()):
            return {"from_sweep": {"time": entry.get("time"), "file": entry.get("file")},
                    "panel": i}
    return None


def mark_kept(out_dir: Path, recipe: Optional[str] = None) -> Optional[dict]:
    """Stamp the latest render or publish (of *recipe*, if given) as the keeper.

    -> the marked entry, or None if there is none. `keep` is the owner saying
    "this one": the cleanest stretch label there is. Same read-modify-atomic-
    replace as record_render.
    """
    import json
    import os
    path = out_dir / RENDER_LOG_NAME
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(entries, list):
        return None
    want = recipe.upper() if recipe else None
    for e in reversed(entries):
        if not isinstance(e, dict) or e.get("kind") not in ("render", "publish"):
            continue
        if want and want not in (str(e.get("recipe") or "").upper(),
                                 str(e.get("tag") or "").upper()):
            continue
        from datetime import datetime
        e["kept"] = datetime.now().astimezone().isoformat(timespec="seconds")
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entries, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, path)
        return e
    return None


def save_sheet(sheet: np.ndarray, path: Path) -> Path:
    """Write a contact sheet as-is.

    Not save_rgb: that flips vertically for the FITS origin convention, which
    would turn the panel labels upside down — the sheet is already in display
    orientation because its panels were flipped individually.
    """
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet.astype(np.uint8)).save(path, quality=92, optimize=True)
    return path


def save_rgb(rgb: np.ndarray, path: Path, max_px: Optional[int] = None) -> Path:
    """Write an RGB float image (0..1) as a JPEG with no text or furniture.

    Oriented by stacker.sky_parity(): the same row order as every other
    picture product, so a colour render and its channel JPEGs match.
    """
    from PIL import Image
    from stacking import stacker
    img = Image.fromarray(stacker.sky_parity((np.clip(rgb, 0, 1) * 255).astype(np.uint8)))
    if max_px and max(img.size) > max_px:
        ratio = max_px / max(img.size)
        img = img.resize((max(1, int(img.width * ratio)),
                          max(1, int(img.height * ratio))), Image.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Full chroma (4:4:4) at quality 95, not the JPEG default of 4:2:0 at 92.
    # Found 2026-09-25 on a denoised NGC 7380: smooth colour gradients with
    # no grain to hide behind quantise into an 8 px lattice (16 px in the
    # subsampled chroma), visible at high zoom and absent from a lossless
    # save. Raw composites hide the same blocks under their noise. The cost
    # is ~1.6x the file size; a denoised deliverable is the wrong place to
    # save it.
    img.save(path, quality=95, subsampling=0, optimize=True)
    return path
