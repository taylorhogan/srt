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
"""

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np

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


def _remove_gradient(chan: np.ndarray, mesh: int = BG_MESH_FRACTION) -> np.ndarray:
    """Subtract a coarse 2-D background model — vignetting and sky gradient.

    Falls back to a flat median subtraction if photutils is unavailable, which
    leaves the gradient in but never makes things worse.
    """
    try:
        from astropy.stats import SigmaClip
        from photutils.background import Background2D, SExtractorBackground
        box = max(64, min(chan.shape) // max(1, mesh))
        bkg = Background2D(
            chan, box_size=box, filter_size=3,
            sigma_clip=SigmaClip(sigma=3.0), bkg_estimator=SExtractorBackground(),
        )
        return chan - bkg.background
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


def apply_white_balance(subbed: dict[str, np.ndarray], mode) -> tuple[dict, dict]:
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
STAR_CHROMA_FLOOR = 0.04        # RGB luminance (0..1) below which the palette keeps its colour


def star_mask(chan: np.ndarray, thresh_sigma: float = STAR_MASK_THRESH_SIGMA,
              max_npix: int = STAR_MASK_MAX_NPIX) -> tuple[np.ndarray, dict]:
    """(mask 0..1, info) — feathered discs on the stars of *chan*.

    *chan* is a linear stack (background-subtracted or not; sep models the
    background itself). Sources are kept if round, not too large and not on
    the edge; the disc radius follows sep's isophotal size, so a bright star
    with a wide profile gets a wider disc than a faint one.
    """
    import sep
    a = np.ascontiguousarray(np.nan_to_num(chan), dtype=np.float32)
    h, w = a.shape
    mask = np.zeros((h, w), np.float32)
    try:
        bkg = sep.Background(a)
        data = a - bkg.back()
        rms = float(bkg.globalrms) or 1.0
        objs = sep.extract(data, thresh_sigma, err=rms, minarea=5)
    except Exception as exc:  # noqa: BLE001
        return mask, {"stars": 0, "why": "star detection failed: %s" % exc}
    keep = (objs["b"] / np.maximum(objs["a"], 1e-6)) >= STAR_MASK_MIN_AXIS_RATIO
    keep &= objs["npix"] <= max_npix
    objs = objs[keep]
    yy_full, xx_full = None, None
    n = 0
    for o in objs:
        r = float(np.clip(STAR_MASK_RADIUS_A * float(o["a"]),
                          STAR_MASK_MIN_RADIUS_PX, STAR_MASK_MAX_RADIUS_PX))
        sig = STAR_MASK_FEATHER * r
        reach = int(np.ceil(r + 3.0 * sig))
        cx, cy = float(o["x"]), float(o["y"])
        x0, x1 = max(0, int(cx) - reach), min(w, int(cx) + reach + 1)
        y0, y1 = max(0, int(cy) - reach), min(h, int(cy) + reach + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.hypot(yy - cy, xx - cx)
        disc = np.where(d <= r, 1.0, np.exp(-0.5 * ((d - r) / sig) ** 2)).astype(np.float32)
        np.maximum(mask[y0:y1, x0:x1], disc, out=mask[y0:y1, x0:x1])
        n += 1
    return mask, {"stars": n, "detected": int(len(keep)), "kept": int(keep.sum()),
                  "coverage": float(mask.mean())}


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
             white_balance: str = WHITE_BALANCE) -> tuple[dict, float]:
    """Background-subtract every channel and find the shared white point.

    Factored out of compose() because the per-channel exports have to be the
    exact same pixels on the exact same scale — a channel JPEG rendered against
    its own white point would look nothing like its contribution to the
    composite, which defeats the point of being able to inspect it.
    """
    if subtract_background:
        subbed = {k: _remove_gradient(v, mesh) for k, v in channels.items()}
    else:
        subbed = {k: v - float(np.nanmedian(v)) for k, v in channels.items()}
    # Balance BEFORE the Ha blend: the excess is measured against R and added
    # to R, so scaling R first changes the ratio and the subtraction together
    # and the excess itself is unchanged; balancing afterwards would rescale it.
    subbed, _ = apply_white_balance(subbed, white_balance)
    # HALRGB: the blend happens here, on subtracted linear data, so the
    # shared white point and every per-channel export see the blended R.
    # ha_gain=None leaves it to the caller (the movie pins one ratio for
    # every depth); 0 adds only the excess map, for inspection.
    if ha_gain is not None:
        subbed, _ = apply_ha_blend(subbed, ha_gain)
    return subbed, shared_white(subbed, white_pct)



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


def sky_noise_model(chan: np.ndarray, bin_factor: int = AUTO_BIN) -> dict:
    """Sky level, per-pixel and binned noise, and the faint/bright signal
    fractions of one background-subtracted channel."""
    import sep
    from math import erf, sqrt

    def phi(v):
        return 0.5 * (1.0 + erf(v / sqrt(2.0)))

    a = np.ascontiguousarray(chan.astype(np.float32))
    binned = np.ascontiguousarray(_block_mean(a, bin_factor).astype(np.float32))
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


def effective_options(**overrides) -> dict:
    """Every compose() display knob with its default filled in.

    The command parser only carries the options the user actually typed, so a
    record built from it says nothing about the settings that did most of the
    work. Reproducing a render means knowing all of them.
    """
    opts = {"black_pct": BLACK_PCT, "white_pct": WHITE_PCT,
            "softening": SOFTENING, "mesh": BG_MESH_FRACTION,
            "subtract_background": SUBTRACT_BACKGROUND, "scnr": SCNR_AMOUNT,
            "ha_gain": HA_GAIN, "white_balance": WHITE_BALANCE}
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


def compose(channels: dict[str, np.ndarray], black_pct: float = BLACK_PCT,
            white_pct: float = WHITE_PCT,
            subtract_background: bool = SUBTRACT_BACKGROUND,
            softening: float = SOFTENING,
            mesh: int = BG_MESH_FRACTION,
            scnr: float = SCNR_AMOUNT,
            ha_gain: float = HA_GAIN,
            white_balance: str = WHITE_BALANCE) -> np.ndarray:
    """Combine channel stacks into an RGB image in 0..1.

    channels holds any of R/G/B plus an optional L, and for HALRGB an HA plane
    that _prepare folds into R and L. Every channel must already be on the
    same pixel grid — that is what the shared reference guarantees.
    """
    subbed, white = _prepare(channels, subtract_background, mesh, white_pct, ha_gain,
                             white_balance)
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

    return np.clip(np.nan_to_num(rgb), 0.0, 1.0)


def sky_anchored_blacks(subbed: dict[str, np.ndarray], black_sigma: float,
                        black_pct: float = BLACK_PCT, log=None) -> dict[str, float]:
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
    """
    if black_sigma is None or black_sigma <= 0:
        return blacks_from(subbed, black_pct)
    import sep
    out = {}
    for c, arr in subbed.items():
        if c in DERIVED_PLANES:
            continue
        a = np.ascontiguousarray(arr.astype(np.float32))
        try:
            sig = float(sep.Background(a).globalrms)
            med = float(np.nanmedian(a))
        except Exception:
            out[c] = float(np.nanpercentile(arr, black_pct))
            continue
        out[c] = med - black_sigma * sig
        if log:
            pct = float(np.mean(a <= out[c]) * 100.0)
            log(f"    {c}: sky {med:+.3f} sigma {sig:.3f} -> black {out[c]:+.3f} ADU "
                f"({pct:.0f}% of pixels, vs {black_pct:.0f}% under the old rule)")
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
                      white_balance: str = WHITE_BALANCE) -> list[Path]:
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
    subbed, white = _prepare(channels, subtract_background, mesh, white_pct, ha_gain,
                             white_balance)
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
