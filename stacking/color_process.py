"""Multi-filter colour processing: stack each channel, combine, stretch.

Backs the webchat ``process <dso> <recipe>`` command. Recipes:

    LRGB  R->R  G->G  B->B, with L substituted as luminance
    HOO   Ha->R, O-III->G and B      (H-alpha region palette)
    SHO   S-II->R, Ha->G, O-III->B   (the "Hubble" palette)

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
from pathlib import Path
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
}

# FITS FILTER values vary by capture software; match on a squashed form.
_ALIASES: dict[str, tuple[str, ...]] = {
    "Ha":    ("HA", "HALPHA", "H-ALPHA", "HALFA"),
    "O-III": ("OIII", "O3", "O-III", "OXYGEN3"),
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


def _squash(name: str) -> str:
    return "".join(ch for ch in str(name).upper() if ch.isalnum() or ch == "-")


def resolve_filter(wanted: str, available: list[str]) -> Optional[str]:
    """Map a recipe's filter name onto whatever this DSO's frames actually say."""
    targets = {_squash(a) for a in _ALIASES.get(wanted, ())} | {_squash(wanted)}
    for name in available:
        if _squash(name) in targets:
            return name
    return None


def _stretch(chan: np.ndarray, black_pct: float, white: float,
             softening: float = SOFTENING) -> np.ndarray:
    """asinh stretch onto 0..1 with a per-channel black and a SHARED white.

    black_pct is a percentile of this channel, so the same share of pixels goes
    to black on any target — see BLACK_PCT.
    """
    black = float(np.nanpercentile(chan, black_pct))
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


def compose(channels: dict[str, np.ndarray], black_pct: float = BLACK_PCT,
            white_pct: float = WHITE_PCT,
            subtract_background: bool = SUBTRACT_BACKGROUND,
            softening: float = SOFTENING,
            mesh: int = BG_MESH_FRACTION) -> np.ndarray:
    """Combine channel stacks into an RGB image in 0..1.

    channels holds any of R/G/B plus an optional L. Every channel must already
    be on the same pixel grid — that is what the shared reference guarantees.
    """
    if subtract_background:
        subbed = {k: _remove_gradient(v, mesh) for k, v in channels.items()}
    else:
        subbed = {k: v - float(np.nanmedian(v)) for k, v in channels.items()}
    colour = [subbed[c] for c in ("R", "G", "B") if c in subbed]
    white = float(np.nanpercentile(np.maximum.reduce(colour), white_pct))
    _logger.info("Compose: shared white point %.2f ADU (p%.1f)", white, white_pct)

    rgb = np.dstack([_stretch(subbed[c], black_pct, white, softening)
                     for c in ("R", "G", "B")])

    if "L" in subbed:
        # Classic LRGB: keep the colour from RGB, take the brightness from L.
        # Scaling by the ratio preserves hue instead of washing it out, which is
        # what simply averaging L into each channel would do.
        lum = _stretch(subbed["L"], black_pct, white, softening)
        rgb_lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(rgb_lum > 1e-4, lum / np.maximum(rgb_lum, 1e-4), 0.0)
        rgb = rgb * np.clip(scale, 0.0, MAX_LUM_BOOST)[:, :, None]

    return np.clip(np.nan_to_num(rgb), 0.0, 1.0)


def channel_cache_path(cache_dir: Path, dso: str, recipe: str, chan: str) -> Path:
    return cache_dir / f"channels_{dso}_{recipe}_{chan}.npy"


def load_cached_channels(cache_dir: Path, dso: str, recipe: str) -> Optional[dict]:
    """Return the stacked channels from a previous run, or None if incomplete.

    Stacking is ~17 minutes and the stretch is a second; caching the channels is
    what makes the display parameters worth exposing at all, because otherwise
    every tweak costs a re-stack.
    """
    mapping = RECIPES.get(recipe.upper())
    if mapping is None:
        return None
    out = {}
    for chan in mapping:
        f = channel_cache_path(cache_dir, dso, recipe.upper(), chan)
        if not f.exists():
            if chan == "L":          # optional
                continue
            return None
        out[chan] = np.load(f)
    return out or None


def process_dso(
    dso_dir: Path,
    recipe: str,
    progress_cb: Optional[Callable[[str], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
    scale: int = 1,
    use_flats: bool = True,
    cache_dir: Optional[Path] = None,
    reuse: bool = False,
    **compose_kw,
) -> tuple[np.ndarray, dict]:
    """Stack every filter a recipe needs and return (rgb 0..1, info).

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
    if reuse and cache_dir is not None:
        cached = load_cached_channels(cache_dir, dso_dir.name, recipe)
        if cached is not None:
            if progress_cb:
                progress_cb(f"reusing cached {recipe} channels "
                            f"({', '.join(sorted(cached))}) — no re-stack")
            rgb = compose(cached, **compose_kw)
            return rgb, {"recipe": recipe, "reused": True,
                         "channels": {c: c for c in cached}, "frames": {},
                         "reference": "(cached)", "shape": rgb.shape[:2],
                         "flats": use_flats}
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
    if missing and progress_cb:
        progress_cb(f"missing {', '.join(missing)} — continuing without")

    # One reference for every filter. Prefer the channel with the most frames,
    # since more frames means a better chance of a genuinely sharp one.
    ref_filter = max(set(resolved.values()), key=lambda f: len(by_filter[f]))
    ref_paths = by_filter[ref_filter]
    arcsec = stacker._get_arcsec_per_pixel()
    from cmd_processing.super_user_commands import _load_precomputed_fwhm_stars
    ref_pre = _load_precomputed_fwhm_stars(dso_dir, ref_paths, arcsec)
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
    for chan in ("R", "G", "B", "L"):
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
        pre = _load_precomputed_fwhm_stars(dso_dir, paths, arcsec)
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

    # Under a shared reference stack() skips its per-filter coverage crop, so
    # every channel is on the identical grid. Trimming to a common size from the
    # corner would silently mis-align them if that ever stopped being true.
    shapes = {v.shape for v in stacks.values()}
    if len(shapes) != 1:
        raise ValueError(f"channels are on different grids: {shapes} — "
                         "they cannot be combined without re-registration")
    h, w = shapes.pop()
    m = int(EDGE_CROP * min(h, w))
    if m > 0:
        stacks = {k: v[m:-m, m:-m] for k, v in stacks.items()}

    if cache_dir is not None:
        for c, v in stacks.items():
            try:
                np.save(channel_cache_path(cache_dir, dso_dir.name, recipe, c),
                        v.astype(np.float32))
            except Exception:
                _logger.warning("could not cache channel %s", c, exc_info=True)

    rgb = compose(stacks, **compose_kw)
    info = {
        "recipe": recipe,
        "flats": use_flats,
        "channels": {c: resolved[c] for c in resolved},
        "frames": used,
        "reference": ref_path.name,
        "shape": rgb.shape[:2],
    }
    return rgb, info


def save_rgb(rgb: np.ndarray, path: Path, max_px: Optional[int] = None) -> Path:
    """Write an RGB float image (0..1) as a JPEG with no text or furniture."""
    from PIL import Image
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[::-1])
    if max_px and max(img.size) > max_px:
        ratio = max_px / max(img.size)
        img = img.resize((max(1, int(img.width * ratio)),
                          max(1, int(img.height * ratio))), Image.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=92, optimize=True)
    return path
