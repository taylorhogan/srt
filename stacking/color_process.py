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

# Display defaults, tuned on sh2-92. A black point at 2 sigma above sky looked
# clean but clipped the whole faint envelope — only 5.8% of pixels survived into
# the display window, against 98.8% for the ZScale preview the stack command
# uses. 0.8 sigma keeps the outer nebula without the sky breaking into colour
# speckle.
BLACK_SIGMA = 0.8
WHITE_PCT = 99.0
SOFTENING = 0.025
EDGE_CROP = 0.02     # dithered border where not every frame contributed


def _squash(name: str) -> str:
    return "".join(ch for ch in str(name).upper() if ch.isalnum() or ch == "-")


def resolve_filter(wanted: str, available: list[str]) -> Optional[str]:
    """Map a recipe's filter name onto whatever this DSO's frames actually say."""
    targets = {_squash(a) for a in _ALIASES.get(wanted, ())} | {_squash(wanted)}
    for name in available:
        if _squash(name) in targets:
            return name
    return None


def _stretch(chan: np.ndarray, black_sigma: float, white: float) -> np.ndarray:
    """asinh stretch onto 0..1 with a per-channel black and a SHARED white."""
    from astropy.stats import mad_std
    sd = float(mad_std(chan, ignore_nan=True)) or 1.0
    black = black_sigma * sd
    y = np.clip((chan - black) / max(white - black, 1e-6), 0.0, 1.0)
    return np.arcsinh(y / SOFTENING) / np.arcsinh(1.0 / SOFTENING)


def compose(channels: dict[str, np.ndarray], black_sigma: float = BLACK_SIGMA,
            white_pct: float = WHITE_PCT) -> np.ndarray:
    """Combine sky-subtracted channel stacks into an RGB image in 0..1.

    channels holds any of R/G/B plus an optional L. Every channel must already
    be on the same pixel grid — that is what the shared reference guarantees.
    """
    subbed = {k: v - float(np.nanmedian(v)) for k, v in channels.items()}
    colour = [subbed[c] for c in ("R", "G", "B") if c in subbed]
    white = float(np.nanpercentile(np.maximum.reduce(colour), white_pct))
    _logger.info("Compose: shared white point %.2f ADU (p%.1f)", white, white_pct)

    rgb = np.dstack([_stretch(subbed[c], black_sigma, white) for c in ("R", "G", "B")])

    if "L" in subbed:
        # Classic LRGB: keep the colour from RGB, take the brightness from L.
        # Scaling by the ratio preserves hue instead of washing it out, which is
        # what simply averaging L into each channel would do.
        lum = _stretch(subbed["L"], black_sigma, white)
        rgb_lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(rgb_lum > 1e-4, lum / np.maximum(rgb_lum, 1e-4), 0.0)
        rgb = rgb * scale[:, :, None]

    return np.clip(np.nan_to_num(rgb), 0.0, 1.0)


def process_dso(
    dso_dir: Path,
    recipe: str,
    progress_cb: Optional[Callable[[str], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
    scale: int = 1,
    use_flats: bool = True,
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

    rgb = compose(stacks)
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
