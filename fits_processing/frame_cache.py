"""Read-side helpers for the per-DSO frame_stats.json cache.

This module exists because `_load_precomputed_fwhm_stars` lived as a PRIVATE
function inside cmd_processing/super_user_commands.py while being imported by
`stacking/color_process.py` and `nn/stacks.py` -- leaf-level science packages
reaching up into the 5,500-line command module for an underscore-prefixed
name. That inverted the dependency direction (science must never depend on the
command layer) and coupled two validated pipelines to the internals of a file
scheduled for dismantling.

The cache itself is written by fits_processing/frame_watcher.py and the
`stats`/`bad` commands; putting its reader beside its writers is the honest
home. (Architecture plan: this becomes iris/science/ alongside the rest of
fits_processing.)
"""
import json
import logging
import os
from pathlib import Path

_logger = logging.getLogger(__name__)


def load_precomputed_fwhm_stars(
    dso_dir: Path, paths: list[Path], arcsec_per_pixel: float
) -> dict[Path, tuple[float, int]]:
    """Build a {path: (fwhm_px, star_count)} map from a DSO's frame_stats.json.

    Lets the stacker reuse the FWHM/star measurements already cached by the
    `stats`/`bad` commands instead of redoing the detection pass. The cache
    stores FWHM in arcseconds; the stacker works in pixels, so convert with
    *arcsec_per_pixel*. Only paths present in *paths* are returned; a missing,
    unreadable, or value-less cache yields an empty map (stacker measures
    normally). Matching is by normalised absolute path so cache keys written on
    a different run still line up.
    """
    cache_path = Path(dso_dir) / "frame_stats.json"
    if not cache_path.exists() or not arcsec_per_pixel:
        return {}
    try:
        with open(cache_path) as fh:
            rows = json.load(fh)
    except Exception:
        _logger.warning("could not read %s", cache_path, exc_info=True)
        return {}
    by_norm = {
        os.path.normcase(os.path.abspath(r["path"])): r
        for r in rows
        if isinstance(r, dict) and r.get("path")
    }
    out: dict[Path, tuple[float, int]] = {}
    for p in paths:
        r = by_norm.get(os.path.normcase(os.path.abspath(str(p))))
        if not r:
            continue
        fa = r.get("fwhm_arcsec")
        fwhm_px = float(fa) / arcsec_per_pixel if fa else 0.0
        out[p] = (fwhm_px, int(r.get("star_count") or 0))
    return out
