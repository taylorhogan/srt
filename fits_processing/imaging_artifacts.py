"""Eagerly-built imaging artifacts: annotated latest-frame JPEG and stats graph.

Both pieces are rendered by the frame_watcher right after each new sub is
analysed, so the web UI can show them without doing a synchronous render on
every card click.  The web server also calls into these helpers for its
on-demand `/api/latest_image` endpoint, so the rendering logic lives in one
place.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

_logger = logging.getLogger(__name__)


LATEST_IMAGE_NAME = "latest_imaging.jpg"
STATS_PLOT_NAME = "imaging_stats.jpg"


def render_latest_image(
    image_dir: Path,
    arcsec_per_pixel: float,
    output_path: Path,
) -> Optional[Path]:
    """Render the newest LIGHT FITS under image_dir to an annotated JPEG.

    Returns the output path on success, None if no FITS were found.
    """
    from fits_processing import fitstojpg, fitsfwhm, sky_brightness as sb

    latest_fits = fitstojpg.get_latest_file(str(image_dir), "fits")
    if latest_fits is None:
        return None

    filter_name = None
    try:
        from astropy.io import fits as _fits
        with _fits.open(str(latest_fits)) as hdul:
            filter_name = str(hdul[0].header.get("FILTER", "")).strip() or None
    except Exception:
        _logger.exception("render_latest_image: header read failed for %s", latest_fits)

    try:
        sky_data = sb.measure_sky(Path(str(latest_fits)), arcsec_per_pixel=arcsec_per_pixel)
    except Exception:
        _logger.exception("render_latest_image: sky measure failed")
        sky_data = None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fitsfwhm.save_fwhm(
        Path(str(latest_fits)), output_path,
        arcsec_per_pixel=arcsec_per_pixel,
        annotate=False,
        filter_name=filter_name,
        sky_data=sky_data,
    )
    return output_path


def render_stats_plot_from_cache_path(
    cache_path: Path,
    output_path: Path,
) -> Optional[Path]:
    """Render the multi-panel stats plot from a frame_stats.json cache.

    Returns output_path on success, None if cache missing/empty or no frames
    had detectable stars (matches save_stats_plot_from_cache's contract).
    """
    from fits_processing import fitsfwhm

    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    try:
        with open(cache_path) as f:
            frames = json.load(f)
    except Exception:
        _logger.exception("render_stats_plot: failed to read %s", cache_path)
        return None
    if not isinstance(frames, list) or not frames:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plot_path, frames_with_stars = fitsfwhm.save_stats_plot_from_cache(
            frames, output_path,
        )
    except Exception:
        _logger.exception("render_stats_plot: plot render failed")
        return None
    if frames_with_stars == 0:
        return None
    return Path(plot_path)
