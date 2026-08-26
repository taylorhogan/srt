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


def gather_dso_frames(
    dso_dir: Path,
    latest_session_only: bool = True,
    session_gap_hours: float = 6.0,
) -> list:
    """Return a DSO's cached per-frame stats, reconciled with disk and ordered.

    Reads ``<dso_dir>/frame_stats.json``, drops entries whose FITS file no longer
    exists, and sorts the survivors chronologically by **file mtime** (the real
    capture time — the DATE-OBS header can be bogus, e.g. an unsynced camera
    clock stamping frames ``2017-12-20``).

    When ``latest_session_only`` is True (the default) only the most recent
    observing session is returned: frames are cut at the last gap in mtime larger
    than ``session_gap_hours``.  Without this, the per-DSO cache accumulates every
    night the target was ever imaged (plus any junk dumped into its LIGHT folder),
    and the frame-count x-axis lets the biggest day dominate the whole plot.

    Both the eager imaging-card render and the ``stats`` command use this so the
    two can't drift.
    """
    cache_path = Path(dso_dir) / "frame_stats.json"
    if not cache_path.exists():
        return []
    try:
        with open(cache_path) as f:
            entries = json.load(f)
    except Exception:
        _logger.exception("gather_dso_frames: failed to read %s", cache_path)
        return []
    if not isinstance(entries, list):
        return []

    dso_dir = Path(dso_dir).resolve()
    on_disk: list[tuple[float, dict]] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        p = e.get("path")
        if not p:
            continue
        fp = Path(p)
        # Defend against cross-target contamination: a DSO's cache must only
        # contain frames physically under that DSO's folder. (Caches have been
        # seen mixing multiple targets, e.g. HAT-P-32 frames in ngc5907's cache.)
        try:
            if not fp.resolve().is_relative_to(dso_dir):
                continue
        except (OSError, ValueError):
            continue
        try:
            mtime = fp.stat().st_mtime
        except OSError:
            continue  # FITS no longer on disk — drop the stale cache entry
        on_disk.append((mtime, e))
    on_disk.sort(key=lambda t: t[0])

    if latest_session_only and len(on_disk) > 1:
        gap = session_gap_hours * 3600.0
        start = 0
        for i in range(len(on_disk) - 1, 0, -1):
            if on_disk[i][0] - on_disk[i - 1][0] > gap:
                start = i
                break
        on_disk = on_disk[start:]

    return [e for _, e in on_disk]


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
    latest_session_only: bool = True,
) -> Optional[Path]:
    """Render the multi-panel stats plot from a frame_stats.json cache.

    *latest_session_only* mirrors the ``stats`` command: True is plain
    ``stats <dso>``, False is ``stats <dso> all``. It is passed through rather
    than fixed here because the two callers genuinely want different things --
    the web-chat card is about the night in progress, while the live page shows
    the target's whole history.

    Returns output_path on success, None if cache missing/empty or no frames
    had detectable stars (matches save_stats_plot_from_cache's contract).
    """
    from fits_processing import fitsfwhm

    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    # Reconcile against disk + sort chronologically so the card matches the
    # fresh `stats` tile (raw cache order can put a day out of place).
    frames = gather_dso_frames(cache_path.parent,
                               latest_session_only=latest_session_only)
    if not frames:
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
