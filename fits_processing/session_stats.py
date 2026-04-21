"""
Session-level time-series statistics.

Produces two plots for the current imaging session:
  1. FWHM (arcsec) vs time, dots coloured by filter name
  2. Sky brightness (instr mag/arcsec² or ADU/s) vs time, dots coloured by filter

Speed strategy
--------------
N.I.N.A writes ``HFR`` (half-flux radius, pixels) to the FITS header when
plate-solving / star detection runs during capture.  Reading the header is
~100× faster than re-running Gaussian fitting on every frame.  This module
tries HFR first; only falls back to fitsfwhm.calculate_fwhm when the keyword
is absent.  Sky brightness always reads pixel data (fast sigma-clip, no fitting).
"""
from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.utils.iers import conf
conf.auto_max_age = None
from astropy.utils import iers
iers.conf.auto_download = False

import os, sys
if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from fits_processing import fitsfwhm
from fits_processing import sky_brightness as sb

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
_DARK_BG   = "#0d0d1a"
_DARK_AXES = "#1a1a2e"

# HFR → FWHM conversion for a Gaussian PSF: FWHM = 2√(2 ln 2) σ, HFR ≈ √(2 ln 2) σ
# so FWHM ≈ 2 × HFR.  N.I.N.A's HFR is in pixels.
_HFR_TO_FWHM = 2.0

# Known filter colours (case-insensitive key)
_FILTER_PALETTE: dict[str, str] = {
    "l":         "#e0e0e0",
    "lum":       "#e0e0e0",
    "luminance": "#e0e0e0",
    "r":         "#ff5555",
    "red":       "#ff5555",
    "g":         "#55dd55",
    "green":     "#55dd55",
    "b":         "#5599ff",
    "blue":      "#5599ff",
    "ha":        "#cc2200",
    "h-alpha":   "#cc2200",
    "halpha":    "#cc2200",
    "sii":       "#ffaa00",
    "oiii":      "#00cccc",
    "hb":        "#6688ff",
    "h-beta":    "#6688ff",
    "hbeta":     "#6688ff",
    "clear":     "#bbbbbb",
    "no filter": "#bbbbbb",
}
_EXTRA_COLORS = [
    "#ff9900", "#aa44ff", "#00ff99", "#ff44aa",
    "#44ffee", "#ffee44", "#aa88ff", "#88ffaa",
]


def _filter_color(name: str, dynamic: dict) -> str:
    key = (name or "").strip().lower()
    if key in _FILTER_PALETTE:
        return _FILTER_PALETTE[key]
    if key not in dynamic:
        dynamic[key] = _EXTRA_COLORS[len(dynamic) % len(_EXTRA_COLORS)]
    return dynamic[key]


def _apply_dark_theme(fig, *axes):
    fig.patch.set_facecolor(_DARK_BG)
    for ax in axes:
        ax.set_facecolor(_DARK_AXES)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")


# ------------------------------------------------------------------
# Per-frame analysis
# ------------------------------------------------------------------

def _analyse_frame(fits_path: Path, arcsec_per_pixel: float) -> Optional[dict]:
    """
    Return a dict with per-frame metrics, or None on failure.

    Keys: time, filter, fwhm_arcsec, sky_adu_per_s, sky_mag_arcsec2
    """
    try:
        with fits.open(fits_path) as hdul:
            hdr = hdul[0].header

        # --- Timestamp ---
        date_obs = hdr.get("DATE-OBS", None)
        if date_obs:
            try:
                obs_dt = datetime.fromisoformat(date_obs.rstrip("Z"))
            except ValueError:
                obs_dt = datetime.fromtimestamp(fits_path.stat().st_mtime)
        else:
            obs_dt = datetime.fromtimestamp(fits_path.stat().st_mtime)

        filter_name = str(hdr.get("FILTER", "Unknown")).strip()

        # --- FWHM: prefer header HFR (fast), fall back to full fitting ---
        fwhm_arcsec: Optional[float] = None
        hfr = hdr.get("HFR", None)
        if hfr is not None:
            try:
                fwhm_arcsec = float(hfr) * _HFR_TO_FWHM * arcsec_per_pixel
            except (ValueError, TypeError):
                pass

        if fwhm_arcsec is None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, mean_arcsec, star_count, _ = fitsfwhm.calculate_fwhm(
                        fits_path, arcsec_per_pixel=arcsec_per_pixel
                    )
                if star_count > 0:
                    fwhm_arcsec = float(mean_arcsec)
            except Exception:
                pass

        # --- Sky brightness ---
        sky = sb.measure_sky(fits_path, arcsec_per_pixel=arcsec_per_pixel)

        return {
            "time":            obs_dt,
            "filter":          filter_name,
            "fwhm_arcsec":     fwhm_arcsec,
            "sky_adu_per_s":   sky["sky_adu_per_s"]         if sky else None,
            "sky_mag_arcsec2": sky.get("sky_mag_arcsec2")   if sky else None,
        }

    except Exception as exc:
        print(f"session_stats: skipping {fits_path.name}: {exc}")
        return None


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def load_session_frames(
    image_dir: Path,
    since_dt: datetime,
    arcsec_per_pixel: float = 0.26,
    max_workers: int = 8,
) -> list[dict]:
    """
    Scan *image_dir* for FITS files modified since *since_dt*, analyse each
    in parallel, and return a list of frame dicts sorted by observation time.
    """
    image_dir = Path(image_dir)
    since_ts  = since_dt.timestamp()

    try:
        fits_files = sorted(
            (f for f in image_dir.rglob("*.fits") if f.stat().st_mtime >= since_ts),
            key=lambda f: f.stat().st_mtime,
        )
    except Exception as exc:
        print(f"session_stats: scan failed for {image_dir}: {exc}")
        return []

    if not fits_files:
        return []

    results: list[dict] = []
    n = min(max_workers, len(fits_files))
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = {pool.submit(_analyse_frame, f, arcsec_per_pixel): f for f in fits_files}
        for fut in as_completed(futures):
            data = fut.result()
            if data is not None:
                results.append(data)

    results.sort(key=lambda d: d["time"])
    return results


def save_session_plots(
    frames: list[dict],
    fwhm_path: Path,
    sky_path: Path,
) -> tuple[Path, Path]:
    """
    Write two JPEG time-series plots and return their paths.

    ``fwhm_path`` — FWHM (arcsec) over time, coloured by filter.
    ``sky_path``  — Sky brightness over time, coloured by filter.
                    Y axis is instr mag/arcsec² (inverted: darker sky at top)
                    when gain data are available, otherwise ADU/s.
    """
    fwhm_path = Path(fwhm_path)
    sky_path  = Path(sky_path)
    dyn: dict[str, str] = {}   # dynamic colour map for unknown filters

    # ----------------------------------------------------------------
    # Plot 1 — FWHM over time
    # ----------------------------------------------------------------
    fwhm_frames = [d for d in frames if d["fwhm_arcsec"] is not None]

    fig, ax = plt.subplots(figsize=(12, 5))
    _apply_dark_theme(fig, ax)

    if fwhm_frames:
        seen: dict[str, str] = {}
        for d in fwhm_frames:
            f     = d["filter"]
            color = _filter_color(f, dyn)
            kw    = dict(color=color, s=35, zorder=3, alpha=0.85)
            if f not in seen:
                seen[f] = color
                ax.scatter(d["time"], d["fwhm_arcsec"], label=f, **kw)
            else:
                ax.scatter(d["time"], d["fwhm_arcsec"], **kw)

        median_fwhm = float(np.median([d["fwhm_arcsec"] for d in fwhm_frames]))
        ax.axhline(median_fwhm, color="#888888", linewidth=1,
                   linestyle="--", alpha=0.6, zorder=2)
        ax.text(
            fwhm_frames[0]["time"], median_fwhm,
            f'  median {median_fwhm:.2f}"',
            color="#aaaaaa", va="bottom", fontsize=8,
        )

        legend = ax.legend(facecolor=_DARK_AXES, labelcolor="white",
                           framealpha=0.8, fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.set_ylabel('FWHM (arcsec)"', color="white")
        ax.set_title(
            f'Session FWHM  |  {len(fwhm_frames)} frames  |  '
            f'median {median_fwhm:.2f}"',
            color="white",
        )
    else:
        ax.text(0.5, 0.5, "No frames with detected stars",
                transform=ax.transAxes, ha="center", va="center", color="#aaaaaa")
        ax.set_title("Session FWHM  |  no data", color="white")

    ax.set_xlabel("Observation time (UTC)", color="white")
    ax.grid(True, color="#333355", alpha=0.5, linewidth=0.6)
    fwhm_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(fwhm_path, format="jpeg", dpi=150,
                bbox_inches="tight", facecolor=_DARK_BG)
    plt.close(fig)

    # ----------------------------------------------------------------
    # Plot 2 — Sky brightness over time
    # ----------------------------------------------------------------
    use_mag   = any(d.get("sky_mag_arcsec2") is not None for d in frames)
    sky_key   = "sky_mag_arcsec2" if use_mag else "sky_adu_per_s"
    y_label   = "Instr mag/arcsec²  (higher = darker sky)" if use_mag else "Sky ADU/s  (lower = darker)"
    sky_frames = [d for d in frames if d.get(sky_key) is not None]

    fig, ax = plt.subplots(figsize=(12, 5))
    _apply_dark_theme(fig, ax)

    if sky_frames:
        seen = {}
        for d in sky_frames:
            f     = d["filter"]
            color = _filter_color(f, dyn)
            kw    = dict(color=color, s=35, zorder=3, alpha=0.85)
            if f not in seen:
                seen[f] = color
                ax.scatter(d["time"], d[sky_key], label=f, **kw)
            else:
                ax.scatter(d["time"], d[sky_key], **kw)

        median_sky = float(np.median([d[sky_key] for d in sky_frames]))
        ax.axhline(median_sky, color="#888888", linewidth=1,
                   linestyle="--", alpha=0.6, zorder=2)
        ax.text(
            sky_frames[0]["time"], median_sky,
            f"  median {median_sky:.2f}",
            color="#aaaaaa", va="bottom", fontsize=8,
        )

        if use_mag:
            ax.invert_yaxis()   # convention: fainter (larger mag) = better = top

        ax.legend(facecolor=_DARK_AXES, labelcolor="white",
                  framealpha=0.8, fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.set_ylabel(y_label, color="white")
        ax.set_title(
            f'Session sky brightness  |  {len(sky_frames)} frames  |  '
            f'median {median_sky:.2f}',
            color="white",
        )
    else:
        ax.text(0.5, 0.5, "No sky brightness data",
                transform=ax.transAxes, ha="center", va="center", color="#aaaaaa")
        ax.set_title("Session sky brightness  |  no data", color="white")

    ax.set_xlabel("Observation time (UTC)", color="white")
    ax.grid(True, color="#333355", alpha=0.5, linewidth=0.6)
    sky_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(sky_path, format="jpeg", dpi=150,
                bbox_inches="tight", facecolor=_DARK_BG)
    plt.close(fig)

    return fwhm_path, sky_path
