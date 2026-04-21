"""
Sky brightness measurement from FITS frames.

Strategy — corner subframes
---------------------------
The centre of a DSO image often contains the target (nebula/galaxy), which
would contaminate a whole-frame background estimate.  Instead we sample four
corner subframes (each ``CORNER_FRAC`` × image size) where only sky is present,
then take the sigma-clipped median across those regions.

Primary metric
--------------
``sky_adu_per_s`` — median sky ADU normalised by exposure time.
Lower = darker sky.  Directly comparable between frames regardless of
integration time.

If the FITS header carries GAIN (e⁻/ADU) the module also reports
``sky_e_per_s_arcsec2`` (electrons/s/arcsec²) and an *instrumental*
``sky_mag_arcsec2`` = -2.5 × log₁₀(sky_e/s/arcsec²).

To convert the instrumental magnitude to an absolute mag/arcsec² you need
the photometric zero-point for your specific camera+telescope system (set
``cfg["sky"]["zero_point"]`` once you have a calibration measurement).
Until then the instrumental value is still useful for night-to-night
comparison — lower number = brighter sky (worse).

Gradient analysis
-----------------
The frame is divided into an 8 × 8 grid and each cell's sky background is
estimated independently.  The RMS and peak-to-peak of the cell medians
quantify the sky gradient (e.g. a nearby light-dome pulling one side bright).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ZScaleInterval
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from astropy.utils.iers import conf
conf.auto_max_age = None
from astropy.utils import iers
iers.conf.auto_download = False

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
_CORNER_FRAC = 0.18      # fraction of each axis used per corner subframe
_GRID_N      = 8         # N × N cells for gradient map
_SIGMA       = 3.0       # sigma for sigma-clipped stats

_DARK_BG   = "#0d0d1a"
_DARK_AXES = "#1a1a2e"


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _corner_subframes(data: np.ndarray) -> list[np.ndarray]:
    """Return the four corner crops of *data*."""
    h, w = data.shape
    rh = max(1, int(h * _CORNER_FRAC))
    rw = max(1, int(w * _CORNER_FRAC))
    return [
        data[:rh,  :rw ],   # bottom-left  (origin = lower in FITS)
        data[:rh,  -rw:],   # bottom-right
        data[-rh:, :rw ],   # top-left
        data[-rh:, -rw:],   # top-right
    ]


def _sky_from_subframes(subframes: list[np.ndarray]) -> tuple[float, float]:
    """Return (sky_median_adu, sky_std_adu) from a list of corner arrays."""
    medians = []
    for sf in subframes:
        _, med, _ = sigma_clipped_stats(sf, sigma=_SIGMA)
        medians.append(med)
    # Median-of-medians is robust against one bad corner (e.g., satellite trail)
    return float(np.median(medians)), float(np.std(medians))


def _cell_sky(data: np.ndarray) -> np.ndarray:
    """Return an (_GRID_N × _GRID_N) array of per-cell sky medians."""
    h, w = data.shape
    cell_h = h // _GRID_N
    cell_w = w // _GRID_N
    grid = np.full((_GRID_N, _GRID_N), np.nan)
    for r in range(_GRID_N):
        for c in range(_GRID_N):
            r0, r1 = r * cell_h, (r + 1) * cell_h
            c0, c1 = c * cell_w, (c + 1) * cell_w
            cell = data[r0:r1, c0:c1]
            _, med, _ = sigma_clipped_stats(cell, sigma=_SIGMA)
            grid[r, c] = med
    return grid


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
# Public API
# ------------------------------------------------------------------

def measure_sky(
    fits_path: Path,
    arcsec_per_pixel: float = 0.26,
) -> Optional[dict]:
    """
    Measure sky background from a FITS frame using corner subframes.

    Parameters
    ----------
    fits_path : Path
        Path to the FITS file.
    arcsec_per_pixel : float
        Plate scale in arcsec / pixel.

    Returns
    -------
    dict with keys:
        sky_adu             — median sky signal (ADU, offset-corrected)
        sky_adu_per_s       — sky_adu / exptime  (primary comparison metric)
        sky_adu_per_s_arcsec2 — further divided by plate_scale²
        sky_e_per_s_arcsec2 — in electrons (None if GAIN absent in header)
        sky_mag_arcsec2     — instrumental mag/arcsec² (None if GAIN absent)
        sky_gradient_rms    — RMS of per-cell sky medians (ADU)
        sky_gradient_ptp    — peak-to-peak of per-cell sky medians (ADU)
        exptime             — exposure time in seconds
        gain                — gain e⁻/ADU from header (None if absent)
        offset              — camera offset ADU from header (0 if absent)
        cell_grid           — (_GRID_N × _GRID_N) ndarray of cell sky medians (ADU)
    Returns None on failure.
    """
    fits_path = Path(fits_path)
    try:
        with fits.open(fits_path) as hdul:
            hdr = hdul[0].header
            raw = hdul[0].data.astype(float)
    except Exception as exc:
        print(f"sky_brightness: failed to open {fits_path.name}: {exc}")
        return None

    data = np.squeeze(raw)
    if data.ndim != 2:
        print(f"sky_brightness: unexpected image shape {data.shape} in {fits_path.name}")
        return None

    # --- FITS header values ---
    exptime = float(hdr.get("EXPTIME", hdr.get("EXPOSURE", 1.0)))
    gain    = hdr.get("GAIN", hdr.get("EGAIN", None))
    gain    = float(gain) if gain is not None else None
    offset  = float(hdr.get("OFFSET", 0.0))

    # --- Sky background from corner subframes ---
    corners = _corner_subframes(data)
    sky_adu_raw, corner_std = _sky_from_subframes(corners)
    sky_signal = max(sky_adu_raw - offset, 0.0)   # offset-corrected

    # --- Derived metrics ---
    sky_adu_per_s          = sky_signal / max(exptime, 1e-6)
    plate_scale_sq         = arcsec_per_pixel ** 2
    sky_adu_per_s_arcsec2  = sky_adu_per_s / plate_scale_sq

    sky_e_per_s_arcsec2: Optional[float] = None
    sky_mag_arcsec2:     Optional[float] = None
    if gain is not None and sky_adu_per_s_arcsec2 > 0:
        sky_e_per_s_arcsec2 = sky_adu_per_s_arcsec2 * gain
        if sky_e_per_s_arcsec2 > 0:
            sky_mag_arcsec2 = -2.5 * math.log10(sky_e_per_s_arcsec2)

    # --- Sky gradient across frame ---
    cell_grid = _cell_sky(data) - offset           # offset-correct each cell
    valid = cell_grid[np.isfinite(cell_grid)]
    sky_gradient_rms = float(np.std(valid))   if len(valid) else 0.0
    sky_gradient_ptp = float(np.ptp(valid))   if len(valid) else 0.0

    return {
        "sky_adu":               float(sky_signal),
        "sky_adu_per_s":         float(sky_adu_per_s),
        "sky_adu_per_s_arcsec2": float(sky_adu_per_s_arcsec2),
        "sky_e_per_s_arcsec2":   sky_e_per_s_arcsec2,
        "sky_mag_arcsec2":       sky_mag_arcsec2,
        "sky_gradient_rms":      sky_gradient_rms,
        "sky_gradient_ptp":      sky_gradient_ptp,
        "exptime":               exptime,
        "gain":                  gain,
        "offset":                offset,
        "cell_grid":             cell_grid,
    }


def save_sky_map(
    fits_path: Path,
    output_path: Path,
    arcsec_per_pixel: float = 0.26,
) -> tuple[Path, Optional[dict]]:
    """
    Render a sky background gradient map and save as JPEG.

    Shows the image with an 8 × 8 grid overlay where each cell is coloured
    by its sky background level (green = dark, red = bright).
    Corner subframes used for the primary sky measurement are outlined in cyan.

    Returns (output_path, sky_data_dict).
    """
    fits_path   = Path(fits_path)
    output_path = Path(output_path)

    try:
        with fits.open(fits_path) as hdul:
            raw = hdul[0].data.astype(float)
    except Exception as exc:
        print(f"sky_brightness: failed to open {fits_path.name}: {exc}")
        return output_path, None

    data = np.squeeze(raw)
    sky_data = measure_sky(fits_path, arcsec_per_pixel)
    if sky_data is None:
        return output_path, None

    cell_grid = sky_data["cell_grid"]
    h, w      = data.shape
    cell_h    = h / _GRID_N
    cell_w    = w / _GRID_N
    rh        = max(1, int(h * _CORNER_FRAC))
    rw        = max(1, int(w * _CORNER_FRAC))

    interval  = ZScaleInterval()
    vmin, vmax = interval.get_limits(data)

    fig, ax = plt.subplots(figsize=(8, 8))
    _apply_dark_theme(fig, ax)
    ax.imshow(data, cmap="gray", vmin=vmin, vmax=vmax, origin="lower",
              aspect="auto", interpolation="nearest")

    # Colour-coded grid overlay
    valid = cell_grid[np.isfinite(cell_grid)]
    cmin  = float(valid.min()) if len(valid) else 0.0
    cmax  = float(valid.max()) if len(valid) else 1.0
    cmap  = plt.cm.RdYlGn_r        # green (dark) → red (bright)
    norm  = Normalize(vmin=cmin, vmax=cmax)

    for r in range(_GRID_N):
        for c in range(_GRID_N):
            val = cell_grid[r, c]
            if not np.isfinite(val):
                continue
            x0 = c * cell_w
            y0 = r * cell_h
            color = cmap(norm(val))
            rect = plt.Rectangle(
                (x0, y0), cell_w, cell_h,
                facecolor=(*color[:3], 0.4),
                edgecolor="white", linewidth=0.5,
            )
            ax.add_patch(rect)
            ax.text(
                x0 + cell_w / 2, y0 + cell_h / 2,
                f"{val:.0f}",
                ha="center", va="center",
                fontsize=7, color="white", fontweight="bold",
            )

    # Highlight the four corner subframes in cyan
    corners_coords = [
        (0,     0,     rw, rh),   # bottom-left
        (w - rw, 0,   rw, rh),   # bottom-right
        (0,     h - rh, rw, rh), # top-left
        (w - rw, h - rh, rw, rh),# top-right
    ]
    for x0, y0, cw, ch in corners_coords:
        rect = plt.Rectangle((x0, y0), cw, ch,
                              facecolor="none", edgecolor="cyan", linewidth=1.5,
                              linestyle="--")
        ax.add_patch(rect)

    # Colour bar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Sky ADU (offset-corrected)", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    # Title
    title_parts = [
        f'Sky: {sky_data["sky_adu"]:.0f} ADU',
        f'{sky_data["sky_adu_per_s"]:.2f} ADU/s',
    ]
    if sky_data.get("sky_mag_arcsec2") is not None:
        title_parts.append(f'{sky_data["sky_mag_arcsec2"]:.2f} instr mag/arcsec²')
    title_parts.append(f'gradient ±{sky_data["sky_gradient_rms"]:.1f} ADU')
    ax.set_title("  |  ".join(title_parts), color="white", fontsize=9, pad=6)
    ax.set_xlabel("X (pixels)", color="white")
    ax.set_ylabel("Y (pixels)", color="white")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_path, format="jpeg", dpi=150, bbox_inches="tight",
                facecolor=_DARK_BG)
    plt.close(fig)

    return output_path, sky_data


def save_sky_heatmap(
    fits_path: Path,
    output_path: Path,
    arcsec_per_pixel: float = 0.26,
    n_cells: int = _GRID_N,
) -> Path:
    """
    Render a sky brightness heatmap in the same visual style as the FWHM and
    eccentricity heatmaps.  Each cell shows sky background in ADU/s;
    green = dark sky (good), red = bright sky (bad).

    Returns (output_path, sky_data_dict).  sky_data_dict is None on failure.
    """
    fits_path   = Path(fits_path)
    output_path = Path(output_path)

    sky_data = measure_sky(fits_path, arcsec_per_pixel)
    if sky_data is None:
        return output_path, None

    try:
        with fits.open(fits_path) as hdul:
            raw = hdul[0].data.astype(float)
    except Exception as exc:
        print(f"sky_brightness: failed to open {fits_path.name}: {exc}")
        return output_path, None

    data = np.squeeze(raw)
    img_h, img_w = data.shape
    vmin, vmax   = ZScaleInterval().get_limits(data)

    exptime   = sky_data["exptime"]
    cell_grid = sky_data["cell_grid"]           # offset-corrected ADU per cell
    # Convert each cell to ADU/s for the display metric
    adu_per_s_grid = cell_grid / max(exptime, 1e-6)

    valid  = adu_per_s_grid[np.isfinite(adu_per_s_grid)]
    norm   = Normalize(vmin=float(valid.min()), vmax=float(valid.max()))
    cmap   = plt.get_cmap("RdYlGn_r")
    cell_w = img_w / n_cells
    cell_h = img_h / n_cells

    fig, ax = plt.subplots(figsize=(10, 10))
    _apply_dark_theme(fig, ax)
    ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
              interpolation="nearest", extent=[0, img_w, 0, img_h])

    for r in range(n_cells):
        for c in range(n_cells):
            val = adu_per_s_grid[r, c]
            if not np.isfinite(val):
                continue
            color = cmap(norm(val))
            rect  = plt.Rectangle(
                (c * cell_w, r * cell_h), cell_w, cell_h,
                facecolor=(*color[:3], 0.55), edgecolor="white", linewidth=0.8,
            )
            ax.add_patch(rect)
            ax.text(
                c * cell_w + cell_w / 2, r * cell_h + cell_h / 2,
                f"{val:.2f}",
                ha="center", va="center",
                fontsize=9, color="white", fontweight="bold",
            )

    cb = plt.colorbar(ScalarMappable(norm=norm, cmap="RdYlGn_r"),
                      ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Sky ADU/s  (lower = darker)", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    title = (
        f"{fits_path.name}  |  Sky brightness grid  |  "
        f"{sky_data['sky_adu_per_s']:.2f} ADU/s"
    )
    if sky_data.get("sky_mag_arcsec2") is not None:
        title += f'  |  {sky_data["sky_mag_arcsec2"]:.2f} instr mag/arcsec²'
    ax.set_title(title, color="white")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, format="jpeg", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path, sky_data


def sky_summary_text(sky_data: dict) -> str:
    """Format a sky_data dict into a short human-readable string."""
    lines = [
        f'Sky background: {sky_data["sky_adu"]:.0f} ADU  '
        f'({sky_data["sky_adu_per_s"]:.2f} ADU/s)',
    ]
    if sky_data.get("sky_mag_arcsec2") is not None:
        lines.append(
            f'Sky brightness: {sky_data["sky_mag_arcsec2"]:.2f} instr mag/arcsec²'
        )
    lines.append(
        f'Sky gradient: ±{sky_data["sky_gradient_rms"]:.1f} ADU RMS  '
        f'({sky_data["sky_gradient_ptp"]:.1f} ADU peak-to-peak)'
    )
    return "\n".join(lines)


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python sky_brightness.py <path/to/frame.fits> [arcsec_per_pixel]")
        sys.exit(1)
    fits_p  = Path(sys.argv[1])
    app_val = float(sys.argv[2]) if len(sys.argv) > 2 else 0.26
    result  = measure_sky(fits_p, app_val)
    if result:
        print(sky_summary_text(result))
        print(f"  exptime : {result['exptime']} s")
        print(f"  gain    : {result['gain']} e/ADU")
        print(f"  offset  : {result['offset']} ADU")
    else:
        print("Failed to measure sky brightness.")
        sys.exit(1)
