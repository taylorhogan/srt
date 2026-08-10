import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.utils.iers import conf
conf.auto_max_age = None
from astropy.utils import iers
iers.conf.auto_download = False
from astropy.io import fits
from astropy.modeling import fitting, models
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ZScaleInterval
from matplotlib.patches import Circle, Patch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from photutils.detection import DAOStarFinder

# FWHM = 2 * sqrt(2 * ln(2)) * sigma
_FWHM_SIGMA_FACTOR = 2.3548

# Approximate FWHM (pixels) used only for the initial star detection pass
_DETECTION_FWHM = 5.0

# Stars must be this many sigma above background to be detected
_DETECTION_THRESHOLD_SIGMA = 10.0

# Half-width (pixels) of the cutout stamp used for Gaussian fitting
# Half-width of the fitting stamp, pixels. Was 15 (a 31x31 box), which is barely
# wider than a real star: measured stellar FWHM on this system is ~14.7 px, and a
# Gaussian fit needs roughly 3x the FWHM of wings to constrain the width. 25 gives
# a 51x51 box. The upper FWHM bound below is tied to this, so at 15 a genuine
# 14.7 px star sat one pixel under the rejection threshold.
_STAMP_HALF = 25

# Nothing narrower than this can be starlight. The 17-inch diffraction limit is
# 0.32" and the plate scale is 0.26"/px, so a real star cannot be under ~1.2 px
# however good the seeing. The old lower bound of 0.5 px admitted exactly the
# single-pixel sensor defects that despiking now removes -- belt and braces,
# because a defect that survives despiking must still not be called a star.
_MIN_PHYSICAL_FWHM_PX = 1.5

# Stamped on records so Gaussian-era and Moffat-era values are never mixed in a
# trend. Untagged historical records are Gaussian.
_PSF_MODEL = "moffat"

# Fitted Gaussian amplitude must be at least this many × background std
_MIN_SNR = 10.0

# Ratio of the larger axis stddev to the smaller must not exceed this (1.0 = perfect circle)
_MAX_ELLIPTICITY = 2.0

_logger = logging.getLogger(__name__)

_DARK_BG   = "#0d0d1a"
_DARK_AXES = "#1a1a2e"


def _apply_dark_theme(fig, *axes):
    """Apply the standard dark observatory theme to a figure and its axes."""
    fig.patch.set_facecolor(_DARK_BG)
    for ax in axes:
        ax.set_facecolor(_DARK_AXES)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")


def _fit_stars(
    fits_path: Path,
    threshold_sigma: float = _DETECTION_THRESHOLD_SIGMA,
    min_snr: float = _MIN_SNR,
    max_ellipticity: float = _MAX_ELLIPTICITY,
) -> tuple[np.ndarray, list[tuple[float, float, float]]]:
    """
    Load a FITS image, detect stars, and fit a 2D Gaussian to each one.

    Filters applied:
      - DAOStarFinder threshold (threshold_sigma × background std)
      - DAOStarFinder built-in sharpness [0.2, 1.0] and roundness [-1, 1]
      - Post-fit amplitude SNR >= min_snr × background std
      - Post-fit ellipticity (long_axis / short_axis) <= max_ellipticity

    Returns:
        (image_data, [(x, y, fwhm_pixels, eccentricity, major_angle_rad, amplitude), ...])
        Eccentricity is sqrt(1 - (short_axis/long_axis)^2); 0 = perfect circle, 1 = line.
    """
    with fits.open(fits_path) as hdul:
        raw = hdul[0].data.astype(float)

    data = np.squeeze(raw)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D image data after squeeze, got shape {data.shape}")

    # Remove single-pixel sensor defects BEFORE detecting anything.
    #
    # No darks are taken, so raw subs carry hot pixels, and hot pixels are
    # sharper than stars. On a NORMAL frame this barely matters -- real stars
    # dominate detection and the measured FWHM shifts by only ~3%. The failure
    # is rare and conditional: when stars go broad and faint, the defects win
    # the detection competition outright and the measurement collapses.
    #
    # Measured 2026-08-06, old code vs new, same frames:
    #
    #     2026-07-12 O-III (mid-night)   775 stars 1.85"  ->  1653 stars 1.91"
    #     2026-07-23 Ha    (mid-night)   903 stars 1.73"  ->  1926 stars 1.78"
    #     2026-08-04 O-III (end, poor)     5 stars 0.21"  ->   256 stars 4.21"
    #
    # So this is a robustness fix, not a systemic correction. It matters because
    # a metric that silently returns nonsense on bad nights is worse than one
    # that is merely noisy: it would poison any trend containing a poor night,
    # and 0.21" is not obviously wrong to anything downstream that consumes it.
    #
    # stacker._despike already existed for the same underlying reason on the
    # registration path, where hot pixels voted for the identity transform. It
    # was simply never applied here. Detection and fitting use the cleaned copy;
    # the ORIGINAL frame is returned for display, so the rendered image still
    # shows what the sensor actually recorded.
    from stacking import stacker  # local import: avoids a circular import at load
    try:
        measured = stacker._despike(data)
    except Exception:
        _logger.exception("despike failed; measuring the raw frame (expect hot-pixel FWHM)")
        measured = data

    _, median, std = sigma_clipped_stats(measured, sigma=3.0)
    background_subtracted = measured - median

    finder = DAOStarFinder(fwhm=_DETECTION_FWHM, threshold=threshold_sigma * std)
    sources = finder(background_subtracted)

    if sources is None or len(sources) == 0:
        return data, []

    height, width = data.shape
    fitter = fitting.LevMarLSQFitter()
    stars: list[tuple[float, float, float, float]] = []

    for source in sources:
        xc = int(round(source["xcentroid"]))
        yc = int(round(source["ycentroid"]))

        x0, x1 = xc - _STAMP_HALF, xc + _STAMP_HALF + 1
        y0, y1 = yc - _STAMP_HALF, yc + _STAMP_HALF + 1

        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            continue

        stamp = background_subtracted[y0:y1, x0:x1]
        y_grid, x_grid = np.mgrid[0:stamp.shape[0], 0:stamp.shape[1]]

        init_sigma = _DETECTION_FWHM / _FWHM_SIGMA_FACTOR
        init_model = models.Gaussian2D(
            amplitude=float(stamp.max()),
            x_mean=float(_STAMP_HALF),
            y_mean=float(_STAMP_HALF),
            x_stddev=init_sigma,
            y_stddev=init_sigma,
        )

        try:
            fitted = fitter(init_model, x_grid, y_grid, stamp)
            sigma_x = abs(fitted.x_stddev.value)
            sigma_y = abs(fitted.y_stddev.value)

            # Width comes from a Moffat fit; SHAPE still comes from the Gaussian
            # above. Two fits per star, deliberately.
            #
            # A real stellar profile has broader wings than a Gaussian can
            # represent, so a Gaussian widens its core to cover them and reports
            # a width that is too large. A Moffat models the wings with its
            # power-law term and keeps the core honest. Measured 2026-08-06 on
            # 118 bright stars in one sh2-92 sub: Gaussian 6.77 px (1.76"),
            # Moffat 6.00 px (1.56") -- 11.5% smaller. This is also what
            # PixInsight fits, so the numbers are now comparable with it.
            #
            # astropy's Moffat2D is circularly symmetric: no x/y widths, no
            # angle. Eccentricity and major-axis angle therefore still come from
            # the Gaussian, which the optics-trend metrics need for tilt, coma
            # and collimation. Swapping wholesale would have silently removed
            # them.
            #
            # The offset is NOT a constant that historical values can be scaled
            # by -- it depends on the fitted power-law index, which moves with
            # seeing. Records carry _PSF_MODEL so the two families are never
            # averaged together; recompute from the frames instead.
            fwhm = _FWHM_SIGMA_FACTOR * np.sqrt(sigma_x * sigma_y)
            try:
                moffat = fitter(
                    models.Moffat2D(amplitude=float(stamp.max()),
                                    x_0=float(_STAMP_HALF), y_0=float(_STAMP_HALF),
                                    gamma=max(sigma_x, 0.5), alpha=3.0),
                    x_grid, y_grid, stamp)
                gamma, alpha = abs(moffat.gamma.value), abs(moffat.alpha.value)
                if gamma > 0 and alpha > 0:
                    m_fwhm = 2.0 * gamma * np.sqrt(2.0 ** (1.0 / alpha) - 1.0)
                    if np.isfinite(m_fwhm) and m_fwhm > 0:
                        fwhm = float(m_fwhm)
            except Exception:
                pass          # keep the Gaussian width if the Moffat will not fit

            # Reject bad FWHM range
            if not (_MIN_PHYSICAL_FWHM_PX < fwhm < _STAMP_HALF):
                continue
            # Reject faint fits (noise spikes that squeaked past detection)
            if fitted.amplitude.value < min_snr * std:
                continue
            # Reject elongated sources (cosmic rays, diffraction spikes, artifacts)
            long_ax, short_ax = max(sigma_x, sigma_y), min(sigma_x, sigma_y)
            if short_ax > 0 and long_ax / short_ax > max_ellipticity:
                continue

            ecc = float(np.sqrt(1.0 - (short_ax / long_ax) ** 2)) if long_ax > 0 else 0.0
            # theta is the CCW rotation of the x_stddev axis from the +x axis.
            # Major axis angle: if x_stddev >= y_stddev it's theta, otherwise theta + 90°.
            theta = float(fitted.theta.value)
            major_angle = theta if sigma_x >= sigma_y else theta + np.pi / 2.0
            # Amplitude rides along as a 6th element so callers can select a
            # fixed, brightness-matched star sample (see
            # compute_optics_trend_metrics). Appended, never inserted: existing
            # callers index s[0..4] positionally.
            stars.append((float(xc), float(yc), fwhm, ecc, major_angle,
                          float(fitted.amplitude.value)))
        except Exception:
            continue

    return data, stars


def calculate_fwhm(
    fits_path: Path,
    arcsec_per_pixel: float = 1.0,
    threshold_sigma: float = _DETECTION_THRESHOLD_SIGMA,
    min_snr: float = _MIN_SNR,
    max_ellipticity: float = _MAX_ELLIPTICITY,
) -> tuple[float, float, int]:
    """
    Detect stars in a FITS image, fit a 2D Gaussian to each one,
    and return the mean FWHM across all stars.

    Args:
        fits_path:         Path to the FITS file.
        arcsec_per_pixel:  Plate scale used to convert FWHM from pixels to arcseconds.
        threshold_sigma:   Detection threshold in units of background std (default 10).
        min_snr:           Minimum fitted amplitude / background std (default 10).
        max_ellipticity:   Maximum long/short axis ratio; rejects non-round sources (default 2).

    Returns:
        (mean_fwhm_pixels, mean_fwhm_arcsec, star_count, mean_eccentricity)
        Values are 0.0 / 0 if no stars are found.
    """
    _, stars = _fit_stars(fits_path, threshold_sigma, min_snr, max_ellipticity)
    if not stars:
        return 0.0, 0.0, 0, 0.0
    mean_px = float(np.mean([s[2] for s in stars]))
    mean_ecc = float(np.mean([s[3] for s in stars]))
    return mean_px, mean_px * arcsec_per_pixel, len(stars), mean_ecc


def display_fwhm(
    fits_path: Path,
    arcsec_per_pixel: float = 1.0,
    threshold_sigma: float = _DETECTION_THRESHOLD_SIGMA,
    min_snr: float = _MIN_SNR,
    max_ellipticity: float = _MAX_ELLIPTICITY,
) -> None:
    """
    Display the FITS image with a circle around each detected star.
    Red    = FWHM more than 10% above mean (soft).
    Yellow = within 10% of mean.
    Green  = FWHM more than 10% below mean (tight).
    Circle radius is scaled to the star's individual FWHM.

    Args:
        fits_path:         Path to the FITS file.
        arcsec_per_pixel:  Plate scale for the title readout.
        threshold_sigma:   Detection threshold in units of background std (default 10).
        min_snr:           Minimum fitted amplitude / background std (default 10).
        max_ellipticity:   Maximum long/short axis ratio; rejects non-round sources (default 2).
    """
    data, stars = _fit_stars(fits_path, threshold_sigma, min_snr, max_ellipticity)

    if not stars:
        print("No stars found.")
        return

    mean_px = float(np.mean([s[2] for s in stars]))
    mean_ecc = float(np.mean([s[3] for s in stars]))

    vmin, vmax = ZScaleInterval().get_limits(data)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")

    for x, y, fwhm, *_ in stars:
        if fwhm > mean_px * 1.1:
            color = "red"
        elif fwhm < mean_px * 0.9:
            color = "green"
        else:
            color = "yellow"
        ax.add_patch(Circle((x, y), radius=fwhm * 2.5, color=color, fill=False, linewidth=1.5))

    mean_arcsec = mean_px * arcsec_per_pixel
    ax.set_title(
        f"{fits_path.name}  |  {len(stars)} stars  |  "
        f'mean FWHM {mean_arcsec:.2f}"  |  '
        f"mean ecc {mean_ecc:.3f}"
    )
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.legend(handles=[
        Patch(facecolor="none", edgecolor="green",  label=f'Good  (< {mean_arcsec * 0.9:.2f}")'),
        Patch(facecolor="none", edgecolor="yellow", label=f'OK    ({mean_arcsec * 0.9:.2f} – {mean_arcsec * 1.1:.2f}")'),
        Patch(facecolor="none", edgecolor="red",    label=f'Soft  (> {mean_arcsec * 1.1:.2f}")'),
    ], loc="upper right")

    plt.tight_layout()
    plt.show()


def save_fwhm(
    fits_path: Path,
    output_path: Path,
    arcsec_per_pixel: float = 1.0,
    threshold_sigma: float = _DETECTION_THRESHOLD_SIGMA,
    min_snr: float = _MIN_SNR,
    max_ellipticity: float = _MAX_ELLIPTICITY,
    annotate: bool = True,
    filter_name: str | None = None,
    sky_data: dict | None = None,
) -> Path:
    """
    Annotate a FITS image with detected stars and save it as a JPG.

    Returns the output_path written.
    """
    data, stars = _fit_stars(fits_path, threshold_sigma, min_snr, max_ellipticity)

    mean_px = float(np.mean([s[2] for s in stars])) if stars else 0.0
    mean_ecc = float(np.mean([s[3] for s in stars])) if stars else 0.0

    vmin, vmax = ZScaleInterval().get_limits(data)

    fig, ax = plt.subplots(figsize=(12, 12))
    _apply_dark_theme(fig, ax)
    ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")

    if annotate:
        for x, y, fwhm, *_ in stars:
            if fwhm > mean_px * 1.1:
                color = "red"
            elif fwhm < mean_px * 0.9:
                color = "green"
            else:
                color = "yellow"
            ax.add_patch(Circle((x, y), radius=fwhm * 2.5, color=color, fill=False, linewidth=1.5))

    mean_arcsec = mean_px * arcsec_per_pixel

    if filter_name or sky_data:
        parts = []
        if filter_name:
            parts.append(filter_name)
        if stars:
            parts.append(f'FWHM {mean_arcsec:.2f}"')
            parts.append(f"ecc {mean_ecc:.3f}")
            parts.append(f"{len(stars)} stars")
        else:
            parts.append("no stars detected")
        if sky_data:
            if sky_data.get("sky_mag_arcsec2") is not None:
                parts.append(f'sky {sky_data["sky_mag_arcsec2"]:.2f} mag/arcsec²')
            elif sky_data.get("sky_adu_per_s") is not None:
                parts.append(f'sky {sky_data["sky_adu_per_s"]:.4f} ADU/s')
            elif sky_data.get("sky_below_pedestal"):
                # Not a dark sky — a frame sitting under its own pedestal. Say so
                # rather than drop the field, or the frame looks like one where
                # sky simply was not measured.
                parts.append("sky below pedestal")
        title = "  |  ".join(parts)
    else:
        title = (
            f"{fits_path.name}  |  {len(stars)} stars  |  "
            f'mean FWHM {mean_arcsec:.2f}"  |  '
            f"mean ecc {mean_ecc:.3f}"
            if stars else f"{fits_path.name}  |  no stars detected"
        )
    ax.set_title(title)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    if annotate and stars:
        legend = ax.legend(handles=[
            Patch(facecolor="none", edgecolor="green",  label=f'Good  (< {mean_arcsec * 0.9:.2f}")'),
            Patch(facecolor="none", edgecolor="yellow", label=f'OK    ({mean_arcsec * 0.9:.2f} – {mean_arcsec * 1.1:.2f}")'),
            Patch(facecolor="none", edgecolor="red",    label=f'Soft  (> {mean_arcsec * 1.1:.2f}")'),
        ], loc="upper right")
        legend.get_frame().set_facecolor(_DARK_AXES)
        for text in legend.get_texts():
            text.set_color("white")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, format="jpeg", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path, mean_px, mean_ecc


def _make_grid(xs: np.ndarray, ys: np.ndarray, values: np.ndarray,
               img_h: int, img_w: int, n_cells: int = 8) -> np.ndarray:
    """
    Divide the image into an n_cells × n_cells grid and fill each cell with
    the median of `values` for stars that fall inside it.
    Cells with no stars are NaN (rendered as transparent/masked).
    Returns a 2-D array shaped (n_cells, n_cells), origin='lower'.
    """
    grid = np.full((n_cells, n_cells), np.nan)
    cell_w = img_w / n_cells
    cell_h = img_h / n_cells
    col_idx = np.clip((xs / cell_w).astype(int), 0, n_cells - 1)
    row_idx = np.clip((ys / cell_h).astype(int), 0, n_cells - 1)
    for r in range(n_cells):
        for c in range(n_cells):
            mask = (row_idx == r) & (col_idx == c)
            if mask.any():
                grid[r, c] = float(np.median(values[mask]))
    return grid


def save_fwhm_heatmaps(
    fits_path: Path,
    fwhm_output_path: Path,
    ecc_output_path: Path,
    arcsec_per_pixel: float = 1.0,
    n_cells: int = 8,
    threshold_sigma: float = _DETECTION_THRESHOLD_SIGMA,
    min_snr: float = _MIN_SNR,
    max_ellipticity: float = _MAX_ELLIPTICITY,
) -> tuple[Path, Path]:
    """
    Produce two grid heatmap images overlaid on the FITS image.
    Each cell is coloured by the median FWHM (or eccentricity) of the
    stars that fall inside it, making field-wide optical patterns obvious.

    Returns (fwhm_output_path, ecc_output_path).
    """
    data, stars = _fit_stars(fits_path, threshold_sigma, min_snr, max_ellipticity)
    vmin, vmax = ZScaleInterval().get_limits(data)
    img_h, img_w = data.shape

    def _render(out_path, label, grid, norm, cbar_label, title):
        fig, ax = plt.subplots(figsize=(10, 10))
        _apply_dark_theme(fig, ax)
        ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
                  interpolation="nearest", extent=[0, img_w, 0, img_h])
        # Draw each cell as a coloured rectangle; skip NaN cells
        cell_w = img_w / n_cells
        cell_h = img_h / n_cells
        cmap = plt.get_cmap("RdYlGn_r")
        for r in range(n_cells):
            for c in range(n_cells):
                val = grid[r, c]
                if np.isnan(val):
                    continue
                color = cmap(norm(val))
                rect = plt.Rectangle(
                    (c * cell_w, r * cell_h), cell_w, cell_h,
                    facecolor=(*color[:3], 0.55), edgecolor="white",
                    linewidth=0.8,
                )
                ax.add_patch(rect)
                ax.text(
                    c * cell_w + cell_w / 2, r * cell_h + cell_h / 2,
                    f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color="white",
                    fontweight="bold",
                )
        cb = plt.colorbar(ScalarMappable(norm=norm, cmap="RdYlGn_r"),
                          ax=ax, fraction=0.03, pad=0.02)
        cb.set_label(cbar_label, color="white")
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
        ax.set_title(title)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, format="jpeg", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    if not stars:
        for out_path, label in [(fwhm_output_path, "FWHM"), (ecc_output_path, "Eccentricity")]:
            fig, ax = plt.subplots(figsize=(10, 10))
            _apply_dark_theme(fig, ax)
            ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
                      interpolation="nearest")
            ax.set_title(f"{fits_path.name}  |  {label} grid heatmap  |  no stars detected")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(out_path, format="jpeg", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
        return fwhm_output_path, ecc_output_path

    xs = np.array([s[0] for s in stars])
    ys = np.array([s[1] for s in stars])
    fwhms = np.array([s[2] for s in stars])
    eccs = np.array([s[3] for s in stars])

    # --- FWHM grid ---
    fwhm_grid = _make_grid(xs, ys, fwhms, img_h, img_w, n_cells)
    median_fwhm = float(np.median(fwhms))
    median_fwhm_arcsec = median_fwhm * arcsec_per_pixel
    fwhm_grid_arcsec = fwhm_grid * arcsec_per_pixel
    _render(
        fwhm_output_path, "FWHM",
        fwhm_grid_arcsec,
        Normalize(vmin=np.nanmin(fwhm_grid_arcsec), vmax=np.nanmax(fwhm_grid_arcsec)),
        'FWHM (arcsec)',
        f"{fits_path.name}  |  FWHM grid  |  {len(stars)} stars  |  "
        f'median {median_fwhm_arcsec:.2f}"',
    )

    # --- Eccentricity grid ---
    ecc_grid = _make_grid(xs, ys, eccs, img_h, img_w, n_cells)
    median_ecc = float(np.median(eccs))
    _render(
        ecc_output_path, "Eccentricity",
        ecc_grid,
        Normalize(vmin=0.0, vmax=max(float(np.nanmax(ecc_grid)), 0.5)),
        "Eccentricity (0=round, 1=elongated)",
        f"{fits_path.name}  |  Eccentricity grid  |  {len(stars)} stars  |  "
        f"median ecc {median_ecc:.3f}",
    )

    return fwhm_output_path, ecc_output_path


def save_fwhm_vs_distance(
    fits_path: Path,
    output_path: Path,
    arcsec_per_pixel: float = 1.0,
    threshold_sigma: float = _DETECTION_THRESHOLD_SIGMA,
    min_snr: float = _MIN_SNR,
    max_ellipticity: float = _MAX_ELLIPTICITY,
) -> Path:
    """
    Scatter plot of FWHM (px) vs distance from image centre (px).
    Useful for diagnosing field curvature: good optics show a flat
    distribution; curvature shows FWHM rising away from centre.
    """
    data, stars = _fit_stars(fits_path, threshold_sigma, min_snr, max_ellipticity)

    fig, ax = plt.subplots(figsize=(8, 6))
    _apply_dark_theme(fig, ax)

    if not stars:
        ax.set_title(f"{fits_path.name}  |  FWHM vs distance  |  no stars detected")
    else:
        cy, cx = data.shape[0] / 2.0, data.shape[1] / 2.0
        distances = np.array([np.hypot(s[0] - cx, s[1] - cy) for s in stars])
        fwhms_arcsec = np.array([s[2] * arcsec_per_pixel for s in stars])

        ax.scatter(distances, fwhms_arcsec, s=18, alpha=0.7, color="steelblue", linewidths=0)

        # Linear trend line
        if len(stars) >= 2:
            coeffs = np.polyfit(distances, fwhms_arcsec, 1)
            x_line = np.linspace(distances.min(), distances.max(), 200)
            ax.plot(x_line, np.polyval(coeffs, x_line), color="tomato", linewidth=1.5,
                    label=f'trend  slope={coeffs[0]:+.4f}"/px')
            ax.legend(fontsize=9)

        median_fwhm_arcsec = float(np.median(fwhms_arcsec))
        ax.axhline(median_fwhm_arcsec, color="gray", linewidth=1, linestyle="--",
                   label=f'median {median_fwhm_arcsec:.2f}"')
        ax.set_xlabel("Distance from centre (px)")
        ax.set_ylabel('FWHM (arcsec)')
        ax.set_title(
            f"{fits_path.name}  |  FWHM vs distance  |  {len(stars)} stars  |  "
            f'median {median_fwhm_arcsec:.2f}"'
        )
        ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, format="jpeg", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def save_eccentricity_angle_map(
    fits_path: Path,
    output_path: Path,
    arcsec_per_pixel: float = 1.0,
    threshold_sigma: float = _DETECTION_THRESHOLD_SIGMA,
    min_snr: float = _MIN_SNR,
    max_ellipticity: float = _MAX_ELLIPTICITY,
) -> Path:
    """
    Plot an arrow at each star's position pointing along the major axis of
    its fitted Gaussian.  Arrow length and colour both scale with eccentricity
    (round stars get a short, green stub; elongated stars get a long, red arrow).
    Useful for diagnosing tilt, collimation, or tracking errors: a systematic
    radial or tangential pattern reveals the underlying aberration.
    """
    data, stars = _fit_stars(fits_path, threshold_sigma, min_snr, max_ellipticity)
    vmin, vmax = ZScaleInterval().get_limits(data)

    fig, ax = plt.subplots(figsize=(10, 10))
    _apply_dark_theme(fig, ax)
    ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")

    if not stars:
        ax.set_title(f"{fits_path.name}  |  Eccentricity angle map  |  no stars detected")
    else:
        eccs = np.array([s[3] for s in stars])
        norm_ecc = Normalize(vmin=0.0, vmax=max(eccs.max(), 0.5))
        cmap = plt.get_cmap("RdYlGn_r")

        # Arrow length is proportional to eccentricity, scaled to ~2 % of image width
        arrow_scale = max(data.shape) * 0.05

        for x, y, fwhm, ecc, angle, *_ in stars:
            length = ecc * arrow_scale
            if length < 1.0:
                length = 1.0          # always draw a tiny stub so every star is visible
            dx = length * np.cos(angle)
            dy = length * np.sin(angle)
            color = cmap(norm_ecc(ecc))
            # Draw a two-headed arrow (positive and negative direction along major axis)
            ax.annotate(
                "", xy=(x + dx, y + dy), xytext=(x - dx, y - dy),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.2),
            )

        cb = plt.colorbar(ScalarMappable(norm=norm_ecc, cmap="RdYlGn_r"), ax=ax,
                          fraction=0.03, pad=0.02)
        cb.set_label("Eccentricity (0=round, 1=elongated)", color="white")
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
        mean_ecc = float(np.median(eccs))
        ax.set_title(
            f"{fits_path.name}  |  Elongation angle map  |  {len(stars)} stars  |  "
            f"median ecc {mean_ecc:.3f}"
        )

    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, format="jpeg", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def compute_optical_metrics(
    fits_path: Path,
    arcsec_per_pixel: float = 1.0,
    n_cells: int = 8,
    threshold_sigma: float = _DETECTION_THRESHOLD_SIGMA,
    min_snr: float = _MIN_SNR,
    max_ellipticity: float = _MAX_ELLIPTICITY,
) -> dict:
    """
    Compute a set of scalar optical-quality metrics from a FITS image.

    Returns a dict with keys:
        star_count          int
        median_fwhm_px      float
        median_fwhm_arcsec  float
        median_ecc          float
        field_uniformity    float  std/mean of per-cell median FWHM — 0=perfect, lower is better
        cv_ecc              float  std/median eccentricity across all stars — lower is better
        tilt_score          float  OLS FWHM(x,y)=ax+by+c gradient×diagonal/median_fwhm
        coma_score          float  Pearson r(ecc, radius) — 0=none, 1=eccentricity grows with radius
        collimation_score   float  mean(cos²(elongation−radial)) — 0.5=random, 1.0=all radial

    Returns an empty dict if fewer than 5 stars are detected.
    """
    data, stars = _fit_stars(fits_path, threshold_sigma, min_snr, max_ellipticity)
    if len(stars) < 5:
        return {}

    h, w = data.shape
    cx, cy = w / 2.0, h / 2.0

    xs     = np.array([s[0] for s in stars])
    ys     = np.array([s[1] for s in stars])
    fwhms  = np.array([s[2] for s in stars])
    eccs   = np.array([s[3] for s in stars])
    angles = np.array([s[4] for s in stars])

    median_fwhm = float(np.median(fwhms))

    # --- Field Uniformity (grid-based) ---
    grid = _make_grid(xs, ys, fwhms, h, w, n_cells)
    populated = grid[~np.isnan(grid)]
    if len(populated) >= 2 and np.nanmean(grid) > 0:
        field_uniformity = float(np.nanstd(populated) / np.nanmean(populated))
    else:
        field_uniformity = 0.0

    # --- CV of Eccentricity ---
    median_ecc = float(np.median(eccs))
    cv_ecc = float(np.std(eccs) / median_ecc) if median_ecc > 0 else 0.0

    # --- Tilt Score (OLS plane fit to FWHM) ---
    # FWHM = a*x + b*y + c  →  gradient magnitude normalised by image diagonal and median FWHM
    A = np.column_stack([xs, ys, np.ones(len(xs))])
    coeffs, _, _, _ = np.linalg.lstsq(A, fwhms, rcond=None)
    a_coef, b_coef = coeffs[0], coeffs[1]
    diagonal = float(np.hypot(w, h))
    tilt_score = float(np.hypot(a_coef, b_coef) * diagonal / median_fwhm) if median_fwhm > 0 else 0.0

    # --- Coma Score (eccentricity vs radial distance) ---
    radii = np.hypot(xs - cx, ys - cy)
    if np.std(radii) > 0 and np.std(eccs) > 0:
        coma_score = float(max(0.0, np.corrcoef(radii, eccs)[0, 1]))
    else:
        coma_score = 0.0

    # --- Collimation Score (radial alignment of elongation vectors) ---
    # For each star, angle between its elongation direction and the radial direction from centre.
    # cos²(Δ) = 1 → radial (coma-like); 0.5 → random; 0 → tangential.
    radial_angles = np.arctan2(ys - cy, xs - cx)
    delta = angles - radial_angles
    collimation_score = float(np.mean(np.cos(delta) ** 2))

    return {
        "star_count":           len(stars),
        "median_fwhm_px":       median_fwhm,
        "median_fwhm_arcsec":   median_fwhm * arcsec_per_pixel,
        "median_ecc":           median_ecc,
        "field_uniformity":     field_uniformity,
        "cv_ecc":               cv_ecc,
        "tilt_score":           tilt_score,
        "coma_score":           coma_score,
        "collimation_score":    collimation_score,
    }


# --- Seeing-robust optical-trend metrics -----------------------------------
#
# compute_optical_metrics above answers "how good is this frame". These answer a
# different question: "has the TELESCOPE changed" — which needs metrics that
# survive the weather, and its metrics do not. Measured on sh2-92 (same target,
# same filter, no optical change in between): from the best night to the worst,
# seeing went 1.88" -> 3.11" and field_uniformity moved +88%, coma_score -41%,
# collimation_score -29% — all in the direction that reads as "optics improved".
# See docs/optics_trend_plan.md for the full table and the reasoning.
#
# Two rules make these different:
#   * fixed-N star sampling — always the same number of stars in the same
#     brightness band, so the population cannot change with the seeing (the star
#     count fell 718 -> 130 across those two nights, which is most of the effect);
#   * quadrature, never ratios — seeing adds isotropic blur that combines in
#     quadrature, so subtracting the best patch of field removes it to first
#     order. Dividing by the mean, as field_uniformity does, builds an inverse
#     seeing dependence straight into the metric.

_TREND_N_STARS = 150            # fixed sample size; a night with fewer is reported as-is
_TREND_SATURATION_SKIP = 0.02   # drop the brightest 2% — flat-topped fits lie
_TREND_GRID = 4                 # 4x4 cells: ~9 stars per cell at N=150


def _fits_shape(fits_path: Path) -> tuple[int, int]:
    """(height, width) of a FITS image without keeping the pixels around."""
    with fits.open(fits_path) as hdul:
        shape = np.squeeze(hdul[0].data).shape
    return int(shape[0]), int(shape[1])


def compute_optics_trend_metrics(
    fits_path: Path,
    arcsec_per_pixel: float = 1.0,
    n_stars: int = _TREND_N_STARS,
    threshold_sigma: float = _DETECTION_THRESHOLD_SIGMA,
    min_snr: float = _MIN_SNR,
    max_ellipticity: float = _MAX_ELLIPTICITY,
) -> dict:
    """Optical metrics designed to hold still when the seeing moves.

    Returns {} when the frame cannot support the measurement (too few stars).

    Keys, and what each is for:

        stars_used            how many stars the fixed sample actually got
        seeing_floor_arcsec   p10 of per-cell FWHM — the best patch of field,
                              where the optics contribute least. A better
                              per-night seeing estimate than median FWHM, which
                              folds in the optical field degradation.
        field_excess_arcsec   sqrt(median_cell^2 - floor^2): the optical blur
                              added across the field, seeing removed.
        edge_excess_arcsec    the same at the p90 cell — the worst of the field.
                              Up while field_excess is flat means tilt/spacing.
        sweet_spot_x/y/r      where FWHM is minimised, in normalised field
                              coordinates (-1..1, 0 = centre). THE collimation
                              indicator: seeing lifts the whole surface without
                              moving its minimum.
        radial_fraction       mean cos(2*delta) between elongation and the radial
                              direction: +1 all radial (coma), 0 random,
                              -1 tangential.
        uniform_fraction      Rayleigh R of doubled elongation angles. 1 = every
                              star elongated the same way, which is NOT optics —
                              that is guiding, wind shake or a cable snag.
        uniform_angle_deg     the direction of that elongation, 0-180.
        median_ecc            eccentricity over the fixed sample.

    Angles are doubled throughout because an elongation axis is defined modulo
    180 degrees, not 360 — a star elongated "north" and one elongated "south" are
    the same measurement, and averaging raw angles would cancel them out.
    """
    sample, detected, shape = trend_sample(
        fits_path, n_stars, threshold_sigma, min_snr, max_ellipticity)
    if not sample:
        return {}
    return optics_trend_from_sample(sample, detected, shape, arcsec_per_pixel)


def trend_sample(
    fits_path: Path,
    n_stars: int = _TREND_N_STARS,
    threshold_sigma: float = _DETECTION_THRESHOLD_SIGMA,
    min_snr: float = _MIN_SNR,
    max_ellipticity: float = _MAX_ELLIPTICITY,
):
    """One frame's fixed-N, brightness-matched star sample.

    Split out from the metrics so a whole night's frames can be POOLED into one
    measurement. That matters more than it looks: the spatial metrics are
    limited by stars-per-cell, and measured across three comparable Ha nights the
    per-frame-then-median approach still scattered 86% on field_excess and 123%
    on radial_fraction. Pooling ~15 frames puts ~10x the stars in each cell.

    Returns (sample, detected_count, (h, w)); sample is empty when the frame
    cannot support a measurement.
    """
    _, stars = _fit_stars(fits_path, threshold_sigma, min_snr, max_ellipticity)
    if len(stars) < 20:
        return [], len(stars), (0, 0)
    # Sort bright-first, drop the top few percent (saturated cores fit badly, and
    # they are the stars most likely to survive when the seeing collapses), then
    # take a fixed count.
    ordered = sorted(stars, key=lambda s: s[5], reverse=True)
    skip = int(len(ordered) * _TREND_SATURATION_SKIP)
    sample = ordered[skip:skip + n_stars]
    if len(sample) < 20:
        sample = ordered[:n_stars]
    if len(sample) < 20:
        return [], len(stars), (0, 0)
    return sample, len(stars), _fits_shape(fits_path)


def optics_trend_from_sample(sample, detected: int, shape, arcsec_per_pixel: float) -> dict:
    """Derive the trend metrics from a star sample — one frame's or a night's.

    Pooling across frames is safe here because everything measured is a function
    of position in the FIELD, and dither offsets are a few pixels against a 4x4
    grid. An equatorial mount adds no field rotation to undo.
    """
    h, w = shape
    if not sample or h <= 0 or w <= 0:
        return {}
    n_stars = len(sample)
    cx, cy = w / 2.0, h / 2.0

    xs = np.array([s[0] for s in sample], float)
    ys = np.array([s[1] for s in sample], float)
    fwhm_as = np.array([s[2] for s in sample], float) * arcsec_per_pixel
    eccs = np.array([s[3] for s in sample], float)
    angles = np.array([s[4] for s in sample], float)

    out: dict = {
        "stars_used": len(sample),
        "stars_detected": detected,
        "median_fwhm_arcsec": float(np.median(fwhm_as)),
        "median_ecc": float(np.median(eccs)),
    }

    # --- quadrature field excess -------------------------------------------
    # Per-cell median FWHM^2, then compare the typical and worst cells against
    # the best. p10/p90 rather than min/max: the extreme of a set of noisy cell
    # medians is biased, the deciles are not.
    sq = fwhm_as ** 2
    cell_x = np.clip((xs / w * _TREND_GRID).astype(int), 0, _TREND_GRID - 1)
    cell_y = np.clip((ys / h * _TREND_GRID).astype(int), 0, _TREND_GRID - 1)
    cell_med = []
    for i in range(_TREND_GRID):
        for j in range(_TREND_GRID):
            m = (cell_x == i) & (cell_y == j)
            if m.sum() >= 3:
                cell_med.append(float(np.median(sq[m])))
    if len(cell_med) >= 4:
        cells = np.array(cell_med)
        floor_sq = float(np.percentile(cells, 10))
        out["seeing_floor_arcsec"] = float(np.sqrt(max(floor_sq, 0.0)))
        out["field_excess_arcsec"] = float(
            np.sqrt(max(float(np.median(cells)) - floor_sq, 0.0)))
        out["edge_excess_arcsec"] = float(
            np.sqrt(max(float(np.percentile(cells, 90)) - floor_sq, 0.0)))
        out["cells_used"] = len(cell_med)

    # --- sweet spot: where the FWHM surface bottoms out ---------------------
    # Quadratic surface fit to FWHM^2 in normalised coordinates. Seeing raises
    # the constant term and leaves the location of the minimum alone, which is
    # what makes this the collimation indicator worth trending.
    u = (xs - cx) / (w / 2.0)
    v = (ys - cy) / (h / 2.0)
    try:
        A = np.column_stack([np.ones_like(u), u, v, u * u, v * v, u * v])
        coef, *_ = np.linalg.lstsq(A, sq, rcond=None)
        _, b, c, d, e, f = coef
        hess = np.array([[2.0 * d, f], [f, 2.0 * e]])
        if np.linalg.det(hess) > 0 and d > 0:   # a genuine minimum, not a saddle
            su, sv = np.linalg.solve(hess, [-b, -c])
            # Clamp: an almost-flat field puts the fitted minimum far outside the
            # frame, where it means nothing. Report the edge rather than a number
            # that would swamp any trend.
            su = float(np.clip(su, -1.5, 1.5))
            sv = float(np.clip(sv, -1.5, 1.5))
            out["sweet_spot_x"] = su
            out["sweet_spot_y"] = sv
            out["sweet_spot_r"] = float(np.hypot(su, sv))
    except Exception:
        _logger.debug("sweet-spot fit failed for %s", fits_path, exc_info=True)

    # --- elongation geometry: optics vs tracking ---------------------------
    # Radial elongation is coma/collimation; one shared direction across the
    # whole field is not optics at all. Separating those is the point — the old
    # collimation_score conflates them.
    radial = np.arctan2(ys - cy, xs - cx)
    out["radial_fraction"] = float(np.mean(np.cos(2.0 * (angles - radial))))
    cs, sn = np.cos(2.0 * angles), np.sin(2.0 * angles)
    out["uniform_fraction"] = float(np.hypot(cs.mean(), sn.mean()))
    out["uniform_angle_deg"] = float(
        (np.degrees(np.arctan2(sn.mean(), cs.mean())) / 2.0) % 180.0)
    return out


# A frame only joins the pool if it supplied the FULL fixed-N sample, and a
# night needs this many such frames to be measured at all.
#
# Gating on the pooled total instead does not work, and the failure is
# instructive: on 2026-07-13 each frame held only ~130 stars, but eleven of them
# pooled to 1489 — comfortably past any total-star threshold, while every single
# frame was too thin for a comparable sample. The whole point of fixed-N is that
# the population is the same every night, and a frame that cannot fill it breaks
# that guarantee no matter how many such frames are added together.
#
# The consequence is deliberate: a bad-seeing or clouded night records NOTHING.
# An honest gap in the series beats a number that is really a seeing measurement
# wearing an optics label.
_TREND_MIN_FULL_FRAMES = 5


def compute_optics_trend_for_frames(
    fits_paths,
    arcsec_per_pixel: float = 1.0,
    n_stars: int = _TREND_N_STARS,
    progress_cb=None,
) -> dict:
    """Pool several frames into ONE optics measurement for the night.

    Per-frame metrics then medians is the obvious approach and it is too noisy:
    across three comparable Ha nights it scattered 86% on field_excess and 123%
    on radial_fraction, because each 4x4 cell only held ~9 stars. Pooling puts
    roughly ten times that in every cell, which is where the precision has to
    come from.

    Returns {} when too few frames could supply a full sample (see
    _TREND_MIN_FULL_FRAMES).
    """
    pooled: list = []
    detected_total = 0
    shape = (0, 0)
    used = 0
    thin = 0
    for path in fits_paths:
        try:
            sample, detected, sh = trend_sample(path, n_stars)
        except Exception:
            _logger.debug("trend sample failed for %s", path, exc_info=True)
            continue
        detected_total += detected
        if len(sample) < n_stars:
            # Short sample = a different star population from a night that could
            # fill it. Pooling it would silently reintroduce the seeing
            # dependence this whole design exists to remove.
            thin += 1
            continue
        if shape == (0, 0):
            shape = sh
        elif sh != shape:
            continue          # different sensor/binning — not poolable
        pooled.extend(sample)
        used += 1
        if progress_cb:
            progress_cb(f"optics trend: {used} frames, {len(pooled)} stars pooled")

    if used < _TREND_MIN_FULL_FRAMES:
        _logger.info(
            "Optics trend: skipped — only %d frames could supply %d stars "
            "(%d too thin, need %d frames)",
            used, n_stars, thin, _TREND_MIN_FULL_FRAMES)
        return {}

    metrics = optics_trend_from_sample(pooled, detected_total, shape, arcsec_per_pixel)
    if metrics:
        metrics["frames_used"] = used
        metrics["stars_pooled"] = len(pooled)
    return metrics


def save_optical_metrics_table(metrics: dict, output_path: Path) -> Path:
    """
    Render the scalar optical-quality metrics as a 4-column table image
    (Metric | Bad | Current | Good).  Each row is coloured on a green→red
    gradient based on how the current value sits between the good and bad
    reference thresholds.

    Returns output_path.
    """
    m = metrics

    def _badness(val: float, good: float, bad: float) -> float:
        """Return 0.0 (best) … 1.0 (worst), clamped."""
        span = bad - good
        if span == 0:
            return 0.0
        return max(0.0, min(1.0, (val - good) / span))

    rows = [
        ("CV FWHM",          ">0.30", f"{m['field_uniformity']:.2f}",  "<0.05"),
        ("CV Ecc",           ">0.30", f"{m['cv_ecc']:.2f}",            "<0.05"),
        ("Tilt score",       ">0.50", f"{m['tilt_score']:.2f}",        "<0.20"),
        ("Coma score",       ">0.50", f"{m['coma_score']:.2f}",        "<0.20"),
        ("Collimation",      ">0.80", f"{m['collimation_score']:.2f}", "~0.50"),
    ]
    scores = [
        _badness(m["field_uniformity"],     good=0.05, bad=0.30),  # lower is better
        _badness(m["cv_ecc"],               good=0.05, bad=0.30),  # lower is better
        _badness(m["tilt_score"],           good=0.20, bad=0.50),  # lower is better
        _badness(m["coma_score"],           good=0.20, bad=0.50),  # lower is better
        _badness(m["collimation_score"],    good=0.50, bad=0.80),  # lower is better
    ]

    cmap = plt.get_cmap("RdYlGn_r")
    cell_text   = [list(row) for row in rows]
    cell_colors = [[cmap(s)] * 4 for s in scores]

    fig, ax = plt.subplots(figsize=(7, 3))
    _apply_dark_theme(fig, ax)
    ax.axis("off")
    ax.set_title(
        f"Optical Quality  |  {m['star_count']} stars  |  "
        f"FWHM {m['median_fwhm_arcsec']:.2f}\"  |  "
        f"ecc {m['median_ecc']:.3f}",
        fontsize=11, pad=12,
    )

    table = ax.table(
        cellText=cell_text,
        colLabels=["Metric", "Bad", "Current", "Good"],
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.2)

    # Style header row
    for col in range(4):
        cell = table[0, col]
        cell.set_facecolor("#2b2b2b")
        cell.set_text_props(color="white", fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, format="jpeg", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def _extract_filter(fits_path: Path) -> str:
    """Extract filter name from a NINA FITS filename.

    NINA naming: DATE_TIME_FILTER__EXPOSURE_SEQUENCE.fits
    e.g. 2026-04-07_23-15-02_Ha__300.00s_0012.fits → 'Ha'
    """
    parts = fits_path.stem.split("_")
    if len(parts) >= 3:
        return parts[2]
    return "?"


# Standard colour mapping for common narrowband/broadband filters.
_FILTER_COLORS = {
    "L":  "#cccccc",
    "R":  "#e74c3c",
    "G":  "#2ecc71",
    "B":  "#3498db",
    "Ha": "#e74c3c",
    "Oiii": "#2ecc71",
    "OIII": "#2ecc71",
    "Sii": "#9b59b6",
    "SII": "#9b59b6",
}


def save_stats_plot(
    fits_files: list[Path],
    output_path: Path,
    arcsec_per_pixel: float = 1.0,
    threshold_sigma: float = _DETECTION_THRESHOLD_SIGMA,
    min_snr: float = _MIN_SNR,
    max_ellipticity: float = _MAX_ELLIPTICITY,
) -> tuple[Path, int]:
    """Produce a three-panel plot of per-frame FWHM, eccentricity, and sky brightness vs time.

    Each frame is a dot coloured by filter. Median lines are drawn across.

    Returns (output_path, frames_with_stars).
    """
    import warnings
    from datetime import datetime as _dt
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import matplotlib.dates as mdates
    from fits_processing import sky_brightness as _sb

    filters = [_extract_filter(f) for f in fits_files]

    def _obs_time(f: Path) -> _dt:
        """Read DATE-OBS from header (UTC) and return as naive local datetime."""
        from datetime import timezone as _utc
        try:
            hdr = fits.getheader(f)
            date_obs = hdr.get("DATE-OBS")
            if date_obs:
                utc_dt = _dt.fromisoformat(date_obs.rstrip("Z")).replace(tzinfo=_utc.utc)
                return utc_dt.astimezone().replace(tzinfo=None)
        except Exception:
            pass
        return _dt.fromtimestamp(f.stat().st_mtime)

    def _analyse(f):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, stars = _fit_stars(f, threshold_sigma, min_snr, max_ellipticity)
        fwhm = float(np.mean([s[2] for s in stars])) if stars else 0.0
        ecc  = float(np.mean([s[3] for s in stars])) if stars else 0.0

        sky = _sb.measure_sky(f, arcsec_per_pixel=arcsec_per_pixel)
        sky_val = None
        use_mag = False
        if sky:
            if sky.get("sky_mag_arcsec2") is not None:
                sky_val = sky["sky_mag_arcsec2"]
                use_mag = True
            elif sky.get("sky_adu_per_s") is not None:
                sky_val = sky["sky_adu_per_s"]

        return fwhm, ecc, len(stars), sky_val, use_mag, _obs_time(f)

    results = [None] * len(fits_files)
    max_workers = min(8, len(fits_files))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {pool.submit(_analyse, f): i for i, f in enumerate(fits_files)}
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception:
                results[idx] = (0.0, 0.0, 0, None, False, _dt.fromtimestamp(fits_files[idx].stat().st_mtime))

    # Separate star and sky data; x-axis is observation time.
    times, fwhms, eccs, frame_filters = [], [], [], []
    sky_times, sky_vals, sky_filters = [], [], []
    use_mag = False
    for i, (fwhm, ecc, count, sky_val, is_mag, obs_time) in enumerate(results):
        if count > 0:
            times.append(obs_time)
            fwhms.append(fwhm)
            eccs.append(ecc)
            frame_filters.append(filters[i])
        if sky_val is not None:
            sky_times.append(obs_time)
            sky_vals.append(sky_val)
            sky_filters.append(filters[i])
            if is_mag:
                use_mag = True

    if not fwhms:
        return output_path, 0

    fwhms_arcsec = [f * arcsec_per_pixel for f in fwhms]
    median_fwhm = float(np.median(fwhms_arcsec))
    median_ecc  = float(np.median(eccs))

    unique_filters = sorted(set(frame_filters) | set(sky_filters))
    star_colors = [_FILTER_COLORS.get(f, "#f39c12") for f in frame_filters]
    sky_colors  = [_FILTER_COLORS.get(f, "#f39c12") for f in sky_filters]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # --- FWHM panel ---

    ax1.scatter(times, fwhms_arcsec, c=star_colors, s=30, zorder=3)
    ax1.axhline(median_fwhm, color="white", linestyle="--", linewidth=1, alpha=0.8)
    ax1.text(times[0], median_fwhm, f' median {median_fwhm:.2f}"',
             va="bottom", color="white", fontsize=9)
    ax1.set_ylabel('FWHM (arcsec)')

    # --- Eccentricity panel ---
    ax2.scatter(times, eccs, c=star_colors, s=30, zorder=3)
    ax2.axhline(median_ecc, color="white", linestyle="--", linewidth=1, alpha=0.8)
    ax2.text(times[0], median_ecc, f" median {median_ecc:.3f}",
             va="bottom", color="white", fontsize=9)
    ax2.set_ylabel("Eccentricity")

    # --- Sky brightness panel ---
    if sky_vals:
        median_sky = float(np.median(sky_vals))
        ax3.scatter(sky_times, sky_vals, c=sky_colors, s=30, zorder=3)
        ax3.axhline(median_sky, color="white", linestyle="--", linewidth=1, alpha=0.8)
        ax3.text(sky_times[0], median_sky, f" median {median_sky:.2f}",
                 va="bottom", color="white", fontsize=9)
        if use_mag:
            ax3.invert_yaxis()
            ax3.set_ylabel("Sky (mag/arcsec²)\nhigher = darker")
        else:
            ax3.set_ylabel("Sky (ADU/s)\nlower = darker")
    else:
        ax3.text(0.5, 0.5, "No sky data", transform=ax3.transAxes,
                 ha="center", va="center", color="#aaaaaa", fontsize=10)
        ax3.set_ylabel("Sky brightness")

    ax3.set_xlabel("Time (local)")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Legend for filters.
    handles = []
    for f in unique_filters:
        c = _FILTER_COLORS.get(f, "#f39c12")
        handles.append(plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=c,
                                  markersize=8, label=f))
    ax1.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.7)

    title = f"{len(fwhms)}/{len(fits_files)} frames with stars"
    ax1.set_title(title, color="white")

    fig.patch.set_facecolor("#0d0d1a")
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.grid(True, alpha=0.2)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#444")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, format="jpeg", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path, len(fwhms)


def save_stats_plot_from_cache(
    frames: list[dict],
    output_path: Path,
) -> tuple[Path, int]:
    """4-panel (FWHM / eccentricity / sky / star count) plot from pre-computed frame dicts.

    X-axis is frame number 1-N.  Background bands lightly shade each calendar
    day so multi-night runs are easy to read.  Returns (output_path, frames_with_stars).
    """
    from datetime import datetime as _dt

    output_path = Path(output_path)

    # Defensive: callers may pass frames out of chronological order. Sort by
    # observation time so the per-day background bands stay contiguous (frames
    # with an unparseable time are kept, ordered last, in their original order).
    def _sort_key(d: dict) -> tuple:
        try:
            return (0, _dt.fromisoformat(d["time"]))
        except Exception:
            return (1, _dt.max)
    frames = sorted(frames, key=_sort_key)

    # Assign sequential frame numbers across all frames (preserving order)
    n_total = len(frames)
    fwhm_nums, fwhms, eccs, star_colors = [], [], [], []
    sky_nums, sky_vals, sky_colors = [], [], []
    star_nums, star_counts, star_count_colors = [], [], []
    frame_dates: list = []   # date per frame (may be None)
    use_mag = False

    for i, d in enumerate(frames, 1):
        try:
            t = _dt.fromisoformat(d["time"])
            frame_dates.append(t.date())
        except Exception:
            frame_dates.append(None)

        filt = d.get("filter", "Unknown")
        fwhm = d.get("fwhm_arcsec")
        ecc  = d.get("eccentricity")
        if fwhm is not None:
            fwhm_nums.append(i)
            fwhms.append(float(fwhm))
            eccs.append(float(ecc) if ecc is not None else 0.0)
            star_colors.append(_FILTER_COLORS.get(filt, "#f39c12"))

        mag = d.get("sky_mag_arcsec2")
        adu = d.get("sky_adu_per_s")
        if mag is not None:
            sky_nums.append(i)
            sky_vals.append(float(mag))
            sky_colors.append(_FILTER_COLORS.get(filt, "#f39c12"))
            use_mag = True
        elif adu is not None:
            sky_nums.append(i)
            sky_vals.append(float(adu))
            sky_colors.append(_FILTER_COLORS.get(filt, "#f39c12"))

        sc = d.get("star_count")
        if sc is not None:
            star_nums.append(i)
            star_counts.append(int(sc))
            star_count_colors.append(_FILTER_COLORS.get(filt, "#f39c12"))

    if not fwhms:
        return output_path, 0

    # Build contiguous date bands for background shading
    def _day_bands(dates):
        bands = []
        if not dates:
            return bands
        current, start, idx = dates[0], 1, 0
        for i, d in enumerate(dates[1:], 2):
            if d != current:
                bands.append((start, i - 1, idx))
                current, start, idx = d, i, 1 - idx
        bands.append((start, len(dates), idx))
        return bands

    day_bands = _day_bands(frame_dates)

    median_fwhm = float(np.median(fwhms))
    median_ecc  = float(np.median(eccs))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 13), sharex=True)

    # Day shading — odd-indexed bands get a subtle tint
    for ax in (ax1, ax2, ax3, ax4):
        for start, end, shade_idx in day_bands:
            if shade_idx == 1:
                ax.axvspan(start - 0.5, end + 0.5, color="#5080b0", alpha=0.12, zorder=0)


    def _draw_filter_lines(ax, xs, ys):
        ax.plot(xs, ys, color="white", linewidth=0.8, alpha=0.35, zorder=2)

    _draw_filter_lines(ax1, fwhm_nums, fwhms)
    ax1.scatter(fwhm_nums, fwhms, c=star_colors, s=30, zorder=3)
    ax1.axhline(median_fwhm, color="white", linestyle="--", linewidth=1, alpha=0.8)
    ax1.text(fwhm_nums[0], median_fwhm, f' median {median_fwhm:.2f}"',
             va="bottom", color="white", fontsize=9)
    ax1.set_ylabel('FWHM (arcsec)')
    ax1.set_title(f"{len(fwhms)} frames · {dso_label(frames)}", color="white")

    _draw_filter_lines(ax2, fwhm_nums, eccs)
    ax2.scatter(fwhm_nums, eccs, c=star_colors, s=30, zorder=3)
    ax2.axhline(median_ecc, color="white", linestyle="--", linewidth=1, alpha=0.8)
    ax2.text(fwhm_nums[0], median_ecc, f" median {median_ecc:.3f}",
             va="bottom", color="white", fontsize=9)
    ax2.set_ylabel("Eccentricity")

    if sky_vals:
        median_sky = float(np.median(sky_vals))
        _draw_filter_lines(ax3, sky_nums, sky_vals)
        ax3.scatter(sky_nums, sky_vals, c=sky_colors, s=30, zorder=3)
        ax3.axhline(median_sky, color="white", linestyle="--", linewidth=1, alpha=0.8)
        ax3.text(sky_nums[0], median_sky, f" median {median_sky:.2f}",
                 va="bottom", color="white", fontsize=9)
        if use_mag:
            ax3.invert_yaxis()
            ax3.set_ylabel("Sky (mag/arcsec²)\nhigher = darker")
        else:
            ax3.set_ylabel("Sky (ADU/s)\nlower = darker")
    else:
        ax3.text(0.5, 0.5, "No sky data", transform=ax3.transAxes,
                 ha="center", va="center", color="#aaaaaa", fontsize=10)
        ax3.set_ylabel("Sky brightness")

    if star_counts:
        median_stars = float(np.median(star_counts))
        _draw_filter_lines(ax4, star_nums, star_counts)
        ax4.scatter(star_nums, star_counts, c=star_count_colors, s=30, zorder=3)
        ax4.axhline(median_stars, color="white", linestyle="--", linewidth=1, alpha=0.8)
        ax4.text(star_nums[0], median_stars, f" median {median_stars:.0f}",
                 va="bottom", color="white", fontsize=9)
        ax4.set_ylabel("Stars detected")
    else:
        ax4.text(0.5, 0.5, "No star count data", transform=ax4.transAxes,
                 ha="center", va="center", color="#aaaaaa", fontsize=10)
        ax4.set_ylabel("Stars detected")

    ax4.set_xlabel("Frame")
    ax4.set_xlim(0.5, n_total + 0.5)

    fig.patch.set_facecolor("#0d0d1a")
    for ax in (ax1, ax2, ax3, ax4):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.grid(True, alpha=0.2)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#444")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, format="jpeg", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path, len(fwhms)


def dso_label(frames: list[dict]) -> str:
    """Extract a date-range label from the frame list for the plot title."""
    from datetime import datetime as _dt
    dates = set()
    for d in frames:
        try:
            dates.add(_dt.fromisoformat(d["time"]).date())
        except Exception:
            pass
    if not dates:
        return ""
    lo, hi = min(dates), max(dates)
    return str(lo) if lo == hi else f"{lo} – {hi}"


if __name__ == "__main__":
    import sys
    import os
    if __package__ is None or __package__ == "":
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
    from configs import config

    cfg = config.data()
    arcsec_per_pixel = cfg["nina"]["arc_sec_per_pixel"]

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/taylor/Documents/srt/local/2026-03-02_01-13-02_R_116_300.00s_0040.fits")
    threshold_sigma = 8.0
    min_snr = 10.0
    max_ellipticity = 4.0
    mean_pixels, mean_arcsec, count, mean_ecc = calculate_fwhm(
        path, arcsec_per_pixel=arcsec_per_pixel, threshold_sigma=threshold_sigma, min_snr=min_snr, max_ellipticity=max_ellipticity
    )
    print(f"Stars found      : {count}")
    print(f'Mean FWHM        : {mean_arcsec:.2f}"')
    print(f"Mean eccentricity: {mean_ecc:.3f}")
    display_fwhm(path, arcsec_per_pixel=arcsec_per_pixel, threshold_sigma=threshold_sigma, min_snr=min_snr, max_ellipticity=max_ellipticity)
