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
_STAMP_HALF = 15

# Fitted Gaussian amplitude must be at least this many × background std
_MIN_SNR = 10.0

# Ratio of the larger axis stddev to the smaller must not exceed this (1.0 = perfect circle)
_MAX_ELLIPTICITY = 2.0


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
        (image_data, [(x, y, fwhm_pixels, eccentricity), ...])
        Eccentricity is sqrt(1 - (short_axis/long_axis)^2); 0 = perfect circle, 1 = line.
    """
    with fits.open(fits_path) as hdul:
        raw = hdul[0].data.astype(float)

    data = np.squeeze(raw)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D image data after squeeze, got shape {data.shape}")

    _, median, std = sigma_clipped_stats(data, sigma=3.0)
    background_subtracted = data - median

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
            fwhm = _FWHM_SIGMA_FACTOR * (sigma_x + sigma_y) / 2.0

            # Reject bad FWHM range
            if not (0.5 < fwhm < _STAMP_HALF):
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
            stars.append((float(xc), float(yc), fwhm, ecc, major_angle))
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

    ax.set_title(
        f"{fits_path.name}  |  {len(stars)} stars  |  "
        f"mean FWHM {mean_px:.2f} px  ({mean_px * arcsec_per_pixel:.2f}\")  |  "
        f"mean ecc {mean_ecc:.3f}"
    )
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.legend(handles=[
        Patch(facecolor="none", edgecolor="green",  label=f"Good  (< {mean_px * 0.9:.1f} px / -{mean_px * arcsec_per_pixel * 0.1:.2f}\")"),
        Patch(facecolor="none", edgecolor="yellow", label=f"OK    ({mean_px * 0.9:.1f} – {mean_px * 1.1:.1f} px)"),
        Patch(facecolor="none", edgecolor="red",    label=f"Soft  (> {mean_px * 1.1:.1f} px / +{mean_px * arcsec_per_pixel * 0.1:.2f}\")"),
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

    title = (
        f"{fits_path.name}  |  {len(stars)} stars  |  "
        f"mean FWHM {mean_px:.2f} px  ({mean_px * arcsec_per_pixel:.2f}\")  |  "
        f"mean ecc {mean_ecc:.3f}"
        if stars else f"{fits_path.name}  |  no stars detected"
    )
    ax.set_title(title)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    if annotate and stars:
        ax.legend(handles=[
            Patch(facecolor="none", edgecolor="green",  label=f"Good  (< {mean_px * 0.9:.1f} px)"),
            Patch(facecolor="none", edgecolor="yellow", label=f"OK    ({mean_px * 0.9:.1f} – {mean_px * 1.1:.1f} px)"),
            Patch(facecolor="none", edgecolor="red",    label=f"Soft  (> {mean_px * 1.1:.1f} px)"),
        ], loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, format="jpeg", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path, mean_px, mean_ecc


def save_fwhm_heatmaps(
    fits_path: Path,
    fwhm_output_path: Path,
    ecc_output_path: Path,
    arcsec_per_pixel: float = 1.0,
    threshold_sigma: float = _DETECTION_THRESHOLD_SIGMA,
    min_snr: float = _MIN_SNR,
    max_ellipticity: float = _MAX_ELLIPTICITY,
) -> tuple[Path, Path]:
    """
    Produce two heatmap images showing per-star FWHM and eccentricity
    as coloured scatter points overlaid on the FITS image.

    Returns (fwhm_output_path, ecc_output_path).
    """
    data, stars = _fit_stars(fits_path, threshold_sigma, min_snr, max_ellipticity)
    vmin, vmax = ZScaleInterval().get_limits(data)

    if not stars:
        # Write blank images with a "no stars" message
        for out_path, label in [(fwhm_output_path, "FWHM"), (ecc_output_path, "Eccentricity")]:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(f"{fits_path.name}  |  {label} heatmap  |  no stars detected")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(out_path, format="jpeg", dpi=150, bbox_inches="tight")
            plt.close(fig)
        return fwhm_output_path, ecc_output_path

    xs = np.array([s[0] for s in stars])
    ys = np.array([s[1] for s in stars])
    fwhms = np.array([s[2] for s in stars])
    eccs = np.array([s[3] for s in stars])

    dot_size = max(data.shape) / 30

    # --- FWHM heatmap ---
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    norm_fwhm = Normalize(vmin=fwhms.min(), vmax=fwhms.max())
    sc = ax.scatter(xs, ys, c=fwhms, cmap="RdYlGn_r", norm=norm_fwhm, s=dot_size, alpha=0.85, linewidths=0)
    cb = plt.colorbar(ScalarMappable(norm=norm_fwhm, cmap="RdYlGn_r"), ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("FWHM (px)")
    mean_px = float(np.median(fwhms))
    ax.set_title(
        f"{fits_path.name}  |  FWHM heatmap  |  {len(stars)} stars  |  "
        f"median {mean_px:.2f} px ({mean_px * arcsec_per_pixel:.2f}\")"
    )
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    fwhm_output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fwhm_output_path, format="jpeg", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Eccentricity heatmap ---
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    norm_ecc = Normalize(vmin=0.0, vmax=max(eccs.max(), 0.5))
    sc = ax.scatter(xs, ys, c=eccs, cmap="RdYlGn_r", norm=norm_ecc, s=dot_size, alpha=0.85, linewidths=0)
    cb = plt.colorbar(ScalarMappable(norm=norm_ecc, cmap="RdYlGn_r"), ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Eccentricity (0=round, 1=elongated)")
    mean_ecc = float(np.median(eccs))
    ax.set_title(
        f"{fits_path.name}  |  Eccentricity heatmap  |  {len(stars)} stars  |  "
        f"median ecc {mean_ecc:.3f}"
    )
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ecc_output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(ecc_output_path, format="jpeg", dpi=150, bbox_inches="tight")
    plt.close(fig)

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

    if not stars:
        ax.set_title(f"{fits_path.name}  |  FWHM vs distance  |  no stars detected")
    else:
        cy, cx = data.shape[0] / 2.0, data.shape[1] / 2.0
        distances = np.array([np.hypot(s[0] - cx, s[1] - cy) for s in stars])
        fwhms = np.array([s[2] for s in stars])

        ax.scatter(distances, fwhms, s=18, alpha=0.7, color="steelblue", linewidths=0)

        # Linear trend line
        if len(stars) >= 2:
            coeffs = np.polyfit(distances, fwhms, 1)
            x_line = np.linspace(distances.min(), distances.max(), 200)
            ax.plot(x_line, np.polyval(coeffs, x_line), color="tomato", linewidth=1.5,
                    label=f"trend  slope={coeffs[0]:+.4f} px/px")
            ax.legend(fontsize=9)

        median_fwhm = float(np.median(fwhms))
        ax.axhline(median_fwhm, color="gray", linewidth=1, linestyle="--",
                   label=f"median {median_fwhm:.2f} px")
        ax.set_xlabel("Distance from centre (px)")
        ax.set_ylabel(f"FWHM (px)   [1 px = {arcsec_per_pixel:.3f}\"]")
        ax.set_title(
            f"{fits_path.name}  |  FWHM vs distance  |  {len(stars)} stars  |  "
            f"median {median_fwhm:.2f} px ({median_fwhm * arcsec_per_pixel:.2f}\")"
        )
        ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, format="jpeg", dpi=150, bbox_inches="tight")
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
    ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")

    if not stars:
        ax.set_title(f"{fits_path.name}  |  Eccentricity angle map  |  no stars detected")
    else:
        eccs = np.array([s[3] for s in stars])
        norm_ecc = Normalize(vmin=0.0, vmax=max(eccs.max(), 0.5))
        cmap = plt.get_cmap("RdYlGn_r")

        # Arrow length is proportional to eccentricity, scaled to ~2 % of image width
        arrow_scale = max(data.shape) * 0.025

        for x, y, fwhm, ecc, angle in stars:
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
        cb.set_label("Eccentricity (0=round, 1=elongated)")
        mean_ecc = float(np.median(eccs))
        ax.set_title(
            f"{fits_path.name}  |  Elongation angle map  |  {len(stars)} stars  |  "
            f"median ecc {mean_ecc:.3f}"
        )

    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, format="jpeg", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


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
    print(f"Mean FWHM        : {mean_pixels:.2f} px  ({mean_arcsec:.2f}\")")
    print(f"Mean eccentricity: {mean_ecc:.3f}")
    display_fwhm(path, arcsec_per_pixel=arcsec_per_pixel, threshold_sigma=threshold_sigma, min_snr=min_snr, max_ellipticity=max_ellipticity)
