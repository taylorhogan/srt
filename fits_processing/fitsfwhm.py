from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.utils import iers
iers.conf.auto_download = False
from astropy.io import fits
from astropy.modeling import fitting, models
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ZScaleInterval
from matplotlib.patches import Circle, Patch
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
        (image_data, [(x, y, fwhm_pixels), ...])
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
    stars: list[tuple[float, float, float]] = []

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

            stars.append((float(xc), float(yc), fwhm))
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
        (mean_fwhm_pixels, mean_fwhm_arcsec, star_count)
        Values are 0.0 / 0 if no stars are found.
    """
    _, stars = _fit_stars(fits_path, threshold_sigma, min_snr, max_ellipticity)
    if not stars:
        return 0.0, 0.0, 0
    mean_px = float(np.mean([s[2] for s in stars]))
    return mean_px, mean_px * arcsec_per_pixel, len(stars)


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

    vmin, vmax = ZScaleInterval().get_limits(data)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")

    for x, y, fwhm in stars:
        if fwhm > mean_px * 1.1:
            color = "red"
        elif fwhm < mean_px * 0.9:
            color = "green"
        else:
            color = "yellow"
        ax.add_patch(Circle((x, y), radius=fwhm * 2.5, color=color, fill=False, linewidth=1.5))

    ax.set_title(
        f"{fits_path.name}  |  {len(stars)} stars  |  "
        f"mean FWHM {mean_px:.2f} px  ({mean_px * arcsec_per_pixel:.2f}\")"
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
) -> Path:
    """
    Annotate a FITS image with detected stars and save it as a JPG.

    Returns the output_path written.
    """
    data, stars = _fit_stars(fits_path, threshold_sigma, min_snr, max_ellipticity)

    mean_px = float(np.mean([s[2] for s in stars])) if stars else 0.0

    vmin, vmax = ZScaleInterval().get_limits(data)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")

    for x, y, fwhm in stars:
        if fwhm > mean_px * 1.1:
            color = "red"
        elif fwhm < mean_px * 0.9:
            color = "green"
        else:
            color = "yellow"
        ax.add_patch(Circle((x, y), radius=fwhm * 2.5, color=color, fill=False, linewidth=1.5))

    title = (
        f"{fits_path.name}  |  {len(stars)} stars  |  "
        f"mean FWHM {mean_px:.2f} px  ({mean_px * arcsec_per_pixel:.2f}\")"
        if stars else f"{fits_path.name}  |  no stars detected"
    )
    ax.set_title(title)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    if stars:
        ax.legend(handles=[
            Patch(facecolor="none", edgecolor="green",  label=f"Good  (< {mean_px * 0.9:.1f} px)"),
            Patch(facecolor="none", edgecolor="yellow", label=f"OK    ({mean_px * 0.9:.1f} – {mean_px * 1.1:.1f} px)"),
            Patch(facecolor="none", edgecolor="red",    label=f"Soft  (> {mean_px * 1.1:.1f} px)"),
        ], loc="upper right")

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
    mean_pixels, mean_arcsec, count = calculate_fwhm(
        path, arcsec_per_pixel=arcsec_per_pixel, threshold_sigma=threshold_sigma, min_snr=min_snr, max_ellipticity=max_ellipticity
    )
    print(f"Stars found : {count}")
    print(f"Mean FWHM   : {mean_pixels:.2f} px  ({mean_arcsec:.2f}\")")
    display_fwhm(path, arcsec_per_pixel=arcsec_per_pixel, threshold_sigma=threshold_sigma, min_snr=min_snr, max_ellipticity=max_ellipticity)
