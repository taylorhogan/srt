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
        cb.set_label(cbar_label)
        ax.set_title(title)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, format="jpeg", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if not stars:
        for out_path, label in [(fwhm_output_path, "FWHM"), (ecc_output_path, "Eccentricity")]:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
                      interpolation="nearest")
            ax.set_title(f"{fits_path.name}  |  {label} grid heatmap  |  no stars detected")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(out_path, format="jpeg", dpi=150, bbox_inches="tight")
            plt.close(fig)
        return fwhm_output_path, ecc_output_path

    xs = np.array([s[0] for s in stars])
    ys = np.array([s[1] for s in stars])
    fwhms = np.array([s[2] for s in stars])
    eccs = np.array([s[3] for s in stars])

    # --- FWHM grid ---
    fwhm_grid = _make_grid(xs, ys, fwhms, img_h, img_w, n_cells)
    median_fwhm = float(np.median(fwhms))
    _render(
        fwhm_output_path, "FWHM",
        fwhm_grid,
        Normalize(vmin=np.nanmin(fwhm_grid), vmax=np.nanmax(fwhm_grid)),
        "FWHM (px)",
        f"{fits_path.name}  |  FWHM grid  |  {len(stars)} stars  |  "
        f"median {median_fwhm:.2f} px ({median_fwhm * arcsec_per_pixel:.2f}\")",
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
        arrow_scale = max(data.shape) * 0.05

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
    ax.axis("off")
    ax.set_title(
        f"Optical Quality  |  {m['star_count']} stars  |  "
        f"FWHM {m['median_fwhm_px']:.2f}px ({m['median_fwhm_arcsec']:.2f}\")  |  "
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
    plt.savefig(output_path, format="jpeg", dpi=150, bbox_inches="tight")
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
    """Produce a two-panel plot of per-frame FWHM and eccentricity.

    Each frame is a dot coloured by filter. Median lines are drawn across.

    Returns (output_path, frames_with_stars).
    """
    import warnings
    from concurrent.futures import ThreadPoolExecutor, as_completed

    filters = [_extract_filter(f) for f in fits_files]

    def _analyse(f):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, stars = _fit_stars(f, threshold_sigma, min_snr, max_ellipticity)
        if not stars:
            return 0.0, 0.0, 0
        return (
            float(np.mean([s[2] for s in stars])),
            float(np.mean([s[3] for s in stars])),
            len(stars),
        )

    results = [None] * len(fits_files)
    max_workers = min(8, len(fits_files))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {pool.submit(_analyse, f): i for i, f in enumerate(fits_files)}
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception:
                results[idx] = (0.0, 0.0, 0)

    # Filter to frames with detected stars, keeping original index for x-axis.
    indices, fwhms, eccs, frame_filters = [], [], [], []
    for i, (fwhm, ecc, count) in enumerate(results):
        if count > 0:
            indices.append(i)
            fwhms.append(fwhm)
            eccs.append(ecc)
            frame_filters.append(filters[i])

    if not fwhms:
        return output_path, 0

    fwhms_arcsec = [f * arcsec_per_pixel for f in fwhms]
    fwhms_arr = np.array(fwhms_arcsec)
    eccs_arr = np.array(eccs)
    median_fwhm = float(np.median(fwhms_arr))
    median_ecc = float(np.median(eccs_arr))

    # Assign colours by filter.
    unique_filters = sorted(set(frame_filters))
    colors = [_FILTER_COLORS.get(f, "#f39c12") for f in frame_filters]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.scatter(indices, fwhms_arcsec, c=colors, s=30, zorder=3)
    ax1.axhline(median_fwhm, color="white", linestyle="--", linewidth=1, alpha=0.8)
    ax1.text(len(fits_files) * 0.01, median_fwhm, f' median {median_fwhm:.2f}"',
             va="bottom", color="white", fontsize=9)
    ax1.set_ylabel('FWHM (arcsec)')
    ax1.set_facecolor("#1a1a2e")
    ax1.grid(True, alpha=0.2)

    ax2.scatter(indices, eccs, c=colors, s=30, zorder=3)
    ax2.axhline(median_ecc, color="white", linestyle="--", linewidth=1, alpha=0.8)
    ax2.text(len(fits_files) * 0.01, median_ecc, f" median {median_ecc:.3f}",
             va="bottom", color="white", fontsize=9)
    ax2.set_ylabel("Eccentricity")
    ax2.set_xlabel("Frame")
    ax2.set_facecolor("#1a1a2e")
    ax2.grid(True, alpha=0.2)

    # Legend for filters.
    handles = []
    for f in unique_filters:
        c = _FILTER_COLORS.get(f, "#f39c12")
        handles.append(plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=c,
                                  markersize=8, label=f))
    ax1.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.7)

    fig.patch.set_facecolor("#0d0d1a")
    ax1.tick_params(colors="white")
    ax2.tick_params(colors="white")
    for ax in (ax1, ax2):
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#444")

    title = f"{len(fwhms)}/{len(fits_files)} frames with stars"
    ax1.set_title(title, color="white")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, format="jpeg", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path, len(fwhms)


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
