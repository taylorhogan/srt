"""
Color–magnitude (H–R) diagram from imaging subs, calibrated against Gaia DR3.

A true Hertzsprung–Russell diagram plots luminosity vs temperature; from images
we build its observable cousin, a *color–magnitude diagram* (CMD): a colour index
(temperature proxy, x-axis) vs an apparent magnitude (brightness proxy, y-axis).
That needs two filters.

Pipeline (see :func:`build_cmd`):
  1. Stack the blue and red filter subs separately (registered).
  2. Plate-solve each stack with ASTAP → a WCS.
  3. Detect + aperture-measure stars in each stack (``sep``) → instrumental mags.
  4. Cross-match the two filters' star lists by sky position → one colour per star.
  5. One Gaia DR3 cone query over the field; cross-match to set a per-filter zero
     point (blue→BP, red→RP), so the axes sit on the Gaia photometric system.
  6. Plot calibrated colour vs magnitude (brightest at top), with the Gaia field
     CMD drawn faintly behind as a reference sequence.

Everything is fail-safe at the edges (missing solver, no Gaia, too few stars):
the caller gets a clear message rather than a traceback. The best results come
from a star *cluster* (open or globular), where all stars share one distance and
the main sequence / giant branch / turnoff actually line up.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np

_logger = logging.getLogger(__name__)

# Rough blue→red ordering by effective wavelength, used to decide which of two
# filters is the "blue" and which the "red" when the caller doesn't say.
_FILTER_WAVELENGTH_RANK = {
    "B": 1, "OIII": 2, "G": 3, "CLEAR": 4, "L": 4, "R": 5, "HA": 6, "SII": 7,
}


def filter_rank(name: str) -> int:
    """Blue→red ordering rank for a filter name (lower = bluer). Unknown → 4."""
    return _FILTER_WAVELENGTH_RANK.get(name.strip().upper(), 4)


def choose_filters(
    by_filter: dict[str, list],
    min_frames: int = 1,
) -> Optional[tuple[str, str]]:
    """Pick a (blue, red) filter pair from the available filter→paths mapping.

    Prefers the two filters with the most subs, ordered blue→red by effective
    wavelength. Returns None if fewer than two usable filters are present.
    """
    usable = {f: p for f, p in by_filter.items() if len(p) >= min_frames}
    if len(usable) < 2:
        return None
    # Two most-imaged filters, then order by wavelength rank.
    top2 = sorted(usable, key=lambda f: len(usable[f]), reverse=True)[:2]
    blue, red = sorted(top2, key=filter_rank)
    if filter_rank(blue) == filter_rank(red):
        return None
    return blue, red


def _plate_solve_wcs(
    data: np.ndarray,
    ra_deg: Optional[float],
    dec_deg: Optional[float],
    fov_deg: float,
    astap_exe: str,
    progress_cb: Optional[Callable[[str], None]] = None,
):
    """Plate-solve a stacked image array with ASTAP and return an astropy WCS.

    Writes the array to a throwaway FITS (the imaging data is never touched),
    passing the field centre + field-of-view as hints. Returns None on any
    failure (missing solver, no solution, unreadable WCS).
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    if not astap_exe or not os.path.exists(astap_exe):
        if progress_cb:
            progress_cb("ASTAP not found — cannot plate-solve")
        return None

    tmp_dir = tempfile.mkdtemp(prefix="cmd_solve_")
    tmp_fits = Path(tmp_dir) / "stack.fits"
    try:
        fits.writeto(tmp_fits, np.asarray(data, dtype=np.float32), overwrite=True)
        cmd = [astap_exe, "-f", str(tmp_fits), "-fov", f"{fov_deg:.4f}",
               "-r", "5", "-update"]
        if ra_deg is not None and dec_deg is not None:
            cmd += ["-ra", f"{ra_deg / 15.0:.6f}", "-spd", f"{dec_deg + 90.0:.6f}"]
        subprocess.run(cmd, capture_output=True, timeout=300)
        solved = fits.getheader(tmp_fits)
        if "CRVAL1" not in solved:
            if progress_cb:
                progress_cb("plate solve failed (no WCS in solution)")
            return None
        return WCS(solved)
    except Exception:
        _logger.exception("cmd_diagram plate solve failed")
        return None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _detect_and_measure(data: np.ndarray, thresh_sigma: float = 5.0):
    """Detect stars and aperture-measure background-subtracted flux with sep.

    Returns (x, y, instrumental_mag) arrays for stars with positive flux. The
    aperture radius is scaled to the typical stellar profile in the frame.
    """
    import sep

    data = np.ascontiguousarray(data, dtype=np.float32)
    bkg = sep.Background(data)
    data_sub = data - bkg.back()
    objs = sep.extract(data_sub, thresh=thresh_sigma, err=bkg.globalrms)
    if len(objs) == 0:
        return np.array([]), np.array([]), np.array([])

    # Aperture ~ a couple of profile sigmas; sep 'a','b' are profile RMS sizes.
    ab = np.sqrt(np.clip(objs["a"] * objs["b"], 1e-6, None))
    aper_r = float(np.median(ab)) * 2.5
    aper_r = max(2.5, min(aper_r, 12.0))

    flux, _flux_err, flag = sep.sum_circle(
        data_sub, objs["x"], objs["y"], aper_r, gain=1.0
    )
    flux = np.asarray(flux, dtype=np.float64)
    good = np.isfinite(flux) & (flux > 0) & (np.asarray(flag) == 0)
    x = np.asarray(objs["x"])[good]
    y = np.asarray(objs["y"])[good]
    mag = -2.5 * np.log10(flux[good])
    return x, y, mag


def _robust_zero_point(offsets: np.ndarray, n_sigma: float = 3.0) -> Optional[float]:
    """Sigma-clipped median of (catalog − instrumental) magnitude offsets."""
    vals = offsets[np.isfinite(offsets)]
    if vals.size < 5:
        return None
    for _ in range(3):
        med = np.median(vals)
        sd = np.std(vals)
        if sd == 0:
            break
        keep = np.abs(vals - med) < n_sigma * sd
        if keep.all() or keep.sum() < 5:
            break
        vals = vals[keep]
    return float(np.median(vals))


def build_cmd(
    blue_paths: list[Path],
    red_paths: list[Path],
    blue_name: str,
    red_name: str,
    dso_name: str,
    output_path: Path,
    astap_exe: str,
    arcsec_per_pixel: float,
    match_radius_arcsec: float = 2.0,
    gaia_mag_limit: float = 19.0,
    progress_cb: Optional[Callable[[str], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
    precomputed_fwhm_stars: Optional[dict] = None,
    observatory_name: str = "this telescope",
) -> dict:
    """Build a Gaia-calibrated colour–magnitude diagram and save it as a JPEG.

    Returns a stats dict: ``{n_stars, n_gaia, zp_blue, zp_red, output}``.
    Raises ``RuntimeError`` with a human-readable message on any step that makes
    a diagram impossible (no WCS, too few stars, no Gaia matches).
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from stacking import stacker

    def _say(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    def _ckpt() -> None:
        if cancel_cb and cancel_cb():
            from utils.cancellation import Cancelled
            raise Cancelled()

    # ── 1. Stack both filters ────────────────────────────────────────────────
    stacks: dict[str, np.ndarray] = {}
    headers: dict[str, fits.Header] = {}
    for name, paths in ((blue_name, blue_paths), (red_name, red_paths)):
        _ckpt()
        _say(f"stacking {len(paths)} {name} subs…")
        data, _info = stacker.stack(
            paths, register=True, progress_cb=lambda m, n=name: _say(f"[{n}] {m}"),
            cancel_cb=cancel_cb, precomputed_fwhm_stars=precomputed_fwhm_stars,
        )
        stacks[name] = data
        headers[name] = fits.getheader(paths[0])

    # ── 2. Plate-solve each stack → WCS ──────────────────────────────────────
    h, w = stacks[red_name].shape
    fov_deg = (max(h, w) * arcsec_per_pixel) / 3600.0
    wcs: dict = {}
    for name in (blue_name, red_name):
        _ckpt()
        _say(f"plate-solving {name} stack with ASTAP…")
        hdr = headers[name]
        try:
            ra_deg = float(hdr.get("RA"))
            dec_deg = float(hdr.get("DEC"))
        except (TypeError, ValueError):
            ra_deg = dec_deg = None
        solved = _plate_solve_wcs(
            stacks[name], ra_deg, dec_deg, fov_deg, astap_exe, progress_cb
        )
        if solved is None:
            raise RuntimeError(
                f"Could not plate-solve the {name} stack — check ASTAP and its "
                f"star database. A CMD needs a WCS to match stars to Gaia."
            )
        wcs[name] = solved

    # ── 3. Detect + measure stars in each stack ──────────────────────────────
    stars: dict = {}
    for name in (blue_name, red_name):
        _ckpt()
        x, y, mag = _detect_and_measure(stacks[name])
        if x.size < 10:
            raise RuntimeError(f"Only {x.size} stars measured in the {name} "
                               f"stack — too few for a diagram.")
        sky = wcs[name].pixel_to_world(x, y)
        stars[name] = {"mag": mag, "sky": SkyCoord(sky)}
        _say(f"{name}: measured {x.size} stars")

    # ── 4. Cross-match blue ↔ red by sky position ────────────────────────────
    _ckpt()
    idx, sep2d, _ = stars[blue_name]["sky"].match_to_catalog_sky(stars[red_name]["sky"])
    paired = sep2d.arcsec <= match_radius_arcsec
    if paired.sum() < 10:
        raise RuntimeError(f"Only {int(paired.sum())} stars matched between "
                           f"{blue_name} and {red_name}. Are the frames of the "
                           f"same field, and did both solve?")
    blue_mag = stars[blue_name]["mag"][paired]
    red_mag = stars[red_name]["mag"][idx[paired]]
    pair_sky = stars[blue_name]["sky"][paired]
    _say(f"matched {paired.sum()} stars across both filters")

    # ── 5. Gaia field query + per-filter zero point (blue→BP, red→RP) ─────────
    _ckpt()
    _say("querying Gaia DR3 for the field…")
    center = SkyCoord(pair_sky.ra.mean(), pair_sky.dec.mean())
    radius_deg = fov_deg * 0.6  # half-FOV plus margin
    gaia_tbl = _query_gaia(center, radius_deg, gaia_mag_limit, progress_cb)
    if gaia_tbl is None or len(gaia_tbl) < 5:
        raise RuntimeError("Gaia query returned too few sources — no calibration "
                           "possible (check network / astroquery).")
    gaia_sky = SkyCoord(ra=np.asarray(gaia_tbl["ra"]) * u.deg,
                        dec=np.asarray(gaia_tbl["dec"]) * u.deg)
    g_bp = np.asarray(gaia_tbl["phot_bp_mean_mag"], dtype=np.float64)
    g_rp = np.asarray(gaia_tbl["phot_rp_mean_mag"], dtype=np.float64)
    g_g = np.asarray(gaia_tbl["phot_g_mean_mag"], dtype=np.float64)

    g_idx, g_sep, _ = pair_sky.match_to_catalog_sky(gaia_sky)
    has_gaia = g_sep.arcsec <= match_radius_arcsec
    zp_blue = _robust_zero_point(g_bp[g_idx[has_gaia]] - blue_mag[has_gaia])
    zp_red = _robust_zero_point(g_rp[g_idx[has_gaia]] - red_mag[has_gaia])
    if zp_blue is None or zp_red is None:
        raise RuntimeError(f"Only {int(has_gaia.sum())} stars matched to Gaia — "
                           f"not enough to calibrate. A star cluster gives the "
                           f"cleanest result.")

    color = (blue_mag + zp_blue) - (red_mag + zp_red)
    mag = red_mag + zp_red
    _say(f"calibrated: ZP({blue_name})={zp_blue:.2f}, ZP({red_name})={zp_red:.2f}; "
         f"{int(has_gaia.sum())} Gaia anchors")

    # ── 6. Plot ──────────────────────────────────────────────────────────────
    _ckpt()
    _plot_cmd(color, mag, g_bp - g_rp, g_g, blue_name, red_name, dso_name,
              int(has_gaia.sum()), output_path, observatory_name=observatory_name)

    return {
        "n_stars": int(color.size),
        "n_gaia": int(has_gaia.sum()),
        "zp_blue": round(zp_blue, 3),
        "zp_red": round(zp_red, 3),
        "output": str(output_path),
    }


def _query_gaia(center, radius_deg: float, mag_limit: float, progress_cb):
    """Cone-search Gaia DR3 for sources in the field, brighter than mag_limit."""
    try:
        from astroquery.gaia import Gaia
        adql = (
            "SELECT source_id, ra, dec, phot_g_mean_mag, "
            "phot_bp_mean_mag, phot_rp_mean_mag "
            "FROM gaiadr3.gaia_source "
            "WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {center.ra.deg:.6f}, {center.dec.deg:.6f}, {radius_deg:.5f})) "
            f"AND phot_g_mean_mag < {mag_limit} "
            "AND phot_bp_mean_mag IS NOT NULL AND phot_rp_mean_mag IS NOT NULL"
        )
        job = Gaia.launch_job_async(adql)
        return job.get_results()
    except Exception:
        _logger.exception("Gaia query failed")
        if progress_cb:
            progress_cb("Gaia query failed")
        return None


def _plot_cmd(color, mag, gaia_color, gaia_g, blue_name, red_name, dso_name,
              n_gaia, output_path: Path, observatory_name: str = "this telescope") -> None:
    """Render the colour–magnitude diagram to a JPEG (dark theme).

    Three panels sharing axes for direct comparison: the measured stars alone,
    the Gaia reference sequence alone, and the two overlaid.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    BG, GAIA, FG = "#0d1117", "#30475e", "#c9d1d9"
    CMAP = "RdYlBu_r"  # blue (small B−R, hot) → red (large B−R, cool)

    finite_g = np.isfinite(gaia_color) & np.isfinite(gaia_g)
    g_color, g_mag = gaia_color[finite_g], gaia_g[finite_g]
    color, mag = np.asarray(color), np.asarray(mag)

    # Shared colour/magnitude ranges over both populations: the panels line up,
    # and a measured point's hue matches its x-position (same scale as the axis).
    all_color = np.concatenate([color, g_color]) if g_color.size else color
    all_mag = np.concatenate([mag, g_mag]) if g_mag.size else mag
    cmin, cmax = (float(np.nanmin(all_color)), float(np.nanmax(all_color))) if all_color.size else (0.0, 1.0)
    mmin, mmax = (float(np.nanmin(all_mag)), float(np.nanmax(all_mag))) if all_mag.size else (0.0, 1.0)

    def _draw_gaia(ax, alpha):
        if g_color.size:
            ax.scatter(g_color, g_mag, s=6, c=GAIA, alpha=alpha, linewidths=0)

    def _draw_measured(ax):
        # False-colour each star by its colour index: blue = hot, red = cool.
        return ax.scatter(color, mag, s=14, c=color, cmap=CMAP, vmin=cmin, vmax=cmax,
                          alpha=0.9, linewidths=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 8), dpi=110,
                             sharex=True, sharey=True, constrained_layout=True)
    fig.patch.set_facecolor(BG)

    sc = _draw_measured(axes[0])
    axes[0].set_title(f"Measured ({observatory_name}) — {color.size} stars", color="#e6edf3")
    _draw_gaia(axes[1], 0.6)
    axes[1].set_title(f"Gaia DR3 reference — {g_color.size} stars", color="#e6edf3")
    _draw_gaia(axes[2], 0.45)          # faint, behind…
    _draw_measured(axes[2])            # …measured on top
    axes[2].set_title("Combined (overlay)", color="#e6edf3")

    if all_color.size:
        cpad, mpad = 0.05 * (cmax - cmin + 1e-6), 0.05 * (mmax - mmin + 1e-6)
        axes[0].set_xlim(cmin - cpad, cmax + cpad)
        axes[0].set_ylim(mmax + mpad, mmin - mpad)  # inverted: brighter at top

    for i, ax in enumerate(axes):
        ax.set_facecolor(BG)
        ax.set_xlabel(f"colour  {blue_name} − {red_name}  (Gaia BP/RP scale)", color=FG)
        if i == 0:
            ax.set_ylabel(f"magnitude  {red_name}  (Gaia RP scale)", color=FG)
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.grid(True, color="#21262d", linewidth=0.6)

    # Grey proxy legend on the combined panel so the backdrop is labelled.
    if g_color.size:
        proxy = Line2D([], [], marker="o", linestyle="", markersize=6,
                       markerfacecolor=GAIA, markeredgecolor="none", label="Gaia DR3 field")
        leg = axes[2].legend(handles=[proxy], facecolor="#161b22", edgecolor="#30363d",
                             labelcolor=FG, fontsize=9, loc="upper right")
        leg.get_frame().set_alpha(0.9)

    # Shared colourbar showing the measured stars' temperature false-colour.
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.025, pad=0.01)
    cbar.set_label(f"measured star colour  {blue_name} − {red_name}   "
                   f"(blue = hot  →  red = cool)", color=FG)
    cbar.ax.tick_params(colors="#8b949e")
    cbar.outline.set_edgecolor("#30363d")

    fig.suptitle(f"Colour–magnitude diagram — {dso_name}   ·   "
                 f"{n_gaia} Gaia calibration stars", color="#e6edf3", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="jpeg", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
