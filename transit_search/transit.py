"""Transit-search side channel.

Given a DSO name and a filter, runs aperture photometry on every star detected
in the field across every saved LIGHT sub-exposure, then searches each star's
differential light curve for transit-like signals (large unstructured dips and
periodic Box-Least-Squares power).

Results are persisted to local/transits.json and a top-N plot is produced for
posting back to the web chat by the calling command.

Reuses the existing FWHM-rejection + astroalign registration pipeline in
stacking/stacker.py (via _prepare_for_convergence) so star positions are
consistent frame to frame — only the photometry, differential correction,
and BLS search are new here.
"""

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from astropy.timeseries import BoxLeastSquares
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from photutils.detection import DAOStarFinder

if __package__ is None or __package__ == "":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from configs import config as _config

_logger = logging.getLogger(__name__)

_DARK_BG = "#0d0d1a"
_DARK_AXES = "#1a1a2e"

_DETECT_FWHM = 5.0
_DETECT_THRESHOLD_SIGMA = 8.0


def _transit_cfg() -> dict:
    return _config.data().get("transit", {})


def _transit_path() -> Path:
    rel = _transit_cfg().get("file", "local/transits.json")
    root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    return root / rel


def load_transits() -> dict:
    path = _transit_path()
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        _logger.exception("Failed to load transits.json")
        return {}


def save_transits(dso_name: str, filter_name: str, entry: dict) -> None:
    """Merge entry under data[dso][filter] and atomically rewrite the file."""
    path = _transit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = load_transits()
    dso_key = dso_name.lower().replace(" ", "")
    existing = data.get(dso_key, {})
    existing[filter_name] = entry
    data[dso_key] = existing
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(path)
    except Exception:
        _logger.exception("Failed to save transits.json")
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _find_dso_dir(image_dir: Path, name: str) -> Optional[Path]:
    target = name.lower().replace(" ", "").replace("_", "")
    candidates = [
        d for d in image_dir.iterdir()
        if d.is_dir() and target in d.name.lower().replace(" ", "").replace("_", "")
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    def _latest(d: Path) -> float:
        try:
            return max(
                f.stat().st_mtime
                for f in d.rglob("*.fits")
                if f.parent.name.upper() == "LIGHT"
            )
        except ValueError:
            return 0.0
    return max(candidates, key=_latest)


def _obs_time_mjd(path: Path) -> float:
    """Read DATE-OBS (UTC) from FITS, fall back to file mtime. Return MJD."""
    try:
        hdr = fits.getheader(path)
        date_obs = hdr.get("DATE-OBS")
        if date_obs:
            iso = date_obs.rstrip("Z")
            return float(Time(iso, scale="utc", format="isot").mjd)
    except Exception:
        pass
    return float(Time(datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)).mjd)


def _build_reference_stack(frames: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """FWHM-weighted mean of registered frames with NaN-aware accumulation."""
    arr = np.stack(frames, axis=0)
    mask = ~np.isnan(arr)
    safe = np.where(mask, arr, 0.0)
    w = weights[:, None, None].astype(np.float32) * mask.astype(np.float32)
    numer = np.sum(safe * w, axis=0)
    denom = w.sum(axis=0)
    out = np.where(denom > 0, numer / np.where(denom > 0, denom, 1.0), np.nan)
    return out


def _detect_reference_stars(stack: np.ndarray) -> tuple[np.ndarray, float]:
    """DAOStarFinder on the reference stack.  Returns (positions [(x,y), ...], median FWHM)."""
    finite = np.isfinite(stack)
    if not finite.any():
        return np.empty((0, 2)), _DETECT_FWHM

    valid = np.where(finite, stack, np.nan)
    _, median, std = sigma_clipped_stats(valid, sigma=3.0)
    if std <= 0 or not np.isfinite(std):
        return np.empty((0, 2)), _DETECT_FWHM

    finder = DAOStarFinder(fwhm=_DETECT_FWHM, threshold=_DETECT_THRESHOLD_SIGMA * std)
    sources = finder(np.nan_to_num(valid - median, nan=0.0))
    if sources is None or len(sources) == 0:
        return np.empty((0, 2)), _DETECT_FWHM

    xs = np.asarray(sources["xcentroid"], dtype=np.float64)
    ys = np.asarray(sources["ycentroid"], dtype=np.float64)
    positions = np.column_stack([xs, ys])
    # Median FWHM hint from DAO's sharpness isn't an FWHM; just use the detection FWHM
    # plus a quick refinement from how clustered the SHARPNESS distribution is.
    return positions, _DETECT_FWHM


def _photometry_one_frame(
    frame: np.ndarray,
    positions: np.ndarray,
    aperture_r: float,
    sky_in: float,
    sky_out: float,
) -> np.ndarray:
    """Background-subtracted aperture flux per position. NaN where aperture invalid."""
    apers = CircularAperture(positions, r=aperture_r)
    annuli = CircularAnnulus(positions, r_in=sky_in, r_out=sky_out)

    # Replace NaN with 0 for photometry but track via a mask aperture-sum so we
    # can blank out any star whose aperture overlaps registration NaN edges.
    nan_mask = np.isnan(frame)
    safe = np.where(nan_mask, 0.0, frame).astype(np.float64)

    phot = aperture_photometry(safe, apers)
    bkg_phot = aperture_photometry(safe, annuli)
    bad_phot = aperture_photometry(nan_mask.astype(np.float64), apers)

    raw_flux = np.asarray(phot["aperture_sum"], dtype=np.float64)
    bkg_sum = np.asarray(bkg_phot["aperture_sum"], dtype=np.float64)
    bad_count = np.asarray(bad_phot["aperture_sum"], dtype=np.float64)

    aper_area = apers.area
    annulus_area = annuli.area
    sky_per_px = bkg_sum / annulus_area if annulus_area > 0 else 0.0
    net = raw_flux - sky_per_px * aper_area
    net = np.where(bad_count > 0.5, np.nan, net)
    return net


def _differential_normalize(
    flux_matrix: np.ndarray,
    comparison_quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-star relative flux = flux / ensemble(comparison_stars), normalised so each
    star's nanmedian == 1.0.  Comparison ensemble = stars in the lowest-RMS
    `comparison_quantile` fraction, excluding the dimmest 10 % (noise-dominated)
    and brightest 5 % (potentially saturated).

    Returns (rel_flux [T, S], comparison_mask [S]).
    """
    n_frames, n_stars = flux_matrix.shape
    median_flux = np.nanmedian(flux_matrix, axis=0)

    # Rank stars by brightness; drop dimmest 10 % and brightest 5 %.
    finite = np.isfinite(median_flux) & (median_flux > 0)
    if finite.sum() < 4:
        # Not enough stars to compute a sensible ensemble; bail to raw flux/median.
        rel = flux_matrix / np.where(median_flux > 0, median_flux, 1.0)
        return rel, np.zeros(n_stars, dtype=bool)

    bright_rank = np.full(n_stars, np.inf)
    bright_rank[finite] = -median_flux[finite]
    sort_idx = np.argsort(bright_rank)  # brightest first
    n_finite = int(finite.sum())
    drop_bright = max(1, int(0.05 * n_finite))
    drop_dim = max(1, int(0.10 * n_finite))
    eligible = set(sort_idx[drop_bright:n_finite - drop_dim].tolist())

    # Per-star raw RMS (after dividing by own median).
    star_norm = flux_matrix / np.where(median_flux > 0, median_flux, np.nan)
    raw_rms = np.array([
        float(np.nanstd(star_norm[:, s])) if s in eligible else np.inf
        for s in range(n_stars)
    ])

    # Take lowest-RMS fraction of eligible stars as the comparison ensemble.
    eligible_idx = np.array(sorted(eligible))
    rms_sorted = eligible_idx[np.argsort(raw_rms[eligible_idx])]
    n_comp = max(3, int(comparison_quantile * len(eligible_idx)))
    comp_idx = rms_sorted[:n_comp]

    comp_mask = np.zeros(n_stars, dtype=bool)
    comp_mask[comp_idx] = True

    # Ensemble = sum of comparison fluxes per frame (NaN-safe).
    comp_slice = flux_matrix[:, comp_idx]
    ensemble = np.nansum(comp_slice, axis=1)
    ensemble = np.where(ensemble > 0, ensemble, np.nan)

    rel = flux_matrix / ensemble[:, None]
    rel_median = np.nanmedian(rel, axis=0)
    rel = rel / np.where(rel_median > 0, rel_median, np.nan)
    return rel, comp_mask


def _max_dip_sigma(rel_curve: np.ndarray) -> float:
    """Strongest dip (median - lowest-rolling-mean-of-3) in units of MAD."""
    finite = rel_curve[np.isfinite(rel_curve)]
    if finite.size < 6:
        return 0.0
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    if mad <= 0:
        return 0.0
    # Rolling mean of 3 (centred), NaN-safe via simple sliding window.
    pad = np.concatenate([[np.nan], rel_curve, [np.nan]])
    rolling = np.array([
        np.nanmean(pad[i:i + 3]) if np.isfinite(pad[i:i + 3]).any() else np.nan
        for i in range(len(rel_curve))
    ])
    lowest = float(np.nanmin(rolling)) if np.isfinite(rolling).any() else med
    return (med - lowest) / (1.4826 * mad)


def _run_bls(
    times_mjd: np.ndarray,
    rel_curve: np.ndarray,
    baseline_days: float,
) -> Optional[dict]:
    """Astropy BLS on a single light curve. Returns top-period summary, or None."""
    mask = np.isfinite(rel_curve)
    if mask.sum() < 20 or baseline_days < 1.0:
        return None
    t = times_mjd[mask]
    y = rel_curve[mask]
    min_dt = float(np.median(np.diff(np.sort(t))))
    if min_dt <= 0:
        return None
    period_min = max(2 * min_dt, 0.2)
    period_max = max(period_min * 1.5, 0.5 * baseline_days)
    if period_max <= period_min:
        return None

    try:
        bls = BoxLeastSquares(t, y)
        periods = np.linspace(period_min, period_max, 1500)
        durations = np.array([0.02, 0.05, 0.1, 0.2])
        result = bls.power(periods, durations)
        power = np.asarray(result.power)
        if not np.isfinite(power).any():
            return None
        best = int(np.nanargmax(power))
        return {
            "period_d": float(result.period[best]),
            "depth": float(result.depth[best]),
            "duration_d": float(result.duration[best]),
            "power": float(power[best]),
            "power_mean": float(np.nanmean(power)),
            "power_std": float(np.nanstd(power)),
        }
    except Exception:
        _logger.exception("BLS failed on a light curve")
        return None


def _plot_top_candidates(
    times_mjd: np.ndarray,
    rel_flux: np.ndarray,
    candidates: list[dict],
    output_path: Path,
    title: str,
) -> None:
    n = len(candidates)
    if n == 0:
        return
    fig = Figure(figsize=(12, max(3.0, 2.0 * n)))
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor(_DARK_BG)

    t0 = float(np.nanmin(times_mjd)) if times_mjd.size else 0.0
    times_rel = times_mjd - t0

    for row, cand in enumerate(candidates):
        s = cand["_star_idx"]
        period = cand.get("bls_period_d")

        ax1 = fig.add_subplot(n, 2, 2 * row + 1)
        ax1.set_facecolor(_DARK_AXES)
        ax1.tick_params(colors="white")
        for spine in ax1.spines.values():
            spine.set_edgecolor("#444466")
        ax1.plot(times_rel, rel_flux[:, s], "o", markersize=3, color="#5fa8d3")
        ax1.axhline(1.0, color="#888", linestyle=":", linewidth=0.8)
        ax1.set_ylabel(f"#{row+1}\n({cand['x']:.0f},{cand['y']:.0f})", color="white")
        if row == 0:
            ax1.set_title(f"{title} — raw light curve (days from first frame)", color="white")
        ax1.grid(True, alpha=0.25, color="#444466")

        ax2 = fig.add_subplot(n, 2, 2 * row + 2)
        ax2.set_facecolor(_DARK_AXES)
        ax2.tick_params(colors="white")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#444466")
        if period and period > 0:
            phase = ((times_mjd - t0) / period) % 1.0
            order = np.argsort(phase)
            ax2.plot(phase[order], rel_flux[order, s], "o", markersize=3, color="#ffb86b")
            ax2.set_xlim(0, 1)
            ax2.axhline(1.0, color="#888", linestyle=":", linewidth=0.8)
            ax2.set_title(
                f"P={period:.3f}d  depth={cand.get('bls_depth', 0):.3f}  "
                f"dip σ={cand['max_dip_sigma']:.1f}",
                color="white", fontsize=9,
            )
        else:
            ax2.text(0.5, 0.5, f"dip σ={cand['max_dip_sigma']:.1f}\n(no BLS)",
                     ha="center", va="center", color="white", transform=ax2.transAxes)
            ax2.set_xticks([])
            ax2.set_yticks([])
        ax2.grid(True, alpha=0.25, color="#444466")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="jpeg", dpi=130, facecolor=fig.get_facecolor())


def run_transit_search(
    dso_name: str,
    filter_name: str,
    image_dir: Path,
    output_plot_path: Path,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Top-level orchestrator. Returns the entry that was saved to transits.json."""
    from stacking import stacker  # local import — avoids circular at module load

    def _notify(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    cfg = _transit_cfg()
    min_frames = int(cfg.get("min_frames", 20))
    ap_mult = float(cfg.get("aperture_fwhm_mult", 1.5))
    sky_mult = cfg.get("sky_annulus_fwhm_mult", (3.0, 5.0))
    sky_in_mult, sky_out_mult = float(sky_mult[0]), float(sky_mult[1])
    comp_q = float(cfg.get("comparison_quantile", 0.25))
    top_n = int(cfg.get("top_n_plot", 5))

    dso_dir = _find_dso_dir(image_dir, dso_name)
    if dso_dir is None:
        raise ValueError(f"No image directory found for '{dso_name}'")

    light_files = sorted(
        (f for f in dso_dir.rglob("*.fits") if f.parent.name.upper() == "LIGHT"),
        key=lambda p: p.stat().st_mtime,
    )
    if not light_files:
        raise ValueError(f"No LIGHT frames under {dso_dir}")

    by_filter = stacker.group_by_filter(light_files)
    paths = by_filter.get(filter_name, [])
    if len(paths) < min_frames:
        raise ValueError(
            f"Need at least {min_frames} '{filter_name}' frames, found {len(paths)}"
        )

    _notify(f"prep: registering {len(paths)} '{filter_name}' frames at full res…")
    frames, accepted, fwhm_values = stacker._prepare_for_convergence(
        paths,
        max_fwhm_multiplier=1.5,
        register=True,
        downscale_to=None,
        progress_cb=progress_cb,
    )
    if len(frames) < min_frames:
        raise ValueError(
            f"Too few frames survived registration ({len(frames)} < {min_frames})"
        )
    weights = stacker._fwhm_weights(fwhm_values, accepted)

    _notify(f"prep: extracting timestamps for {len(accepted)} frames…")
    times_mjd = np.array([_obs_time_mjd(p) for p in accepted], dtype=np.float64)
    baseline_days = float(times_mjd.max() - times_mjd.min())

    _notify("photometry: building reference stack…")
    reference = _build_reference_stack(frames, weights)

    _notify("photometry: detecting stars on reference stack…")
    positions, det_fwhm = _detect_reference_stars(reference)
    if len(positions) < 4:
        raise ValueError(f"Only {len(positions)} stars detected; cannot build a comparison ensemble")

    measured_fwhm = float(np.median([v for v in fwhm_values.values() if v > 0]) or det_fwhm)
    aperture_r = max(2.0, ap_mult * measured_fwhm)
    sky_in = sky_in_mult * measured_fwhm
    sky_out = sky_out_mult * measured_fwhm

    _notify(
        f"photometry: {len(positions)} stars  ap={aperture_r:.1f}px  "
        f"sky=({sky_in:.1f},{sky_out:.1f})px  …"
    )

    n_frames = len(frames)
    n_stars = len(positions)
    flux_matrix = np.full((n_frames, n_stars), np.nan, dtype=np.float64)

    def _do_one(i: int) -> tuple[int, np.ndarray]:
        return i, _photometry_one_frame(frames[i], positions, aperture_r, sky_in, sky_out)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, row in pool.map(_do_one, range(n_frames)):
            flux_matrix[i] = row

    _notify(f"photometry: done — {np.isfinite(flux_matrix).mean()*100:.1f}% valid")

    _notify(f"differential photometry: comparison_quantile={comp_q}…")
    rel_flux, comp_mask = _differential_normalize(flux_matrix, comp_q)

    _notify(f"transit search: per-star dip + BLS  (baseline={baseline_days:.2f} d)…")

    def _per_star(s: int) -> dict:
        curve = rel_flux[:, s]
        out: dict = {"_star_idx": s,
                     "x": float(positions[s, 0]),
                     "y": float(positions[s, 1]),
                     "max_dip_sigma": _max_dip_sigma(curve)}
        bls = _run_bls(times_mjd, curve, baseline_days)
        if bls is not None:
            power_z = (bls["power"] - bls["power_mean"]) / bls["power_std"] if bls["power_std"] > 0 else 0.0
            out.update({
                "bls_period_d": round(bls["period_d"], 5),
                "bls_depth": round(bls["depth"], 5),
                "bls_duration_d": round(bls["duration_d"], 5),
                "bls_power": round(bls["power"], 4),
                "bls_power_z": round(power_z, 3),
            })
        return out

    with ThreadPoolExecutor(max_workers=4) as pool:
        per_star = list(pool.map(_per_star, range(n_stars)))

    for cand in per_star:
        dip = float(cand.get("max_dip_sigma", 0.0))
        bls_z = float(cand.get("bls_power_z", 0.0))
        cand["score"] = round(0.5 * dip + 0.5 * bls_z, 3)

    per_star.sort(key=lambda c: c["score"], reverse=True)
    top = per_star[:top_n]

    _notify(f"plotting top {len(top)} candidates…")
    _plot_top_candidates(
        times_mjd, rel_flux, top, output_plot_path,
        title=f"{dso_name} [{filter_name}]",
    )

    entry = {
        "updated": date.today().isoformat(),
        "frame_count": n_frames,
        "baseline_days": round(baseline_days, 3),
        "n_stars": n_stars,
        "n_comparison": int(comp_mask.sum()),
        "aperture_px": round(aperture_r, 2),
        "candidates": [
            {k: v for k, v in c.items() if not k.startswith("_")}
            for c in top
        ],
    }
    save_transits(dso_dir.name, filter_name, entry)
    _notify("done.")
    return entry
