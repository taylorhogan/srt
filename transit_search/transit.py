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
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _sanitize_rel_flux(rel_flux: np.ndarray, high_sigma: float = 5.0) -> np.ndarray:
    """Blank unphysical differential-flux samples before the transit search.

    Differential flux is normalised so each star's median ≈ 1. Two kinds of
    samples are blanked to NaN because they corrupt both the dip statistic and
    BLS without representing a real transit:

      * non-finite or non-positive values (over-subtracted sky / registration
        glitches — a star cannot have ≤ 0 net flux), and
      * points more than ``high_sigma`` robust sigma *above* the baseline — a
        star never brightens during a transit, so high spikes are glitches.

    Low excursions (the actual dips/eclipses we want to find) are left intact.
    Clipping is per-star (per column).
    """
    clean = rel_flux.astype(np.float64, copy=True)
    clean[~np.isfinite(clean)] = np.nan
    clean[clean <= 0] = np.nan
    med = np.nanmedian(clean, axis=0)
    mad = np.nanmedian(np.abs(clean - med[None, :]), axis=0)
    sigma = 1.4826 * mad
    # Where sigma is 0/NaN (flat or all-NaN column) disable high clipping.
    high = np.where(sigma > 0, med + high_sigma * sigma, np.inf)
    with np.errstate(invalid="ignore"):
        clean[clean > high[None, :]] = np.nan
    return clean


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


def _single_transit_score(
    times_mjd: np.ndarray,
    rel_curve: np.ndarray,
    min_dur_h: float = 0.3,
    max_dur_h: float = 4.0,
    n_widths: int = 16,
    min_in_points: int = 12,
    min_side_points: int = 8,
    min_side_h: float = 0.4,
    baseline_quality_floor: float = 0.01,
    max_depth: float = 0.5,
) -> dict:
    """Matched-filter search for a single transit-shaped dip in one light curve.

    A real transit is a flat-bottomed box bracketed by flat baseline on *both*
    sides, sitting on a photometrically clean baseline. This scores every
    candidate (centre, width) box and keeps the best, rewarding exactly those
    properties — which separates a shallow planet transit on a bright star from
    the deeper artifacts that a plain "lowest dip" statistic favours:

      * ``snr``     = depth·√n_in / sigma_oot      — significance vs baseline noise
      * ``shape``   = sigma_oot / max(sigma_in, sigma_oot)
                      ≈ 1 for a flat bottom, < 1 for a ramp or a noise spike
      * ``quality`` = min(1, q0 / sigma_oot)        — down-weights noisy baselines
                      (faint stars, blends), where a real transit can't be detected
      * ``score``   = snr · shape · quality

    Requiring baseline on both sides rejects start/end-of-night systematic ramps;
    the depth cap rejects unphysical (blended / over-subtracted) excursions.
    Times are in days (MJD); duration parameters are in hours.

    Returns a dict with keys score, snr, depth, duration_d, t_center_mjd,
    n_in, shape, sigma_oot (all zero when nothing qualifies).
    """
    zero = {"score": 0.0, "snr": 0.0, "depth": 0.0, "duration_d": 0.0,
            "t_center_mjd": 0.0, "n_in": 0, "shape": 0.0, "sigma_oot": 0.0}
    finite = np.isfinite(rel_curve)
    if finite.sum() < 30:
        return zero
    t = times_mjd[finite]
    y = rel_curve[finite]
    order = np.argsort(t)
    t, y = t[order], y[order]
    tlo, thi = float(t[0]), float(t[-1])
    baseline = thi - tlo
    if baseline <= 0:
        return zero

    day = 1.0 / 24.0
    min_side_d = min_side_h * day
    # Don't let the widest box leave too little room for two-sided baseline.
    max_w = min(max_dur_h * day, max(0.0, baseline - 2.0 * min_side_d))
    min_w = min_dur_h * day
    if max_w <= min_w:
        return zero

    best = dict(zero)
    for w in np.linspace(min_w, max_w, n_widths):
        half = w / 2.0
        for tc in t:
            if tc - half < tlo + min_side_d or tc + half > thi - min_side_d:
                continue
            inwin = np.abs(t - tc) <= half
            n_in = int(inwin.sum())
            if n_in < min_in_points:
                continue
            n_before = int(np.sum(t < tc - half))
            n_after = int(np.sum(t > tc + half))
            if n_before < min_side_points or n_after < min_side_points:
                continue
            oot = ~inwin
            base = float(np.median(y[oot]))
            sig_oot = 1.4826 * float(np.median(np.abs(y[oot] - base)))
            if sig_oot <= 0:
                continue
            yin = y[inwin]
            depth = base - float(np.mean(yin))
            if depth <= 0 or depth > max_depth:
                continue
            sig_in = 1.4826 * float(np.median(np.abs(yin - np.median(yin))))
            snr = depth * np.sqrt(n_in) / sig_oot
            shape = sig_oot / max(sig_in, sig_oot)
            quality = min(1.0, baseline_quality_floor / sig_oot)
            # Baseline flatness: a real transit returns to the SAME level on both
            # sides; a start/end-of-night ramp dips low then rises to baseline (or
            # vice-versa), so its pre- and post-dip medians differ. Penalise that
            # mismatch — this is what separates the planet transit from the
            # per-star extinction/settling ramps common-mode can't fully remove.
            base_before = float(np.median(y[t < tc - half]))
            base_after = float(np.median(y[t > tc + half]))
            trend = abs(base_before - base_after)
            flatness = sig_oot / max(sig_oot, trend)
            score = snr * shape * quality * flatness
            if score > best["score"]:
                best = {"score": float(score), "snr": float(snr),
                        "depth": float(depth), "duration_d": float(w),
                        "t_center_mjd": float(tc), "n_in": n_in,
                        "shape": float(shape), "sigma_oot": float(sig_oot)}
    return best


def _run_bls(
    times_mjd: np.ndarray,
    rel_curve: np.ndarray,
    baseline_days: float,
    max_depth: float = 0.8,
    min_cycles: float = 2.0,
    depth_snr_min: float = 3.0,
    min_transit_points: int = 3,
    min_power: float = 0.01,
    max_in_transit_factor: float = 2.5,
) -> Optional[dict]:
    """Astropy BLS on a single light curve. Returns top-period summary, or None.

    Beyond fitting, the best period must clear a significance floor so sparse,
    gappy sampling can't manufacture a confident-looking detection from a
    near-flat periodogram: the baseline must span ``min_cycles`` periods, the
    depth must be significant versus the curve's robust scatter
    (``depth_snr_min``), there must be ``min_transit_points`` in-transit, and
    the absolute power must exceed ``min_power``.
    """
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
        # Astropy requires every trial duration to be strictly shorter than the
        # minimum trial period. With minute-cadence subs period_min hits its
        # 0.2 d floor, which collides with the 0.2 d duration and makes
        # bls.power() raise for every star. Keep only durations below period_min.
        durations = np.array([0.02, 0.05, 0.1, 0.2])
        durations = durations[durations < period_min]
        if durations.size == 0:
            return None
        result = bls.power(periods, durations)
        power = np.asarray(result.power)
        depth = np.asarray(result.depth)
        # Keep only periods with a physically plausible depth: flux is normalised
        # to ~1, so a real dip has 0 < depth < max_depth. Depth ≥ 1 means negative
        # in-transit flux (an outlier artifact); depth ≤ 0 is a brightening.
        physical = np.isfinite(power) & (depth > 0) & (depth < max_depth)
        if not physical.any():
            return None
        best = int(np.argmax(np.where(physical, power, -np.inf)))
        best_period = float(result.period[best])
        best_depth = float(result.depth[best])
        best_duration = float(result.duration[best])
        best_power = float(power[best])

        # --- Significance floor -------------------------------------------- #
        # Reject detections that sparse/gappy sampling can fake. Without this a
        # near-flat periodogram (tiny power_std) turns a ~0-power, ~1% "dip"
        # into a huge z-score.
        # 1) the baseline must actually show the period repeat a few times.
        if best_period <= 0 or baseline_days / best_period < min_cycles:
            return None
        # 2) absolute-power backstop.
        if best_power < min_power:
            return None
        # 3) the depth must be significant vs. the curve's robust scatter,
        #    given how many points actually fall in the transit window.
        sigma = 1.4826 * float(np.median(np.abs(y - np.median(y))))
        transit_time = float(result.transit_time[best])
        phase = ((t - transit_time) / best_period + 0.5) % 1.0 - 0.5
        n_in = int(np.sum(np.abs(phase) * best_period <= best_duration / 2.0))
        if n_in < min_transit_points:
            return None
        # A real transit spends most of its time OUT of transit: the in-transit
        # fraction should be ≈ duration/period. If the sampling instead piles
        # far more points than that into the window, the period is aliasing
        # clustered data (sparse/gappy window function), not a real dip.
        expected_in = (best_duration / best_period) * len(y)
        if n_in > max_in_transit_factor * max(expected_in, 1.0):
            return None
        if sigma > 0 and best_depth < depth_snr_min * sigma / np.sqrt(n_in):
            return None

        # Compute the significance baseline over physical periods only, so a few
        # artifact spikes don't inflate the mean/std and wash out the z-score.
        phys_power = power[physical]
        return {
            "period_d": best_period,
            "depth": best_depth,
            "duration_d": best_duration,
            "power": best_power,
            "power_mean": float(np.nanmean(phys_power)),
            "power_std": float(np.nanstd(phys_power)),
            "n_in_transit": n_in,
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
        # Overlay the detected single-transit box: shade the in-transit window
        # and draw the fitted depth level.
        tc = cand.get("transit_t_center_mjd")
        dur = cand.get("transit_duration_d")
        depth = cand.get("transit_depth")
        if tc and dur and depth:
            tc_rel = float(tc) - t0
            half = float(dur) / 2.0
            ax1.axvspan(tc_rel - half, tc_rel + half, color="#ffb86b", alpha=0.12)
            ax1.hlines(1.0 - float(depth), tc_rel - half, tc_rel + half,
                       color="#ffb86b", linewidth=1.2)
        label = f"#{row+1}"
        if cand.get("ra_deg") is not None and cand.get("dec_deg") is not None:
            label += f"\nRA {cand['ra_deg']:.4f}\nDec {cand['dec_deg']:+.4f}"
        else:
            label += f"\n({cand['x']:.0f},{cand['y']:.0f})"
        if cand.get("gaia_g_mag") is not None:
            label += f"\nG={cand['gaia_g_mag']:.1f}"
        ax1.set_ylabel(label, color="white")
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
            # No periodic BLS detection (e.g. single-night run): summarise the
            # single-transit box fit instead.
            depth = cand.get("transit_depth", 0.0) or 0.0
            dur_h = (cand.get("transit_duration_d", 0.0) or 0.0) * 24.0
            txt = (
                f"single-transit box\n"
                f"depth={depth*100:.2f}%\n"
                f"dur={dur_h:.2f} h\n"
                f"SNR={cand.get('transit_snr', 0):.1f}\n"
                f"shape={cand.get('transit_shape', 0):.2f}\n"
                f"baseline σ={cand.get('transit_sigma_oot', 0)*100:.2f}%"
            )
            ax2.text(0.5, 0.5, txt, ha="center", va="center", color="white",
                     transform=ax2.transAxes, fontsize=9)
            ax2.set_xticks([])
            ax2.set_yticks([])
        ax2.grid(True, alpha=0.25, color="#444466")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="jpeg", dpi=130, facecolor=fig.get_facecolor())


def _plate_solve_wcs(
    reference_path: Path,
    astap_exe: str,
    progress_cb: Optional[Callable[[str], None]] = None,
):
    """Plate-solve ``reference_path`` with ASTAP and return an astropy WCS, or None.

    Solves a temporary copy (so the original imaging data is never touched) using
    the frame's own RA/DEC header as a hint. ASTAP reads pixel size + focal length
    from the FITS header to determine the field of view, so ``-fov 0`` is reliable
    for N.I.N.A frames. Returns None on any failure (missing solver, no solution,
    unreadable WCS).
    """
    from astropy.wcs import WCS

    if not astap_exe or not os.path.exists(astap_exe):
        if progress_cb:
            progress_cb("identify: ASTAP not found, skipping star IDs")
        return None

    try:
        hdr = fits.getheader(reference_path)
        ra_deg = float(hdr.get("RA"))
        dec_deg = float(hdr.get("DEC"))
    except Exception:
        ra_deg = dec_deg = None

    tmp_dir = tempfile.mkdtemp(prefix="transit_solve_")
    tmp_fits = Path(tmp_dir) / "ref.fits"
    try:
        shutil.copy2(reference_path, tmp_fits)
        cmd = [astap_exe, "-f", str(tmp_fits), "-fov", "0", "-r", "5", "-update"]
        if ra_deg is not None and dec_deg is not None:
            cmd += ["-ra", f"{ra_deg / 15.0:.6f}", "-spd", f"{dec_deg + 90.0:.6f}"]
        if progress_cb:
            progress_cb("identify: plate-solving reference frame with ASTAP…")
        subprocess.run(cmd, capture_output=True, timeout=300)
        solved = fits.getheader(tmp_fits)
        if "CRVAL1" not in solved:
            if progress_cb:
                progress_cb("identify: plate solve failed, skipping star IDs")
            return None
        return WCS(solved)
    except Exception:
        _logger.exception("Plate solve failed")
        return None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _identify_candidates(
    reference_path: Path,
    candidates: list[dict],
    astap_exe: str,
    match_radius_arcsec: float = 3.0,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> None:
    """Add RA/Dec and the nearest Gaia source to each candidate, in place.

    Plate-solves the reference frame (whose pixel grid the candidate x,y live in),
    converts each (x, y) to sky coordinates, and cone-searches Gaia DR3 for the
    nearest source. Fully fail-safe: any solver/network/query failure leaves the
    candidates unchanged apart from any RA/Dec already filled in.
    """
    wcs = _plate_solve_wcs(reference_path, astap_exe, progress_cb)
    if wcs is None:
        return

    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.gaia import Gaia

    if progress_cb:
        progress_cb(f"identify: Gaia lookup for {len(candidates)} candidates…")

    radius = u.Quantity(match_radius_arcsec, u.arcsec)
    for c in candidates:
        try:
            sky = wcs.pixel_to_world(c["x"], c["y"])
            c["ra_deg"] = round(float(sky.ra.deg), 6)
            c["dec_deg"] = round(float(sky.dec.deg), 6)
        except Exception:
            continue
        try:
            res = Gaia.cone_search_async(sky, radius=radius).get_results()
            if len(res) == 0:
                c["gaia_source_id"] = None
                continue
            res.sort("dist")
            row = res[0]
            gmag = row["phot_g_mean_mag"]
            c["gaia_source_id"] = str(row["source_id"])
            c["gaia_g_mag"] = (None if np.ma.is_masked(gmag)
                               else round(float(gmag), 3))
            c["gaia_sep_arcsec"] = round(float(row["dist"]) * 3600.0, 2)
        except Exception:
            _logger.exception("Gaia lookup failed for a candidate")
            if progress_cb:
                progress_cb("identify: Gaia query failed, leaving remaining IDs blank")
            break


def _register_only(
    paths: list,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> tuple[list, list]:
    """Load and astroalign-register frames WITHOUT per-frame FWHM fitting.

    The transit search supplies its own clean-baseline weighting, so the slow
    per-star Gaussian FWHM measurement in ``stacker._prepare_for_convergence``
    (~15 min for ~150 small frames) is pure overhead here; registration alone is
    ~30 s. Frames are weighted uniformly downstream. Returns
    (registered_frames, accepted_paths) aligned index-for-index.
    """
    from stacking import stacker
    raw: list = []
    good: list = []
    for p in paths:
        try:
            raw.append(stacker._load_fits_2d(p))
            good.append(p)
        except Exception:
            _logger.warning("register_only: failed to load %s", getattr(p, "name", p))
    if len(raw) < 2:
        return raw, good
    if progress_cb:
        progress_cb(f"registering {len(raw)} frames (fast path, no per-frame FWHM)…")
    registered, surviving = stacker._register_frames(raw)
    accepted = [good[i] for i in surviving]
    if progress_cb:
        progress_cb(f"registration done — {len(registered)}/{len(raw)} frames aligned")
    return registered, accepted


def _transit_significance(
    times_mjd: np.ndarray,
    rel_curve: np.ndarray,
    field_scores: np.ndarray,
    observed_score: float,
    st_kwargs: dict,
    n_permutations: int = 500,
    rng_seed: int = 0,
) -> dict:
    """Quantify how likely the top box detection is real rather than chance.

    Three complementary tests (no single one is sufficient on its own):

      * ``perm_fap`` — scramble THIS curve in time many times and re-run the box
        search; the fraction reaching the observed score is the false-alarm
        probability under a no-time-structure (white-noise) null. Small ⇒ the
        dip is real structure, not random scatter.
      * ``field_fap`` / ``field_z`` — where the observed score falls in the
        distribution of *all* searched field stars, which share this night's
        systematics. A strong outlier ⇒ not a systematic the whole field shows.
      * ``dip_bump`` — observed (downward) score ÷ the best UPWARD box score on
        the same curve. A transit is a one-sided dip, so ≫ 1; symmetric noise or
        a sinusoid gives ≈ 1.

    Definitive confirmation still needs a second transit at the predicted period;
    a single night can't establish periodicity.
    """
    out = {"perm_fap": None, "field_fap": None, "field_z": None,
           "dip_bump": None, "n_permutations": n_permutations}

    # --- permutation FAP (white-noise null) -------------------------------- #
    finite_idx = np.where(np.isfinite(rel_curve))[0]
    if finite_idx.size >= 30 and n_permutations > 0:
        vals = rel_curve[finite_idx]
        rng = np.random.default_rng(rng_seed)
        hits = 0
        for _ in range(n_permutations):
            perm = rel_curve.copy()
            perm[finite_idx] = rng.permutation(vals)
            s = _single_transit_score(times_mjd, perm, **st_kwargs)["score"]
            if s >= observed_score:
                hits += 1
        out["perm_fap"] = (hits + 1) / (n_permutations + 1)

    # --- dip vs. bump (one-sidedness) -------------------------------------- #
    med = float(np.nanmedian(rel_curve))
    bump = _single_transit_score(times_mjd, 2.0 * med - rel_curve, **st_kwargs)["score"]
    out["dip_bump"] = float(observed_score / max(bump, 1e-6))

    # --- field outlier ----------------------------------------------------- #
    fs = np.asarray(field_scores, dtype=float)
    fs = fs[np.isfinite(fs)]
    if fs.size >= 5:
        # exclude one instance of the top score (the candidate itself)
        rest = np.sort(fs)[:-1] if fs.size > 1 else fs
        out["field_fap"] = float((np.sum(fs >= observed_score)) / fs.size)
        med_f = float(np.median(rest))
        mad_f = 1.4826 * float(np.median(np.abs(rest - med_f)))
        out["field_z"] = float((observed_score - med_f) / mad_f) if mad_f > 0 else None
    return out


def _plot_candidate_field(
    reference: np.ndarray,
    candidate: dict,
    output_path: Path,
    title: str,
) -> None:
    """Render the reference stack with the top candidate circled (+ a zoom inset)."""
    from matplotlib.patches import Circle, Rectangle

    finite = reference[np.isfinite(reference)]
    if finite.size == 0:
        return
    vmin, vmax = np.percentile(finite, [10.0, 99.5])
    img = np.nan_to_num(reference, nan=float(vmin))
    h, w = reference.shape
    x, y = float(candidate["x"]), float(candidate["y"])

    fig = Figure(figsize=(10, 5.2))
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor(_DARK_BG)

    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    r = 16.0
    ax.add_patch(Circle((x, y), r, fill=False, color="#ff5050", lw=1.8))
    ax.annotate("candidate", (x, min(y + r + 10, h - 4)), color="#ff5050",
                ha="center", va="bottom", fontsize=10, weight="bold")
    sub = 45
    ax.add_patch(Rectangle((x - sub, y - sub), 2 * sub, 2 * sub,
                           fill=False, color="#ffb86b", lw=0.8, ls="--"))
    ax.set_title("reference stack", color="white", fontsize=10)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444466")

    # Zoom inset around the candidate.
    ax2 = fig.add_subplot(1, 2, 2)
    x0, x1 = int(max(0, x - sub)), int(min(w, x + sub))
    y0, y1 = int(max(0, y - sub)), int(min(h, y + sub))
    ax2.imshow(img[y0:y1, x0:x1], origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
               extent=[x0, x1, y0, y1])
    ax2.add_patch(Circle((x, y), r, fill=False, color="#ff5050", lw=1.8))
    loc = ""
    if candidate.get("ra_deg") is not None:
        loc = f"RA {candidate['ra_deg']:.4f}  Dec {candidate['dec_deg']:+.4f}"
    if candidate.get("gaia_g_mag") is not None:
        loc += f"   G={candidate['gaia_g_mag']:.1f}"
    ax2.set_title("zoom" + (f"   {loc}" if loc else ""), color="white", fontsize=9)
    ax2.tick_params(colors="white")
    for sp in ax2.spines.values():
        sp.set_edgecolor("#444466")

    fig.suptitle(title, color="white", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="jpeg", dpi=130, facecolor=fig.get_facecolor())


def _notify_push(
    candidate: dict,
    dso_name: str,
    filter_name: str,
    field_z: float,
    field_image_path: Optional[Path],
) -> None:
    """Push a strong-detection alert (best-effort; never raises into the search)."""
    try:
        from utils import pushover
    except Exception:
        _logger.exception("pushover import failed; skipping transit alert")
        return

    loc = ""
    if candidate.get("ra_deg") is not None and candidate.get("dec_deg") is not None:
        loc = f" @ RA {candidate['ra_deg']:.4f} Dec {candidate['dec_deg']:+.4f}"
    if candidate.get("gaia_g_mag") is not None:
        loc += f" (G={candidate['gaia_g_mag']:.1f})"
    sig = candidate.get("significance", {})
    fap = sig.get("perm_fap")
    bump = sig.get("dip_bump")
    extra = []
    if fap is not None:
        extra.append(f"FAP {fap}")
    if bump is not None:
        extra.append(f"dip/bump {bump}")
    extra_str = (" | " + ", ".join(extra)) if extra else ""
    msg = (f"Possible transit in {dso_name} [{filter_name}]{loc}: "
           f"field_z={field_z:.1f}{extra_str}. Candidate only — needs a 2nd "
           f"transit to confirm.")

    img = (str(field_image_path) if field_image_path is not None
           and field_image_path.exists() else None)
    try:
        pushover.push_message(msg, image=img)
    except Exception:
        _logger.exception("transit push notification failed")


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
    min_valid_fraction = float(cfg.get("min_valid_fraction", 0.8))
    edge_margin_mult = float(cfg.get("edge_margin_mult", 1.0))
    outlier_high_sigma = float(cfg.get("outlier_high_sigma", 5.0))
    max_bls_depth = float(cfg.get("max_bls_depth", 0.8))
    common_mode_correction = bool(cfg.get("common_mode_correction", True))
    shared_period_frac = float(cfg.get("shared_period_frac", 0.05))
    min_bls_cycles = float(cfg.get("min_bls_cycles", 2))
    depth_snr_min = float(cfg.get("depth_snr_min", 3.0))
    min_transit_points = int(cfg.get("min_transit_points", 3))
    min_bls_power = float(cfg.get("min_bls_power", 0.01))
    max_in_transit_factor = float(cfg.get("max_in_transit_factor", 2.5))
    identify_candidates = bool(cfg.get("identify_candidates", True))
    gaia_match_radius_arcsec = float(cfg.get("gaia_match_radius_arcsec", 3.0))
    st_min_dur_h = float(cfg.get("single_transit_min_dur_h", 0.3))
    st_max_dur_h = float(cfg.get("single_transit_max_dur_h", 4.0))
    st_n_widths = int(cfg.get("single_transit_n_widths", 16))
    st_min_in_points = int(cfg.get("single_transit_min_in_points", 12))
    st_min_side_points = int(cfg.get("single_transit_min_side_points", 8))
    st_min_side_h = float(cfg.get("single_transit_min_side_h", 0.4))
    st_quality_floor = float(cfg.get("single_transit_baseline_quality_floor", 0.01))
    st_max_depth = float(cfg.get("single_transit_max_depth", 0.5))
    bls_score_weight = float(cfg.get("bls_score_weight", 0.5))
    skip_fwhm = bool(cfg.get("skip_fwhm_registration", True))
    significance_permutations = int(cfg.get("significance_permutations", 500))
    field_z_alert = float(cfg.get("field_z_alert", 6.0))
    astap_exe = _config.data().get("hardware", {}).get("astap_exe", "")
    st_kwargs = dict(
        min_dur_h=st_min_dur_h, max_dur_h=st_max_dur_h, n_widths=st_n_widths,
        min_in_points=st_min_in_points, min_side_points=st_min_side_points,
        min_side_h=st_min_side_h, baseline_quality_floor=st_quality_floor,
        max_depth=st_max_depth,
    )

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
    if filter_name == "*":
        # No filter specified (or subs may lack FILTER headers): use every LIGHT
        # sub regardless of filter, and label the run "all".
        paths = light_files
        filter_name = "all"
    else:
        paths = by_filter.get(filter_name, [])
        if not paths:
            # Case-insensitive fallback (e.g. user types "l", FITS header says "L")
            for key, val in by_filter.items():
                if key.lower() == filter_name.lower():
                    paths = val
                    filter_name = key  # use the canonical name going forward
                    break
    if len(paths) < min_frames:
        available = ", ".join(sorted(by_filter.keys())) or "none"
        raise ValueError(
            f"Need at least {min_frames} '{filter_name}' frames, found {len(paths)}"
            f" (available filters: {available})"
        )

    sample_fwhm = 0.0
    if skip_fwhm:
        # Fast path: register only (no per-frame FWHM fitting — the search does its
        # own clean-baseline weighting). Measure FWHM on a few frames just to size
        # the photometry aperture sensibly. ~30 s vs ~15 min.
        _notify(f"prep: registering {len(paths)} '{filter_name}' frames (fast path)…")
        frames, accepted = _register_only(paths, progress_cb=progress_cb)
        fwhm_values = {}
        sample_paths = accepted[:: max(1, len(accepted) // 3)][:3]
        sample_vals = [stacker._measure_fwhm(p) for p in sample_paths]
        sample_vals = [v for v in sample_vals if v > 0]
        sample_fwhm = float(np.median(sample_vals)) if sample_vals else 0.0
    else:
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

    _measured = [v for v in fwhm_values.values() if v > 0]
    if _measured:
        measured_fwhm = float(np.median(_measured))
    elif sample_fwhm > 0:
        measured_fwhm = sample_fwhm
    else:
        measured_fwhm = det_fwhm
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

    tick = max(1, n_frames // 5)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(_do_one, i): i for i in range(n_frames)}
        done = 0
        for fut in as_completed(futs):
            i, row = fut.result()
            flux_matrix[i] = row
            done += 1
            if done % tick == 0:
                _notify(f"photometry: {done}/{n_frames} frames done…")

    _notify(f"photometry: done — {np.isfinite(flux_matrix).mean()*100:.1f}% valid")

    _notify(f"differential photometry: comparison_quantile={comp_q}…")
    rel_flux, comp_mask = _differential_normalize(flux_matrix, comp_q)
    # Blank unphysical (≤0) and high-spike outlier samples before searching, so
    # they can't drive fake dips or unphysically deep BLS detections.
    rel_flux = _sanitize_rel_flux(rel_flux, high_sigma=outlier_high_sigma)

    # --- Reject stars that can't yield a trustworthy light curve ------------ #
    # Otherwise edge artifacts dominate the results: stars near the frame border
    # drift into the NaN regions left by astroalign registration on some frames,
    # so their aperture is blanked there (bad_count > 0.5 → NaN). The few surviving
    # points read as a huge fake "dip", and BLS is starved of its 20-point minimum.
    # Keep only stars whose full aperture stays on-chip and that have enough
    # finite measurements for both the dip statistic and BLS to be meaningful.
    frame_h, frame_w = frames[0].shape
    edge_margin = edge_margin_mult * sky_out
    valid_counts = np.isfinite(rel_flux).sum(axis=0)
    # 20 is the floor required by _run_bls; the fraction dominates for longer runs.
    min_valid = max(20, int(np.ceil(min_valid_fraction * n_frames)))
    xs, ys = positions[:, 0], positions[:, 1]
    on_chip = (
        (xs >= edge_margin) & (xs < frame_w - edge_margin)
        & (ys >= edge_margin) & (ys < frame_h - edge_margin)
    )
    keep_mask = on_chip & (valid_counts >= min_valid)
    kept_indices = np.flatnonzero(keep_mask)
    n_kept = int(kept_indices.size)
    _notify(
        f"filtering: {n_kept}/{n_stars} stars pass "
        f"(≥{min_valid} valid pts, ≥{edge_margin:.0f}px from edge)"
    )
    if n_kept == 0:
        raise ValueError(
            f"No stars survived the validity/edge filter "
            f"(needed ≥{min_valid} of {n_frames} valid points and "
            f"≥{edge_margin:.0f}px from the edge). Try more frames or lower "
            f"min_valid_fraction / edge_margin_mult."
        )

    # --- Common-mode detrending -------------------------------------------- #
    # A flux change shared by many stars at the same epoch (start-of-night ramp,
    # a between-session offset, transparency variation) is never a transit — that
    # affects ONE star. Estimate the common mode as the per-frame median over the
    # kept stars, divide it out of every curve, and renormalise each star to
    # median 1. This removes the systematics that otherwise dominate both the dip
    # statistic and BLS (via sampling aliases) on this kind of data.
    if common_mode_correction and n_kept >= 20:
        cm = np.nanmedian(rel_flux[:, kept_indices], axis=1)
        cm = np.where(np.isfinite(cm) & (cm > 0), cm, 1.0)
        rel_flux = rel_flux / cm[:, None]
        med = np.nanmedian(rel_flux, axis=0)
        rel_flux = rel_flux / np.where((med > 0) & np.isfinite(med), med, 1.0)
        _notify("common-mode: divided out per-frame median across kept stars")
    elif common_mode_correction:
        _notify(f"common-mode: skipped (only {n_kept} kept stars, need ≥20)")

    _notify(f"transit search: per-star box + BLS  (baseline={baseline_days:.2f} d)…")

    def _per_star(s: int) -> dict:
        curve = rel_flux[:, s]
        st = _single_transit_score(times_mjd, curve, **st_kwargs)
        out: dict = {"_star_idx": s,
                     "x": float(positions[s, 0]),
                     "y": float(positions[s, 1]),
                     "max_dip_sigma": _max_dip_sigma(curve),
                     "transit_score": round(st["score"], 3),
                     "transit_snr": round(st["snr"], 3),
                     "transit_depth": round(st["depth"], 5),
                     "transit_duration_d": round(st["duration_d"], 5),
                     "transit_t_center_mjd": round(st["t_center_mjd"], 6),
                     "transit_sigma_oot": round(st["sigma_oot"], 5),
                     "transit_shape": round(st["shape"], 3),
                     "transit_n_in": st["n_in"]}
        bls = _run_bls(
            times_mjd, curve, baseline_days,
            max_depth=max_bls_depth,
            min_cycles=min_bls_cycles,
            depth_snr_min=depth_snr_min,
            min_transit_points=min_transit_points,
            min_power=min_bls_power,
            max_in_transit_factor=max_in_transit_factor,
        )
        if bls is not None:
            power_z = (bls["power"] - bls["power_mean"]) / bls["power_std"] if bls["power_std"] > 0 else 0.0
            out.update({
                "bls_period_d": round(bls["period_d"], 5),
                "bls_depth": round(bls["depth"], 5),
                "bls_duration_d": round(bls["duration_d"], 5),
                "bls_power": round(bls["power"], 4),
                "bls_power_z": round(power_z, 3),
                "bls_n_in_transit": bls["n_in_transit"],
            })
        return out

    tick = max(1, n_kept // 5)
    per_star: list[dict] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(_per_star, s): s for s in kept_indices}
        done = 0
        for fut in as_completed(futs):
            per_star.append(fut.result())
            done += 1
            if done % tick == 0:
                _notify(f"transit search: BLS {done}/{n_kept} stars…")

    # --- Flag BLS periods shared across many stars (alias systematics) ------ #
    # Real periodic signals don't cluster at one period across unrelated field
    # stars; sparse/irregular sampling does (window-function aliases). Bin
    # periods at 0.001 d and flag any bin holding ≥ shared_period_frac of the
    # stars that produced a BLS detection, then drop those stars' BLS score
    # contribution. Backstop to common-mode detrending.
    from collections import Counter
    bls_bins = [round(c["bls_period_d"], 3) for c in per_star if c.get("bls_period_d") is not None]
    if bls_bins:
        counts = Counter(bls_bins)
        threshold = max(3, int(shared_period_frac * len(bls_bins)))
        shared_periods = {p for p, n in counts.items() if n >= threshold}
        if shared_periods:
            _notify(
                f"common-mode: {len(shared_periods)} BLS period(s) shared by "
                f"≥{threshold} stars flagged as systematic"
            )
        for cand in per_star:
            p = cand.get("bls_period_d")
            if p is not None and round(p, 3) in shared_periods:
                cand["bls_shared_period"] = True

    for cand in per_star:
        # Primary signal: the single-transit box score (flat-bottomed dip on a
        # clean baseline, bracketed by baseline on both sides). This separates a
        # shallow planet transit on a bright star from the deeper artifacts a
        # plain "lowest dip" statistic ranks first.
        st_score = float(cand.get("transit_score", 0.0))
        # A period shared across many stars is a systematic alias, not a transit;
        # zero its BLS contribution so it can't win on periodicity alone. BLS adds
        # a bonus when a genuine repeat is present (multi-night runs).
        bls_z = 0.0 if cand.get("bls_shared_period") else float(cand.get("bls_power_z", 0.0))
        cand["score"] = round(st_score + bls_score_weight * bls_z, 3)

    per_star.sort(key=lambda c: c["score"], reverse=True)
    top = per_star[:top_n]

    # Tag the top candidates with sky coordinates + nearest Gaia source.
    if identify_candidates:
        _identify_candidates(
            accepted[0], top, astap_exe,
            match_radius_arcsec=gaia_match_radius_arcsec,
            progress_cb=progress_cb,
        )

    # --- Significance of the top candidate --------------------------------- #
    field_image_path = None
    if top:
        best = top[0]
        _notify(f"significance: testing top candidate ({significance_permutations} permutations)…")
        sig = _transit_significance(
            times_mjd, rel_flux[:, best["_star_idx"]],
            np.array([c["score"] for c in per_star], dtype=float),
            float(best["score"]), st_kwargs,
            n_permutations=significance_permutations,
        )
        best["significance"] = {k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in sig.items()}
        # Annotated reference frame with the candidate circled.
        field_image_path = output_plot_path.with_name(
            output_plot_path.stem + "_field" + output_plot_path.suffix)
        try:
            ftitle = f"{dso_name} [{filter_name}] — top candidate"
            _plot_candidate_field(reference, best, field_image_path, ftitle)
        except Exception:
            _logger.exception("field image generation failed")
            field_image_path = None

        # Strong single-night detection → push notification. field_z is the
        # robust field-outlier z-score: a high value means this star stands out
        # from its neighbours that share the same night's systematics. Still a
        # candidate (needs a 2nd transit at the predicted period to confirm).
        field_z = sig.get("field_z")
        if field_z is not None and field_z >= field_z_alert:
            _notify_push(best, dso_name, filter_name, field_z, field_image_path)

    _notify(f"plotting top {len(top)} candidate(s)…")
    _plot_top_candidates(
        times_mjd, rel_flux, top, output_plot_path,
        title=f"{dso_name} [{filter_name}]",
    )

    entry = {
        "updated": date.today().isoformat(),
        "frame_count": n_frames,
        "baseline_days": round(baseline_days, 3),
        "n_stars": n_stars,
        "n_stars_searched": n_kept,
        "n_comparison": int(comp_mask.sum()),
        "aperture_px": round(aperture_r, 2),
        "candidates": [
            {k: v for k, v in c.items() if not k.startswith("_")}
            for c in top
        ],
    }
    if field_image_path is not None and field_image_path.exists():
        entry["field_image"] = str(field_image_path)
    save_transits(dso_dir.name, filter_name, entry)
    _notify("done.")
    return entry
