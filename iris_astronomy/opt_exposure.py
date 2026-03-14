"""
opt_exposure.py — Optimum sub-exposure time calculator for Iris / QHY600M.

Theory
------
The total noise per pixel in a single sub-exposure of length *t* seconds is:

    sigma_total^2 = R^2 + (S + D) * t

where
    R  = read noise            (electrons RMS)
    S  = sky background rate   (electrons / pixel / second)
    D  = dark current rate     (electrons / pixel / second)  -- negligible for cooled sensor

The "sky-limited" regime is reached when sky noise dominates read noise by a
factor of at least *k* (default k = 3):

    sqrt(S * t) / R  >=  k
    =>  t_opt = (k * R)^2 / S

Sky rate S is measured empirically from real FITS frames via sigma-clipped
background statistics.  NINA writes camera metadata (EGAIN, RDNOISE) into the
FITS header; if present those values are used in preference to the defaults.

Usage
-----
    from iris_astronomy.opt_exposure import optimum_exposure

    result = optimum_exposure("latest_frame.fits")
    print(result["summary"])

    # or average over several frames for a more stable sky rate:
    result = optimum_exposure(["frame1.fits", "frame2.fits", "frame3.fits"])

CLI
---
    python iris_astronomy/opt_exposure.py path/to/frame.fits [frame2.fits ...]
"""

import os
import sys
from pathlib import Path
from typing import Union

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

# ── QHY600M defaults (Sony IMX455, Unity Gain mode, NINA gain = 56) ──────────
_DEFAULT_READ_NOISE_E   = 1.65   # electrons RMS  (datasheet: 1.6-1.7 e-)
_DEFAULT_GAIN_E_PER_ADU = 1.0    # electrons per ADU at unity gain


def measure_sky(fits_path: Union[str, Path]) -> dict:
    """Measure sky background statistics from a single FITS file.

    Reads the image data and relevant header keywords written by N.I.N.A
    (EXPTIME / EXPOSURE, EGAIN, RDNOISE) and returns a dict of sky metrics.

    Args:
        fits_path: Path to the FITS file.

    Returns:
        dict with keys:
            sky_adu     -- sigma-clipped median sky level (ADU)
            sky_std_adu -- sky noise standard deviation (ADU)
            exptime_sec -- exposure time in seconds (from header)
            egain       -- e-/ADU from FITS header, or None if absent
            rdnoise_e   -- read noise in e- from FITS header, or None if absent
            fits_path   -- str path of the file
    """
    fits_path = Path(fits_path)

    with fits.open(fits_path) as hdul:
        header = hdul[0].header
        data   = hdul[0].data.astype(float)

    # Exposure time -- NINA writes EXPTIME; fall back to EXPOSURE
    exptime = float(header.get("EXPTIME", header.get("EXPOSURE", 0)))
    if exptime <= 0:
        raise ValueError(f"Could not read a positive exposure time from {fits_path.name}")

    # Camera metadata written by NINA (may be absent in older sequences)
    egain   = float(header["EGAIN"])   if "EGAIN"   in header else None
    rdnoise = float(header["RDNOISE"]) if "RDNOISE" in header else None

    # Sky background via sigma-clipped statistics (same approach as fitsfwhm.py)
    _, sky_median, sky_std = sigma_clipped_stats(data, sigma=3.0)

    return {
        "sky_adu":     float(sky_median),
        "sky_std_adu": float(sky_std),
        "exptime_sec": exptime,
        "egain":       egain,
        "rdnoise_e":   rdnoise,
        "fits_path":   str(fits_path),
    }


def optimum_exposure(
    fits_paths: Union[str, Path, list],
    read_noise_e:   float = _DEFAULT_READ_NOISE_E,
    gain_e_per_adu: float = _DEFAULT_GAIN_E_PER_ADU,
    sky_ratio:      float = 3.0,
    bias_adu:       float = 0.0,
) -> dict:
    """Compute the optimum sub-exposure time from one or more FITS frames.

    Measures the sky background rate from each supplied frame and averages
    across frames for a robust estimate, then applies:

        t_opt = (sky_ratio * R)^2 / S

    where R is read noise (e-) and S is the sky rate (e-/px/s).

    If EGAIN / RDNOISE are present in the FITS header (written by N.I.N.A)
    they take precedence over the gain_e_per_adu / read_noise_e arguments.

    Args:
        fits_paths:     Single path or list of paths to calibrated FITS frames
                        from the same session and filter.
        read_noise_e:   Camera read noise in electrons (default: QHY600M unity gain).
        gain_e_per_adu: Sensor gain in e-/ADU (default: QHY600M unity gain).
        sky_ratio:      Target sky-noise / read-noise ratio k (default 3.0).
                        k=3 is a common minimum; k=10 gives a very conservative
                        sky-dominated exposure.
        bias_adu:       Bias pedestal in ADU to subtract before computing sky rate.
                        Pass 0 if frames are already bias-subtracted.

    Returns:
        dict with keys:
            optimum_exposure_sec  -- recommended sub-exposure length (seconds)
            current_exposure_sec  -- exposure time of the supplied frame(s)
            current_sky_ratio     -- measured sqrt(S*t)/R for the supplied frames
            is_sky_limited        -- True if current_sky_ratio >= sky_ratio
            sky_rate_e_per_sec    -- measured sky background rate (e-/px/s)
            sky_background_adu    -- average sky level across frames (ADU)
            read_noise_e          -- read noise value used in calculation
            gain_e_per_adu        -- gain value used in calculation
            sky_ratio_target      -- the k value requested
            n_frames              -- number of FITS files analysed
            summary               -- human-readable one-line result string
    """
    if isinstance(fits_paths, (str, Path)):
        fits_paths = [fits_paths]
    fits_paths = [Path(p) for p in fits_paths]

    sky_rates_e = []
    sky_adus    = []
    exptimes    = []
    r_used      = read_noise_e
    g_used      = gain_e_per_adu

    for path in fits_paths:
        m = measure_sky(path)

        # Prefer header-supplied camera parameters when available
        g = m["egain"]    if m["egain"]    is not None else gain_e_per_adu
        r = m["rdnoise_e"] if m["rdnoise_e"] is not None else read_noise_e

        r_used = r
        g_used = g

        sky_rate_adu_s = (m["sky_adu"] - bias_adu) / m["exptime_sec"]
        sky_rate_e_s   = sky_rate_adu_s * g

        sky_rates_e.append(sky_rate_e_s)
        sky_adus.append(m["sky_adu"])
        exptimes.append(m["exptime_sec"])

    sky_rate_e   = float(np.mean(sky_rates_e))
    sky_bg_adu   = float(np.mean(sky_adus))
    current_t    = float(np.mean(exptimes))

    if sky_rate_e <= 0:
        raise ValueError("Sky rate is zero or negative -- check bias_adu and FITS data")

    t_opt          = (sky_ratio * r_used) ** 2 / sky_rate_e
    current_ratio  = np.sqrt(sky_rate_e * current_t) / r_used
    is_sky_limited = current_ratio >= sky_ratio

    if is_sky_limited:
        verdict = f"already sky-limited (ratio {current_ratio:.1f}x)"
    else:
        verdict = f"under-exposed for sky limit (ratio {current_ratio:.1f}x < target {sky_ratio}x)"

    summary = (
        f"Sky rate {sky_rate_e:.2f} e-/px/s  ->  "
        f"t_opt = {t_opt:.0f}s  (current {current_t:.0f}s, {verdict})"
    )

    return {
        "optimum_exposure_sec": round(t_opt, 1),
        "current_exposure_sec": round(current_t, 1),
        "current_sky_ratio":    round(float(current_ratio), 2),
        "is_sky_limited":       bool(is_sky_limited),
        "sky_rate_e_per_sec":   round(sky_rate_e, 4),
        "sky_background_adu":   round(sky_bg_adu, 1),
        "read_noise_e":         r_used,
        "gain_e_per_adu":       g_used,
        "sky_ratio_target":     sky_ratio,
        "n_frames":             len(fits_paths),
        "summary":              summary,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python opt_exposure.py <frame.fits> [frame2.fits ...]")
        sys.exit(1)

    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    result = optimum_exposure(sys.argv[1:])

    print(f"\n{'─'*60}")
    print(f"  Optimum Exposure Analysis  ({result['n_frames']} frame(s))")
    print(f"{'─'*60}")
    for k, v in result.items():
        if k != "summary":
            print(f"  {k:<26} {v}")
    print(f"\n  {result['summary']}")
    print(f"{'─'*60}\n")
