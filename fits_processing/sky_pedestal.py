"""
Bias + dark pedestal calibration for sky-brightness measurement.

Why this exists
---------------
``sky_brightness.measure_sky`` used to compute the sky signal as
``corner_median - hdr["OFFSET"]``.  On the QHY600M (and ZWO cameras generally)
the FITS ``OFFSET`` keyword is the camera's *unitless offset dial setting* — 10
here — **not** an ADU pedestal.  The real bias pedestal measured from this rig's
BIAS frames is ~152 ADU, so subtracting 10 left ~96% of the pedestal sitting in
the "sky" number: a reported 150.5 ADU of sky was really ~154 ADU of pedestal
plus ~6.5 ADU of actual sky.  The metric was overstated ~23x and, far worse, the
genuine night-to-night variation was buried under a constant — it read
0.480–0.500 ADU/s across seven very different nights.

This module builds a real pedestal from the BIAS/DARK frames on disk and lets
``measure_sky`` subtract that instead.

Model
-----
    pedestal(T, t) = bias(T) + dark_rate(T) * t

``bias`` comes from the BIAS frames (exposure ~0) and is treated as
temperature-independent, which holds well for a regulated CMOS sensor — it is
the electronic offset, not a thermal signal.  ``dark_rate`` comes from
``(dark_level - bias) / exptime`` on the DARK frames and *is* strongly
temperature-dependent, so when a frame's sensor temperature does not match a
calibrated one the rate is scaled by the standard halving rule:

    dark_rate(T) = dark_rate(T0) * 2 ** ((T - T0) / DARK_DOUBLING_C)

Extrapolated lookups are flagged (``extrapolated: True``) so callers can tell
measured from modelled.  Because the real sky signal here is only a few ADU, a
1–2 ADU pedestal error is a large *relative* error — shoot BIAS/DARK sets at
each sensor temperature you actually image at rather than relying on the
extrapolation.

Usage
-----
    python fits_processing/sky_pedestal.py --build      # scan and write the table
    python fits_processing/sky_pedestal.py              # show the current table
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
CAL_FILENAME = "sky_pedestal.json"

# Dark current halves per this many degrees C. 6.3 is the conventional figure
# for silicon image sensors and is only used to extrapolate to temperatures
# with no calibration frames of their own.
DARK_DOUBLING_C = 6.3

# A frame's sensor temperature must be within this of a calibrated entry for the
# lookup to count as measured rather than extrapolated.
TEMP_TOLERANCE_C = 1.5

# Fraction of each axis used per corner sample — must match sky_brightness.
_CORNER_FRAC = 0.18
_SIGMA = 3.0

# Anything at or below this exposure counts as a bias frame.
_BIAS_MAX_EXPTIME = 0.01


def _cal_path(path: Optional[Path] = None) -> Path:
    if path is not None:
        return Path(path)
    root = Path(__file__).resolve().parent.parent
    return root / "local" / CAL_FILENAME


# ------------------------------------------------------------------
# Frame measurement
# ------------------------------------------------------------------

def corner_level_from_array(data: np.ndarray) -> Optional[float]:
    """Sigma-clipped corner background level in ADU from an in-memory frame.

    Uses the sigma-clipped *mean* of each corner rather than the median: the
    data are 16-bit integers, so a median quantises to 0.5 ADU steps, and on a
    real sky signal of only a few ADU that quantisation is a large fraction of
    the measurement.  The four corner means are then combined with a median,
    which keeps the robustness against one corner being spoiled by nebulosity
    or a satellite trail.

    This is the single estimator shared by calibration and measurement — the
    pedestal is only valid if both sides measure the frame the same way.
    """
    if data is None or data.ndim != 2:
        return None
    h, w = data.shape
    rh = max(1, int(h * _CORNER_FRAC))
    rw = max(1, int(w * _CORNER_FRAC))
    blocks = [
        data[:rh, :rw], data[:rh, w - rw:],
        data[h - rh:, :rw], data[h - rh:, w - rw:],
    ]
    means = []
    for b in blocks:
        mean, _, _ = sigma_clipped_stats(np.asarray(b, dtype=np.float64), sigma=_SIGMA)
        means.append(float(mean))
    return float(np.median(means))


def corner_level(fits_path: Path) -> Optional[float]:
    """Sigma-clipped corner background level in ADU, read straight from disk.

    Reads only the four corner blocks, and reads them unscaled so that a
    memory-map can be used (astropy refuses to memory-map a frame carrying
    BZERO/BSCALE, which N.I.N.A always writes).  BZERO is re-applied by hand.
    """
    try:
        with fits.open(fits_path, memmap=True, do_not_scale_image_data=True) as hdul:
            bzero = float(hdul[0].header.get("BZERO", 0.0))
            bscale = float(hdul[0].header.get("BSCALE", 1.0))
            data = hdul[0].data
            if data is None or data.ndim != 2:
                return None
            h, w = data.shape
            rh = max(1, int(h * _CORNER_FRAC))
            rw = max(1, int(w * _CORNER_FRAC))
            # Scale only the corner blocks, so the full frame is never realised.
            corners = np.empty((2 * rh, 2 * rw), dtype=np.float64)
            corners[:rh, :rw] = data[:rh, :rw]
            corners[:rh, rw:] = data[:rh, w - rw:]
            corners[rh:, :rw] = data[h - rh:, :rw]
            corners[rh:, rw:] = data[h - rh:, w - rw:]
            corners = corners * bscale + bzero
        return corner_level_from_array(corners)
    except Exception as exc:
        print(f"sky_pedestal: failed to read {Path(fits_path).name}: {exc}")
        return None


def _frame_key(hdr) -> Optional[tuple]:
    """(gain, offset, ccd_temp, exptime) from a header, or None if unusable."""
    try:
        gain = hdr.get("GAIN")
        offset = hdr.get("OFFSET")
        temp = hdr.get("CCD-TEMP")
        exp = hdr.get("EXPTIME", hdr.get("EXPOSURE"))
        if gain is None or offset is None or temp is None or exp is None:
            return None
        return (float(gain), float(offset), float(temp), float(exp))
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------
# Building the calibration
# ------------------------------------------------------------------

def find_calibration_dirs(roots: list[Path]) -> tuple[list[Path], list[Path]]:
    """Locate BIAS and DARK directories beneath *roots*."""
    bias_dirs: list[Path] = []
    dark_dirs: list[Path] = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for d in root.rglob("*"):
            if not d.is_dir():
                continue
            name = d.name.upper()
            if name == "BIAS":
                bias_dirs.append(d)
            elif name == "DARK":
                dark_dirs.append(d)
    return sorted(bias_dirs), sorted(dark_dirs)


def build_calibration(
    roots: Optional[list[Path]] = None,
    out_path: Optional[Path] = None,
    max_frames: int = 25,
) -> dict:
    """Scan BIAS/DARK folders and write a pedestal table keyed by gain/offset/temp.

    Frames are grouped by (gain, offset) and by sensor temperature rounded to the
    nearest degree, since the setpoint is regulated and the reported value dithers
    a few tenths around it.
    """
    if roots is None:
        try:
            from configs import config
            roots = [Path(config.data()["nina"]["image_dir"])]
        except Exception:
            roots = []
    out_path = _cal_path(out_path)

    bias_dirs, dark_dirs = find_calibration_dirs([Path(r) for r in roots])
    print(f"sky_pedestal: {len(bias_dirs)} BIAS dir(s), {len(dark_dirs)} DARK dir(s)")

    # --- bias levels, grouped by (gain, offset, round(temp)) ---
    bias_groups: dict[tuple, list[float]] = {}
    bias_sources: dict[tuple, set] = {}
    for d in bias_dirs:
        files = sorted(d.glob("*.fits"))[:max_frames]
        for f in files:
            try:
                hdr = fits.getheader(f)
            except Exception:
                continue
            key = _frame_key(hdr)
            if key is None or key[3] > _BIAS_MAX_EXPTIME:
                continue
            lvl = corner_level(f)
            if lvl is None:
                continue
            gk = (key[0], key[1], round(key[2]))
            bias_groups.setdefault(gk, []).append(lvl)
            bias_sources.setdefault(gk, set()).add(str(d))

    # --- dark levels: convert to a rate using the matching bias ---
    dark_groups: dict[tuple, list[float]] = {}
    dark_sources: dict[tuple, set] = {}
    for d in dark_dirs:
        files = sorted(d.glob("*.fits"))[:max_frames]
        for f in files:
            try:
                hdr = fits.getheader(f)
            except Exception:
                continue
            key = _frame_key(hdr)
            if key is None or key[3] <= _BIAS_MAX_EXPTIME:
                continue
            gk = (key[0], key[1], round(key[2]))
            bias_for_temp = bias_groups.get(gk)
            if not bias_for_temp:
                # No bias at this exact temperature — fall back to the nearest
                # bias at the same gain/offset, since bias is ~temp-independent.
                cands = [(abs(k[2] - gk[2]), v) for k, v in bias_groups.items()
                         if k[0] == gk[0] and k[1] == gk[1]]
                if not cands:
                    continue
                bias_for_temp = min(cands, key=lambda c: c[0])[1]
            lvl = corner_level(f)
            if lvl is None:
                continue
            rate = (lvl - float(np.median(bias_for_temp))) / max(key[3], 1e-6)
            dark_groups.setdefault(gk, []).append(rate)
            dark_sources.setdefault(gk, set()).add(str(d))

    entries = []
    for gk in sorted(set(bias_groups) | set(dark_groups)):
        gain, offset, temp = gk
        bias_vals = bias_groups.get(gk, [])
        dark_vals = dark_groups.get(gk, [])
        if not bias_vals:
            cands = [(abs(k[2] - temp), v) for k, v in bias_groups.items()
                     if k[0] == gain and k[1] == offset]
            if not cands:
                continue
            bias_vals = min(cands, key=lambda c: c[0])[1]
        entries.append({
            "gain": gain,
            "offset": offset,
            "temp_c": float(temp),
            "bias_adu": round(float(np.median(bias_vals)), 4),
            "bias_std_adu": round(float(np.std(bias_vals)), 4) if len(bias_vals) > 1 else 0.0,
            "dark_rate_adu_per_s": round(float(np.median(dark_vals)), 8) if dark_vals else 0.0,
            "n_bias": len(bias_vals),
            "n_dark": len(dark_vals),
            "sources": sorted(bias_sources.get(gk, set()) | dark_sources.get(gk, set())),
        })

    cal = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dark_doubling_c": DARK_DOUBLING_C,
        "temp_tolerance_c": TEMP_TOLERANCE_C,
        "corner_frac": _CORNER_FRAC,
        "entries": entries,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    print(f"sky_pedestal: wrote {out_path} ({len(entries)} entries)")
    return cal


# ------------------------------------------------------------------
# Using the calibration
# ------------------------------------------------------------------

_cache: Optional[dict] = None
_cache_mtime: float = 0.0


def load_calibration(path: Optional[Path] = None) -> Optional[dict]:
    """Load (and cache) the pedestal table. Returns None if it does not exist."""
    global _cache, _cache_mtime
    p = _cal_path(path)
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return None
    if _cache is not None and mtime == _cache_mtime:
        return _cache
    try:
        _cache = json.loads(p.read_text(encoding="utf-8"))
        _cache_mtime = mtime
        return _cache
    except Exception as exc:
        print(f"sky_pedestal: could not read {p}: {exc}")
        return None


def lookup(
    gain: Optional[float],
    offset: Optional[float],
    ccd_temp: Optional[float],
    exptime: float,
    cal: Optional[dict] = None,
) -> Optional[dict]:
    """Pedestal in ADU for a frame, or None if nothing matches its gain/offset.

    Returns a dict with ``pedestal_adu``, the ``bias_adu`` and ``dark_adu`` parts,
    the temperature offset actually used, and whether the dark term had to be
    extrapolated from a different temperature.
    """
    if cal is None:
        cal = load_calibration()
    if not cal or not cal.get("entries"):
        return None
    if gain is None or offset is None:
        return None

    matches = [e for e in cal["entries"]
               if e["gain"] == float(gain) and e["offset"] == float(offset)]
    if not matches:
        return None

    if ccd_temp is None:
        # No temperature in the header: use the coolest entry and say so.
        best = min(matches, key=lambda e: e["temp_c"])
        delta = 0.0
        extrapolated = True
    else:
        best = min(matches, key=lambda e: abs(e["temp_c"] - float(ccd_temp)))
        delta = float(ccd_temp) - best["temp_c"]
        extrapolated = abs(delta) > cal.get("temp_tolerance_c", TEMP_TOLERANCE_C)

    doubling = cal.get("dark_doubling_c", DARK_DOUBLING_C)
    rate = best["dark_rate_adu_per_s"] * (2.0 ** (delta / doubling))
    dark_adu = rate * max(float(exptime), 0.0)
    bias_adu = best["bias_adu"]

    return {
        "pedestal_adu": bias_adu + dark_adu,
        "bias_adu": bias_adu,
        "dark_adu": dark_adu,
        "dark_rate_adu_per_s": rate,
        "cal_temp_c": best["temp_c"],
        "temp_delta_c": delta,
        "extrapolated": extrapolated,
        "n_bias": best["n_bias"],
        "n_dark": best["n_dark"],
    }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    if "--build" in sys.argv:
        roots = None
        if "--root" in sys.argv:
            roots = [Path(sys.argv[sys.argv.index("--root") + 1])]
        cal = build_calibration(roots=roots)
    else:
        cal = load_calibration()
        if cal is None:
            print(f"No calibration at {_cal_path()}. Run with --build first.")
            sys.exit(1)

    print(f"\ngenerated {cal['generated']}  "
          f"(dark doubling {cal['dark_doubling_c']} degC, "
          f"temp tolerance +/-{cal['temp_tolerance_c']} degC)")
    hdr = (f"{'gain':>6}{'offset':>8}{'temp C':>9}{'bias ADU':>11}"
           f"{'dark ADU/s':>13}{'dark@300s':>11}{'nB':>5}{'nD':>4}")
    print(hdr)
    print("-" * len(hdr))
    for e in cal["entries"]:
        print(f"{e['gain']:>6.0f}{e['offset']:>8.0f}{e['temp_c']:>9.1f}"
              f"{e['bias_adu']:>11.3f}{e['dark_rate_adu_per_s']:>13.6f}"
              f"{e['dark_rate_adu_per_s'] * 300:>11.3f}{e['n_bias']:>5}{e['n_dark']:>4}")
