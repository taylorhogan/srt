"""
Live FITS frame watcher — imaging ticker + per-DSO stats cache.

Runs a single background thread while imaging is active.  For each new LIGHT
frame that lands in the NINA image directory it:
  1. Runs full star detection + Gaussian fitting (FWHM, eccentricity).
  2. Measures sky brightness.
  3. Appends a stats dict to <dso_dir>/frame_stats.json (atomic write).
  4. Updates an in-memory "last frame" record read by the web ticker.

The JSON cache means that the `stats` command can skip re-opening every FITS
file — it reads pre-computed values from the cache instead.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

_POLL_SECONDS   = 5    # how often to scan for new files
_STABILITY_WAIT = 2.5  # seconds to confirm file isn't still being written

# ── module-level state (protected by _lock) ──────────────────────────────────
_lock   = threading.Lock()
_active = False
_frame_count    = 0
_last_frame: Optional[dict] = None
_stop_event: Optional[threading.Event] = None
_ticker_path: Optional[Path] = None  # file-based IPC so other processes can read state

# Coalescing single-slot rebuild trigger: setting the event asks a worker to
# rebuild the latest-image + stats artifacts. Multiple new frames during one
# render coalesce into a single follow-up rebuild.
_artifact_request = threading.Event()
_artifact_worker_started = False
_artifact_image_dir: Optional[Path] = None
_artifact_arcsec: float = 1.0
_last_artifact_cache: Optional[Path] = None  # cache path of the most recent frame's DSO


# ── public API ────────────────────────────────────────────────────────────────

def start(image_dir: Path, arcsec_per_pixel: float) -> None:
    """Start the watcher thread.  No-op if already running."""
    global _active, _frame_count, _last_frame, _stop_event, _ticker_path
    global _artifact_image_dir, _artifact_arcsec, _artifact_worker_started
    with _lock:
        if _active:
            return
        _active      = True
        _frame_count = 0
        _last_frame  = None
        _ticker_path = Path(image_dir) / "frame_ticker.json"
        _artifact_image_dir = Path(image_dir)
        _artifact_arcsec = float(arcsec_per_pixel)

    _stop_event = threading.Event()
    _write_ticker(True)
    threading.Thread(
        target=_run,
        args=(Path(image_dir), arcsec_per_pixel, _stop_event),
        daemon=True,
        name="frame-watcher",
    ).start()
    # Long-lived single-slot artifact rebuilder. Survives across imaging
    # sessions: stop()/start() just flip _active.
    if not _artifact_worker_started:
        _artifact_worker_started = True
        threading.Thread(
            target=_artifact_worker,
            daemon=True,
            name="frame-watcher-artifacts",
        ).start()


def stop() -> None:
    """Signal the watcher thread to stop."""
    global _active
    with _lock:
        _active = False
    if _stop_event:
        _stop_event.set()
    _write_ticker(False)


def get_ticker() -> dict:
    """Return current ticker state (thread-safe copy)."""
    with _lock:
        return {
            "active":       _active,
            "frame_count":  _frame_count,
            "last_frame":   dict(_last_frame) if _last_frame else None,
        }


# ── internal helpers ──────────────────────────────────────────────────────────

def _write_ticker(active: bool) -> None:
    """Write ticker state to a file so other processes (web server) can read it."""
    if _ticker_path is None:
        return
    with _lock:
        count = _frame_count
        last  = dict(_last_frame) if _last_frame else None
    tmp = _ticker_path.with_suffix(".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump({
                "active":         active,
                "frame_count":    count,
                "last_frame":     last,
                "last_heartbeat": datetime.utcnow().isoformat(),
            }, f, default=str)
        tmp.replace(_ticker_path)
    except Exception:
        pass


def _is_light(p: Path) -> bool:
    return p.parent.name.upper() == "LIGHT"


def _dso_dir(fits_path: Path, image_dir: Path) -> Optional[Path]:
    """Walk up from fits_path until the parent is image_dir (= DSO-level dir)."""
    d = fits_path.parent
    while d.parent != image_dir and d.parent != d:
        d = d.parent
    return d if d.parent == image_dir else None


def _stable(path: Path) -> bool:
    """Return True if the file size hasn't changed after _STABILITY_WAIT seconds."""
    try:
        size_before = path.stat().st_size
        time.sleep(_STABILITY_WAIT)
        return path.stat().st_size == size_before and size_before > 0
    except OSError:
        return False


def _load_cache(cache_path: Path) -> list:
    try:
        with open(cache_path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_cache(cache_path: Path, entries: list) -> None:
    tmp = cache_path.with_suffix(".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(entries, f, default=str, indent=2)
        tmp.replace(cache_path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _analyse(fits_path: Path, arcsec_per_pixel: float) -> Optional[dict]:
    """Open a FITS file and extract per-frame quality metrics."""
    from astropy.io import fits as _fits
    from fits_processing import fitsfwhm, sky_brightness as sb

    try:
        with _fits.open(fits_path) as hdul:
            hdr = hdul[0].header

        date_obs = hdr.get("DATE-OBS")
        try:
            obs_dt = datetime.fromisoformat(date_obs.rstrip("Z")) if date_obs else None
        except (ValueError, AttributeError):
            obs_dt = None
        if obs_dt is None:
            obs_dt = datetime.fromtimestamp(fits_path.stat().st_mtime)

        filter_name = str(hdr.get("FILTER", "Unknown")).strip()

        # Full star detection → FWHM + eccentricity in one pass
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, fwhm_arcsec, star_count, ecc = fitsfwhm.calculate_fwhm(
                fits_path, arcsec_per_pixel=arcsec_per_pixel
            )

        if star_count == 0:
            # Fall back to NINA's HFR header when no stars detected
            hfr = hdr.get("HFR")
            fwhm_arcsec = float(hfr) * 2.0 * arcsec_per_pixel if hfr else None
            ecc = None
        else:
            fwhm_arcsec = round(float(fwhm_arcsec), 3)
            ecc = round(float(ecc), 3)

        sky = sb.measure_sky(fits_path, arcsec_per_pixel=arcsec_per_pixel)

        return {
            "path":            str(fits_path),
            "time":            obs_dt.isoformat(),
            "filter":          filter_name,
            "fwhm_arcsec":     fwhm_arcsec,
            "eccentricity":    ecc,
            "star_count":      star_count,
            "sky_adu_per_s":   round(sky["sky_adu_per_s"], 2)       if sky else None,
            "sky_mag_arcsec2": round(sky["sky_mag_arcsec2"], 2)
                               if sky and sky.get("sky_mag_arcsec2") is not None else None,
        }

    except Exception as exc:
        print(f"frame_watcher: skipping {fits_path.name}: {exc}")
        return None


def _run(image_dir: Path, arcsec_per_pixel: float, stop_event: threading.Event) -> None:
    global _frame_count, _last_frame

    # Pre-populate the seen set so we don't re-process files from before imaging started
    seen: set[Path] = set()
    try:
        for f in image_dir.rglob("*.fits"):
            if _is_light(f):
                seen.add(f)
    except Exception:
        pass

    while not stop_event.is_set():
        try:
            current: set[Path] = set()
            for f in image_dir.rglob("*.fits"):
                if _is_light(f):
                    current.add(f)

            # Sort new files by modification time so we process them in order
            new_files = sorted(current - seen, key=lambda f: f.stat().st_mtime)

            for fits_path in new_files:
                if stop_event.is_set():
                    break
                if not _stable(fits_path):
                    continue   # file still being written; catch it next poll
                seen.add(fits_path)

                frame = _analyse(fits_path, arcsec_per_pixel)
                if frame is None:
                    continue

                with _lock:
                    _last_frame  = frame
                    _frame_count += 1

                dso = _dso_dir(fits_path, image_dir)
                if dso:
                    cache_path = dso / "frame_stats.json"
                    entries = _load_cache(cache_path)
                    entries.append(frame)
                    _save_cache(cache_path, entries)
                    _request_artifact_rebuild(cache_path)

        except Exception as exc:
            print(f"frame_watcher: poll error: {exc}")

        _write_ticker(True)  # heartbeat so the web server can detect staleness
        stop_event.wait(timeout=_POLL_SECONDS)


def _request_artifact_rebuild(cache_path: Path) -> None:
    """Schedule a single coalesced rebuild of the imaging artifacts."""
    global _last_artifact_cache
    with _lock:
        _last_artifact_cache = cache_path
    _artifact_request.set()


def _images_output_dir() -> Optional[Path]:
    """Resolve the web server's images dir (where /images is mounted from)."""
    try:
        from configs import config as _config
        cfg = _config.data()
        rel = cfg.get("web_chat", {}).get("upload_dir", "saved_dso")
        root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        out = root / rel
        out.mkdir(parents=True, exist_ok=True)
        return out
    except Exception:
        return None


def _artifact_worker() -> None:
    """Single-slot rebuilder: render latest-image and stats-plot artifacts.

    Coalescing: many sub arrivals during one render produce a single follow-up
    render, because _artifact_request is an Event (not a queue).
    """
    from fits_processing import imaging_artifacts as _ia

    while True:
        _artifact_request.wait()
        _artifact_request.clear()

        with _lock:
            cache_path = _last_artifact_cache
            image_dir = _artifact_image_dir
            arcsec = _artifact_arcsec
        out_dir = _images_output_dir()
        if image_dir is None or out_dir is None:
            continue

        try:
            _ia.render_latest_image(
                image_dir=image_dir,
                arcsec_per_pixel=arcsec,
                output_path=out_dir / _ia.LATEST_IMAGE_NAME,
            )
        except Exception:
            pass  # render_latest_image already logs

        if cache_path is not None:
            try:
                _ia.render_stats_plot_from_cache_path(
                    cache_path=cache_path,
                    output_path=out_dir / _ia.STATS_PLOT_NAME,
                )
            except Exception:
                pass
