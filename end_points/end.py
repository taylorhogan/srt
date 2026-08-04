import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
import sys
import time
import warnings

import numpy as np

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from configs import config

from fits_processing import fitsfwhm, sky_brightness as sb
from hardware_control import pwi4_utils, kasa_utils as ku, utl_shelly
from cmd_processing import jobs, super_user_commands, social_server
from sentry import vision_safety
from utils import pushover, utils


def determine_roof_state_visually(account):
    cfg = config.data()

    is_parked, is_closed, is_open, mod_date = vision_safety.visual_status()
    if is_parked:
        reply = "Scope is Parked"
        if is_closed:
            reply += "\nRoof is closed"
        else:
            reply += "\nRoof is not closed"
        if is_open:
            reply += "\nRoof is open"
        else:
            reply += "\nRoof is not open"
    else:
        reply = "Scope is not parked"

    lm = vision_safety.last_match
    if lm and "min_conf" in lm:
        reply += (
            f"\n━━ Match confidence (≥ {lm['min_conf']:.2f}) ━━\n"
            f"Parked : {lm['parked']['conf']:.2f}\n"
            f"Closed : {lm['closed']['conf']:.2f}\n"
            f"Open   : {lm['open']['conf']:.2f}"
        )
    elif lm and lm.get("error"):
        # Snapshot was unreadable — visual_status() failed safe (all-False), so
        # the "not parked" above means "couldn't see", not a confirmed state.
        reply += f"\n⚠ Vision unavailable: {lm['error']}"

    reply += "\nPicture Date:" + mod_date + "\n"
    logger = cfg["logger"]["logging"]
    logger.info(account)
    if account in cfg["Super Users"]:
        social_server.post_social_message(reply, cfg["camera safety"]["scope_view"])
    else:
        social_server.post_social_message(reply)


def _post_imaging_summary(imaging_start: datetime) -> None:
    """Post end-of-night quality metrics, reading from the frame_watcher cache first."""
    import json as _json

    logger = utils.set_logger()
    cfg = config.data()
    image_dir = Path(cfg["nina"]["image_dir"])
    arcsec_per_pixel = cfg["nina"]["arc_sec_per_pixel"]
    start_ts = imaging_start.timestamp()

    social_server.post_social_message(
        f"Scanning for FITS since {imaging_start.strftime('%Y-%m-%d %H:%M')}"
    )

    def _is_light(f: Path) -> bool:
        return f.parent.name.upper() == "LIGHT"

    def _dso_dir(fits_path: Path) -> Path | None:
        d = fits_path.parent
        while d.parent != image_dir and d.parent != d:
            d = d.parent
        return d if d.parent == image_dir else None

    try:
        fits_files = sorted(
            (f for f in image_dir.rglob("*.fits")
             if _is_light(f) and f.stat().st_mtime >= start_ts),
            key=lambda f: f.stat().st_mtime,
        )
    except Exception:
        logger.exception("Failed to scan image directory %s", image_dir)
        social_server.post_social_message("Imaging complete — could not scan image directory")
        return

    if not fits_files:
        social_server.post_social_message("Imaging complete — no new FITS files found")
        return

    logger.info("Found %d FITS files since imaging start", len(fits_files))

    # Scope to the imaged DSO only. The scan above globs the whole image tree,
    # so without this any *other* target's frames whose mtime falls in the window
    # (e.g. a dataset imported mid-day for transit analysis) would be written into
    # this DSO's frame_stats.json and pollute its stats graph.
    dso_dir = _dso_dir(fits_files[-1])
    if dso_dir is not None:
        fits_files = [f for f in fits_files if dso_dir in f.parents]
    if not fits_files:
        social_server.post_social_message("Imaging complete — no new FITS files found")
        return

    # Load the per-frame cache written by frame_watcher during imaging
    cached: dict[str, dict] = {}
    cache_path: Path | None = (dso_dir / "frame_stats.json") if dso_dir else None
    if cache_path and cache_path.exists():
        try:
            with open(cache_path) as f:
                entries = _json.load(f)
            if isinstance(entries, list):
                for e in entries:
                    if "path" in e:
                        cached[str(Path(e["path"]))] = e
        except Exception:
            pass

    # Analyse only frames missing from the cache (should be rare)
    need_analysis = [f for f in fits_files if str(f) not in cached]
    if need_analysis:
        social_server.post_social_message(
            f"{len(cached)} cached, {len(need_analysis)} new — analysing…"
        )

        def _analyse(fits_path: Path) -> dict:
            from astropy.io import fits as _fits
            try:
                with _fits.open(fits_path) as hdul:
                    hdr = hdul[0].header
                filter_name = str(hdr.get("FILTER", "Unknown")).strip()
                date_obs = hdr.get("DATE-OBS")
                try:
                    obs_dt = datetime.fromisoformat(date_obs.rstrip("Z")) if date_obs else None
                except (ValueError, AttributeError):
                    obs_dt = None
                if obs_dt is None:
                    obs_dt = datetime.fromtimestamp(fits_path.stat().st_mtime)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, fwhm_arcsec, star_count, ecc = fitsfwhm.calculate_fwhm(
                        fits_path, arcsec_per_pixel=arcsec_per_pixel
                    )
                if star_count == 0:
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
                    # 5 dp, not 2: pedestal-corrected sky runs ~0.00-0.05 ADU/s.
                    "sky_adu_per_s":   round(sky["sky_adu_per_s"], 5)
                                       if sky and sky.get("sky_adu_per_s") is not None else None,
                    "sky_mag_arcsec2": round(sky["sky_mag_arcsec2"], 2)
                                       if sky and sky.get("sky_mag_arcsec2") is not None else None,
                    "pedestal_source": sky.get("pedestal_source") if sky else None,
                }
            except Exception as exc:
                logger.warning("Could not analyse %s: %s", fits_path.name, exc)
                return {"path": str(fits_path), "fwhm_arcsec": None, "eccentricity": None,
                        "sky_adu_per_s": None, "sky_mag_arcsec2": None}

        with ThreadPoolExecutor(max_workers=min(8, len(need_analysis))) as pool:
            for entry in pool.map(_analyse, need_analysis):
                cached[str(Path(entry["path"]))] = entry

        if cache_path:
            tmp = cache_path.with_suffix(".tmp")
            try:
                with open(tmp, "w") as f:
                    _json.dump(list(cached.values()), f, default=str, indent=2)
                tmp.replace(cache_path)
            except Exception:
                pass
    else:
        social_server.post_social_message(f"All {len(fits_files)} frames loaded from cache")

    frames = [cached[str(f)] for f in fits_files if str(f) in cached]
    fwhm_list    = [float(e["fwhm_arcsec"])     for e in frames if e.get("fwhm_arcsec")     is not None]
    ecc_list     = [float(e["eccentricity"])     for e in frames if e.get("eccentricity")    is not None]
    sky_adu_list = [float(e["sky_adu_per_s"])    for e in frames if e.get("sky_adu_per_s")   is not None]
    sky_mag_list = [float(e["sky_mag_arcsec2"])  for e in frames if e.get("sky_mag_arcsec2") is not None]

    if not fwhm_list:
        social_server.post_social_message(
            f"Imaging complete — {len(fits_files)} frames, no stars detected in any frame"
        )
        return

    summary = (
        f"Imaging complete — {len(fits_files)} frames ({len(fwhm_list)} with stars)\n"
        f'Median FWHM: {float(np.median(fwhm_list)):.2f}"\n'
        f"Median eccentricity: {float(np.median(ecc_list)):.3f}"
    )
    if sky_mag_list:
        summary += f"\nMedian sky: {float(np.median(sky_mag_list)):.2f} mag/arcsec²"
    elif sky_adu_list:
        summary += f"\nMedian sky: {float(np.median(sky_adu_list)):.2f} ADU/s"

    social_server.post_social_message(summary)
    logger.info(
        "Imaging summary: %d frames, median FWHM=%.2f\", ecc=%.3f",
        len(fits_files), float(np.median(fwhm_list)), float(np.median(ecc_list)),
    )

    # Best-effort and last: an optics measurement is never worth risking the
    # rest of the shutdown, and by here the roof is closed and the dehumidifier
    # is on, so the couple of minutes it takes costs nothing.
    try:
        _record_optics_trend(dso_dir, frames, arcsec_per_pixel, logger)
    except Exception:
        logger.exception("Optics trend measurement failed")


# How many of the night's frames feed the pooled optics measurement. 12 frames
# is ~3 min of star fitting for ~1800 pooled stars; more would be tighter still
# but this runs on every shutdown.
_OPTICS_TREND_FRAMES = 12

# A frame claiming better than this is lying — one 08-03 frame measured 0.22",
# which is physically impossible and would drag the "best frames" selection
# straight onto the worst data.
_OPTICS_MIN_PLAUSIBLE_FWHM = 0.5


def _record_optics_trend(dso_dir, frames: list[dict], arcsec_per_pixel: float,
                         logger) -> None:
    """Measure tonight's seeing-robust optics metrics and append them to history.

    Logged and persisted only — nothing consumes this yet, deliberately. The
    thresholds a comparison command would need do not exist until there is a
    baseline to set them from, and a change detector that cries wolf gets
    ignored. See docs/optics_trend_plan.md.

    Uses the night's SHARPEST frames, not a random sample: optical field errors
    are easiest to see when the seeing is not smearing them, and the frames are
    already measured so choosing them is free.
    """
    import json as _json

    if dso_dir is None:
        return
    candidates = [
        e for e in frames
        if e.get("fwhm_arcsec") and float(e["fwhm_arcsec"]) >= _OPTICS_MIN_PLAUSIBLE_FWHM
        and Path(e["path"]).exists()
    ]
    if len(candidates) < 4:
        logger.info("Optics trend: skipped — only %d usable frames", len(candidates))
        return
    candidates.sort(key=lambda e: float(e["fwhm_arcsec"]))
    chosen = candidates[:_OPTICS_TREND_FRAMES]

    metrics = fitsfwhm.compute_optics_trend_for_frames(
        [Path(e["path"]) for e in chosen], arcsec_per_pixel=arcsec_per_pixel,
    )
    if not metrics:
        return

    filters = sorted({e.get("filter") for e in chosen if e.get("filter")})
    night = Path(chosen[0]["path"]).parent.parent.name
    record = {
        "night": night,
        "dso": dso_dir.name,
        "filters": filters,
        "computed": datetime.now().astimezone().isoformat(timespec="seconds"),
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
    }

    logger.info(
        "Optics trend [%s %s]: %d frames / %d stars — seeing floor %.2f\", "
        "field excess %.2f\", edge excess %.2f\", sweet spot (%+.2f, %+.2f) r=%.2f, "
        "radial %.2f, uniform %.2f @ %.0f deg",
        night, ",".join(filters) or "?", metrics.get("frames_used", 0),
        metrics.get("stars_pooled", 0), metrics.get("seeing_floor_arcsec", float("nan")),
        metrics.get("field_excess_arcsec", float("nan")),
        metrics.get("edge_excess_arcsec", float("nan")),
        metrics.get("sweet_spot_x", float("nan")), metrics.get("sweet_spot_y", float("nan")),
        metrics.get("sweet_spot_r", float("nan")), metrics.get("radial_fraction", float("nan")),
        metrics.get("uniform_fraction", float("nan")),
        metrics.get("uniform_angle_deg", float("nan")),
    )

    # One row per night, newest last; a re-run of the same night replaces its row
    # rather than adding a second one the trend would read as two nights.
    hist_path = dso_dir / "optics_trend.json"
    history = []
    if hist_path.exists():
        try:
            loaded = _json.load(open(hist_path))
            if isinstance(loaded, list):
                history = [r for r in loaded if r.get("night") != night]
        except Exception:
            logger.warning("optics_trend.json unreadable — starting a new history")
    history.append(record)
    history.sort(key=lambda r: r.get("night", ""))
    tmp = hist_path.with_suffix(".tmp")
    try:
        with open(tmp, "w") as f:
            _json.dump(history, f, indent=1)
        tmp.replace(hist_path)
    except Exception:
        logger.exception("Could not write %s", hist_path)
        return

    social_server.post_social_message(
        f"Optics: field excess {metrics.get('field_excess_arcsec', 0):.2f}\", "
        f"sweet spot r={metrics.get('sweet_spot_r', 0):.2f}, "
        f"radial {metrics.get('radial_fraction', 0):+.2f} "
        f"({metrics.get('stars_pooled', 0)} stars, night {len(history)} of history)"
    )


def do_main():
    logger = utils.set_logger()

    cfg = config.data()

    logger.info('Begin End Sequence')

    # Tag this run's posts (roof-close spectrogram, imaging summary, …) with
    # the imaging job that started the night, so they land on that job's card
    # in the webchat instead of the buried system feed. No-op when called
    # in-process from a thread that already has a job (e.g. emergency stop).
    jobs.adopt_imaging_job()

    try:
        with open("safety.txt", "w") as file:
            file.write("USER SAFE")
        logger.info("before discovery")
        dev_map = asyncio.run(ku.make_discovery_map())
        logger.info("after discovery")
        parked = pwi4_utils.get_is_parked()
        try:
            if parked:
                social_server.post_social_message("Mount says Iris is parked")
                instructions = (dict
                    (
                    {
                        "Telescope mount": 'off',
                        "Roof motor": 'on',
                        "Iris inside light": 'on'
                    }
                ))
                logger.info("step 1")
                asyncio.run(ku.kasa_do(dev_map, instructions))
                logger.info("step 2")
                # make sure lights are on
                time.sleep(60)
                cfg = config.data()

                inside_view = cfg["camera safety"]["scope_view"]
                # visual_status() retries internally on a garbage (torn/starved/
                # unreadable) frame: the mount is already confirmed parked by PWI4
                # above, so a single vision frame reading "not parked" is a corrupt
                # snapshot, not a real state. One such frame once left the roof open.
                parked, closed, is_open, mod_date = vision_safety.visual_status()
                pushover.push_message("Investigating if scope is parked", inside_view)
                if parked:
                    logger.info("step 3")
                    social_server.post_social_message("Vision Safety says Scope is parked, closing roof")
                    super_user_commands.announce_roof_movement("The roof will be closing in one minute")
                    super_user_commands.toggle_roof(dev_map, capture_direction="close")
                    # Judge the close exactly as roof!! close does — a single
                    # frame 30s after the relay is not a verdict, and reporting
                    # one as though it were is what told the imaging card the
                    # roof had not closed when it had (2026-08-04). Runs with
                    # imaging_run=True so a cancel on the imaging card can never
                    # unwind the shutdown half-done: each check posts its own
                    # annotated snapshot, and the loop gives the roof 20 minutes.
                    closed = super_user_commands.confirm_roof_state(
                        "closed", imaging_run=True)
                    if closed:
                        social_server.post_social_message("Vision Safety says roof is closed")
                    else:
                        social_server.post_social_message("Vision Safety says roof is NOT closed")
                        # An unclosed roof at end of night is the one thing here
                        # that needs a human tonight, not in the morning log.
                        pushover.push_message(
                            "ROOF NOT CONFIRMED CLOSED after the end sequence — check the observatory",
                            inside_view)
                    logger.info("step 4")
                    # Turn off the inside light + roof motor FIRST so a later
                    # failure (e.g. an unreachable dehumidifier relay) can never
                    # skip the light-off and leave the observatory lit.
                    try:
                        instructions = (dict
                            (
                            {
                                "Iris inside light": 'off',
                                "Roof motor": 'off',
                            }
                        ))
                        logger.info("step 5")
                        asyncio.run(ku.kasa_do(dev_map, instructions))
                        logger.info("step 6")
                    except Exception:
                        logger.exception("Failed to turn off inside light / roof motor")
                    # Turn on dehumidifier (best-effort; bounded + isolated so a
                    # relay blip can't abort the rest of the shutdown).
                    utl_shelly.set_dehumidifier(True)
                else:
                    social_server.post_social_message("Vision Safety says Scope is NOT parked")
                    # Roof stays open (scope not confirmed parked), but the inside
                    # light was switched on for vision and must still be turned off.
                    try:
                        asyncio.run(ku.kasa_do(dev_map, {"Iris inside light": 'off'}))
                    except Exception:
                        logger.exception("Failed to turn off inside light (scope-not-parked path)")

            else:
                social_server.post_social_message("Mount says Iris is NOT parked, roof will remain open")
                instructions = (dict
                    (
                    {
                        "Telescope mount": 'off',
                        "Roof motor": 'off',
                        "Iris inside light": 'off',
                    }
                ))
                logger.info("step 7")
                asyncio.run(ku.kasa_do(dev_map, instructions))
                logger.info("step 8")



        except:
            logger.info('Problem')
            logger.exception("Exception")


    except:
        logger.info('Problem')
        logger.exception("Exception")




    # Prefer the actual imaging start time persisted by doit_cmd; this is the
    # correct boundary for "frames from tonight" regardless of when the end
    # sequence runs. Fall back to yesterday's sunset only if the marker file is
    # missing or unreadable.
    imaging_start = None
    try:
        _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        with open(os.path.join(_root, "imaging_start.txt")) as _f:
            imaging_start = datetime.fromisoformat(_f.read().strip())
    except (FileNotFoundError, ValueError):
        imaging_start = None

    if imaging_start is None:
        from astral import LocationInfo
        from astral.sun import sun
        cfg = config.data()
        loc = cfg["location"]
        city = LocationInfo(loc["city"], "USA", loc["timezone"], loc["latitude"], loc["longitude"])
        yesterday = (datetime.now() - timedelta(days=1)).date()
        imaging_start = sun(city.observer, date=yesterday)["sunset"]

    _post_imaging_summary(imaging_start)
    logger.info('End End Sequence')
    super_user_commands.set_imaging_state(super_user_commands.ImagingState.NONE)



if __name__ == "__main__":
    do_main()
