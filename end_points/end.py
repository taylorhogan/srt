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
                    "sky_adu_per_s":   round(sky["sky_adu_per_s"], 2) if sky else None,
                    "sky_mag_arcsec2": round(sky["sky_mag_arcsec2"], 2)
                                       if sky and sky.get("sky_mag_arcsec2") is not None else None,
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
                parked, closed, is_open, mod_date = vision_safety.visual_status()
                pushover.push_message("Investigating if scope is parked", inside_view)
                if parked:
                    logger.info("step 3")
                    social_server.post_social_message("Vision Safety says Scope is parked, closing roof")
                    super_user_commands.announce_roof_movement("The roof will be closing in one minute")
                    super_user_commands.toggle_roof(dev_map, capture_direction="close")
                    # wait for roof to close
                    time.sleep(30)

                    parked, closed, is_open, mod_date = vision_safety.visual_status()
                    pushover.push_message("Investigating if roof is closed", inside_view)
                    if closed:
                        social_server.post_social_message("Vision Safety says roof is closed")
                    else:
                        social_server.post_social_message("Vision Safety says roof is NOT closed")
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
