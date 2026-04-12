import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import requests

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from configs import config

from fits_processing import fitsfwhm
from hardware_control import pwi4_utils, kasa_utils as ku
from cmd_processing import super_user_commands, social_server
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
    reply += "\nPicture Date:" + mod_date + "\n"
    logger = cfg["logger"]["logging"]
    logger.info(account)
    if account in cfg["Super Users"]:
        social_server.post_social_message(reply, cfg["camera safety"]["scope_view"])
    else:
        social_server.post_social_message(reply)


def _post_imaging_summary(imaging_start: datetime) -> None:
    """Scan FITS files written since imaging_start and post quality metrics to Mastodon."""
    logger = utils.set_logger()
    cfg = config.data()
    image_dir = Path(cfg["nina"]["image_dir"])
    arcsec_per_pixel = cfg["nina"]["arc_sec_per_pixel"]
    start_ts = imaging_start.timestamp()
    social_server.post_social_message(
        f"Scanning for FITS since {imaging_start.strftime('%Y-%m-%d %H:%M')}"
    )

    try:
        fits_files = sorted(
            (f for f in image_dir.rglob("*.fits") if f.stat().st_mtime >= start_ts),
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
    social_server.post_social_message(
        f"Processing {len(fits_files)} FITS files, first: {fits_files[0].name}"
    )
    fwhm_px_list, fwhm_arcsec_list, star_count_list, ecc_list = [], [], [], []

    def _analyse(f):
        return fitsfwhm.calculate_fwhm(f, arcsec_per_pixel=arcsec_per_pixel)

    total = len(fits_files)
    max_workers = min(8, total)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_analyse, f): f for f in fits_files}
            for n, fut in enumerate(as_completed(futures), start=1):
                logger.info("Analysing frames: %d/%d", n, total)
                try:
                    mean_px, mean_arcsec, star_count, mean_ecc = fut.result()
                    if star_count > 0:
                        fwhm_px_list.append(mean_px)
                        fwhm_arcsec_list.append(mean_arcsec)
                        star_count_list.append(star_count)
                        ecc_list.append(mean_ecc)
                except Exception:
                    logger.exception("FWHM analysis failed for %s", futures[fut])

    if not fwhm_px_list:
        social_server.post_social_message(
            f"Imaging complete — {len(fits_files)} frames, no stars detected in any frame"
        )
        return

    median_fwhm_px     = float(np.median(fwhm_px_list))
    median_fwhm_arcsec = float(np.median(fwhm_arcsec_list))
    median_stars       = float(np.median(star_count_list))
    median_ecc         = float(np.median(ecc_list))

    social_server.post_social_message(
        f"Imaging complete — {len(fits_files)} frames ({len(fwhm_px_list)} with stars)\n"
        f'Median FWHM: {median_fwhm_arcsec:.2f}"\n'
        f"Median stars/frame: {median_stars:.0f}\n"
        f"Median eccentricity: {median_ecc:.3f}"
    )
    logger.info(
        "Imaging summary: %d frames, median FWHM=%.2f px (%.2f\"), stars=%.0f, ecc=%.3f",
        len(fits_files), median_fwhm_px, median_fwhm_arcsec, median_stars, median_ecc,
    )


def do_main():
    logger = utils.set_logger()

    cfg = config.data()

    logger.info('Begin End Sequence')

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
                    super_user_commands.toggle_roof(dev_map)
                    # wait for roof to close
                    time.sleep(30)

                    parked, closed, is_open, mod_date = vision_safety.visual_status()
                    pushover.push_message("Investigating if roof is closed", inside_view)
                    if closed:
                        social_server.post_social_message("Vision Safety says roof is closed")
                    else:
                        social_server.post_social_message("Vision Safety says roof is NOT closed")
                    logger.info("step 4")
                    # turn on dehumidifier
                    r = requests.get('http://192.168.87.28/relay/0?turn=on')
                    # turn off lights
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
                else:
                    social_server.post_social_message("Vision Safety says Scope is NOT parked")

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




    from astral import LocationInfo
    from astral.sun import sun
    cfg = config.data()
    loc = cfg["location"]
    city = LocationInfo(loc["city"], "USA", loc["timezone"], loc["latitude"], loc["longitude"])
    yesterday = datetime.now() - timedelta(days=1)
    sunset = sun(city.observer, date=yesterday)["sunset"]
    _post_imaging_summary(sunset.replace(tzinfo=None))
    logger.info('End End Sequence')
    super_user_commands.set_imaging_state(super_user_commands.ImagingState.NONE)



if __name__ == "__main__":
    do_main()
