import asyncio
from datetime import datetime
import logging
import os
from pathlib import Path
import sys
import time

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
    fwhm_px_list, fwhm_arcsec_list, star_count_list, ecc_list = [], [], [], []

    for fits_file in fits_files:
        try:
            mean_px, mean_arcsec, star_count, mean_ecc = fitsfwhm.calculate_fwhm(
                fits_file, arcsec_per_pixel=arcsec_per_pixel
            )
            if star_count > 0:
                fwhm_px_list.append(mean_px)
                fwhm_arcsec_list.append(mean_arcsec)
                star_count_list.append(star_count)
                ecc_list.append(mean_ecc)
        except Exception:
            logger.exception("FWHM analysis failed for %s", fits_file)

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
        f"Median FWHM: {median_fwhm_px:.2f} px ({median_fwhm_arcsec:.2f}\")\n"
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

                # Read imaging start time written by doit_cmd
                try:
                    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                    with open(os.path.join(_root, "imaging_start.txt")) as _f:
                        imaging_start = datetime.fromisoformat(_f.read().strip())
                except Exception:
                    imaging_start = datetime.now()  # fallback: won't match any old FITS files
                _post_imaging_summary(imaging_start)

        except:
            logger.info('Problem')
            logger.exception("Exception")


    except:
        logger.info('Problem')
        logger.exception("Exception")

    super_user_commands.set_imaging_state(super_user_commands.ImagingState.NONE)
    logger.info('End End Sequence')



if __name__ == "__main__":
    do_main()
