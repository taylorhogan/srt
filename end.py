import asyncio

import inside_camera_server
import social_server
import config
import logging
import vision_safety
import kasa_utils as ku
import pwi4_utils
import os


def determine_roof_state_visually():
    cfg = config.data()

    is_parked, is_closed, is_open, mod_date = vision_safety.visual_status()
    if is_parked:
        if is_closed:
            reply = "Scope is parked and roof is closed"
        else:
            reply = "Scope is parked and roof is not closed"
        if is_open:
            reply = "Scope is parked and roof is open"
        else:
            reply = "Scope is parked and roof is not open"
    else:
        reply = "Scope is not parked"


    reply += "\nPicture Date:" + mod_date + "\n"
    social_server.post_social_message(reply, cfg["camera safety"]["scope_view"])


if __name__ == "__main__":

    cfg = config.data()
    path = os.path.join(cfg["Install"], 'iris.log')
    logging.basicConfig(filename=path, level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logger = logging.getLogger(__name__)
    cfg["logger"]["logging"] = logger

    logger.info('Begin End Sequence')

    try:
        dev_map = asyncio.run(ku.make_discovery_map())

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

                asyncio.run(ku.kasa_do(dev_map, instructions))
            else:
                social_server.post_social_message("Mount says Iris is NOT parked")
                instructions = (dict
                    (
                    {
                        "Telescope mount": 'off',
                        "Roof motor": 'off',
                        "Iris inside light": 'on'
                    }
                ))

                asyncio.run(ku.kasa_do(dev_map, instructions))
        except:
            logger.info('Problem')
            logger.exception("Exception")

        determine_roof_state_visually()

    except:
        logger.info('Problem')
        logger.exception("Exception")

    logger.info('End End Sequence')
