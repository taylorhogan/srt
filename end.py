import asyncio
import os.path

import inside_camera_server
import vision_safety
import social_server
import config
import logging
import pwi4_utils
import os
import kasa_utils as ku


def determine_roof_state():
    cfg = config.data()
    inside_camera_server.take_snapshot()
    is_closed, is_parked, is_open, mod_date = vision_safety.analyse_safety(cfg["camera safety"]["scope_view"])
    reply = "Roof Closed Visual: " + str(is_closed) + "\n"
    reply += "Roof Open Visual: " + str(is_open) + "\n"
    reply += "Scope Parked Visual:" + str(is_parked) + "\n"
    reply += "Copied Date:" + mod_date + "\n"
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

        parked = pwi4_utils.get_is_parked(logger)
        if parked:
            social_server.post_social_message("Mount says Iris is parked")
            dev_map = asyncio.run(ku.make_discovery_map())
            instructions = (dict
                (
                {
                    "Telescope mount": 'off',
                    "Roof motor": 'on'
                }
            ))

            asyncio.run(ku.kasa_do(dev_map, instructions))
        else:
            social_server.post_social_message("Mount says Iris is NOT parked")

    except:
        logger.info('Problem')
        logger.exception("Exception")

    logger.info('End End Sequence')
