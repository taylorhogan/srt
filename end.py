import asyncio
from collections import OrderedDict
import inside_camera_server
import vision_safety
import social_server
import config
import sys
import os
import logging
import pwi4_utils
import kasa_utils as ku


from kasa import Discover

_super_user_config = OrderedDict(
    {
        "Super Users": {
            'Thogan', 'tmhobservatory'
        },
    })


async def mount_off():
    ip = _super_user_config["name_map"]["Telescope mount"]
    dev = await Discover.discover_single(ip)
    await dev.turn_off()
    await dev.update()




async def roof_motor_on():
    ip = _super_user_config["name_map"]["Roof motor"]
    dev = await Discover.discover_single(ip)
    await dev.turn_on()
    await dev.update()




async def lights_on():
    ip = _super_user_config["name_map"]["Iris back lights"]
    dev = await Discover.discover_single(ip)
    await dev.turn_on()
    await dev.update()

    ip = _super_user_config["name_map"]["Iris inside light"]
    dev = await Discover.discover_single(ip)
    await dev.turn_on()
    await dev.update()


async def lights_off():
    ip = _super_user_config["name_map"]["Iris back lights"]
    dev = await Discover.discover_single(ip)
    await dev.turn_off()
    await dev.update()

    ip = _super_user_config["name_map"]["Iris inside light"]
    dev = await Discover.discover_single(ip)
    await dev.turn_off()
    await dev.update()



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

    logging.basicConfig(filename='iris.log', level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    cfg = config.data()
    logger = logging.getLogger(__name__)
    cfg["logger"]["logging"] = logger

    logger.info('End Sequence')

    try:

        parked = pwi4_utils.get_is_parked(logger)
        if parked:
            social_server.post_social_message("Mount says Iris is parked")
        else:
            social_server.post_social_message("Mount says Iris is not parked")



        if parked:
            dev_map = asyncio.run(ku.make_discovery_map())
            instructions = (dict
                (
                {
                    "Telescope mount": 'off',
                    "Roof motor": 'on'
                }
            ))

            asyncio.run(ku.kasa_do(dev_map, instructions))


    except:
        logger.info('Problem')
        logger.exception("Exception")



    print("End of end")
