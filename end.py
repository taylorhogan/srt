import asyncio
import logging
import os
import time

import requests

import config
import kasa_utils as ku
import pwi4_utils
import social_server
import super_user_commands
import vision_safety


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


def do_main():
    cfg = config.data()
    path = os.path.join(cfg["Install"], 'iris.log')
    logging.basicConfig(filename=path, level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logger = logging.getLogger(__name__)
    cfg["logger"]["logging"] = logger

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
                parked, closed, is_open, mod_date = vision_safety.visual_status()
                if parked:
                    logger.info("step 3")
                    social_server.post_social_message("Vision Safety says Scope is parked, closing roof")
                    super_user_commands.toggle_roof(dev_map)
                    # wait for roof to close
                    time.sleep(30)
                    parked, closed, is_open, mod_date = vision_safety.visual_status()
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

    logger.info('End End Sequence')


if __name__ == "__main__":
    do_main()
