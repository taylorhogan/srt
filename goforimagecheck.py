import asyncio

import social_server
import logging

import kasa_utils as ku
import utils

if __name__ == "__main__":

    utils.set_install_dir()

    logging.basicConfig(filename='../iris.log', level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logger = logging.getLogger(__name__)
    logger.info('Go For Image Check')



    dev_map = asyncio.run(ku.make_discovery_map())
    instructions = (dict
        (
        {
            "Telescope mount": 'ison',
            "Iris door light": 'isoff',
            "Iris inside light": "isoff",
            "Driveway lights": "isoff"
        }
    ))

    check_ok = asyncio.run(ku.kasa_check(dev_map, instructions))
    if check_ok:
        social_server.post_social_message("Iris is go for imaging!")
    else:
        social_server.post_social_message("Iris is NOT go for imaging!")

