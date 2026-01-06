import asyncio
import os, sys
if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from cmd_processing import social_server
import logging

from hardware_control import kasa_utils as ku
from utils import utils

if __name__ == "__main__":

    utils.set_install_dir()
    logger = utils.set_logger()
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
        logger.info('Iris is go for imaging!')
    else:
        social_server.post_social_message("Iris is NOT go for imaging!")
        logger.info('Iris is NOT go for imaging!')

