import asyncio
import logging

import os,sys

import requests

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from cmd_processing import super_user_commands
from hardware_control import kasa_utils as ku
from configs import config
from utils import utils

if __name__ == "__main__":

    cfg = config.data()
    logger = utils.set_logger()

    cfg["logger"]["logging"] = logger
    logger.info('Start Start Sequence')
    utils.set_install_dir()

    super_user_commands.safe_cmd (None, None)
    super_user_commands.imaging_state(False)
    logger.info('Setting safety to safe and not imaging')

    dev_map = asyncio.run(ku.make_discovery_map())
    instructions = (dict
        (
        {
            "Telescope mount": 'on',
            "Iris door light": 'off',
            "Iris inside light": "off",
            "Driveway lights": "off",
            "Deck lights": "off",
            "Grill Lights":"off",
            "Iris landscape lights": "off",
            "Main landscape lights": "off"

        }
    ))

    asyncio.run(ku.kasa_do(dev_map, instructions))
    logger.info('Turning off lights')

    r = requests.get('http://192.168.87.28/relay/0?turn=off')
    logger.info('Turning off dehumidifier')
    logger.info('End Start Sequence')
    print ("Done with startup")
