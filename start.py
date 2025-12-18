import asyncio
import logging
import kasa_utils as ku
import config
import os
import requests
import utils
import super_user_commands





if __name__ == "__main__":

    cfg = config.data()
    path = os.path.join(cfg["Install"], 'iris.log')

    logging.basicConfig(filename=path, level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logger = logging.getLogger(__name__)
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
