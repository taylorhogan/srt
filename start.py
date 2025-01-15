import asyncio
import sys
import social_server
import logging
import config
import kasa_utils as ku
import config
import os



if __name__ == "__main__":

    cfg = config.data()
    path = os.path.join(cfg["Install"], 'iris.log')
    print (path)
    logging.basicConfig(filename=path, level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logger = logging.getLogger(__name__)
    cfg["logger"]["logging"] = logger
    logger.info('Start Start Sequence')

    dev_map = asyncio.run(ku.make_discovery_map())
    instructions = (dict
        (
        {
            "Telescope mount": 'on',
            "Iris back lights": 'off',
            "Iris door light": 'off',
            "Iris inside light": "off",
            "Driveway lights": "off"
        }
    ))

    asyncio.run(ku.kasa_do(dev_map, instructions))
    logger.info('End Start Sequence')
