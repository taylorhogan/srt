import asyncio
import logging
import kasa_utils as ku
import config
import os
import requests




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
            "Iris door light": 'off',
            "Iris inside light": "off",
            "Driveway lights": "off",
            "Iris inside camera":"off"
        }
    ))

    asyncio.run(ku.kasa_do(dev_map, instructions))

# turn off recepticle
    r = requests.get('http://192.168.87.28/relay/0?turn=off')

    logger.info('End Start Sequence')
