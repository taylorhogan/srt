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

    with open("safety.txt", "w") as file:
        file.write("USER SAFE")
    logger.info('Setting safety to safe')

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
