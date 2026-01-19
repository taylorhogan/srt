import os
import random
from hardware_control import kasa_utils as ku
import asyncio
import paho.mqtt.client as paho
import logging
from configs import config
from pathlib import Path
from datetime import datetime



topic_to_sched = "iris/to_sched"
topic_from_sched = "iris/from_sched"
__logger = None


def create_timestamped_filename(base_name, extension=""):
    """Create a filename with base name plus current date and time."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if extension and not extension.startswith("."):
        extension = "." + extension
    return f"{base_name}_{timestamp}{extension}"

def set_logger ():
    global __logger
    if __logger is None:
        cfg = config.data()
        path = Path (__file__).parent.parent.resolve() / 'iris.log'
        print("logging at " + str(path))
        logging.basicConfig(filename=path, level=logging.INFO, format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p')
        logger = logging.getLogger(__name__)
        cfg["logger"]["logging"] = logger
        __logger = logger

    return __logger

def set_install_dir():
    path = Path(__file__).parent.parent.resolve()


    if os.path.exists(path):
        os.chdir(path)

    return path

def get_device_map ():

    logger = set_logger()
    dev_map = {}
    try:
        dev_map = asyncio.run(ku.make_discovery_map())
    except:
        logger.info('Problem')
        logger.exception("Exception")
    return dev_map

def connect_mqtt():
    def on_connect(client, userdata, flags, rc, properties):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    # Set Connecting Client ID
    client_id = f'publish-{random.randint(0, 1000)}'

    # For paho-mqtt 2.0.0, you need to set callback_api_version.
    client = paho.Client(client_id=client_id, callback_api_version=paho.CallbackAPIVersion.VERSION2)

    # client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect('localhost', 1883)
    return client

if __name__ == '__main__':
    set_install_dir()
    dev_map = get_device_map()
    print  (dev_map)