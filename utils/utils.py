import os
import random
from hardware_control import kasa_utils as ku
import asyncio
import paho.mqtt.client as paho
import logging
import config

topic_to_sched = "iris/to_sched"
topic_from_sched = "iris/from_sched"

def set_install_dir():
    path = os.path.dirname(__file__)

    if os.path.exists(path):
        os.chdir(path)

    return path

def get_device_map ():
    cfg = config.data()
    path = os.path.join(set_install_dir(), 'iris.log')
    logging.basicConfig(filename=path, level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logger = logging.getLogger(__name__)
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
    dev_map = get_device_map()
    print  (dev_map)