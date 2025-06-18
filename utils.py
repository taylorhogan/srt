import os
import random

import paho.mqtt.client as paho

topic_to_sched = "iris/to_sched"
topic_from_sched = "iris/from_sched"

def set_install_dir():
    path = os.path.dirname(__file__)

    if os.path.exists(path):
        os.chdir(path)

    return path


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
