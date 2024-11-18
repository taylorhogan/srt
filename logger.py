# python3.6
# Input flow/log
# Output none
# side effects prints to log file

import random
import config.baseconfig as cfg
from paho.mqtt import client as mqtt_client

config = cfg.FlowConfig().config

broker = config["mqtt"]["broker_url"]
port = config["mqtt"]["port"]
topic = "flow/log"

# Generate a Client ID with the subscribe prefix.
client_id = f'subscribe-{random.randint(0, 100)}'
username = config["mqtt"]["user_name"]
password = config["mqtt"]["password"]


def connect_mqtt() -> mqtt_client:
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")

    client.subscribe(topic)
    client.on_message = on_message


def run():
    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()


if __name__ == '__main__':
    run()
