import random

from paho.mqtt import client as mqtt_client

import baseconfig as cfg
import http
import urllib

config = cfg.FlowConfig().config

broker = config["mqtt"]["broker_url"]
port = config["mqtt"]["port"]
topic = "flow/pushover"


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

    conn = http.client.HTTPSConnection("api.pushover.net:443")
    conn.request("POST", "/1/messages.json",
                 urllib.parse.urlencode({
                     "token": "a7fycu94si1ctfnubk3sfqhbsioct2",
                     "user": "ggd66ig5wrpo8z9y7eyncfihor4b33",
                     "message": "hello world",
                 }), {"Content-type": "application/x-www-form-urlencoded"})
    output = conn.getresponse().read().decode('utf-8')
    print(output)
