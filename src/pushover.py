import http
import random
import urllib

from paho.mqtt import client as mqtt_client

from src import public as cfg

config = cfg.FlowConfig().config

broker = config["mqtt"]["broker_url"]
port = config["mqtt"]["port"]
topic = "flow/pushover"

client_id = f'subscribe-{random.randint(0, 100)}'
username = config["mqtt"]["user_name"]
password = config["mqtt"]["password"]

token = config['pushover']['token']
user = config['pushover']['user']


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
                     "token": token,
                     "user": user,
                     "message": "hello world",
                 }), {"Content-type": "application/x-www-form-urlencoded"})
    output = conn.getresponse().read().decode('utf-8')
    print(output)


def push_message(message):
    conn = http.client.HTTPSConnection("api.pushover.net:443")
    conn.request("POST", "/1/messages.json",

                 urllib.parse.urlencode({
                     "token": token,
                     "user": user,
                     "message": message,
                     "verify": False,
                 }), {"Content-type": "application/x-www-form-urlencoded", "verify": False})
    output = conn.getresponse().read().decode('utf-8')
    print(output)


def push_message_with_picture(message, image):
    conn = http.client.HTTPSConnection("api.pushover.net:443")
    conn.request("POST", "/1/messages.json",

                 urllib.parse.urlencode({
                     "token": token,
                     "user": user,
                     "message": message,
                     "verify": False,
                     "attachment": ("image.jpg", open(image, "rb"), "image/jpeg")
                 }), {"Content-type": "application/x-www-form-urlencoded", "verify": False})
    output = conn.getresponse().read().decode('utf-8')
    print(output)

    # conn = http.client.HTTPSConnection("api.pushover.net:443")
    # r = conn.request("POST","https://api.pushover.net/1/messages.json",
    #                        data={
    #                            "token": token,
    #                            "user": user,
    #                            "message": message
    #                            "attachment": ("image.jpg", open(image, "rb"), "image/jpeg")
    #                        },
    #                        files={
    #                            "attachment": ("image.jpg", open(image, "rb"), "image/jpeg")
    #                        })
    #
    # output = conn.getresponse().read().decode('utf-8')
    # print(output)


def main():
    push_message_with_picture("hi", "./base_images/inside.jpg")


if __name__ == '__main__':
    main()
