import http
import random
import urllib

from paho.mqtt import client as mqtt_client

import config

cfg = config.data()



client_id = f'subscribe-{random.randint(0, 100)}'
token = cfg['pushover']['token']
user = cfg['pushover']['user']




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
    # conn = http.client.HTTPSConnection("api.pushover.net:443")
    # conn.request("POST", "/1/messages.json",
    #
    #              urllib.parse.urlencode({
    #                  "token": token,
    #                  "user": user,
    #                  "message": message,
    #                  "verify": False,
    #                  "attachment": ("image.jpg", open(image, "rb"), "image/jpeg")
    #              }), {"Content-type": "application/x-www-form-urlencoded", "verify": False})
    # output = conn.getresponse().read().decode('utf-8')
    # print(output)

    conn = http.client.HTTPSConnection("api.pushover.net:443")
    r = conn.request("POST","https://api.pushover.net/1/messages.json",
                           data={
                               "token": token,
                               "user": user,
                               "message": message

                           },
                           files={
                               "attachment":
                                   ("image.jpg",
                                    open(image, "rb"),
                                    "image/jpeg")
                           }
                     )

    output = conn.getresponse().read().decode('utf-8')
    print(output)


def main():
    push_message_with_picture("hi", "./base_images/inside.jpg")


if __name__ == '__main__':
    main()
