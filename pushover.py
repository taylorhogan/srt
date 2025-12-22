import http
import random
import urllib

import requests

import config

cfg = config.data()

client_id = f'subscribe-{random.randint(0, 100)}'
token = cfg['pushover']['token']
user = cfg['pushover']['user']


def push_message(message):
    try:
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
    except Exception as e:
        print(f"Error during pushover: {e}")


def push_message_with_picture(message, image):
    try:

        response = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": token,
                "user": user,
                "message": message,  # Required: Your message text (up to 1024 chars)
                "title": "News From Iris",  # Optional: Notification title (up to 250 chars)
            },
            files={
                "attachment":
                    ("image.jpg",
                     open(image, "rb"),
                     "image/jpeg")
            }
        )

        print(response.text)
    except Exception as e:
        print(f"Error during pushover: {e}")


def main():
    push_message_with_picture("hi", "./base_images/inside.jpg")


if __name__ == '__main__':
    main()
