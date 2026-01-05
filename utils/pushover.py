import http
import random
import urllib

import requests

import configs
from utils import rate_limit

cfg = configs.data()

client_id = f'subscribe-{random.randint(0, 100)}'
token = cfg['pushover']['token']
user = cfg['pushover']['user']

api_limiter = rate_limit.RateLimiter(max_calls=6, period=60.0)
def push_message(message, image=None):
    if not api_limiter():
        print("Rate limit exceeded - ignoring this call")
        return None

    if image is not None:
        push_message_with_picture(message, image)
        return

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
    for i in range(10):
        print (i)
        push_message ("hi", "./base_images/inside.jpg")


if __name__ == '__main__':
    main()
