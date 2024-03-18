import sys
import logging
import time
import paho.mqtt.publish as publish
import ssl
from pathlib import Path
import requests
import http.client, urllib
import baseconfig as cfg
from paho.mqtt import client as mqtt_client

config = cfg.FlowConfig().config


# from pprint import pformat


# Enter your API key here
api_key = config["weather"]["api_key"]
city_name = config["location"]["city"]
# base_url variable to store url
base_url = "http://api.openweathermap.org/data/2.5/weather?"

# Give city name
# city_name = input("Enter city name : ")
# complete_url variable to store
# complete url address
complete_url = base_url + "appid=" + api_key + "&q=" + city_name

# get method of requests module
# return response object
response = requests.get(complete_url)

# json method of response object
# convert json format data into
# python format data
x = response.json()

# Now x contains list of nested dictionaries
# Check the value of "cod" key is equal to
# "404", means city is found otherwise,
# city is not found
if x["cod"] != "404":
    # store the value of "main"
    # key in variable y
    y = x["main"]

    # store the value corresponding
    # to the "temp" key of y
    current_temperature = y["temp"]

    # store the value corresponding
    # to the "pressure" key of y
    current_pressure = y["pressure"]

    # store the value corresponding
    # to the "humidity" key of y
    current_humidity = y["humidity"]

    # store the value of "weather"
    # key in variable z
    z = x["weather"]

    # store the value corresponding
    # to the "description" key at
    # the 0th index of z
    weather_description = z[0]["description"]

    # print following values
    print(" Temperature (F) = " +
          str(int((current_temperature - 273.15) * 9 / 5 + 32)) +
          "\n atmospheric pressure (in hPa unit) = " +
          str(current_pressure) +
          "\n humidity (in percentage) = " +
          str(current_humidity) +
          "\n description = " +
          str(weather_description))

    message_list = list()
    message_list.append({
        'topic': 'flow/weather',
        'payload': str(weather_description),
        'qos': 1,
        'retain': True,
    })



    publish.multiple(
        message_list,
        transport='tcp',
        hostname='localhost',
        port=8883,
        client_id='',
        keepalive=60,
        auth={'username': 'indi-allsky', 'password': 'Foo14me!'},
        tls={'ca_certs': '/etc/ssl/certs/ca-certificates.crt', 'cert_reqs': ssl.CERT_NONE, 'insecure': True},

    )



else:
    print(" City Not Found ")
