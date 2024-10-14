from pyShelly import pyShelly

import requests
import time

r = requests.get('http://192.168.86.41/relay/0?turn=on')
time.sleep(3)
r = requests.get('http://192.168.86.41/relay/0?turn=off')
