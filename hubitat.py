import requests
import time

on_url = "http://192.168.86.44/apps/api/48/devices/4/on?access_token=986ed6ff-c4df-43a5-9bc6-ec5727acb888"
off_url = "http://192.168.86.44/apps/api/48/devices/4/off?access_token=986ed6ff-c4df-43a5-9bc6-ec5727acb888"
play_url ="http://192.168.86.44/apps/api/48/devices/25/speak?hello?50?hans?access_token=986ed6ff-c4df-43a5-9bc6-ec5727acb888"



for idx in range(10):
     requests.get(on_url)
     time.sleep (.1)
     requests.get(off_url)
     time.sleep(.1)


