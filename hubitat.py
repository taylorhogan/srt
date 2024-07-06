import requests
import time


on_url = "http://192.168.86.20/apps/api/48/devices/4/on?access_token=9bfdd7cb-cd28-4c6a-8825-cde784e4e68c"
off_url = "http://192.168.86.20/apps/api/48/devices/4/off?access_token=9bfdd7cb-cd28-4c6a-8825-cde784e4e68c"
play_url ="http://192.168.86.20/apps/api/48/devices/25/speak?hello?50?hans?access_token=9bfdd7cb-cd28-4c6a-8825-cde784e4e68c"



for idx in range(10):
     requests.get(on_url)
     time.sleep (1)
     requests.get(off_url)
     time.sleep(1)


