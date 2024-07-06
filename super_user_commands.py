import os
import time
from collections import OrderedDict

import requests

_super_user_config = OrderedDict(
    {
        "hubitat_keys": {
            "mount_id": "20",
            "driveway_id": "4",
            "landscape_id": "17",
            "sonos_id": "24"
        },
        "hubitat_token": "986ed6ff-c4df-43a5-9bc6-ec5727acb888",
        "hubitat_url": "http://192.168.86.20/"
    })



def sonos_say_command(phrase):
    config = _super_user_config
    url = config["hubitat_url"] + "/apps/api/48/devices/" + config["hubitat_keys"][
        "sonos_id"] + "/speak?" + phrase + "?access_token=" + \
          config["hubitat_token"]
    requests.get(url)


def mount_on_command():
    config = _super_user_config
    url = config["hubitat_url"] + "/apps/api/48/devices/" + config["hubitat_keys"]["mount_id"] + "/on?access_token=" + \
          config["hubitat_token"]
    requests.get(url)


def mount_off_command():
    config = _super_user_config
    url = config["hubitat_url"] + "/apps/api/48/devices/" + config["hubitat_keys"]["mount_id"] + "/off?access_token=" + \
          config["hubitat_token"]
    requests.get(url)


def kasa_exterior_lights_on():
    os.system("kasa --host 192.168.86.59 on")
    os.system("kasa --host 192.168.86.43 on")


def kasa_exterior_lights_off():
    os.system("kasa --host 192.168.86.59 off")
    os.system("kasa --host 192.168.86.43 off")


def kasa_interior_lights_on():
    os.system("kasa --host 192.168.86.47 on")


def kasa_interior_lights_off():
    os.system("kasa --host 192.168.86.47 off")


def hubitat_lights_on():
    config = _super_user_config
    url1 = config["hubitat_url"] + "apps/api/48/devices/" + config["hubitat_keys"][
        "driveway_id"] + "/on?access_token=" + config["hubitat_token"]
    url2 = config["hubitat_url"] + "apps/api/48/devices/" + config["hubitat_keys"][
        "landscape_id"] + "/on?access_token=" + config["hubitat_token"]
    requests.get(url1)
    requests.get(url2)


def hubitat_lights_off():
    config = _super_user_config
    url1 = config["hubitat_url"] + "apps/api/48/devices/" + config["hubitat_keys"][
        "driveway_id"] + "/off?access_token=" + config["hubitat_token"]
    url2 = config["hubitat_url"] + "apps/api/48/devices/" + config["hubitat_keys"][
        "landscape_id"] + "/off?access_token=" + config["hubitat_token"]
    requests.get(url1)
    requests.get(url2)


def all_lights_on_command():
    hubitat_lights_on()
    kasa_exterior_lights_on()


def all_lights_off_command():
    hubitat_lights_off()
    kasa_exterior_lights_off()


def all_lights_blink_command():
    for idx in range(3):
        all_lights_on_command()
        time.sleep(1)
        all_lights_off_command()
        time.sleep(1)


def prepare_for_start_command():
    all_lights_blink_command()
    all_lights_off_command()
    mount_on_command()


def prepare_for_stop_command():
    mount_off_command()
    all_lights_blink_command()
    all_lights_off_command()

_super_user_commands ={
    "start!":  prepare_for_start_command,
    "stop!" : prepare_for_stop_command
}

def do_super_user_command(words):
    action = _super_user_commands.get(words[1], "no_key")
    if action != "no_key":
        action()
        return True
    else:
        return False




#
#
# async def all_lights_off_command_await():
#     dev = await Discover.discover_single("192.168.86.59")
#     await dev.turn_on()
#     await dev.update()
# async def all_lights_on_command_await():
#     dev = await Discover.discover_single("192.168.86.59")
#     await dev.turn_on()
#     await dev.update()
#
#
# def all_lights_on_command():
#     asyncio.run(all_lights_on_command_await())
#
#
# def all_lights_off_command():
#     asyncio.run(all_lights_off_command_await())
# if __name__ == "__main__":
#     for idx in range(10):
#         os.system("kasa --host 192.168.86.59 on")
#         os.system("kasa --host 192.168.86.43 on")
#         # all_lights_on_command()
#         time.sleep(1)
#         # all_lights_off_command_await()
#         os.system("kasa --host 192.168.86.59 off")
#         os.system("kasa --host 192.168.86.43 off")
#
#         time.sleep(1)



