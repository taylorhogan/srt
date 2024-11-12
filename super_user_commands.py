import asyncio
import os
import subprocess
import time
import traceback
from collections import OrderedDict

import requests
from kasa import Discover
import end

_super_user_config = OrderedDict(
    {
        "Super Users": {
            'Thogan', 'tmhobservatory'
        },
    })


async def make_discovery_map():
    traceback.print_stack()
    print("make map")
    map_from_name_to_ip = dict()
    devices = await Discover.discover()
    for dev in devices.values():
        # await dev.update()
        print(dev.host + " " + dev.alias)
        map_from_name_to_ip.update({dev.alias: dev})
    _super_user_config["name_map"] = map_from_name_to_ip


def kasa_switch_command(device_name, on_off):
    if not _super_user_config.get("name_map"):
        asyncio.run(make_discovery_map())
    dev = _super_user_config["name_map"][device_name]
    if dev is not None:
        string = "kasa --host " + dev.host + " " + on_off
        # os.system(string)
        if on_off == "on":
            dev.turn_on()
        else:
            dev.turn_off()


def kasa_lights_on ():
    asyncio.run(make_discovery_map())
    asyncio.run(end.lights_on())


def kasa_lights_off ():
    asyncio.run(make_discovery_map())
    asyncio.run(end.lights_off())


def toggle_roof_command():
    print ("toggle roof command")
    r = requests.get('http://192.168.86.41/relay/0?turn=on')
    time.sleep(3)
    r = requests.get('http://192.168.86.41/relay/0?turn=off')



def start_nina():
    print("Starting Nina1")
    subprocess.run(["C:/home/taylorhogan/Documents/tmh/runnina.bat"])
    print("Done with Nina")


def get_super_user_commands():
    return {
        "roof!": toggle_roof_command,
        "lights_on!": kasa_lights_on,
        "lights_off!": kasa_lights_off,
        "nina!": start_nina
    }


def do_super_user_command(words, account):
    config = _super_user_config
    su_commands = get_super_user_commands()
    print (str(su_commands))
    action = su_commands.get(words[1], "no_key")
    print("action is " + str (action) + " word " + str(words[1]))
    if action != "no_key":
        super_users = config["Super Users"]
        if account in super_users:
            print("Whoot")
            action()
            return True
        else:
            # social_server.post_social_message(account + " Is not authorized\n")
            print("no auth")
            return False
    else:
        return False


