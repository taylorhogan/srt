import os
import time
from collections import OrderedDict
import social_server
from kasa import Discover
import requests
import asyncio
import requests

_super_user_config = OrderedDict(
    {
        "Super Users": {
            'Thogan', 'tmhobservatory'
        },
    })






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



def all_lights_on_command():

    kasa_exterior_lights_on()


def all_lights_off_command():

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



def prepare_for_stop_command():

    all_lights_blink_command()
    all_lights_off_command()


def toggle_roof_command ():
    all_lights_blink_command()
    all_lights_off_command()
    r = requests.get('http://192.168.86.41/relay/0?turn=on')
    time.sleep(3)
    r = requests.get('http://192.168.86.41/relay/0?turn=off')

_super_user_commands ={
    "start!":  prepare_for_start_command,
    "stop!" : prepare_for_stop_command,
    "roof!": toggle_roof_command
}

def do_super_user_command(words, account):
    config = _super_user_config
    action = _super_user_commands.get(words[1], "no_key")
    if action != "no_key":
        super_users=config["Super Users"]
        if account in super_users:
            action()
            return True
        else:
            social_server.post_social_message(account + " Is not authorized\n")
        return False
    else:
        return False





async def make_discovery_map():
    map_from_name_to_ip = dict ()
    devices = await Discover.discover()
    for dev in devices.values():
        await dev.update()
        print(dev.host + " " + dev.alias)
        map_from_name_to_ip.update({dev.alias: dev.host})
    _super_user_config["name_map"] = map_from_name_to_ip





def turn_inside_light (onoff):
    if not _super_user_config.get("name_map"):
        asyncio.run (make_discovery_map())
    host = _super_user_config["name_map"]["Iris inside light"]
    if host != None:
        string = "kasa --host " + host + " " + onoff
        os.system(string)


