import asyncio
import time
from collections import OrderedDict
import inside_camera_server
import vision_safety
import social_server
import baseconfig as cfg
import sys
import os

from kasa import Discover

_super_user_config = OrderedDict(
    {
        "Super Users": {
            'Thogan', 'tmhobservatory'
        },
    })


async def lights_on_mount_off():
    ip = _super_user_config["name_map"]["Telescope mount"]
    dev = await Discover.discover_single(ip)
    await dev.turn_off()
    await dev.update()

    ip = _super_user_config["name_map"]["Iris back lights"]
    dev = await Discover.discover_single(ip)
    await dev.turn_on()
    await dev.update()

    ip = _super_user_config["name_map"]["Iris inside light"]
    dev = await Discover.discover_single(ip)
    await dev.turn_on()
    await dev.update()


async def lights_off():

    ip = _super_user_config["name_map"]["Iris back lights"]
    dev = await Discover.discover_single(ip)
    await dev.turn_off()
    await dev.update()

    ip = _super_user_config["name_map"]["Iris inside light"]
    dev = await Discover.discover_single(ip)
    await dev.turn_off()
    await dev.update()


async def make_discovery_map():
    map_from_name_to_ip = dict()
    devices = await Discover.discover()
    for dev in devices.values():
        await dev.update()
        print(dev.host + " " + dev.alias)
        map_from_name_to_ip.update({dev.alias: dev.host})
    _super_user_config["name_map"] = map_from_name_to_ip


def determine_roof_state():
    config = cfg.FlowConfig().config
    inside_camera_server.take_snapshot()
    is_closed, is_parked, is_open, mod_date = vision_safety.analyse_safety(config["camera safety"]["scope_view"])
    reply = "Roof Closed: " + str(is_closed) + "\n"
    reply += "Roof Open: " + str(is_open) + "\n"
    reply += "Scope Parked:" + str(is_parked) + "\n"
    reply += "Copied Date:" + mod_date + "\n"
    social_server.post_social_message(reply, config["camera safety"]["scope_view"])


if __name__ == "__main__":

    print("dir is " + str(sys.argv[1]))
    os.chdir(sys.argv[1])

    asyncio.run(make_discovery_map())
    asyncio.run(lights_on_mount_off())
    determine_roof_state()
    asyncio.run(lights_off())

    print("End of end")
