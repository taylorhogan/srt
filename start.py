import asyncio
import time
import sys
import social_server
from collections import OrderedDict
from kasa import Discover
import baseconfig as cfg

_super_user_config = OrderedDict(
    {
        "Super Users": {
            'Thogan', 'tmhobservatory'
        },
    })


async def doit():
    ip = _super_user_config["name_map"]["Telescope mount"]
    dev = await Discover.discover_single(ip)
    await dev.turn_on()
    await dev.update()

    ip = _super_user_config["name_map"]["Iris back lights"]
    dev = await Discover.discover_single(ip)
    await dev.turn_off()
    await dev.update()

    ip = _super_user_config["name_map"]["Iris door light"]
    dev = await Discover.discover_single(ip)
    await dev.turn_off()
    await dev.update()

    ip = _super_user_config["name_map"]["Iris inside light"]
    dev = await Discover.discover_single(ip)
    await dev.turn_off()
    await dev.update()

    ip = _super_user_config["name_map"]["Driveway lights"]
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


if __name__ == "__main__":
    print("Start of start")
    print ("Telling Mastodon something")
    if len(sys.argv) < 2:
        social_server.post_social_message("Starting imaging run of debug")
    else:
        social_server.post_social_message("Starting imaging run of " + sys.argv[1])
    asyncio.run(make_discovery_map())
    asyncio.run(doit())
    print("End of start")
