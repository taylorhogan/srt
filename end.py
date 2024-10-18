import asyncio
import time
from collections import OrderedDict

from kasa import Discover

_super_user_config = OrderedDict(
    {
        "Super Users": {
            'Thogan', 'tmhobservatory'
        },
    })


async def doit():
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


async def make_discovery_map():
    map_from_name_to_ip = dict()
    devices = await Discover.discover()
    for dev in devices.values():
        await dev.update()
        print(dev.host + " " + dev.alias)
        map_from_name_to_ip.update({dev.alias: dev.host})
    _super_user_config["name_map"] = map_from_name_to_ip


if __name__ == "__main__":
    print ("Start of end")
    asyncio.run(make_discovery_map())
    asyncio.run(doit())
    time.sleep(30)
    print("End of end")
