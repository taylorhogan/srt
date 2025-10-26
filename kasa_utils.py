import asyncio
from kasa import Discover


async def make_discovery_map():

        map_from_name_to_ip = dict()
        try:
            devices = await Discover.discover()
        except:
            print("Problem in discover")
        for dev in devices.values():

            await dev.update()
            try:
                map_from_name_to_ip.update({dev.alias: dev.host})
            except:
                print("Problem in update")

        return map_from_name_to_ip





async def kasa_do(cfg, instructions):
    for key in instructions.keys():
        try:
            ip = cfg[key]
            dev = await Discover.discover_single(ip)
            if instructions[key] == "on":
                await dev.turn_on()
            else:
                await dev.turn_off()
            await dev.update()
        except KeyError:
            print("Key " + key + " not found")


async def kasa_check(cfg, instructions):
    for key in instructions.keys():
        try:
            ip = cfg[key]
            dev = await Discover.discover_single(ip)
            ison = dev.is_on
            isoff = dev.is_off

            if instructions[key] == "ison" and isoff:
                return False
            if instructions[key] == "isoff" and ison:
                return False

        except KeyError:
            print("Key " + key + " not found")
            return False
        return True
if __name__ == "__main__":
    dev_map = asyncio.run(make_discovery_map())
    print (dev_map)