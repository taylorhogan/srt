import asyncio
import os
import time
from collections import OrderedDict

import requests
from kasa import Discover

async def test():
    dev = await Discover.discover_single("192.168.86.20")
    await dev.turn_on()
    await dev.update()


if __name__ == "__main__":
    asyncio.run(test())