"""
kasa_switch.py
Command-line utility to turn a Kasa smart plug on or off by device name.

Usage:
    python kasa_switch.py <device_name> <0|1>

Arguments:
    device_name   The alias of the Kasa device (e.g. "Telescope mount")
    state         1 = on, 0 = off

Example:
    python kasa_switch.py "Telescope mount" 1
"""

import asyncio
import sys
import os

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from hardware_control import kasa_utils as ku


async def kasa_switch(device_name: str, state: int) -> None:
    """Turn a Kasa device on (state=1) or off (state=0) by its alias name."""
    dev_map = await ku.make_discovery_map()
    if device_name not in dev_map:
        print(f"Device '{device_name}' not found. Available devices:")
        for name in dev_map:
            print(f"  {name}")
        sys.exit(1)

    action = "on" if state else "off"
    ok = (await ku.kasa_do(dev_map, {device_name: action})).get(device_name)
    if ok:
        print(f"{device_name} turned {action} (verified)")
    else:
        print(f"{device_name} was NOT switched {action} -- the relay does not "
              f"read back in that state")
        sys.exit(2)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python kasa_switch.py <device_name> <0|1>")
        sys.exit(1)

    name = sys.argv[1]
    try:
        state = int(sys.argv[2])
        if state not in (0, 1):
            raise ValueError
    except ValueError:
        print("State must be 0 (off) or 1 (on)")
        sys.exit(1)

    asyncio.run(kasa_switch(name, state))
