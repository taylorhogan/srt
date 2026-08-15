import asyncio
import json
import socket
import struct

from kasa import Discover


def _xor(payload: bytes, decrypt: bool = False) -> bytes:
    """TP-Link's autokey XOR cipher, the whole of the legacy protocol's security."""
    key = 171
    out = bytearray()
    for c in payload:
        out.append(key ^ c)
        key = c if decrypt else (key ^ c)
    return bytes(out)


def legacy_alias(host: str, timeout: float = 4.0):
    """Ask *host* its name over the legacy protocol on TCP 9999, or None.

    Exists because ``Discover.discover()`` is NOT deterministic on these plugs.
    They answer both the legacy broadcast and the newer KLAP one, and whichever
    reply the library takes decides whether an alias comes back at all: the
    legacy payload carries ``system.get_sysinfo`` with the name in it, the KLAP
    payload carries device_id, model and mac and no name.

    Measured 2026-08-15: four consecutive discoveries resolved Telescope mount
    and Roof motor, and a run half an hour later resolved neither, while a
    direct query to 9999 returned both names every single time. Without this the
    map is a coin toss, and a missing key means every command to that device
    silently does nothing.
    """
    try:
        s = socket.create_connection((host, 9999), timeout=timeout)
    except OSError:
        return None
    try:
        body = _xor(b'{"system":{"get_sysinfo":{}}}')
        s.sendall(struct.pack(">I", len(body)) + body)
        head = s.recv(4)
        if len(head) < 4:
            return None
        n = struct.unpack(">I", head)[0]
        if n > 1 << 20:                      # not a reply we understand
            return None
        buf = b""
        while len(buf) < n:
            chunk = s.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        info = json.loads(_xor(buf, decrypt=True))["system"]["get_sysinfo"]
        return info.get("alias") or None
    except (OSError, ValueError, KeyError, struct.error):
        return None
    finally:
        s.close()


async def make_discovery_map():
    """{alias: ip} for every device that will answer to a name.

    Addressing by NAME rather than IP is deliberate: it is what lets DHCP move
    these devices around without breaking anything, and why only the sky camera
    needs a reservation. So an alias that discovery failed to return is worth a
    second, direct attempt rather than a silently absent key.
    """
    map_from_name_to_ip = dict()

    devices = {}
    try:
        devices = await Discover.discover()
    except Exception as e:
        print(f"Error during discovery: {e}")

    for host, device in devices.items():
        alias = device.alias
        if not alias:
            # KLAP reply, or a device that simply did not say. Ask it directly.
            alias = await asyncio.to_thread(legacy_alias, host)
            if alias:
                print(f"Device found at {host}: {alias} ({device.model}) [via 9999]")
        else:
            print(f"Device found at {host}: {alias} ({device.model})")
        if alias:
            map_from_name_to_ip[alias] = host

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


async def kasa_get_states(cfg, device_names):
    """Return a dict of device_name -> 'on'/'off' for each named device."""
    states = {}
    for name in device_names:
        try:
            ip = cfg[name]
            dev = await Discover.discover_single(ip)
            await dev.update()
            states[name] = "on" if dev.is_on else "off"
        except KeyError:
            print("Key " + name + " not found")
    return states


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
    print(dev_map)
