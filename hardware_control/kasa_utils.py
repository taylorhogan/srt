import asyncio
import json
import logging
import socket
import struct
import time

from kasa import Discover

# Failures are logged HERE rather than left to each caller. There are fourteen
# call sites and every one of them used to log success unconditionally on the
# next line, so a device that did nothing produced a log saying it had. Logging
# at the source means a silent no-op cannot happen again even at a call site
# nobody updated.
_logger = logging.getLogger(__name__)


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



def legacy_relay(host, state=None, timeout=2.0, retries=1):
    """Read or set a plug's relay over the legacy protocol on TCP 9999.

    *state* None reads and returns 0/1; 0 or 1 sets and returns the state read
    back afterwards. None is returned when the device cannot be reached.

    This exists because `Discover.discover_single()` re-negotiates the protocol
    on every call and these plugs answer BOTH the legacy port and KLAP on 80.
    Which one the library picks is not deterministic, and KLAP has never once
    authenticated here -- that is what threw AuthenticationError at 03:14 on
    2026-08-16 while the end sequence was trying to switch the mount off, and it
    is the same non-determinism that made the alias lookup a coin toss until
    legacy_alias() was added. Port 9999 answered 14 of 14 devices on every
    attempt measured. So control uses it directly rather than asking the library
    to choose.

    Retried, because these devices accept very few concurrent connections: the
    "port 9999 refuses" reading on 2026-08-14 that sent a night's debugging into
    a firmware theory was almost certainly contention from repeated probing, not
    a closed port.
    """
    for attempt in range(retries + 1):
        try:
            s = socket.create_connection((host, 9999), timeout=timeout)
        except OSError:
            time.sleep(0.5)
            continue
        try:
            if state is None:
                req = {"system": {"get_sysinfo": {}}}
            else:
                req = {"system": {"set_relay_state": {"state": int(state)}}}
            body = _xor(json.dumps(req).encode())
            s.sendall(struct.pack(">I", len(body)) + body)
            head = s.recv(4)
            if len(head) < 4:
                raise OSError("short header")
            n = struct.unpack(">I", head)[0]
            if n > (1 << 20):
                raise OSError("implausible length")
            buf = b""
            while len(buf) < n:
                chunk = s.recv(n - len(buf))
                if not chunk:
                    raise OSError("truncated")
                buf += chunk
            reply = json.loads(_xor(buf, decrypt=True))
        except (OSError, ValueError, struct.error):
            time.sleep(0.5)
            continue
        finally:
            s.close()

        if state is None:
            try:
                return int(reply["system"]["get_sysinfo"]["relay_state"])
            except (KeyError, TypeError, ValueError):
                return None
        # Setting: confirm it took, rather than trusting the acknowledgement.
        if reply.get("system", {}).get("set_relay_state", {}).get("err_code") == 0:
            return legacy_relay(host, None, timeout, retries=1)
        time.sleep(0.5)
    return None


async def kasa_do(cfg, instructions):
    """Switch named devices. Returns {name: True/False}, True only if VERIFIED.

    The old version caught KeyError, printed, and returned nothing, so a device
    missing from the map was a command that silently did nothing -- while the
    caller logged success on the next line regardless. Every "Mount powered
    on/off" line in iris.log was written whether or not anything happened, which
    is why the fault stayed invisible for so long and why those lines cannot be
    used to date a regression.

    So this returns a result per device, and "success" means the relay was READ
    BACK in the requested state, not that a command was accepted.
    """
    results = {}
    for key, want in instructions.items():
        on = str(want).lower() in ("on", "1", "true")
        ip = cfg.get(key)
        if ip is None:
            _logger.error("kasa_do: %r is not in the discovery map -- NOT switched "
                          "%s. Check 'allow third-party apps to control' on that "
                          "plug in the Kasa app.", key, want)
            results[key] = False
            continue
        state = await asyncio.to_thread(legacy_relay, ip, 1 if on else 0)
        if state is None:
            # Legacy route unavailable: fall back to the library, then verify.
            try:
                dev = await Discover.discover_single(ip)
                await (dev.turn_on() if on else dev.turn_off())
                await dev.update()
                state = 1 if dev.is_on else 0
                _logger.warning("kasa_do: %r fell back to the library transport", key)
            except Exception as exc:
                _logger.error("kasa_do: %r NOT switched %s -- %s: %s",
                              key, want, type(exc).__name__, exc)
                results[key] = False
                continue
        ok = (state == (1 if on else 0))
        results[key] = ok
        if ok:
            _logger.info("kasa_do: %r switched %s (verified)", key, want)
        else:
            _logger.error("kasa_do: %r commanded %s but reads back %s",
                          key, want, "on" if state else "off")
    return results


async def kasa_get_states(cfg, device_names):
    """{name: 'on'/'off'/None}. None means the device could not be read."""
    states = {}
    for name in device_names:
        ip = cfg.get(name)
        if ip is None:
            print("Key " + name + " not found")
            states[name] = None
            continue
        state = await asyncio.to_thread(legacy_relay, ip, None)
        if state is None:
            try:
                dev = await Discover.discover_single(ip)
                await dev.update()
                state = 1 if dev.is_on else 0
            except Exception as exc:
                print("kasa_get_states %s: %s" % (name, type(exc).__name__))
                states[name] = None
                continue
        states[name] = "on" if state else "off"
    return states


async def kasa_check(cfg, instructions):
    """True only if EVERY named device is in the requested state.

    The old version had `return True` inside the loop, so a multi-key check only
    ever evaluated the first key -- goforimagecheck passes four conditions and
    three of them were never tested. This evaluates all of them.

    A device that cannot be read returns False. That is the safe direction for
    the roof gate, but note it makes "not readable" and "wrong state"
    indistinguishable to the caller; _mount_power_blocked_reason distinguishes
    them itself by checking the map before calling.
    """
    for key, want in instructions.items():
        ip = cfg.get(key)
        if ip is None:
            print("Key " + key + " not found")
            return False
        state = await asyncio.to_thread(legacy_relay, ip, None)
        if state is None:
            try:
                dev = await Discover.discover_single(ip)
                await dev.update()
                state = 1 if dev.is_on else 0
            except Exception as exc:
                print("kasa_check %s: %s" % (key, type(exc).__name__))
                return False
        if str(want).lower() == "ison" and state != 1:
            return False
        if str(want).lower() == "isoff" and state != 0:
            return False
    return True


if __name__ == "__main__":
    dev_map = asyncio.run(make_discovery_map())
    print(dev_map)
