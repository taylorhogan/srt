"""Local control of a MRCOOL DIY mini-split via its WiFi USB dongle.

MRCOOL DIY units are Midea OEM, so the dongle speaks the Midea LAN protocol
(TCP port 6444) and we drive it with ``msmart-ng`` — entirely on the LAN, no
cloud. This keeps the heat pump in the same no-cloud posture as the Kasa and
Shelly gear, which matters for unattended operation.

Design notes (match the rest of hardware_control/):
- ``msmart`` is imported lazily inside each function, so merely having this file
  in the repo never breaks anything if the dependency isn't installed yet.
- Every public call is best-effort: it logs and returns None/False on failure
  rather than raising into a caller. This is comfort control, not a safety mover
  — it must never block the imaging state machine.
- Stateless per call (connect → authenticate → act). Setpoints change rarely, so
  the per-call handshake overhead is irrelevant and we avoid holding a socket
  across the scheduler's long idle periods.

One-time setup:
    pip install msmart-ng
    # Find the unit + its token/key (V3 devices need them; a one-off cloud
    # lookup via your Midea/SmartHVAC account credentials fetches them):
    python hardware_control/mrcool_utils.py discover
    # Put ip / id / token / key into config_private.py under cfg["hardware"]:
    #   "mrcool_ip": "192.168.87.xx",
    #   "mrcool_id": "0000000000000",
    #   "mrcool_token": "....",
    #   "mrcool_key": "....",
    # then:
    python hardware_control/mrcool_utils.py status
    python hardware_control/mrcool_utils.py set --mode cool --temp 21 --fan auto --power on
"""

import asyncio
import logging
import os
import sys

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from configs import config

logger = logging.getLogger(__name__)

DEFAULT_PORT = 6444


# --------------------------------------------------------------------------- #
# Device construction / auth (async, internal)
# --------------------------------------------------------------------------- #
def _conf():
    """MRCOOL settings from cfg["hardware"], or None if not configured."""
    hw = config.data().get("hardware", {})
    ip = hw.get("mrcool_ip")
    if not ip:
        return None
    return {
        "ip": ip,
        "device_id": int(hw.get("mrcool_id", 0)),
        "port": int(hw.get("mrcool_port", DEFAULT_PORT)),
        "token": hw.get("mrcool_token"),
        "key": hw.get("mrcool_key"),
    }


async def _connect(c):
    """Build an authenticated AirConditioner device, or None on failure."""
    from msmart.device import AirConditioner as AC  # lazy: optional dependency

    dev = AC(ip=c["ip"], port=c["port"], device_id=c["device_id"])
    if c.get("token") and c.get("key"):
        await dev.authenticate(c["token"], c["key"])  # V3 handshake
    return dev


def _run(coro):
    """Run an async coroutine to completion, swallowing errors (best-effort)."""
    c = _conf()
    if c is None:
        logger.warning("MRCOOL not configured (cfg['hardware']['mrcool_ip'] missing)")
        return None
    try:
        return asyncio.run(coro(c))
    except Exception as e:  # noqa: BLE001 — comfort control must never raise upstream
        logger.warning("MRCOOL control failed: %s", e)
        return None


# --------------------------------------------------------------------------- #
# Public API (sync wrappers)
# --------------------------------------------------------------------------- #
def get_state():
    """Return a dict of the unit's current state, or None on failure.

    Includes the indoor temperature sensor — useful to log alongside weather
    for dew-point / thermal-stability tracking.
    """
    async def _do(c):
        dev = await _connect(c)
        await dev.refresh()
        return {
            "power": dev.power_state,
            "mode": getattr(dev.operational_mode, "name", str(dev.operational_mode)),
            "target_c": dev.target_temperature,
            "indoor_c": dev.indoor_temperature,
            "outdoor_c": dev.outdoor_temperature,
            "fan": getattr(dev.fan_speed, "name", str(dev.fan_speed)),
        }

    return _run(_do)


def set_power(on: bool):
    async def _do(c):
        dev = await _connect(c)
        await dev.refresh()
        dev.power_state = bool(on)
        await dev.apply()
        logger.info("MRCOOL power -> %s", "on" if on else "off")
        return True

    return _run(_do) or False


def set_temp(celsius: float):
    async def _do(c):
        dev = await _connect(c)
        await dev.refresh()
        dev.target_temperature = float(celsius)
        await dev.apply()
        logger.info("MRCOOL target temp -> %.1f C", celsius)
        return True

    return _run(_do) or False


def set_mode(mode: str):
    """mode: one of auto, cool, heat, dry, fan_only (case-insensitive)."""
    async def _do(c):
        from msmart.device import AirConditioner as AC
        modes = {
            "auto": AC.OperationalMode.AUTO,
            "cool": AC.OperationalMode.COOL,
            "heat": AC.OperationalMode.HEAT,
            "dry": AC.OperationalMode.DRY,
            "fan_only": AC.OperationalMode.FAN_ONLY,
            "fan": AC.OperationalMode.FAN_ONLY,
        }
        key = mode.lower()
        if key not in modes:
            logger.warning("MRCOOL unknown mode %r (valid: %s)", mode, ", ".join(modes))
            return False
        dev = await _connect(c)
        await dev.refresh()
        dev.operational_mode = modes[key]
        await dev.apply()
        logger.info("MRCOOL mode -> %s", key)
        return True

    return _run(_do) or False


def set_fan(speed: str):
    """speed: one of auto, low, medium, high (case-insensitive)."""
    async def _do(c):
        from msmart.device import AirConditioner as AC
        speeds = {
            "auto": AC.FanSpeed.AUTO,
            "low": AC.FanSpeed.LOW,
            "medium": AC.FanSpeed.MEDIUM,
            "high": AC.FanSpeed.HIGH,
        }
        key = speed.lower()
        if key not in speeds:
            logger.warning("MRCOOL unknown fan %r (valid: %s)", speed, ", ".join(speeds))
            return False
        dev = await _connect(c)
        await dev.refresh()
        dev.fan_speed = speeds[key]
        await dev.apply()
        logger.info("MRCOOL fan -> %s", key)
        return True

    return _run(_do) or False


# --------------------------------------------------------------------------- #
# CLI — validate the unit before wiring it into anything
# --------------------------------------------------------------------------- #
def _discover(ip=None):
    """Discover Midea devices and print id/ip (+ token/key if available).

    With ``ip``, probe that single host directly (``discover_single``) — this
    bypasses the UDP broadcast, which doesn't cross subnets/VLANs. Without it,
    broadcast-scan the local network.
    """
    from msmart.discover import Discover

    def _show(d):
        print(f"  ip={d.ip}  id={d.device_id}  type={getattr(d, 'type', '?')}")
        tok, key = getattr(d, "token", None), getattr(d, "key", None)
        if tok and key:
            print(f"    token={tok}\n    key={key}")
        else:
            print("    (no token/key — V3 device needs a one-off cloud lookup;"
                  " see msmart-ng 'midea-discover --account ...')")

    async def _do():
        if ip:
            dev = await Discover.discover_single(ip)
            if dev is None:
                print(f"No Midea/MRCOOL device answered at {ip}.")
            else:
                _show(dev)
        else:
            devices = await Discover.discover()
            if not devices:
                print("No Midea/MRCOOL devices found on the LAN.")
                return
            for d in devices:
                _show(d)

    asyncio.run(_do())


def main():
    import argparse

    ap = argparse.ArgumentParser(description="MRCOOL mini-split local control")
    sub = ap.add_subparsers(dest="cmd", required=True)
    dsc = sub.add_parser("discover", help="find Midea devices on the LAN")
    dsc.add_argument("--ip", help="probe this IP directly instead of broadcasting")
    sub.add_parser("status", help="print current unit state")
    s = sub.add_parser("set", help="change unit settings")
    s.add_argument("--power", choices=["on", "off"])
    s.add_argument("--mode", help="auto|cool|heat|dry|fan_only")
    s.add_argument("--temp", type=float, help="target temperature, Celsius")
    s.add_argument("--fan", help="auto|low|medium|high")

    args = ap.parse_args()
    if args.cmd == "discover":
        _discover(args.ip)
    elif args.cmd == "status":
        st = get_state()
        print(st if st is not None else "(no state — unconfigured or unreachable)")
    elif args.cmd == "set":
        if args.power is not None:
            set_power(args.power == "on")
        if args.mode is not None:
            set_mode(args.mode)
        if args.temp is not None:
            set_temp(args.temp)
        if args.fan is not None:
            set_fan(args.fan)
        print(get_state())


if __name__ == "__main__":
    main()
