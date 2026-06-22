import logging
import os, sys

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import requests

from configs import config

logger = logging.getLogger(__name__)


def fire_roof_relay(timeout=10):
    """Fire the Shelly relay that toggles the roof motor.

    The roof relay is momentary: GETting the configured URL pulses the relay
    and the roof moves in whichever direction its current position dictates.
    There is no separate open/close — callers must enforce the hardware safety
    rule (scope parked) before calling this.
    """
    url = config.data()["hardware"]["roof_relay_url"]
    return _get(url, timeout=timeout)


def set_dehumidifier(on, timeout=10):
    """Turn the dehumidifier Shelly relay on or off.

    Args:
        on: True to energise the relay (turn=on), False for turn=off.
    """
    base = config.data()["hardware"]["dehumidifier_relay_url"]
    url = f"{base}?turn={'on' if on else 'off'}"
    return _get(url, timeout=timeout)


def read_current_monitor(channel=None, timeout=10):
    """Read one channel of the Shelly 3EM Gen3 current monitor.

    Unlike the roof/dehumidifier relays (Gen1 REST), this device uses the
    Gen2/3 RPC API: GET {base}/rpc/EM1.GetStatus?id=<channel>. Returns the
    parsed status dict (voltage, current, act_power, pf, freq, ...) or None
    on error.
    """
    cfg = config.data()["hardware"]
    base = cfg["current_monitor_url"]
    if channel is None:
        channel = cfg.get("current_monitor_channel", 0)
    url = f"{base}/rpc/EM1.GetStatus?id={channel}"
    response = _get(url, timeout=timeout)
    if response is None:
        return None
    try:
        return response.json()
    except ValueError as e:
        logger.warning("Bad JSON from current monitor (%s): %s", url, e)
        return None


def _get(url, timeout=10):
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        logger.info("Shelly control successful: %s", url)
        return response
    except requests.exceptions.RequestException as e:
        logger.warning("Error controlling Shelly (%s): %s", url, e)
        return None
