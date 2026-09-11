import base64
import os
import sys
from urllib.parse import quote

import requests

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from configs import config

_UNITY_URL = "http://localhost:32000"


def _get_unity_url():
    return config.data().get("pegasus", {}).get("unity_url", _UNITY_URL)


def _get_driver_info():
    """Query the Unity DeviceManager and return (driver_name, driver_key) for the first connected device."""
    unity_url = _get_unity_url()
    try:
        r = requests.get(f"{unity_url}/Server/DeviceManager/Connected", timeout=5)
        r.raise_for_status()
        data = r.json()
        devices = data.get("data") or data.get("devices") or []
        if isinstance(devices, list) and devices:
            device = devices[0]
            name = device.get("name", "")
            # Try all plausible key field names
            key = (device.get("uniqueId") or device.get("uniqueID")
                   or device.get("uniqueKey") or device.get("uuid")
                   or device.get("DriverUniqueKey") or device.get("driverUniqueKey")
                   or device.get("uid") or device.get("id") or "")
            return name, key
    except (requests.RequestException, ValueError, KeyError):
        pass
    return None, None


def get_temperature_humidity():
    """Query the Pegasus UPBv3 environment sensor via the Unity local HTTP API.

    Returns:
        dict with keys 'temperature' (°C) and 'humidity' (%), or None on failure.
    """
    cfg = config.data().get("pegasus", {})
    unity_url = cfg.get("unity_url", _UNITY_URL)
    driver_key = cfg.get("driver_key", "")

    # Auto-discover driver name and key if not configured
    driver_name, discovered_key = _get_driver_info()
    if not driver_name:
        return None
    if not driver_key:
        driver_key = discovered_key

    url = f"{unity_url}/Driver/{driver_name}/Report/Environment"
    if driver_key:
        url += f"?DriverUniqueKey={driver_key}"

    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            msg = r.json()["data"]["message"]
            temp_c = float(msg["temperature"])
            return {
                "temperature_f": round(temp_c * 9 / 5 + 32, 1),
                "temperature_c": temp_c,
                "humidity": float(msg["humidity"]),
            }
    except (requests.RequestException, KeyError, ValueError, TypeError):
        pass

    return None


def _resolve_driver():
    """Return (unity_url, driver_name, driver_key) for the connected Pegasus
    device, or (None, None, None) if it can't be reached."""
    cfg = config.data().get("pegasus", {})
    unity_url = cfg.get("unity_url", _UNITY_URL)
    driver_key = cfg.get("driver_key", "")
    driver_name, discovered_key = _get_driver_info()
    if not driver_name:
        return None, None, None
    if not driver_key:
        driver_key = discovered_key
    return unity_url, driver_name, driver_key


def _send_raw_command(cmd):
    """Send a raw UPBv3 serial command through the Unity passthrough.

    The Unity local API exposes device control only as a base64-encoded raw
    serial command sent with the HTTP OPTIONS verb to
    ``/Driver/{name}/Command/{base64(cmd)}`` (GET/POST/PUT return 405). Returns
    the device's echoed response string, or None on failure.
    """
    unity_url, driver_name, driver_key = _resolve_driver()
    if not driver_name:
        return None
    b64 = base64.b64encode(cmd.encode("ascii")).decode("ascii")
    url = f"{unity_url}/Driver/{driver_name}/Command/{quote(b64, safe='')}"
    if driver_key:
        url += f"?DriverUniqueKey={driver_key}"
    try:
        r = requests.options(url, timeout=5)
        if r.status_code == 200:
            return r.json()["data"]["message"]["response"]
    except (requests.RequestException, KeyError, ValueError, TypeError):
        pass
    return None


def set_power_port(port, level):
    """Set a Pegasus power output port to a PWM level (0 = off, 1-100 = on).

    Ports: 1=camera, 2=gemini, 3=fan, 4-6=spare. Returns True if the device
    acknowledged the command.
    """
    level = max(0, min(100, int(level)))
    return _send_raw_command(f"P{int(port)}:{level}") is not None


# Power ports carrying the imaging train: 1=camera, 2=gemini, 3=fan (the
# names the box itself reports). These are "the 3 Pegasus switches" a stop!
# must leave off once the roof is closed. They are powered off LAST in any
# shutdown, after every roof/mount/vision step, because nothing after them
# needs the train and (until the webcam retired on 2026-09-07) the safety
# camera drew its power through this box; the Kasa camera that replaced it
# does not -- it read the roof fine on 2026-09-11 with all three ports off.
IMAGING_TRAIN_PORTS = (1, 2, 3)


def read_power_port_details():
    """{port_number: (name, level)} for every power port, from the aggregate report.

    ``level`` is the LIVE setting (0 = off, 1-100 = on); ``bootStrap`` in the
    same report is the power-on default and is ignored here. ``name`` is the
    label the box carries (camera / gemini / fan / Output4..6). None if Unity
    or the box cannot be reached.
    """
    unity_url, driver_name, driver_key = _resolve_driver()
    if not driver_name:
        return None
    url = f"{unity_url}/Driver/{driver_name}/Report"
    if driver_key:
        url += f"?DriverUniqueKey={driver_key}"
    try:
        r = requests.get(url, timeout=5)
        hub = r.json()["data"]["message"]["pwmHubStatus"]["hub"]
        return {int(p["portNumber"]): (str(p.get("name", "")), int(p["level"])) for p in hub}
    except (requests.RequestException, KeyError, ValueError, TypeError):
        return None


def read_power_ports():
    """{port_number: level} for every power port; None if unreachable."""
    details = read_power_port_details()
    if details is None:
        return None
    return {port: level for port, (_name, level) in details.items()}


def power_off_imaging_train():
    """Power off the imaging-train ports (camera/gemini/fan) and VERIFY.

    Every port is attempted regardless of individual failures, then the box
    is read back. Returns (ok, levels): ok is True only when the read-back
    shows every train port at level 0; levels is the read-back
    {port: level} (None if the box could not be read, in which case ok is
    False -- an acknowledged command is not evidence the port is off).
    """
    for port in IMAGING_TRAIN_PORTS:
        set_power_port(port, 0)
    levels = read_power_ports()
    if levels is None:
        return False, None
    ok = all(levels.get(p, 1) == 0 for p in IMAGING_TRAIN_PORTS)
    return ok, {p: levels.get(p) for p in IMAGING_TRAIN_PORTS}


if __name__ == "__main__":
    unity_url = _get_unity_url()
    print(f"Querying Pegasus Unity at {unity_url}\n")

    print("--- Device discovery (raw) ---")
    try:
        r = requests.get(f"{unity_url}/Server/DeviceManager/Connected", timeout=5)
        print(f"  Status: {r.status_code}")
        print(f"  Body:   {r.text}")  # full body
        # Print each field of the first device
        data = r.json()
        devices = data.get("data") or data.get("devices") or []
        if isinstance(devices, list) and devices:
            print("\n  Device fields:")
            for k, v in devices[0].items():
                print(f"    {k!r}: {v!r}")
    except requests.RequestException as e:
        print(f"  Error: {e}")

    name, key = _get_driver_info()
    print(f"\n  Parsed driver name: {name}")
    print(f"  Parsed driver key:  {key!r}")

    if name:
        suffix = f"?DriverUniqueKey={key}" if key else ""
        url = f"{unity_url}/Driver/{name}/Report/Environment{suffix}"
        print(f"\n--- Raw environment response ---")
        print(f"  GET {url}")
        try:
            r = requests.get(url, timeout=5)
            print(f"  Status: {r.status_code}  Body: {r.text[:300]}")
        except requests.RequestException as e:
            print(f"  Error: {e}")

    print("\n--- Temperature / Humidity ---")
    result = get_temperature_humidity()
    if result:
        print(f"Temperature: {result['temperature_f']} °F ({result['temperature_c']} °C)")
        print(f"Humidity:    {result['humidity']} %")
    else:
        print("Failed to read environment data.")
