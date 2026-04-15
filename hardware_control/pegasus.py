import os
import sys
import requests

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from configs import config

# Pegasus Unity runs a local HTTP server on this port
_UNITY_URL = "http://localhost:32000"
# Driver names to probe during discovery (Powerbox v3, UPBv2, PPBA fallbacks)
_DRIVER_NAMES = ["PBV3", "PBAV3", "PowerboxV3", "UPBv2", "PPBA"]


def discover_driver_key(unity_url=None):
    """Query the Unity device manager to find the active Pegasus driver key.

    Returns (driver_name, driver_key) or (None, None) if not found.
    """
    if unity_url is None:
        unity_url = config.data().get("pegasus", {}).get("unity_url", _UNITY_URL)

    # Try the device manager endpoint to list connected drivers
    try:
        r = requests.get(f"{unity_url}/Server/DeviceManager/Connected", timeout=5)
        print(f"DeviceManager response ({r.status_code}): {r.text[:500]}")
    except requests.RequestException as e:
        print(f"Could not reach Unity at {unity_url}: {e}")
        return None, None

    # Try environment endpoint for each known driver name
    for driver in _DRIVER_NAMES:
        try:
            url = f"{unity_url}/Driver/{driver}/Report/Environment"
            r = requests.get(url, timeout=5)
            print(f"  {driver}: {r.status_code} {r.text[:200]}")
            if r.status_code == 200:
                return driver, None  # no key needed
        except requests.RequestException:
            pass

    return None, None


def get_temperature_humidity():
    """Query the Pegasus Powerbox via the Unity local HTTP API.

    Returns:
        dict with keys 'temperature' (°C) and 'humidity' (%), or None on failure.
    """
    cfg = config.data().get("pegasus", {})
    unity_url = cfg.get("unity_url", _UNITY_URL)
    driver_key = cfg.get("driver_key", "")

    for driver in _DRIVER_NAMES:
        url = f"{unity_url}/Driver/{driver}/Report/Environment"
        if driver_key:
            url += f"?DriverUniqueKey={driver_key}"
        try:
            r = requests.get(url, timeout=5)
            if r.status_code != 200:
                continue
            data = r.json()
            return {
                "temperature": float(data["temperature"]),
                "humidity": float(data["humidity"]),
            }
        except (requests.RequestException, KeyError, ValueError, TypeError):
            continue

    return None


if __name__ == "__main__":
    cfg = config.data().get("pegasus", {})
    unity_url = cfg.get("unity_url", _UNITY_URL)

    print(f"Querying Pegasus Unity at {unity_url}\n")
    print("--- Driver discovery ---")
    discover_driver_key(unity_url)

    print("\n--- Temperature / Humidity ---")
    result = get_temperature_humidity()
    if result:
        print(f"Temperature: {result['temperature']} °C")
        print(f"Humidity:    {result['humidity']} %")
    else:
        print("Failed to read environment data.")
        print("\nHint: run discover_driver_key() output above to find the right driver name and key.")
