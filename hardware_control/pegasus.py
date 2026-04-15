import os
import sys
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
        # Response: {"data": [{"name": "UPBv3", "id": "<key>", ...}]}
        devices = data.get("data") or data.get("devices") or []
        if isinstance(devices, list) and devices:
            device = devices[0]
            name = device.get("name", "")
            key = device.get("id") or device.get("driverUniqueKey") or device.get("uuid") or ""
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
        # Unity uses PUT for report endpoints
        r = requests.put(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            return {
                "temperature": float(data["temperature"]),
                "humidity": float(data["humidity"]),
            }
    except (requests.RequestException, KeyError, ValueError, TypeError):
        pass

    return None


if __name__ == "__main__":
    unity_url = _get_unity_url()
    print(f"Querying Pegasus Unity at {unity_url}\n")

    print("--- Device discovery ---")
    name, key = _get_driver_info()
    print(f"  Driver name: {name}")
    print(f"  Driver key:  {key}")

    if name and key:
        # Probe the environment endpoint and print raw response for debugging
        url = f"{unity_url}/Driver/{name}/Report/Environment?DriverUniqueKey={key}"
        print(f"\n--- Raw environment response ---")
        print(f"  PUT {url}")
        try:
            r = requests.put(url, timeout=5)
            print(f"  Status: {r.status_code}")
            print(f"  Body:   {r.text[:500]}")
        except requests.RequestException as e:
            print(f"  Error: {e}")

    print("\n--- Temperature / Humidity ---")
    result = get_temperature_humidity()
    if result:
        print(f"Temperature: {result['temperature']} °C")
        print(f"Humidity:    {result['humidity']} %")
    else:
        print("Failed to read environment data.")
