import requests
import json

def control_shelly(ip_address, command, parameter=None):
    """Controls a Shelly device.

    Args:
        ip_address: The IP address of the Shelly device.
        command: The command to send (e.g., "on", "off", "toggle").
        parameter: Optional parameter for the command (e.g., brightness level).
    """
    url = f"http://{ip_address}/relay/"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "method": "Switch.Set",
        "params": {
            "id": 0,
            "on": command == "on"
        }
    }

    if parameter is not None:
      payload["params"]["brightness"] = parameter

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        print("Shelly control successful.")
    except requests.exceptions.RequestException as e:
        print(f"Error controlling Shelly: {e}")
