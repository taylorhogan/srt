import os
import sys
import serial
import serial.tools.list_ports

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from configs import config

_BAUD_RATE = 9600
_TIMEOUT = 2


def _send_command(ser, cmd):
    """Send a command and return the stripped response line."""
    ser.reset_input_buffer()
    ser.write((cmd + "\n").encode())
    return ser.readline().decode(errors="replace").strip()


def discover_com_port():
    """Scan all serial ports and return the first one that responds as a Pegasus device.

    Returns the port name (e.g. 'COM3' or '/dev/ttyUSB0'), or None if not found.
    """
    for port_info in serial.tools.list_ports.comports():
        port = port_info.device
        try:
            with serial.Serial(port, _BAUD_RATE, timeout=_TIMEOUT) as ser:
                response = _send_command(ser, "PA")
                if response.upper().startswith("UPB"):
                    return port
        except (serial.SerialException, OSError):
            continue
    return None


def _get_com_port():
    """Return the configured COM port, discovering it if the config entry is blank."""
    cfg = config.data()
    port = cfg.get("pegasus", {}).get("com_port", "")
    if not port:
        port = discover_com_port()
    return port


def get_temperature_humidity(com_port=None):
    """Query the Pegasus power box for temperature and humidity.

    Args:
        com_port: Serial port string. If None, uses config or auto-discovers.

    Returns:
        dict with keys 'temperature' (°C) and 'humidity' (%), or None on failure.

    The Pegasus UPBv2 'PA' response format (colon-delimited fields after 'UPB2_fw:'):
        u1,u2,u3,u4,adj_out,dew1,dew2,focus,autodew,voltage,current,
        temperature,humidity,dewpoint,auto_adj,pwm
    Index (0-based after splitting on ','): temperature=11, humidity=12
    """
    if com_port is None:
        com_port = _get_com_port()
    if com_port is None:
        return None

    try:
        with serial.Serial(com_port, _BAUD_RATE, timeout=_TIMEOUT) as ser:
            response = _send_command(ser, "PA")
    except (serial.SerialException, OSError):
        return None

    # Response looks like: UPB2_3.15:1,1,1,1,1,0,0,0,0,12.1,0.4,22.5,55.0,13.2,0,100
    if ":" not in response:
        return None

    fields_str = response.split(":", 1)[1]
    fields = fields_str.split(",")

    try:
        temperature = float(fields[11])
        humidity = float(fields[12])
    except (IndexError, ValueError):
        return None

    return {"temperature": temperature, "humidity": humidity}


if __name__ == "__main__":
    port = discover_com_port()
    if port:
        print(f"Pegasus found on {port}")
        result = get_temperature_humidity(port)
        if result:
            print(f"Temperature: {result['temperature']} °C")
            print(f"Humidity:    {result['humidity']} %")
        else:
            print("Failed to read environment data.")
    else:
        print("No Pegasus device found.")
