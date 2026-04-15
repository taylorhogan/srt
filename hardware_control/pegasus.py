import os
import sys
import time
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
    ser.write((cmd + "\r\n").encode())
    return ser.readline().decode(errors="replace").strip()


def _is_pegasus_response(response):
    """Return True if the response looks like it came from a Pegasus device."""
    upper = response.upper()
    return upper.startswith("UPB") or upper.startswith("PA:")


def discover_com_port():
    """Scan all serial ports and return the first one that responds as a Pegasus device.

    Returns the port name (e.g. 'COM3' or '/dev/ttyUSB0'), or None if not found.
    """
    for port_info in serial.tools.list_ports.comports():
        port = port_info.device
        try:
            with serial.Serial(port, _BAUD_RATE, timeout=_TIMEOUT) as ser:
                time.sleep(0.5)  # allow USB-serial adapter to settle
                response = _send_command(ser, "PA")
                if _is_pegasus_response(response):
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

    Tries the SE (sensor/environment) command first (PPB-family devices),
    then falls back to parsing the PA status line (UPBv2).

    Args:
        com_port: Serial port string. If None, uses config or auto-discovers.

    Returns:
        dict with keys 'temperature' (°C) and 'humidity' (%), or None on failure.
    """
    if com_port is None:
        com_port = _get_com_port()
    if com_port is None:
        return None

    try:
        with serial.Serial(com_port, _BAUD_RATE, timeout=_TIMEOUT) as ser:
            time.sleep(0.5)
            se_response = _send_command(ser, "SE")
            pa_response = _send_command(ser, "PA")
    except (serial.SerialException, OSError):
        return None

    # SE response: SE:[temp]:[humidity]:[dewpoint]  (PPB / PPBA)
    if se_response.upper().startswith("SE:"):
        fields = se_response.split(":")[1:]
        try:
            return {"temperature": float(fields[0]), "humidity": float(fields[1])}
        except (IndexError, ValueError):
            pass

    # PA response (colon-delimited PPB/PPBA):
    # PA:[U1]:[U2]:[U3]:[U4]:[Dew1]:[Dew2]:[AutoDew]:[Voltage]:[Current]:[Temp]:[Humidity]
    if pa_response.upper().startswith("PA:"):
        fields = pa_response.split(":")[1:]
        try:
            return {"temperature": float(fields[9]), "humidity": float(fields[10])}
        except (IndexError, ValueError):
            pass

    # PA response (comma-delimited UPBv2):
    # UPB2_fw:[u1,u2,...,temp,humidity,...]  temperature=index 11, humidity=12
    if ":" in pa_response:
        fields = pa_response.split(":", 1)[1].split(",")
        try:
            return {"temperature": float(fields[11]), "humidity": float(fields[12])}
        except (IndexError, ValueError):
            pass

    return None


if __name__ == "__main__":
    # Diagnostic: show every port and its raw response before attempting discovery
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found at all.")
    for port_info in ports:
        port = port_info.device
        print(f"\nProbing {port} ({port_info.description}) ...")
        try:
            with serial.Serial(port, _BAUD_RATE, timeout=_TIMEOUT) as ser:
                time.sleep(0.5)
                raw_pa = _send_command(ser, "PA")
                raw_se = _send_command(ser, "SE")
                print(f"  PA: {repr(raw_pa)}")
                print(f"  SE: {repr(raw_se)}")
        except (serial.SerialException, OSError) as e:
            print(f"  Error: {e}")

    print("\n--- Discovery ---")
    port = discover_com_port()
    if port:
        print(f"Pegasus found on {port}")
        result = get_temperature_humidity(port)
        if result:
            print(f"Temperature: {result['temperature']} °C")
            print(f"Humidity:    {result['humidity']} %")
        else:
            print("Failed to parse environment data.")
    else:
        print("No Pegasus device found.")
