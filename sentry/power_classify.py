import requests
import time
from datetime import datetime
import matplotlib.pyplot as plt
import csv

# Configuration
SHELLY_IP = "192.168.1.XXX"  # Replace with your Shelly EM Gen3 IP address
POLL_INTERVAL = 5  # Seconds between measurements
DURATION_MINUTES = 10  # Total duration for data collection

# Calculated values
num_readings = (DURATION_MINUTES * 60) // POLL_INTERVAL


def fetch_status(ip):
    url = f"http://{ip}/rpc"
    payload = {
        "id": 1,
        "method": "Shelly.GetStatus",
        "params": {}
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching status: {e}")
        return None


# Data storage
times = []
voltages = []
currents = []
powers = []

print(f"Starting data collection from Shelly EM Gen3 at {SHELLY_IP}")
print(f"Collecting {num_readings} readings every {POLL_INTERVAL} seconds...\n")

for i in range(num_readings):
    status = fetch_status(SHELLY_IP)
    if status:
        # Extract data from channel 0 (voltage is the same for both channels)
        ch0 = status.get("em1:0", {})
        ch1 = status.get("em1:1", {})

        voltage = ch0.get("voltage", None)
        current0 = ch0.get("current", 0.0) or 0.0
        current1 = ch1.get("current", 0.0) or 0.0
        power0 = ch0.get("act_power", 0.0) or 0.0  # Active power (can be negative for export)
        power1 = ch1.get("aprt_power", 0.0) or 0.0

        total_current = current0 + current1
        total_power = power0 + power1

        current_time = datetime.now()

        times.append(current_time)
        voltages.append(voltage)
        currents.append(total_current)
        powers.append(total_power)

        print(f"{current_time.strftime('%H:%M:%S')} | "
              f"Voltage: {voltage:.1f} V | "
              f"Current: {total_current:.3f} A | "
              f"Power: {total_power:.1f} W")
    else:
        print("Failed to retrieve data")

    if i < num_readings - 1:
        time.sleep(POLL_INTERVAL)

# Analysis and output
if len(voltages) > 0 and None not in voltages:
    avg_voltage = sum(voltages) / len(voltages)
    min_voltage = min(voltages)
    max_voltage = max(voltages)
else:
    avg_voltage = min_voltage = max_voltage = "N/A"

if len(currents) > 0:
    avg_current = sum(currents) / len(currents)
    max_current = max(currents)
    total_energy_wh = (sum(powers) / len(powers)) * (DURATION_MINUTES / 60) if powers else 0  # Rough estimate
else:
    avg_current = max_current = total_energy_wh = "N/A"

print("\n--- Analysis Summary ---")
print(f"Average Voltage: {avg_voltage:.1f} V (Min: {min_voltage:.1f} V, Max: {max_voltage:.1f} V)")
print(f"Average Current: {avg_current:.3f} A (Max: {max_current:.3f} A)")
print(f"Average Power: {sum(powers) / len(powers):.1f} W")
print(f"Estimated Energy Used (over {DURATION_MINUTES} min): {total_energy_wh:.2f} Wh")

# Save to CSV
with open("shelly_em_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Voltage (V)", "Total Current (A)", "Total Power (W)"])
    for t, v, c, p in zip(times, voltages, currents, powers):
        writer.writerow([t.strftime('%Y-%m-%d %H:%M:%S'), v, c, p])
print("\nData saved to 'shelly_em_data.csv'")

# Plot
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(times, voltages, label="Voltage (V)", color="blue")
plt.ylabel("Voltage (V)")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(times, currents, label="Total Current (A)", color="green")
plt.ylabel("Current (A)")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(times, powers, label="Total Power (W)", color="red")
plt.ylabel("Power (W)")
plt.xlabel("Time")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()