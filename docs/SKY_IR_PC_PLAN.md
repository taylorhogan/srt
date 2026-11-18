# Sky IR cloud sensor: bring-up on the observatory PC

Written 2026-11-16. Status: **sensors wired, firmware not yet flashed**.
The board is an ESP32-S3-DevKitC-1 (N16R8); the MLX90614-BCC and the
BME280 are already on it. The heater and its MOSFET wait for a 2 W
resistor and are not part of this plan.

The design pages are `docs/sky_ir/sky_ir_sensor.html` (the full write-up:
parts, wiring, enclosure, firmware) and `docs/sky_ir/sky_ir_wiring_print.html`
(one printable sheet). The firmware is `docs/sky_ir/sky-ir.yaml`; its
secrets template is `docs/sky_ir/secrets.example.yaml`. Open the pages
from the PC's checkout after `git pull`. A KiCad 8 project for a carrier board (the DevKit in two sockets, the
pull-ups, the MOSFET and headers for the off-board sensors and heater) is in
`docs/sky_ir/kicad/`; its README lists what to confirm before ordering.

## Pins as wired (S3 numbering)

| Signal | Pin | Note |
|---|---|---|
| SDA, both sensors | GPIO8 | 4.7 kΩ to 3V3 |
| SCL, both sensors | GPIO9 | 4.7 kΩ to 3V3 |
| Sensor supply | 3V3 | the MLX is a 3 V part, never 5 V |
| Ground | GND | one ground for everything |
| Heater gate (later) | GPIO10 | through 220 Ω, 10 kΩ to GND |
| Heater supply (later) | 5V | USB 5 V through the board's diode |

GPIO 22 to 25 do not exist on the S3; 26 to 37 are the module's flash
and PSRAM, 19 and 20 are USB, 0, 3, 45 and 46 are strapping pins.

## Why the PC and not the browser installer

The browser installer at web.esphome.io only writes a generic ESPHome
firmware and gets it onto Wi-Fi. The sensor config has to be compiled by
the ESPHome tool, so the tool is installed once on the PC. After the first
flash every update goes over the air and the USB lead is only power.

## Steps

1. **Plug in.** One USB-C lead from the PC into the board's port marked
   **COM** (or UART). That port has the serial bridge and needs no button
   presses. The other port is the chip's native USB and only takes a
   first flash with BOOT held through a reset; leave it alone. The
   sensors stay wired; they draw a few milliamps from the PC port.
2. **COM port.** Device Manager should show a new COM port. If not,
   Windows wants the bridge driver, CH343 or CP2102 depending on the
   board; Windows Update usually finds it.
3. **Install ESPHome**, once, in its own environment:
   ```
   py -3 -m venv %USERPROFILE%\esphome
   %USERPROFILE%\esphome\Scripts\pip install esphome
   ```
4. **Files.** Make `%USERPROFILE%\esphome\sky-ir\`, copy `sky-ir.yaml`
   from the repo into it, copy `secrets.example.yaml` beside it as
   `secrets.yaml` and fill in the Wi-Fi name, the Wi-Fi password and an
   OTA password of your choosing. `secrets.yaml` stays on the PC.
5. **First flash**, from that folder:
   ```
   %USERPROFILE%\esphome\Scripts\esphome run sky-ir.yaml
   ```
   Pick the COM port when asked. The first run downloads the ESP32-S3
   toolchain, several minutes and a couple of gigabytes; later runs are
   quick. It flashes, then tails the log on the same port.
6. **Read the log.** Within a minute it should print the I2C scan with
   `0x5A` (MLX) and `0x76` (BME280), then `sky_temp`, `ir_case_temp`,
   `air_temp`, `humidity`, `pressure`, `sky_delta` and `dew_point`
   every 10 to 30 s. Indoors the sky reading is the ceiling, a few
   degrees under the air. Missing `0x5A` or `0x76` is the wiring:
   swapped SDA/SCL, a pull-up not on 3V3, or a cold joint on the can.
7. **Wi-Fi and address.** The log shows the IP it got. Reserve it in
   DHCP, as for Iris North, so the Python side has a fixed host. Then
   `http://sky-ir.local/` (or the IP) shows the readings, and
   `http://<ip>/sensor/sky_temp` answers JSON.
8. **Done with the PC.** Unplug, move the board to the enclosure on the
   5 V supply at the Kasa outlet. Later config changes are
   `esphome run sky-ir.yaml` again; it finds the device over the air.

## Later, when the 2 W resistor arrives

Fit the IRLZ44N, the 220 Ω, the 10 kΩ and the heater per the wiring
sheet. Nothing in the firmware changes: the dew rule already drives
GPIO10, which until then switches nothing.

## Reading it from SRT

Out of scope here. The sensor publishes over local HTTP like the Shelly
relays; a `sky_delta` below about -20 °C is clear sky, above about -5 °C
is overcast. Where it joins the weather gate (`iris_astronomy`,
`cfg["weather"]`) is its own plan once a week of readings exists.
