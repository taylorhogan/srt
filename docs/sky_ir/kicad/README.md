# Sky IR sensor carrier board (KiCad)

A KiCad 8 project for the sky IR sensor: the ESP32-S3-DevKitC-1 plugs into two 1x22 sockets on a small carrier that holds the I2C pull-ups, the heater MOSFET and its gate parts, and screw-free pin headers for the three things that live off the board on leads: the MLX90614 (it must see the sky), the BME280 breakout and the 47 ohm window heater. Electrically it is the same circuit as `../sky_ir_wiring_print.html`; the firmware is `../sky-ir.yaml`.

## Files

| File | What |
|---|---|
| `sky_ir.kicad_pro` | project (open this) |
| `sky_ir.kicad_sch` | schematic, one sheet, nets by label |
| `sky_ir.kicad_pcb` | board, 70.9 x 72 mm, two layers, routed, GND pour on the back |
| `sky_ir.kicad_sym`, `sky_ir.pretty/` | the project's own symbols and footprints, wired in by `sym-lib-table` / `fp-lib-table` |
| `board.png`, `sch.png` | previews drawn from the files |

## Circuit

| Ref | Part | Nets |
|---|---|---|
| U1 | ESP32-S3-DevKitC-1 (N16R8) in two 1x22 female headers | 3V3 (pins 1-2), GPIO8 SDA, GPIO9 SCL, GPIO10 gate, 5V, GND |
| R1, R2 | 4.7 k | SDA and SCL pull-ups to 3V3 |
| R3 | 220 R | GPIO10 to the MOSFET gate |
| R4 | 10 k | gate to GND (holds the heater off at boot) |
| Q1 | IRLZ44N, TO-220, G D S | low-side switch for the heater |
| J2 | 1x04 header | MLX90614 leads: VDD GND SCL SDA |
| J4 | 1x04 header | BME280 breakout: VIN GND SCL SDA |
| J5 | 1x02 header | heater: +5V, drain |
| H1-H4 | 3.2 mm holes | M3 standoffs |

Unused DevKit pins are left open on purpose (no-connect flags in the schematic).

## Before ordering

1. **Socket row spacing.** The two DevKit columns are drawn 22.86 mm (0.9 in) apart. Measure pin 1 of each row on your board and check; if it differs, change `ROW` in the generator or move J3's column in the board editor. Everything this build uses is in the left column, so the right socket is mechanical only.
2. Open the board, press **B** to fill the GND pour, then run **DRC**. The file was checked here for clearance and connectivity with a script, not with KiCad.
3. The DevKit's USB-C connectors overhang the socket end marked on the silkscreen; leave that board edge clear in the box.
4. The IRLZ44N tab is the drain: do not bolt it to a grounded box wall without an insulator.

## How it was made

Written by a generator script without KiCad (none on the Spark); the S-expressions are plain KiCad 8 format and KiCad 9 opens them. Clearance (0.2 mm), net connectivity and label placement were checked with shapely and kiutils, and the previews were plotted from the files.
