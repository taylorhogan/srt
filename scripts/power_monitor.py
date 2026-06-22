#!/usr/bin/env python3
"""Live power-draw monitor for the Shelly 3EM current monitor.

Polls one channel every second and prints a line ONLY when the real power
changes by more than --threshold watts — so you see the moments the load
transitions (mount slewing, roof motor, dehumidifier cycling) rather than a
wall of identical readings.

Usage:
    python scripts/power_monitor.py                  # channel 0, 1s, 0.5W threshold
    python scripts/power_monitor.py --channel 1
    python scripts/power_monitor.py --interval 0.5 --threshold 2

Ctrl-C to stop.
"""

import argparse
import os
import sys
import time
from datetime import datetime

# Project root
if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from hardware_control import utl_shelly


def main() -> None:
    ap = argparse.ArgumentParser(description="Live Shelly power-draw monitor (prints on change)")
    ap.add_argument("--channel", type=int, default=None, help="EM1 channel (default: config value)")
    ap.add_argument("--interval", type=float, default=1.0, help="poll interval in seconds (default 1)")
    ap.add_argument("--threshold", type=float, default=0.5, help="min watt change to report (default 0.5)")
    args = ap.parse_args()

    print(f"# Watching channel {args.channel if args.channel is not None else '(config)'} "
          f"every {args.interval}s — reporting changes > {args.threshold} W. Ctrl-C to stop.",
          flush=True)
    print(f"# {'time':19}  {'power':>9}  {'delta':>9}  {'current':>8}  {'volts':>6}", flush=True)

    last_power = None
    failed = False
    while True:
        loop_start = time.monotonic()
        s = utl_shelly.read_current_monitor(args.channel)

        if s is None:
            if not failed:  # report a dropout once, not every second
                print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}  read failed — retrying…", flush=True)
                failed = True
        else:
            failed = False
            power = s["act_power"]
            if last_power is None or abs(power - last_power) > args.threshold:
                delta = 0.0 if last_power is None else power - last_power
                print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}  "
                      f"{power:7.1f} W  {delta:+8.1f}  {s['current']:6.3f} A  {s['voltage']:5.1f}",
                      flush=True)
                last_power = power

        # keep a steady cadence even if the HTTP request took a while
        time.sleep(max(0.0, args.interval - (time.monotonic() - loop_start)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n# stopped")
