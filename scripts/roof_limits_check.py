#!/usr/bin/env python3
"""Live readout for commissioning the roof limit switches (sheet RLS-1).

    python scripts/roof_limits_check.py            # poll once
    python scripts/roof_limits_check.py --watch    # 1 Hz live display

Run this during the commissioning drills: press each lever, pull a spade,
jumper NO to NC, press both levers -- and watch each induced failure read as
FAULT with the right reason. The switches earn a vote in roof-state
decisions only after every drill shows the expected verdict here.
"""
import os
import sys
import time

if __package__ is None or __package__ == "":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from hardware_control import roof_limit_switches as rls


def show(r):
    flags = ("".join("1" if v else "0" for v in r.raw)) if r.raw else "----"
    line = f"[{flags}] {r.state.upper():14s} {r.detail}"
    print(time.strftime("%H:%M:%S"), line, flush=True)


def main() -> int:
    watch = "--watch" in sys.argv
    print("inputs shown as [closed_no closed_nc open_no open_nc]")
    while True:
        show(rls.read())
        if not watch:
            return 0
        time.sleep(1)


if __name__ == "__main__":
    sys.exit(main())
