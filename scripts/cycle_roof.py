#!/usr/bin/env python3
"""Cycle the observatory roof once and bank its current signature.

A lightweight alternative to a full imaging run: it does NOT start NINA or pick
a target. It just runs the normal roof toggle sequence (power motor -> fire
relay -> wait for travel -> power off), which now captures the roof motor's
current trace into sentry/roof_signatures/ for anomaly comparison.

SAFETY: the roof must never move while the mount is unparked (collision risk).
This script refuses to run unless the VISUAL check (vision_safety, template
match on the indoor camera) confirms the scope is parked, unless you pass
--force. The visual check is used rather than pwi4_utils.get_is_parked()
because the latter needs the mount powered and connected; the camera does not.

    python scripts/cycle_roof.py                  # parked-checked toggle + capture
    python scripts/cycle_roof.py --direction open # label the captured signature
    python scripts/cycle_roof.py --force          # skip the parked check (DANGEROUS)
"""

import argparse
import asyncio
import os
import sys

# Project root
if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from hardware_control import kasa_utils as ku
from cmd_processing import super_user_commands as suc
from sentry import roof_current_signature as rcs


def main() -> None:
    ap = argparse.ArgumentParser(description="Cycle the roof once and capture its current signature")
    ap.add_argument("--direction", choices=["open", "close"], default=None,
                    help="label for the captured signature (does NOT set roof direction)")
    ap.add_argument("--force", action="store_true",
                    help="skip the scope-parked safety check (DANGEROUS)")
    args = ap.parse_args()

    # --- Absolute safety rule: roof moves only with the scope parked ---------
    # Use the visual check (camera template match) — it works whether or not the
    # mount is powered/connected, unlike pwi4_utils.get_is_parked().
    if args.force:
        print("WARNING: --force set — skipping the scope-parked check.")
    else:
        try:
            parked, closed, is_open, _ = suc.get_status_with_lights()
        except Exception as e:  # noqa: BLE001
            print(f"Could not get visual safety status ({e}). Refusing to move the roof.")
            sys.exit(1)
        if not parked:
            print("Vision Safety: scope is NOT parked — refusing to move the roof (safety rule). "
                  "Park the scope first, or pass --force if you are certain.")
            sys.exit(1)
        roof_state = "closed" if closed else ("open" if is_open else "unknown")
        print(f"Vision Safety: scope parked; roof currently {roof_state}.")

    dev_map = asyncio.run(ku.make_discovery_map())

    print(f"Toggling roof (motor on -> relay -> ~45s travel -> motor off), "
          f"capturing current signature [{args.direction or 'unknown'}]...")
    suc.toggle_roof(dev_map, capture_direction=args.direction)

    # --- Report the signature just banked ------------------------------------
    sig = _newest_signature(args.direction)
    if sig is None:
        print("No signature was captured (is current_monitor_url configured and reachable?).")
        return
    print("\nCaptured signature:")
    for k, v in (sig.get("features") or {}).items():
        print(f"  {k:22} {v}")
    res = rcs.compare(sig)
    print(f"\nAnomaly: {res['is_anomaly']}")
    for r in res.get("reasons", []):
        print(f"  - {r}")
    if res.get("note"):
        print(f"  ({res['note']})")
    print("\nIf this was a clean run, label it good:")
    print(f"  python sentry/roof_current_signature.py label <file> --good")


def _newest_signature(direction):
    """Load the most recently saved unlabeled signature, if any."""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "sentry", "roof_signatures", "unlabeled", direction or "unknown")
    base = os.path.abspath(base)
    if not os.path.isdir(base):
        return None
    files = [os.path.join(base, f) for f in os.listdir(base) if f.endswith(".json")]
    if not files:
        return None
    import json
    with open(max(files, key=os.path.getmtime)) as fh:
        return json.load(fh)


if __name__ == "__main__":
    main()
