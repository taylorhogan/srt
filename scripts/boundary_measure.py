"""
boundary_measure.py
Log the safe/unsafe collision-boundary measurement session: AprilTag corners
(inside Kasa cam) + PWI4 mount alt/az, sampled every few seconds while the
user drives the mount in ~2 degree steps and calls each pose
clears / marginal / would-hit.

Purpose: replace the placeholder SAFE_DEGREES = 3.0 in scope_marker_check.py
with a measured boundary, and replace the single-estimate 23 px/degree with a
fitted scale. Known anchors going in: hand-park 1.75 deg = safe, OTA on the
roof rail 24.6 deg = unsafe; everything between is unmeasured.

Usage (two terminals, or loop in background + notes from another shell):

    python scripts/boundary_measure.py                  # sampling loop, Ctrl+C to stop
    python scripts/boundary_measure.py --note "pose 3: marginal, ~4 inches"
    python scripts/boundary_measure.py --note "would-hit" --pose 5

Everything appends to local/boundary_session.jsonl (one JSON object per line,
kind = "sample" | "note"). Notes carry a timestamp, so they join to the
samples by time — say the verdict while the scope is HOLDING at the pose, not
while it is slewing (samples record is_slewing, so a mistimed note is
recoverable but noisy).

Records per sample:
    when, alt, az, is_slewing, is_tracking      from PWI4 (localhost:8220)
    markers {id: 4x2 corners}, centres          from the Kasa frame
    disp_px                                     worst-corner displacement vs the
                                                parked reference, if one exists
Camera or mount read failures log as nulls and the loop keeps going — a gap
in one stream must not lose the other.

After the session, the analysis (px/degree fit, marginal displacement,
tolerance = marginal x 0.7) is done from this file; nothing here decides
anything.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import numpy as np

from scripts.scope_marker_check import (PARKED_PATH, PX_PER_DEGREE,
                                        find_markers, grab)

LOG_PATH = "local/boundary_session.jsonl"
SNAP_PATH = "local/boundary_frame.jpg"


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _append(rec):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def _mount_status():
    """(alt, az, is_slewing, is_tracking) or Nones if PWI4 is unreachable."""
    try:
        from hardware_control.pwi4_client import PWI4
        s = PWI4().status()
        if not s.mount.is_connected:
            return None, None, None, None
        return (s.mount.altitude_degs, s.mount.azimuth_degs,
                bool(s.mount.is_slewing), bool(s.mount.is_tracking))
    except Exception:
        return None, None, None, None


def _parked_reference():
    if not os.path.exists(PARKED_PATH):
        return None
    stored = json.load(open(PARKED_PATH))
    return {int(k): np.array(v) for k, v in stored["markers"].items()}


def sample(parked):
    alt, az, slewing, tracking = _mount_status()
    img = grab(SNAP_PATH)
    markers = find_markers(img) if img is not None else {}
    rec = {"kind": "sample", "when": _now(),
           "alt": alt, "az": az,
           "is_slewing": slewing, "is_tracking": tracking,
           "camera_ok": img is not None,
           "marker_ids": sorted(markers),
           "markers": {str(i): np.asarray(c).tolist() for i, c in markers.items()},
           "centres": {str(i): [float(np.asarray(c)[:, 0].mean()),
                                float(np.asarray(c)[:, 1].mean())]
                       for i, c in markers.items()}}
    disp = None
    if parked and markers:
        worst = 0.0
        for i, ref in parked.items():
            if i in markers:
                d = np.linalg.norm(np.array(markers[i]) - ref, axis=1)
                worst = max(worst, float(d.max()))
        disp = worst if any(i in markers for i in parked) else None
    rec["disp_px"] = disp
    _append(rec)
    return rec


def _fmt(rec):
    alt = "%.2f" % rec["alt"] if rec["alt"] is not None else "----"
    az = "%.2f" % rec["az"] if rec["az"] is not None else "----"
    if not rec["camera_ok"]:
        tag = "NO FRAME"
    elif not rec["marker_ids"]:
        tag = "tag NOT DETECTED"
    elif rec["disp_px"] is not None:
        tag = "disp %.0f px (~%.1f deg @ old scale)" % (
            rec["disp_px"], rec["disp_px"] / PX_PER_DEGREE)
    else:
        tag = "ids %s (no parked ref)" % rec["marker_ids"]
    move = " SLEWING" if rec["is_slewing"] else ""
    return "%s  alt %s  az %s  %s%s" % (rec["when"][11:19], alt, az, tag, move)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--note", default=None,
                    help="append a timestamped note (the user's verdict for the "
                         "current pose: clears / marginal / would-hit, plus "
                         "inches if measured) instead of running the loop")
    ap.add_argument("--pose", default=None,
                    help="optional pose label to attach to --note")
    ap.add_argument("--interval", type=float, default=5.0,
                    help="seconds between samples (default 5)")
    args = ap.parse_args()

    if args.note is not None:
        rec = {"kind": "note", "when": _now(), "note": args.note}
        if args.pose is not None:
            rec["pose"] = args.pose
        # Snapshot the mount pose into the note itself so a verdict never
        # depends on time-joining alone.
        alt, az, slewing, _ = _mount_status()
        rec.update({"alt": alt, "az": az, "is_slewing": slewing})
        _append(rec)
        print("noted: %s  (alt %s az %s)" % (args.note, alt, az))
        return 0

    parked = _parked_reference()
    if parked is None:
        print("WARNING: no parked reference at %s -- corners still logged, "
              "but disp_px will be null. Run scope_marker_check.py --set-parked "
              "with the scope parked first if you want live displacement."
              % PARKED_PATH)
    print("logging to %s every %.0fs -- Ctrl+C to stop" % (LOG_PATH, args.interval))
    n = 0
    try:
        while True:
            t0 = time.monotonic()
            try:
                rec = sample(parked)
                n += 1
                print(_fmt(rec), flush=True)
            except Exception as e:  # noqa: BLE001 -- one bad sample must not end the session
                print("sample failed: %s" % e, flush=True)
            time.sleep(max(0.0, args.interval - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        print("\nstopped after %d samples -> %s" % (n, LOG_PATH))
    return 0


if __name__ == "__main__":
    sys.exit(main())
