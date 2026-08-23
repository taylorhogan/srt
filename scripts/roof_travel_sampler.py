"""
roof_travel_sampler.py
Sample the Kasa cam's roof verdict as fast as it will grab, through a roof
move, to measure how green_excess transitions between the shut (~-3.6) and
open (~+5.2) plateaus during the ~45 s of travel.

Why: the daylight roof rule is an aperture statistic, not a position proof.
A template match (webcam) only fires when the roof edge is AT the fully
open/closed mark; green_excess is a mean over a fixed box, so in principle a
partially open roof could read "open" as soon as the aperture's patch of sky
is exposed — early or late in the travel depending on where the box sits
relative to the travel direction. Which of those it is has never been
measured; the one mid-travel sample so far (2026-08-23 14:30:31) landed in
the dead-band ("unknown"), which is the safe answer, but one point is not a
curve.

Run it across a roof move (start it just before roof!! open/close):

    python scripts/roof_travel_sampler.py                 # ~150 s, Ctrl+C early
    python scripts/roof_travel_sampler.py --duration 300

Appends to local/roof_travel_green_excess.jsonl. Read-only; touches no
hardware. KNOWN COST: each grab opens the camera's 19443 stream, which may
collide with the kasa_audio shadow capture toggle_roof starts for the same
move — worst case that move's Kasa-mic spectrogram is lost. That capture is
a shadow observer (nothing gates on it), and the main-mic audio + current
signature are unaffected, so losing one is an acceptable price for the curve.

What the curve decides: if the verdict flips to "open"/"shut" only near the
end of travel (dead-band covers most of the transit), the aperture stat is
trustworthy as a full-travel indicator; if it flips early, the Kasa roof
verdict must stay corroboration-only and the cutover keeps the webcam's
positional templates for the roof half.
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

import contextlib
import io

import cv2

LOG_PATH = "local/roof_travel_green_excess.jsonl"
SNAP_PATH = "local/roof_travel_frame.jpg"


def _grab_once(timeout=15):
    """One frame, own snapshot path so kasa_state's frame is not clobbered."""
    from configs import config
    from scripts.probe_kasa_camera import probe_kc_stream
    from sentry.kasa_state import HOST
    from sentry.sky_camera import credentials
    user, pw = credentials(config.data())
    with contextlib.redirect_stdout(io.StringIO()):
        ok = probe_kc_stream(HOST, user, pw, snapshot_path=SNAP_PATH, timeout=timeout)
    if not ok:
        return None
    img = cv2.imread(SNAP_PATH)
    return img if img is not None and img.size else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=float, default=150.0,
                    help="seconds to keep sampling (default 150; Ctrl+C stops early)")
    ap.add_argument("--label", default=None,
                    help="optional label written into every record (e.g. 'open 2026-08-24')")
    args = ap.parse_args()

    from sentry.kasa_state import _roof_verdict

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    print("sampling roof verdict for %.0fs -> %s (Ctrl+C to stop)"
          % (args.duration, LOG_PATH))
    t_end = time.monotonic() + args.duration
    n = 0
    try:
        while time.monotonic() < t_end:
            when = datetime.now().astimezone().isoformat(timespec="seconds")
            img = _grab_once()
            if img is None:
                rec = {"when": when, "grab_ok": False}
                line = "%s  NO FRAME (stream busy or camera down)" % when[11:19]
            else:
                verdict, detail = _roof_verdict(img)
                rec = {"when": when, "grab_ok": True, "verdict": verdict, **detail}
                line = "%s  %-7s  green_excess %-7s chroma %s" % (
                    when[11:19], verdict, detail.get("green_excess"), detail.get("chroma"))
            if args.label:
                rec["label"] = args.label
            with open(LOG_PATH, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(line, flush=True)
            n += 1
    except KeyboardInterrupt:
        pass
    print("done: %d samples -> %s" % (n, LOG_PATH))
    return 0


if __name__ == "__main__":
    sys.exit(main())
