"""
kasa_state.py
Scope and roof state from the inside Kasa camera, in one verdict.

kasa_status() mirrors vision_safety.visual_status()'s shape:

    safe, closed, is_open, when = kasa_state.kasa_status()

so the two can run side by side and eventually one can replace the other.
This is the candidate replacement for the webcam + template system; as of
2026-08-20 it DECIDES NOTHING -- it is called shadow-only, its verdicts
logged next to the webcam's so disagreements can be counted before anything
gates on it.

Two independent read-outs from one frame:

SCOPE -- the AprilTag on the OTA against the parked reference recorded from a
real PWI4 mount_park() (local/scope_marker_parked.json). Binary detection,
corners repeat to 0.1-0.5 px against a 69 px tolerance, works day and night
and survives the flat panel lit beside it. Structurally better than mount
telemetry here: a mount power cycle destroys PWI4's home reference and
get_is_parked() then returns False while the scope sits parked; the tag
measures the physical scope against a fixed camera and does not care.

ROOF -- statistics of a fixed aperture (the region that shows roof underside
when shut, sky when open), with the rule chosen by regime, because the
camera's night mode changes the physics:

  daylight (sun above SUN_DAY_DEG, colour mode)
      green_excess = g - (b+r)/2 over the aperture. Open shows foliage
      (positive), shut shows grey roof underside (negative). Validated
      2026-08-16 across a day of open/close cycles: sign-separated with a
      gap of 5.6-6.3 against a +/-1 dead-band.

  night (sun below, camera in IR: chroma == 0)
      The IR illuminator's own light is the signal. Shut, it reflects off
      the roof underside: bright (p99 195-250) with structure (edge 2.4-3.7%).
      Open, it leaves through the roof and nothing comes back: p99 144-157,
      edge 0.9-1.2%. Measured 2026-08-20 03:xx on 218 open / 18 shut
      samples -- ONE night, shut side contaminated by the flat panel, so the
      thresholds are provisional and both metrics must agree or the answer
      is unknown.

  night with the room lit (sun below but chroma > 0)
      UNKNOWN, deliberately. Measured 2026-08-20 03:49-03:51 with the roof
      OPEN and the inside light on for the close gate: green_excess read
      -10, which the daylight rule calls SHUT. Interior reflections dominate
      the aperture and mimic the shut signature. Nothing here is allowed to
      guess in that regime.

Every ambiguous path returns unknown, and unknown must always mean "refuse
to move anything". A camera failure, a missing marker, metrics that
disagree, an unvalidated regime -- all indistinguishable, by design, from
"not confirmed".

    python -m sentry.kasa_state          # one-shot verdict, printed
"""
import contextlib
import io
import json
import os
import sys
from datetime import datetime

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import cv2
import numpy as np

from configs import config
from utils import utils

_logger = utils.set_logger()

HOST = "192.168.87.65"
SNAP_PATH = "local/kasa_state_frame.jpg"

# The aperture: roof underside when shut, sky when open. Same box as
# roof_region_stats.py and iriscam_shadow_log.py so every log stays comparable.
REGION_Y = (100, 1400)
REGION_X = (0, 700)

# Daylight rule. Sign separates (open foliage positive, shut roof negative,
# measured gap 5.6-6.3); the dead-band turns "barely positive" into unknown
# rather than a guess.
GREEN_OPEN = +1.0
GREEN_SHUT = -1.0

# Night rule, PROVISIONAL: one night's close (2026-08-20), 18 shut samples
# taken during flats. Both metrics must land on the same side.
#   open  p99 144-157  edge 0.9-1.2
#   shut  p99 195-250  edge 2.4-3.7
NIGHT_P99_OPEN, NIGHT_P99_SHUT = 170.0, 180.0
NIGHT_EDGE_OPEN, NIGHT_EDGE_SHUT = 1.8, 2.0

# Above this sun altitude the daylight rule applies. Civil twilight downward
# the camera drifts toward IR and neither rule is validated; the gap between
# SUN_DAY_DEG and "camera actually in IR" resolves as unknown, which is the
# honest answer until the dusk/dawn shadow data exists.
SUN_DAY_DEG = 0.0

# Torn/starved frames happen (a fresh grab returned an undecodable frame the
# day the marker check went in). A retry is cheap; a wrong verdict is not.
RETRIES = 3

# Introspection for callers that want to log why, mirroring
# vision_safety.last_match.
last_detail: dict = {}


def _sun_altitude():
    import pytz
    from astral import LocationInfo
    from astral.sun import elevation
    loc = config.data()["location"]
    li = LocationInfo("obs", "", loc["timezone"], loc["latitude"], loc["longitude"])
    return float(elevation(li.observer, datetime.now(pytz.timezone(loc["timezone"]))))


def _grab(retries=RETRIES, timeout=20):
    from scripts.probe_kasa_camera import probe_kc_stream
    from sentry.sky_camera import credentials
    user, pw = credentials(config.data())
    for _ in range(retries):
        with contextlib.redirect_stdout(io.StringIO()):
            ok = probe_kc_stream(HOST, user, pw, snapshot_path=SNAP_PATH, timeout=timeout)
        if ok:
            img = cv2.imread(SNAP_PATH)
            if img is not None and img.size:
                return img
    return None


def _scope_verdict(img):
    """'safe' / 'UNSAFE' / 'unknown' from the AprilTag, plus detail."""
    from scripts.scope_marker_check import PARKED_PATH, compare, find_markers
    if not os.path.exists(PARKED_PATH):
        return "unknown", {"why": "no parked reference on file"}
    found = find_markers(img)
    stored = json.load(open(PARKED_PATH))
    parked = {int(k): np.array(v) for k, v in stored["markers"].items()}
    return compare(found, parked)


def _roof_verdict(img):
    """'open' / 'shut' / 'unknown' by regime, plus detail."""
    reg = img[REGION_Y[0]:REGION_Y[1], REGION_X[0]:REGION_X[1]]
    f = reg.astype(np.float64)
    grey = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY)
    chroma = float(np.mean(np.max(f, 2) - np.min(f, 2)))
    detail = {"chroma": round(chroma, 2)}
    try:
        sun = _sun_altitude()
        detail["sun_alt"] = round(sun, 1)
    except Exception as e:  # no sun, no regime, no verdict
        detail["why"] = "sun altitude unavailable: %r" % (e,)
        return "unknown", detail

    if sun >= SUN_DAY_DEG:
        b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
        ge = float((g - (b + r) / 2.0).mean())
        detail.update(regime="day", green_excess=round(ge, 2))
        if ge >= GREEN_OPEN:
            return "open", detail
        if ge <= GREEN_SHUT:
            return "shut", detail
        detail["why"] = "green_excess inside the dead-band"
        return "unknown", detail

    if chroma > 0.5:
        # Night but the frame has colour: the room is lit. Interior
        # reflections mimic the shut signature (measured: roof open read
        # green_excess -10 under the inside light). Refuse.
        detail.update(regime="night-lit",
                      why="room lit at night; aperture rules not valid")
        return "unknown", detail

    p99 = float(np.percentile(grey, 99))
    edges = cv2.Canny(cv2.GaussianBlur(grey, (5, 5), 0), 40, 120)
    edge_pct = 100.0 * float((edges > 0).mean())
    detail.update(regime="night", p99=round(p99, 1), edge_pct=round(edge_pct, 2))
    open_votes = (p99 <= NIGHT_P99_OPEN) + (edge_pct <= NIGHT_EDGE_OPEN)
    shut_votes = (p99 >= NIGHT_P99_SHUT) + (edge_pct >= NIGHT_EDGE_SHUT)
    if open_votes == 2 and shut_votes == 0:
        return "open", detail
    if shut_votes == 2 and open_votes == 0:
        return "shut", detail
    detail["why"] = "night metrics disagree or sit between the thresholds"
    return "unknown", detail


def kasa_status(quick=False):
    """(safe, closed, is_open, when) -- vision_safety.visual_status()'s shape.

    safe means "the scope is within tolerance of the recorded park pose", the
    Kasa system's sharper version of parked. Any failure anywhere returns
    (False, False, False, when): unconfirmed on every axis.

    quick=True caps the grab at ONE short attempt. The shadow comparison in
    the roof flow runs seconds before toggle_roof opens its own 50 s audio
    stream on this same one-stream camera; a retrying grab could still hold
    the stream at that moment and cost a roof-move audio capture, which is
    worth more than a shadow sample. A missed quick grab logs camera=False
    and the comparison for that gate is simply absent.
    """
    global last_detail
    when = datetime.now().astimezone()
    img = _grab(retries=1, timeout=8) if quick else _grab()
    if img is None:
        last_detail = {"camera": False}
        _logger.warning("kasa_status: no frame from the camera")
        return False, False, False, when

    scope, sdet = _scope_verdict(img)
    roof, rdet = _roof_verdict(img)
    last_detail = {"camera": True, "scope": scope, "scope_detail": sdet,
                   "roof": roof, "roof_detail": rdet}
    _logger.info("kasa_status: scope=%s roof=%s (%s)", scope, roof,
                 rdet.get("regime", "?"))
    return scope == "safe", roof == "shut", roof == "open", when


def main():
    safe, closed, is_open, when = kasa_status()
    print("scope : %s" % ("SAFE (at park pose)" if safe else
                          last_detail.get("scope", "unknown")))
    print("roof  : %s" % ("CLOSED" if closed else ("OPEN" if is_open else "unknown")))
    for k, v in (last_detail.get("roof_detail") or {}).items():
        print("   %-12s %s" % (k, v))
    sd = last_detail.get("scope_detail") or {}
    if "worst_corner_px" in sd:
        print("   %-12s %.1f px" % ("scope_off", sd["worst_corner_px"]))
    return 0 if (safe and (closed or is_open)) else 1


if __name__ == "__main__":
    sys.exit(main())
