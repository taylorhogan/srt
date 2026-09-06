"""roof_marker_check.py — is the roof SHUT, from the AprilTag on the roof panel.

    python scripts/roof_marker_check.py --set-shut   # with the roof shut
    python scripts/roof_marker_check.py             # check

Sibling of scope_marker_check.py. That one asks "is the scope safe"; this asks
"is the roof shut", off a tag (36h11 id 1) on the roof panel itself, seen by
the indoor camera.

WHY A TAG, AND WHY THIS ANSWER IS A READING RATHER THAN AN INFERENCE
The camera had no positive signal for a shut roof. The gold star marker shows
when the roof is OPEN and hides behind the roof structure when it is shut, so
shut had to be inferred from that marker's ABSENCE -- and a fogged lens, a dead
camera, and a camera pointed somewhere else all look exactly like absence. A
tag that is PRESENT when shut turns that inference into a measurement.

The old colour rule cannot do this job: the camera delivers greyscale infrared
on 91% of frames (943 of 1041 sampled) and on those the colour channels are
identically zero, so anything separating by hue is blind most of the night.

THE SECOND TAG IS THE POINT. Absence alone is weak evidence, so the SCOPE's tag
(id 0) is used as a witness: it is on the mount, always in view, and its
presence proves the camera is working, pointed correctly, and can see. Then

    roof tag found            -> SHUT      (a positive reading)
    roof tag gone, scope seen -> OPEN      (also positive: the camera CAN see,
                                            and the roof is not there)
    neither found             -> UNKNOWN   (blind: lens, power, pointing)

so every verdict rests on something detected, never on nothing detected.

POSE FIRST, ALWAYS. The stored corners describe the roof as seen from ONE
camera position, so the camera is driven there before the frame is taken rather
than trusted to still be there. A record of where it was last put is not
knowledge of where it is: measured 2026-09-05, the local record read
(-123, 363) while the camera sat at (-666, 306) after a move from the phone app.
"""
import argparse
import json
import os
import sys
from datetime import datetime

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import numpy as np

from scripts import kasa_pose
from scripts.scope_marker_check import PARKED_PATH, find_markers, grab

SHUT_PATH = "local/roof_marker_shut.json"
ROOF_ID = 1
SCOPE_ID = 0

# PROVISIONAL, and it must be measured rather than left at this. The roof is a
# rolling panel and does not necessarily stop in exactly the same place twice,
# so the honest tolerance is the spread of the tag's position across many real
# closes -- which cannot be known until many real closes have been logged.
# Wide on purpose meanwhile: the discriminator here is PRESENCE (the tag is
# only visible at all when the roof is over the aperture), and position is a
# sanity check on top of that, not the primary evidence.
TOLERANCE_PX = 200.0


def _reference():
    try:
        with open(SHUT_PATH) as fh:
            return json.load(fh)
    except Exception:       # noqa: BLE001
        return None


def verdict(found, ref, pose):
    """('shut'|'open'|'unknown', detail) from the markers in one frame."""
    if ref is None:
        return "unknown", {"why": "no shut reference recorded; run --set-shut "
                                  "with the roof shut"}
    if pose is None:
        return "unknown", {"why": "camera pose could not be established"}
    if tuple(pose) != tuple(ref["ptz"]):
        return "unknown", {"why": "camera is not at the reference pose",
                           "ref_pose": tuple(ref["ptz"]), "now": tuple(pose)}

    roof_seen = ROOF_ID in found
    scope_seen = SCOPE_ID in found
    if roof_seen:
        ref_c = np.array(ref["markers"][str(ROOF_ID)])
        d = float(np.linalg.norm(np.array(found[ROOF_ID]) - ref_c, axis=1).max())
        if d <= ref.get("tolerance_px", TOLERANCE_PX):
            return "shut", {"worst_corner_px": round(d, 1), "witness": scope_seen}
        # The tag is visible but not where the roof puts it when shut. That is
        # not "open" -- open means the tag is GONE -- it is something
        # unexplained, and unexplained must refuse.
        return "unknown", {"why": "roof tag found, but %.0f px from its shut "
                                  "position (tolerance %.0f)"
                                  % (d, ref.get("tolerance_px", TOLERANCE_PX)),
                           "worst_corner_px": round(d, 1)}
    if scope_seen:
        return "open", {"why": "roof tag absent while the scope tag is visible, "
                               "so the camera can see and the roof is not there"}
    return "unknown", {"why": "neither tag visible; the camera is blind, not "
                              "looking at an open roof"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set-shut", action="store_true",
                    help="record the roof tag's corners as the SHUT reference")
    ap.add_argument("--label", default="roofcheck")
    args = ap.parse_args()

    scope_ref = json.load(open(PARKED_PATH))
    cam = scope_ref.get("camera", "Iris cam")
    want = tuple(scope_ref["ptz"])
    pose = kasa_pose.ensure_at(cam, want)
    print("camera %r driven to %s -> %s" % (cam, want, pose))
    if pose is None:
        print("could not place the camera; refusing to judge the frame")
        return 2

    img = grab("local/%s.jpg" % args.label)
    if img is None:
        print("no frame from the camera -- verdict unknown")
        return 2
    found = find_markers(img, scope_ref.get("dict", "APRILTAG_36h11"))
    print("markers detected: %s" % (sorted(found) or "NONE"))

    if args.set_shut:
        if ROOF_ID not in found:
            print("roof tag (id %d) not visible; cannot record a shut pose" % ROOF_ID)
            return 1
        data = {"when": datetime.now().astimezone().isoformat(timespec="seconds"),
                "tolerance_px": TOLERANCE_PX,
                "tolerance_provisional": True,
                "dict": scope_ref.get("dict", "APRILTAG_36h11"),
                "camera": cam, "ptz": list(pose),
                "roof_id": ROOF_ID, "witness_id": SCOPE_ID,
                "witness_seen_at_record": SCOPE_ID in found,
                "markers": {str(ROOF_ID): np.asarray(found[ROOF_ID]).tolist()}}
        with open(SHUT_PATH, "w") as fh:
            json.dump(data, fh, indent=2)
        c = np.asarray(found[ROOF_ID])
        print("recorded shut reference -> %s" % SHUT_PATH)
        print("  roof tag centre (%.1f, %.1f), mean side %.0f px"
              % (c[:, 0].mean(), c[:, 1].mean(),
                 np.mean([np.linalg.norm(c[k] - c[(k + 1) % 4]) for k in range(4)])))
        print("  witness (scope tag id %d) visible: %s" % (SCOPE_ID, SCOPE_ID in found))
        print("  tolerance %.0f px is PROVISIONAL -- measure it across real closes"
              % TOLERANCE_PX)
        return 0

    v, detail = verdict(found, _reference(), pose)
    print("\nverdict: %s" % v.upper())
    for k, val in detail.items():
        print("   %-18s %s" % (k, val))
    return 0


if __name__ == "__main__":
    sys.exit(main())
