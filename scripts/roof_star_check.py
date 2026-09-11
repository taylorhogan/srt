"""roof_star_check.py -- is the roof OPEN, from the gold star on the wall.

    python scripts/roof_star_check.py --set-open                 # live, roof OPEN, light on
    python scripts/roof_star_check.py --set-open --from-frame F  # from an archived lit open frame
    python scripts/roof_star_check.py                            # live check
    python scripts/roof_star_check.py --replay "sentry/roof_frames_kasa/*/*.jpg"

Sibling of roof_marker_check.py (roof SHUT, from the AprilTag on the panel)
and scope_marker_check.py (scope parked). This one records and checks the
POSITIVE evidence for an open roof.

WHY A POSITIVE MARKER FOR OPEN
Until 2026-09-11 the Kasa camera called the roof OPEN when the roof tag was
absent and the scope tag was visible. Absence is also what a tag that failed
to decode, a smeared lens over that part of the frame, or a roof stopped half
way looks like, and a false OPEN with the roof shut is the collision case.
The user's direction on 2026-09-10: "you need to see the open star".

THE STAR. The gold star the retired webcam used is still on the dark wall to
the left of the aperture. It is in view when the roof is open and hidden by
the roof structure when the roof is shut (every archived frame agrees: seen
in 8/8 lit open frames within 15 px of one spot, and in 0/9 shut frames,
where the nearest yellow thing is the pine roof 60+ px away). It is YELLOW,
so it needs a colour frame: gating reads switch the inside light on, and the
camera drops out of greyscale IR a few seconds later. On an IR frame the
answer is "unknown", never "seen".

THE RECORD. local/roof_star_open.json holds the star's golden position (blob
centroid), its search window, a tolerance, the camera pose the picture was
taken from, and points at local/roof_star_template.png, a crop of the star
with margin. A check finds the nearest star-sized yellow blob to the golden
position, requires it inside the tolerance, and then requires the template
(a bright compact blob on the dark wall) to correlate there -- position AND
context, so pine over the window (bright everywhere, no dark surround, ncc
0.33 or less on every shut frame) cannot pass even where it is yellow.
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import cv2

from scripts import kasa_pose
from scripts.scope_marker_check import PARKED_PATH, find_markers, grab
from sentry import kasa_state as ks

# Where the star sat on 2026-09-10 in the 2560x1440 gating frame; only a hint
# for --set-open, which measures the real centroid.
DEFAULT_NEAR = (638, 718)


def record(img, near, pose, cam, source):
    """Measure the star near *near* in a lit open frame and write the reference."""
    if ks.is_ir(img):
        print("frame is greyscale (IR); the star needs a colour frame -- refusing")
        return 1
    found = find_markers(img, json.load(open(PARKED_PATH)).get("dict", "APRILTAG_36h11"))
    print("markers detected: %s" % (sorted(found) or "NONE"))
    if ks.ROOF_ID in found:
        print("roof tag (id %d) visible: the roof is not open; refusing" % ks.ROOF_ID)
        return 1
    blob = ks.find_star_blob(img, near, ks.STAR_SEARCH_PX)
    if blob is None:
        print("no star-sized yellow blob within %d px of %s" % (ks.STAR_SEARCH_PX, near))
        return 1
    d, area, cx, cy, (bx, by, bw, bh) = blob
    mg = ks.STAR_TEMPLATE_MARGIN
    x0, y0 = max(0, bx - mg), max(0, by - mg)
    x1, y1 = min(img.shape[1], bx + bw + mg), min(img.shape[0], by + bh + mg)
    tmpl = img[y0:y1, x0:x1]
    os.makedirs(os.path.dirname(ks.STAR_TEMPLATE_PATH), exist_ok=True)
    cv2.imwrite(ks.STAR_TEMPLATE_PATH, tmpl)
    data = {"when": datetime.now().astimezone().isoformat(timespec="seconds"),
            "camera": cam, "ptz": list(pose) if pose else None,
            "source": source,
            "centre": [round(float(cx), 1), round(float(cy), 1)],
            "bbox": [int(bx), int(by), int(bw), int(bh)],
            "area_px": int(area),
            "search_px": ks.STAR_SEARCH_PX,
            "tolerance_px": ks.STAR_TOLERANCE_PX,
            "ncc_min": ks.STAR_NCC_MIN,
            "template": ks.STAR_TEMPLATE_PATH,
            "template_origin": [int(x0), int(y0)],
            "witness_seen_at_record": ks.SCOPE_ID in found}
    with open(ks.STAR_PATH, "w") as fh:
        json.dump(data, fh, indent=2)
    print("recorded open-star reference -> %s" % ks.STAR_PATH)
    print("  star centre (%.1f, %.1f), %d px, bbox %s, %d px from the hint"
          % (cx, cy, area, (bx, by, bw, bh), d))
    print("  template %dx%d -> %s" % (tmpl.shape[1], tmpl.shape[0], ks.STAR_TEMPLATE_PATH))
    print("  witness (scope tag id %d) visible: %s" % (ks.SCOPE_ID, ks.SCOPE_ID in found))
    # The recording frame must pass its own check, or the reference is useless.
    v, det = ks.star_verdict(img, data)
    print("  self-check: %s %s" % (v.upper(), det))
    return 0 if v == "seen" else 1


def replay(pattern, ref):
    rows = []
    for p in sorted(glob.glob(pattern)):
        img = cv2.imread(p)
        if img is None:
            continue
        v, det = ks.star_verdict(img, ref)
        rows.append((os.path.relpath(p), v, det))
    for p, v, det in rows:
        brief = {k: det[k] for k in ("star_px", "star_area", "star_ncc") if k in det}
        print("%-7s %-70s %s" % (v, p, brief or det.get("why", "")))
    print("%d frames: %d seen, %d absent, %d unknown"
          % (len(rows), sum(1 for _, v, _ in rows if v == "seen"),
             sum(1 for _, v, _ in rows if v == "absent"),
             sum(1 for _, v, _ in rows if v == "unknown")))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set-open", action="store_true",
                    help="record the star's position and template as the OPEN reference")
    ap.add_argument("--from-frame", help="record from this saved frame instead of the live camera")
    ap.add_argument("--near", type=int, nargs=2, default=DEFAULT_NEAR, metavar=("X", "Y"),
                    help="where to look for the star when recording (default %s)" % (DEFAULT_NEAR,))
    ap.add_argument("--replay", metavar="GLOB", help="run the check over saved frames and tabulate")
    ap.add_argument("--label", default="starcheck")
    args = ap.parse_args()

    if args.replay:
        return replay(args.replay, ks._star_reference())

    scope_ref = json.load(open(PARKED_PATH))
    cam = scope_ref.get("camera", "Iris cam")
    want = tuple(scope_ref["ptz"])

    if args.set_open and args.from_frame:
        img = cv2.imread(args.from_frame)
        if img is None:
            print("cannot read %s" % args.from_frame)
            return 2
        print("recording from %s; pose taken as the parked reference's %s (frames on file "
              "are taken after a pose-verified read, from the same camera position)"
              % (args.from_frame, want))
        return record(img, tuple(args.near), want, cam, args.from_frame)

    pose = kasa_pose.ensure_at(cam, want)
    print("camera %r driven to %s -> %s" % (cam, want, pose))
    if pose is None:
        print("could not place the camera; refusing to judge the frame")
        return 2
    img = grab("local/%s.jpg" % args.label)
    if img is None:
        print("no frame from the camera -- verdict unknown")
        return 2
    if args.set_open:
        return record(img, tuple(args.near), pose, cam, "live")

    v, detail = ks.star_verdict(img, ks._star_reference())
    print("\nopen star: %s" % v.upper())
    for k, val in detail.items():
        print("   %-14s %s" % (k, val))
    return 0


if __name__ == "__main__":
    sys.exit(main())
