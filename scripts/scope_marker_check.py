"""
scope_marker_check.py
Decide whether the scope is SAFE for the roof to close, from an ArUco marker
on the OTA.

Safe, not parked. Parked is one pose; safe is the region around it in which the
roof can close without striking the scope, and that region is what a roof gate
actually needs to know about. A scope 1.75 deg off park is not parked but is
almost certainly safe, and refusing to close on that costs a night for nothing.

This replaces the correlation approach in scope_parked_probe.py, which asked
how much the scope AREA of the image resembles a stored parked frame. That
works, but it conflates pose with illumination badly enough to need a library
of references per lighting condition, and even then leaves 0.234 of margin
between parked and moved.

A marker removes the problem rather than tuning around it:

  * Detection is binary and far harder to fool than correlation, though NOT
    impossible: 4X4_50 produced a spurious id 17 on a real frame here (see
    DICTS below). It is harmless because only ids recorded in the parked pose
    are compared, but "foliage cannot be a marker" would be an overstatement.
    Contrast the greyscale template matcher, where a tree scored 0.63 against
    a real marker's 0.65 and there was no id to disagree about.
  * It binarises, so it does not care about colour, exposure or the day/night
    IR switch. No reference library.
  * It returns four corners, so the answer has UNITS: how many pixels off park,
    not a correlation coefficient needing calibration.

Measured 2026-08-16 with the scope stationary: corner positions repeat to
sd 0.12 px over four captures. Any real slew moves them by orders more.

MOUNT IT FLAT AND RIGID. This is the failure mode that actually happens, and it
happened here within 19 hours. A tag taped to the curved shroud read cleanly at
13:41 and 13:51 on 2026-08-16 and was undetectable by 08:48 the next morning,
the paper having buckled into a visible fold. The detector still FOUND the
quadrilateral — a 145 px rejected candidate sat exactly on the marker — but the
decode failed, because sampling the cell grid uses a planar homography and the
surface was no longer planar. No relaxation of the detector parameters
recovered it.

Note what that rules out: the geometry had IMPROVED. It decoded at 89x136 px
and roughly 49 degrees oblique, and failed at 145 px and closer to face-on. So
size and viewing angle were not the constraint; flatness was. Angling the tag
toward the camera is worth doing for margin, but it does not substitute for a
rigid backing that stands proud of the shroud's curve instead of conforming to
it.

A tag that goes unreadable when the scope slews is fine, incidentally — the
verdict becomes 'unknown' and the caller refuses, which is the safe direction.
Optimise the mounting for the parked pose.

    python scope_marker_check.py --set-parked     # with the scope parked
    python scope_marker_check.py                  # check
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

import cv2
import numpy as np

from configs import config
from scripts.probe_kasa_camera import probe_kc_stream
from sentry.sky_camera import credentials

HOST = "192.168.87.65"
PARKED_PATH = "local/scope_marker_parked.json"
LOG_PATH = "local/scope_marker_log.jsonl"

# The dictionary is recorded in the parked file, so changing this default
# cannot silently stop detecting a marker that is already mounted.
#
# 4X4_50 is the WEAKEST choice and it showed: on 2026-08-16 it reported a
# spurious id 17 -- a 42 px phantom up in the ceiling -- in one frame out of
# four. 16 bits over 50 ids leaves little Hamming distance, so "the error
# correction makes false positives impossible" was an overstatement. The
# phantom was harmless here because only ids present in the parked reference
# are compared, but a stronger dictionary removes the question. 6X6_250 and
# APRILTAG_36h11 produced zero false positives on the same frames.
#
# Detection is not the constraint: 4X4 still resolved at 30 px, and a 6X6 at
# the mounted marker's ~88 px still gives ~11 px per module.
DICTS = {
    "4X4_50": cv2.aruco.DICT_4X4_50,
    "5X5_100": cv2.aruco.DICT_5X5_100,
    "6X6_250": cv2.aruco.DICT_6X6_250,
    "APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}
DEFAULT_DICT = "APRILTAG_36h11"   # what is physically mounted today
#
# Was 4X4_50, and stayed that way after the mounted tag changed -- which made
# the checker silently blind: it detected nothing, said "unknown", and looked
# like a camera problem rather than a stale constant. The parked reference
# recorded alongside it was retired at the same time and for the same reason.
# Changing only one of the two is worse than changing neither: the old
# reference holds the corners of a DIFFERENT tag in a DIFFERENT place, so a
# working detector compared against it would report a confident UNSAFE.

# The question is NOT "is the scope parked" -- it is "can the roof close
# without hitting the scope". Parked is one pose; SAFE is a region around it,
# and the threshold belongs at the collision boundary, not at park precision.
# Getting this backwards made the gate call a hand-park 1.75 deg off "MOVED",
# which was the wrong answer to the wrong question: it was almost certainly
# safe, just not exactly parked.
#
# MEASURED 2026-08-24. The boundary session finally ran: roof open, mount
# UNPOWERED and the OTA positioned by hand, operator calling each pose while
# looking at the clearance against the roof. Log:
# local/boundary_session.jsonl. Full detail in
# local/boundary_session_ABANDONED_20260824.jsonl is NOT part of this -- that
# run was discarded because a verdict got attached to the wrong pose.
#
#     12.4 px  ( 0.5 deg)  clears
#     94.3 px  ( 4.1 deg)  clears
#    266.8 px  (11.6 deg)  MARGINAL -- operator: "close but would clear"
#    296.8 px  (12.9 deg)  MARGINAL -- leaning hit
#    603.7 px  (26.2 deg)  would hit
#
# Direction: a mix, mostly toward the roof -- i.e. the collision geometry
# itself, which is the axis that matters. The tolerance is a single scalar
# compared against displacement in ANY direction, so it is keyed to the tight
# axis and is therefore conservative for others rather than permissive.
#
# The two independent marginal calls sit 30 px apart, 11% of the lower. That
# spread IS the error bar -- the judgement is subjective, by eye -- and the 30%
# margin taken below comfortably exceeds it.
#
# THE MEASUREMENT IS IN PIXELS, so the tolerance is defined in pixels and the
# degree figure is derived for display only. That is the honest direction of
# the conversion: PX_PER_DEGREE is still the unrefined single-estimate 23 from
# 2026-08-16, because the mount was unpowered for this session and there is no
# alt/az to fit a real scale against. Nothing in the safety path uses degrees.
PX_PER_DEGREE = 23.0

# The last pose the operator would still accept, times a 0.7 safety factor.
# Was 69 px (a 3.0 degree placeholder chosen to clear a hand-park), which this
# session shows was far too tight: 94 px was called an unambiguous clear, so
# the gate had been refusing a wide band of genuinely safe poses -- and a
# refusal to close the roof costs a night, or a wet telescope.
_MARGINAL_PX = 266.8
_SAFETY_FACTOR = 0.7
TOLERANCE_PX = _MARGINAL_PX * _SAFETY_FACTOR      # 186.8 px, ~8.1 deg
SAFE_DEGREES = TOLERANCE_PX / PX_PER_DEGREE       # display only


# The default adaptive-threshold sweep (win 3..23 step 10) loses the tag when
# the flat panel is lit. Measured 2026-08-19 during flats: the panel sits ~54%
# saturated a few hundred pixels from the marker, and at the default window
# sizes the local threshold near the tag is set by the glare rather than by the
# tag -- the detector still found 14 quadrilateral candidates and decoded none
# of them. Sweeping more window sizes, small ones included, recovers it.
#
# This is margin, NOT a safety fix. The panel is not lit when park and roof
# state are actually checked (confirmed by the user 2026-08-19), so the gate
# never meets this lighting -- an earlier version of this comment claimed the
# opposite and overstated it. What the flats frame provided was a free
# adverse-lighting stress test, and the detector failed it at default settings.
#
# Checked for regressions on six earlier frames (day, night, three poses): every
# one still detects, at the same centre to 0.3 px. Strictly more sensitive.
_WIN_MIN, _WIN_MAX, _WIN_STEP = 3, 53, 4


def detector(name=DEFAULT_DICT):
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = _WIN_MIN
    params.adaptiveThreshWinSizeMax = _WIN_MAX
    params.adaptiveThreshWinSizeStep = _WIN_STEP
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICTS[name]),
                                   params)


def grab(path):
    user, pw = credentials(config.data())
    if not probe_kc_stream(HOST, user, pw, snapshot_path=path, timeout=15):
        return None
    return cv2.imread(path)


def find_markers(img, dict_name=DEFAULT_DICT):
    """{id: 4x2 corner array} for every marker in the frame."""
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector(dict_name).detectMarkers(grey)
    if ids is None:
        return {}
    return {int(i): c[0] for c, i in zip(corners, ids.ravel())}


def compare(found, parked, ref_pose=None, now_pose=None):
    """(verdict, detail). Verdict is 'safe', 'UNSAFE' or 'unknown'.

    *ref_pose* is the camera pan/tilt position recorded alongside the parked
    corners; *now_pose* is where the camera is pointing now. Both are (x, y)
    or None.

    The corner comparison below is pure pixel space, so it cannot tell a scope
    that moved from a CAMERA that moved. That matters in both directions.
    Pointed away and not restored, every corner shifts and a stationary scope
    is reported UNSAFE -- safe, but it blames the wrong thing and costs a
    night. Worse, restored badly it can still pass: the same coordinate
    re-entered from the opposite side put the frame 121 px out (kasa_ptz's
    backlash note), which quietly eats two thirds of a %.0f px budget that
    exists to describe how far the SCOPE may drift, not how badly the camera
    was parked. Re-entered from a consistent side it repeats to 5 px, which is
    why kasa_ptz.goto re-approaches and why a restored pose is trustworthy.

    So a pose mismatch is UNKNOWN, not UNSAFE: the reference does not describe
    this view, and that is ignorance rather than a finding.

    A reference recorded before poses were stored has ref_pose None. The check
    is then skipped and said so in the detail, so this lands without
    invalidating the reference in use -- it arms itself the next time
    --set-parked runs, which is exactly when the pose is known good.
    """
    pose_note = None
    if ref_pose is None:
        pose_note = ("parked reference predates pose recording; camera "
                     "pointing NOT verified -- re-run --set-parked to arm it")
    else:
        if now_pose is None:
            return "unknown", {"why": "camera pose unknown; the parked "
                                      "reference is tied to pose %s"
                                      % (tuple(ref_pose),)}
        if tuple(now_pose) != tuple(ref_pose):
            return "unknown", {"why": "camera has been pointed elsewhere",
                               "ref_pose": tuple(ref_pose),
                               "now_pose": tuple(now_pose),
                               "fix": "kasa_ptz.py goto '<cam>' %d %d"
                                      % (ref_pose[0], ref_pose[1])}

    missing = [i for i in parked if i not in found]
    if missing:
        # A marker that cannot be seen is not evidence that the scope moved --
        # it could be occluded, or the camera could be off. Unknown, not MOVED:
        # the caller refuses either way, but the two need different fixes.
        return "unknown", {"missing_ids": missing, "found_ids": sorted(found),
                           "pose_note": pose_note}
    worst = 0.0
    per_id = {}
    for i, ref in parked.items():
        d = np.linalg.norm(np.array(found[i]) - np.array(ref), axis=1)
        per_id[i] = {"max_px": float(d.max()), "mean_px": float(d.mean())}
        worst = max(worst, float(d.max()))
    return ("safe" if worst <= TOLERANCE_PX else "UNSAFE",
            {"worst_corner_px": worst, "per_id": per_id,
             "pose_note": pose_note})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set-parked", action="store_true",
                    help="record the current marker corners as the parked pose")
    ap.add_argument("--label", default="check")
    ap.add_argument("--camera", default="Iris cam",
                    help="pan/tilt camera whose pose ties this reference to a "
                         "view (default: %(default)r)")
    ap.add_argument("--dict", default=None, choices=sorted(DICTS),
                    help="marker dictionary; defaults to the one recorded in "
                         "the parked file, else %s" % DEFAULT_DICT)
    args = ap.parse_args()

    img = grab("local/scope_marker_%s.jpg" % args.label)
    if img is None:
        print("no frame from the camera -- verdict unknown")
        return 2
    dict_name = args.dict
    if dict_name is None and os.path.exists(PARKED_PATH):
        dict_name = json.load(open(PARKED_PATH)).get("dict", DEFAULT_DICT)
    dict_name = dict_name or DEFAULT_DICT
    found = find_markers(img, dict_name)
    print("dictionary: %s" % dict_name)
    print("markers detected: %s" % (sorted(found) or "NONE"))

    if args.set_parked:
        if not found:
            print("no marker visible; cannot record a parked pose")
            return 1
        # Record WHERE THE CAMERA WAS LOOKING alongside the corners. Without
        # it the corners are coordinates with no frame of reference, and a
        # later comparison cannot tell a scope that moved from a camera that
        # was pointed elsewhere. Read from the cloud here (imported lazily) --
        # this is an interactive operator command, unlike the verdict path,
        # which must never wait on the internet.
        ptz_pose = None
        try:
            from scripts import kasa_ptz, kasa_pose
            dev = kasa_ptz._device(args.camera)
            ptz_pose = [int(v) for v in kasa_ptz.position(dev)]
            kasa_pose.record(args.camera, ptz_pose)
            print("  camera %r pose %s recorded with the reference"
                  % (args.camera, tuple(ptz_pose)))
        except Exception as exc:        # noqa: BLE001
            print("  WARNING: could not read the camera pose (%s: %s)."
                  % (type(exc).__name__, exc))
            print("  The reference will be saved WITHOUT a pose, so pointing "
                  "stays unverified. Fix the camera link and re-run.")
        data = {"when": datetime.now().astimezone().isoformat(timespec="seconds"),
                "tolerance_px": TOLERANCE_PX, "dict": dict_name,
                "camera": args.camera, "ptz": ptz_pose,
                "markers": {str(i): np.asarray(c).tolist() for i, c in found.items()}}
        with open(PARKED_PATH, "w") as fh:
            json.dump(data, fh, indent=2)
        for i, c in found.items():
            c = np.asarray(c)
            print("  id %d centre (%.1f, %.1f)" % (i, c[:, 0].mean(), c[:, 1].mean()))
        print("recorded parked pose -> %s" % PARKED_PATH)
        return 0

    if not os.path.exists(PARKED_PATH):
        print("no parked pose recorded; run --set-parked with the scope parked")
        return 1
    stored = json.load(open(PARKED_PATH))
    parked = {int(k): np.array(v) for k, v in stored["markers"].items()}
    ref_pose = stored.get("ptz")
    from scripts import kasa_pose
    now_pose = kasa_pose.last(stored.get("camera", args.camera))
    verdict, detail = compare(found, parked, ref_pose, now_pose)

    print("\nverdict: %s" % verdict)
    if verdict == "unknown":
        print("  markers missing: %s" % detail["missing_ids"])
    else:
        print("  worst corner displacement: %.1f px = %.2f deg  "
              "(safe below %.0f px = %.1f deg)"
              % (detail["worst_corner_px"], detail["worst_corner_px"] / PX_PER_DEGREE,
                 TOLERANCE_PX, SAFE_DEGREES))
        for i, d in sorted(detail["per_id"].items()):
            print("    id %d  max %.1f px  mean %.1f px" % (i, d["max_px"], d["mean_px"]))

    with open(LOG_PATH, "a") as fh:
        fh.write(json.dumps({"label": args.label, "verdict": verdict,
                             "when": datetime.now().astimezone().isoformat(timespec="seconds"),
                             **{k: v for k, v in detail.items() if k != "per_id"}}) + "\n")
    return 0 if verdict == "safe" else 3


if __name__ == "__main__":
    sys.exit(main())
