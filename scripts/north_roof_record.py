#!/usr/bin/env python3
"""Re-record one of the north camera's roof references (shut or open).

    python scripts/north_roof_record.py open            # measure, print, write nothing
    python scripts/north_roof_record.py open --write    # ... and replace the open reference

WHY. sentry/north_roof.py decides the roof from AprilTag 2 at two recorded
positions in local/north_roof_marker.json, with a 100 px tolerance. Those
positions are pixel coordinates at one camera pointing. Between 2026-09-24 and
09-27 Iris North's picture moved ~95 px down and ~16 px right in two steps
(09-25 07:36-07:41, 09-26/27), the same at both ends of travel and with the
tag the same size: the camera's aim moved, not the roof. By 2026-10-01 a
correct open read was 99 px off its reference, one bad nudge from UNKNOWN. The
references recorded on 09-22 had no tool behind them; this is that tool.

A REFERENCE IS A SAFETY INPUT, so this refuses unless all of these hold:

  * Iris cam, independently, says the roof is in that state right now
    (open: roof tag gone AND the gold star seen; shut: roof tag at its shut
    position). Recording "open" while the roof is part way would teach the
    north camera that part way is open. Iris cam's read switches the inside
    light on, so it is refused while imaging runs; then, and when the scope
    hides the star, the operator's own eyes stand in (--operator-confirmed).
  * Iris North is AT the recorded pan/tilt pose (the references mean nothing
    at another pointing).
  * The tag decodes in most frames, and the frames agree with each other to
    MAX_SPREAD_PX: a wobbling or moving tag is not a reference.
  * The new position is a re-aim of the old one, not somewhere else: within
    MAX_REAIM_PX of the reference it replaces, the same size to SIDE_TOLERANCE,
    and at least MIN_SEPARATION_PX from the OTHER state's reference.

--write backs the file up first (local/north_roof_marker.<stamp>.json), keeps
the other state untouched, and records what was replaced. Re-record the other
state at the next chance: until both are re-recorded at the same pointing, the
one left alone still carries the old offset.
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

MAX_SPREAD_PX = 5.0
MAX_REAIM_PX = 150.0
MIN_SEPARATION_PX = 400.0
SIDE_TOLERANCE = 0.10
MIN_DECODE_FRACTION = 0.7


def median_corners(per_frame):
    """Per-corner median over frames: [[x, y] x 4]. Pure (no numpy, so CI tests it)."""
    from statistics import median
    return [[round(median(f[k][j] for f in per_frame), 1) for j in (0, 1)]
            for k in range(4)]


def side_px(corners):
    import math
    return math.hypot(corners[0][0] - corners[1][0], corners[0][1] - corners[1][1])


def check(state, per_frame, ref, frames_tried):
    """(new_corners, problems, facts). Pure: no camera, no files."""
    from sentry.north_roof import worst_corner
    other = "shut" if state == "open" else "open"
    tag = str(ref.get("tag_id", 2))
    problems, facts = [], {}
    if len(per_frame) < max(3, MIN_DECODE_FRACTION * frames_tried):
        problems.append("tag decoded in only %d/%d frames" % (len(per_frame), frames_tried))
        return None, problems, facts
    new = median_corners(per_frame)
    spread = max(worst_corner(c, new) for c in per_frame)
    old = ref[state]["markers"][tag]
    oth = ref[other]["markers"][tag]
    facts.update(frames=len(per_frame), spread_px=round(spread, 2),
                 moved_px=round(worst_corner(new, old), 1),
                 from_other_px=round(worst_corner(new, oth), 1),
                 side_px=round(side_px(new), 1), old_side_px=round(side_px(old), 1))
    if spread > MAX_SPREAD_PX:
        problems.append("frames disagree by %.1f px (> %.0f): the tag is not still"
                        % (spread, MAX_SPREAD_PX))
    if facts["moved_px"] > MAX_REAIM_PX:
        problems.append("%.0f px from the old %s reference (> %.0f): not a re-aim"
                        % (facts["moved_px"], state, MAX_REAIM_PX))
    if facts["from_other_px"] < MIN_SEPARATION_PX:
        problems.append("only %.0f px from the %s reference (< %.0f)"
                        % (facts["from_other_px"], other, MIN_SEPARATION_PX))
    if abs(facts["side_px"] / facts["old_side_px"] - 1) > SIDE_TOLERANCE:
        problems.append("tag is %.0f px across, was %.0f: a different distance or zoom"
                        % (facts["side_px"], facts["old_side_px"]))
    return new, problems, facts


def imaging_in_progress():
    """Why imaging may be running, or None. Fails closed."""
    import subprocess
    try:
        with open("imaging.txt") as fh:
            line = fh.readline().strip()
        if line and line != "IMAGING_STATE NONE":
            return "imaging state is %s" % line.split()[-1]
    except OSError:
        pass
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq NINA.exe"],
                             capture_output=True, text=True, timeout=30).stdout or ""
    except Exception:  # noqa: BLE001
        return "could not check for N.I.N.A"
    return "N.I.N.A is running" if "NINA.exe" in out else None


def iris_cam_says(state):
    """(agrees, detail) -- Iris cam's own verdict, with the inside light handled
    exactly as a gating read handles it."""
    from sentry import kasa_state, vision_safety
    _safe, closed, is_open, _when = vision_safety._with_inside_light(
        lambda: kasa_state.kasa_status(verify_pose=True, frames=kasa_state.GATE_FRAMES))
    detail = "Iris cam: closed=%s open=%s" % (closed, is_open)
    if state == "open":
        return bool(is_open and not closed), detail
    return bool(closed and not is_open), detail


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("state", choices=("open", "shut"))
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--write", action="store_true", help="replace the reference")
    ap.add_argument("--operator-confirmed", action="store_true",
                    help="the operator has SEEN the roof in this state; skip Iris cam "
                         "(and its inside light). Every other check still applies.")
    args = ap.parse_args(argv)

    from utils import utils
    utils.set_logger()
    from configs import config
    from scripts import kasa_pose
    from scripts.scope_marker_check import find_markers
    from sentry import north_roof
    from sentry.sky_camera import credentials

    path = north_roof.REFERENCE_PATH
    ref = north_roof.reference(path)
    if ref is None:
        print("no reference file at %s" % path)
        return 2

    if args.operator_confirmed:
        print("roof %s: confirmed by the operator by eye; Iris cam not used" % args.state)
    else:
        busy = imaging_in_progress()
        if busy:
            # The Iris cam read switches the inside light on; during a run that
            # is light in the frame (it happened once, 2026-10-01 20:20:51).
            print("REFUSED: %s -- Iris cam's read would switch the inside light on. "
                  "Confirm the roof by eye and pass --operator-confirmed." % busy)
            return 1
        agrees, why = iris_cam_says(args.state)
        print(why)
        if not agrees:
            print("REFUSED: Iris cam does not confirm the roof is %s" % args.state)
            return 1

    want = tuple(ref["ptz"])
    pose = kasa_pose.ensure_at(ref.get("camera", "Iris North"), want)
    if pose is None or tuple(pose) != want:
        print("REFUSED: Iris North not at %s (got %s)" % (want, pose))
        return 1

    user, pw = credentials(config.data())
    tag = int(ref.get("tag_id", 2))
    per_frame = []
    for i in range(args.frames):
        img = north_roof._grab(ref["host"], user, pw, ref.get("snapshot", "local/north_roof_frame.jpg"),
                               camera=None if i else ref.get("camera", "Iris North"))
        if img is None:
            continue
        found = find_markers(img, ref.get("dict", "APRILTAG_36h11"))
        if tag in found:
            per_frame.append([list(map(float, c)) for c in found[tag]])
    new, problems, facts = check(args.state, per_frame, ref, args.frames)
    print(json.dumps(facts, indent=1))
    if problems:
        print("REFUSED:\n  " + "\n  ".join(problems))
        return 1
    print("new %s reference: %s" % (args.state, new))
    if not args.write:
        print("dry run: nothing written (add --write)")
        return 0

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup = path.replace(".json", ".%s.json" % stamp)
    shutil.copy2(path, backup)
    old = ref[args.state]
    ref[args.state] = {
        "markers": {str(tag): new},
        "frames": facts["frames"],
        "side_px": facts["side_px"],
        "spread_px": facts["spread_px"],
        "recorded": datetime.now().astimezone().isoformat(timespec="seconds"),
        "replaced": {"recorded": old.get("recorded"), "moved_px": facts["moved_px"],
                     "backup": backup},
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(ref, fh, indent=1)
    os.replace(tmp, path)
    print("wrote %s (backup %s)" % (path, backup))

    state, detail = north_roof.read()
    offs = [f.get("off_%s_px" % args.state) for f in detail.get("per_frame", [])]
    print("north read now: %s, %s px off %s" % (state.upper(), offs, args.state))
    return 0 if state == args.state else 1


if __name__ == "__main__":
    sys.exit(main())
