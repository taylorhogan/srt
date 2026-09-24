"""north_roof.py -- the roof's position from the north camera, as a positive decode.

    state, detail = north_roof.read()        # 'shut' | 'open' | 'unknown'

WHY THIS EXISTS. Until now OPEN meant "the roof tag is absent AND the gold star
is visible". Absence is the weakest evidence there is -- a tag that fails to
decode looks exactly like a tag that is not there -- and the star was bolted on
to supply the positive half. It is a template correlation on a painted wall, and
on 2026-09-17 late-morning sun washed the wall out: the star scored 0.37 against
a 0.40 floor, the open never confirmed, the conductor faulted, and under roof
authority the close had to be forced. Before that, on 2026-09-17 00:33, the
unparked scope stood between Iris cam and BOTH tags and the star, so a stop!
could not tell where the roof was at all.

Iris North looks south from the north wall at a tag on the moving roof truss,
and reads it at BOTH ends of travel. So each state is a decode at a recorded
position, and absence never means anything:

    tag at the shut corners   -> SHUT
    tag at the open corners   -> OPEN
    anything else             -> UNKNOWN, and every caller refuses

Measured 2026-09-22 (local/north_roof_marker.json): shut 72.6 px at
(1011, 854), open 350.7 px at (992, 355), 650 px apart on the worst matched
corner, frame-to-frame scatter <= 3 px, and after a full open/close the shut
reading reproduced to 1-2 px against a reference taken before the cycle.

NO LIGHT, NO COLOUR WAIT. The star needed colour, which is why a gating read on
Iris cam switches the inside light on and waits up to 25 s to leave greyscale.
An AprilTag does not care: measured 6/6 decodes in pure IR in an unlit building
on 2026-09-21, with the IR and colour readings agreeing to 0.7 px. This read
therefore touches no lights and leaves nothing to restore -- which matters here
because Iris North takes ~9 s to leave IR, against ~3 s for Iris cam.

THE CAMERA IS DRIVEN TO ITS POSE FIRST, every time, and a pose it cannot reach
is UNKNOWN rather than a verdict. The references are pixel coordinates and mean
nothing at another pointing; the Kasa app moves this camera whenever someone
looks at it.
"""
import json
import logging
import math
import os
import sys

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

_logger = logging.getLogger(__name__)

REFERENCE_PATH = "local/north_roof_marker.json"
# Every frame of every read is kept, because the one read that mattered on
# 2026-09-24 (0/3 decodes at 04:13, a minute after 3/3) had its frames
# overwritten by the next read before anyone could look. Same lesson as the
# 2026-09-17 stop! frame that survived only as a Pushover attachment.
ARCHIVE_DIR = "local/north_archive"
ARCHIVE_KEEP_DAYS = 60
FRAMES = 3
GRAB_RETRIES = 2          # per frame; a dropped stream is noise, not a verdict

# Introspection for callers that want to log why, mirroring kasa_state.last_detail.
last_detail: dict = {}


# ------------------------------------------------------------------ pure

def worst_corner(corners, reference):
    """Largest distance between matched corners, in pixels. Pure.

    Matched, not nearest: the detector returns corners in a consistent order,
    so comparing them pairwise also catches a tag that is the right size in the
    right place but rotated, which a centre-and-size comparison would pass.
    """
    return max(math.hypot(a[0] - b[0], a[1] - b[1])
               for a, b in zip(corners, reference))


def classify(corners, shut_ref, open_ref, tolerance):
    """'shut' | 'open' | 'elsewhere' for one frame's decoded tag. Pure.

    'elsewhere' is a real answer, not a failure: the roof caught part way
    through its travel puts the tag in the wide band that belongs to neither
    reference, and that must read as unknown rather than snapping to whichever
    end happens to be nearer.
    """
    d_shut = worst_corner(corners, shut_ref)
    d_open = worst_corner(corners, open_ref)
    if d_shut <= tolerance and d_shut <= d_open:
        return "shut"
    if d_open <= tolerance:
        return "open"
    return "elsewhere"


def decide(north_state, cam_closed, cam_open):
    """(closed, is_open, source): the roof from both cameras. Pure.

    Operator decision 2026-09-24: the north tag DECIDES. Iris cam may veto
    but never overrule it, and the gold star is a fallback for when the north
    camera cannot answer at all -- so a dead Iris North degrades to the
    behaviour of 2026-09-23, not to "cannot confirm open", which under roof
    authority would mean a forced close.

      north OPEN  and Iris cam SHUT  -> contradiction, refuse
      north SHUT  and Iris cam OPEN  -> contradiction, refuse
      north OPEN                     -> open   (a positive decode; the star is not consulted)
      north SHUT                     -> closed (Iris cam's tag 1 agreeing is a second witness)
      north unknown                  -> whatever Iris cam says, as before
    """
    north_open = north_state == "open"
    north_shut = north_state == "shut"
    if (north_open and cam_closed) or (north_shut and cam_open):
        return False, False, "contradiction"
    if north_open:
        return False, True, "north"
    if north_shut:
        return True, False, "north+cam" if cam_closed else "north"
    if cam_closed or cam_open:
        return bool(cam_closed), bool(cam_open), "cam-fallback"
    return False, False, "none"


def combine(frames):
    """One verdict from per-frame answers ('shut'/'open'/'elsewhere'/'absent'/'blind').

    Deliberately asymmetric, for the same reason kasa_state is: a false OPEN is
    the collision case -- it is what lets the mount move -- while a false SHUT
    costs at worst a roof left closed. So

      OPEN  needs EVERY frame to say open, and at least one frame at all.
      SHUT  needs one frame at the shut corners and no frame contradicting it;
            a frame that simply failed to decode is noise and is tolerated.

    Any 'elsewhere', any disagreement between the two ends, and any read with
    no usable frame is unknown.
    """
    seen = [f for f in frames if f != "blind"]
    if not seen:
        return "unknown"
    if "elsewhere" in seen:
        return "unknown"
    if "open" in seen and "shut" in seen:
        return "unknown"
    if seen and all(f == "open" for f in seen) and len(seen) == len(frames):
        return "open"
    if "shut" in seen and "open" not in seen:
        return "shut"
    return "unknown"


# ------------------------------------------------------------------ io

def reference(path=REFERENCE_PATH):
    """The recorded shut/open corners, or None if they are not on file."""
    try:
        with open(path) as fh:
            ref = json.load(fh)
        if not (ref.get("shut") and ref.get("open")):
            return None
        return ref
    except (OSError, ValueError):
        return None


def _grab(host, user, pw, path, retries=GRAB_RETRIES):
    import contextlib
    import io as _io
    import cv2
    from scripts.probe_kasa_camera import probe_kc_stream
    for _ in range(retries + 1):
        with contextlib.redirect_stdout(_io.StringIO()):
            ok = probe_kc_stream(host, user, pw, snapshot_path=path, timeout=25)
        if ok:
            img = cv2.imread(path)
            if img is not None and img.size:
                return img
    return None


VIEW_WIDTH = 1280        # Pushover caps attachments at 2.5 MB; a full 2560 frame sits near it


def _write_view(img, corners, state, detail_line, view_path):
    """The picture the operator sees: the north frame with the tag and verdict drawn on."""
    try:
        import cv2
        import numpy as np
        out = img.copy()
        if corners is not None:
            pts = np.array(corners, np.int32).reshape(-1, 1, 2)
            colour = (0, 220, 0) if state in ("shut", "open") else (0, 140, 255)
            cv2.polylines(out, [pts], True, colour, 4)
        colour = (0, 220, 0) if state in ("shut", "open") else (0, 0, 255)
        cv2.putText(out, "Iris North  roof: %s" % state.upper(), (30, out.shape[0] - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, colour, 4)
        cv2.putText(out, detail_line, (30, out.shape[0] - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 3)
        if out.shape[1] > VIEW_WIDTH:
            h = int(out.shape[0] * VIEW_WIDTH / out.shape[1])
            out = cv2.resize(out, (VIEW_WIDTH, h), interpolation=cv2.INTER_AREA)
        os.makedirs(os.path.dirname(view_path) or ".", exist_ok=True)
        cv2.imwrite(view_path, out, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return view_path
    except Exception:  # noqa: BLE001 -- a picture must never cost a verdict
        _logger.exception("north_roof: could not write the decision picture")
        return None


def read(frames=FRAMES, path=REFERENCE_PATH):
    """(state, detail): 'shut' | 'open' | 'unknown' from the north camera.

    Never raises and never fabricates: every failure returns 'unknown' with
    detail saying why, and callers refuse exactly as they do on a blind
    vision read.
    """
    global last_detail
    ref = reference(path)
    if ref is None:
        last_detail = {"why": "no north roof reference on file (%s)" % path}
        _logger.warning("north_roof: %s", last_detail["why"])
        return "unknown", dict(last_detail)

    tag = int(ref.get("tag_id", 2))
    tol = float(ref.get("tolerance_px", 100))
    shut_ref = ref["shut"]["markers"][str(tag)]
    open_ref = ref["open"]["markers"][str(tag)]

    from scripts import kasa_pose
    want = tuple(ref["ptz"])
    pose = kasa_pose.ensure_at(ref.get("camera", "Iris North"), want)
    if pose is None or tuple(pose) != want:
        last_detail = {"why": "camera not at %s (got %s); the references mean "
                              "nothing at another pointing" % (want, pose)}
        _logger.warning("north_roof: %s", last_detail["why"])
        return "unknown", dict(last_detail)

    from configs import config
    from sentry.sky_camera import credentials
    from scripts.scope_marker_check import find_markers
    user, pw = credentials(config.data())
    snap = ref.get("snapshot", "local/north_roof_frame.jpg")

    from datetime import datetime, timedelta
    from sentry.kasa_state import is_ir
    stamp = datetime.now()
    day = os.path.join(ARCHIVE_DIR, stamp.strftime("%Y-%m-%d"))
    per_frame, verdicts, kept = [], [], []
    shown = None                              # (img, corners, off_shut, off_open) of the deciding frame
    for i in range(max(1, frames)):
        img = _grab(ref["host"], user, pw, snap)
        if img is None:
            verdicts.append("blind")
            per_frame.append({"camera": False})
            continue
        ir = bool(is_ir(img))
        found = find_markers(img, ref.get("dict", "APRILTAG_36h11"))
        if tag not in found:
            v = "absent"
            per_frame.append({"camera": True, "ir": ir, "tags": sorted(found), "tag": False})
            if shown is None:
                shown = (img, None, None, None)
        else:
            corners = [list(map(float, c)) for c in found[tag]]
            v = classify(corners, shut_ref, open_ref, tol)
            off_s, off_o = worst_corner(corners, shut_ref), worst_corner(corners, open_ref)
            per_frame.append({"camera": True, "ir": ir, "tags": sorted(found), "verdict": v,
                              "off_shut_px": round(off_s, 1), "off_open_px": round(off_o, 1)})
            shown = (img, corners, off_s, off_o)
        verdicts.append(v)
        try:
            import cv2
            os.makedirs(day, exist_ok=True)
            name = "%s_%d_%s%s.jpg" % (stamp.strftime("%H%M%S"), i, v, "_ir" if ir else "")
            cv2.imwrite(os.path.join(day, name), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            kept.append(os.path.join(day, name))
        except Exception:  # noqa: BLE001 -- the archive must never cost a verdict
            _logger.warning("north_roof: could not archive a frame", exc_info=True)
    try:
        import shutil
        cutoff = (stamp - timedelta(days=ARCHIVE_KEEP_DAYS)).strftime("%Y-%m-%d")
        for d in (os.listdir(ARCHIVE_DIR) if os.path.isdir(ARCHIVE_DIR) else []):
            if len(d) == 10 and d < cutoff:
                shutil.rmtree(os.path.join(ARCHIVE_DIR, d), ignore_errors=True)
    except Exception:  # noqa: BLE001
        pass

    state = combine(verdicts)
    view = None
    if shown is not None:
        img, corners, off_s, off_o = shown
        decoded = sum(1 for f in per_frame if f.get("verdict"))
        if corners is None:
            line = "tag not decoded in %d/%d frames  %s" % (len(per_frame) - decoded, len(per_frame),
                                                            stamp.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            line = ("tag %.0f px off shut, %.0f px off open | %d/%d frames | %s"
                    % (off_s, off_o, decoded, len(per_frame), stamp.strftime("%Y-%m-%d %H:%M:%S")))
        view_path = (config.data().get("camera safety", {}).get("scope_view")
                     or "./base_images/scope_view.jpg")
        view = _write_view(img, corners, state, line, view_path)
    last_detail = {"state": state, "verdicts": verdicts, "per_frame": per_frame,
                   "pose": list(want), "tolerance_px": tol, "archive": kept, "view": view}
    _logger.info("north_roof: %s (%s; tolerance %.0f px; %d/%d frames decoded id %d)",
                 state, ", ".join(verdicts), tol,
                 sum(1 for f in per_frame if f.get("verdict")), len(per_frame), tag)
    return state, dict(last_detail)


def main():
    from utils import utils
    utils.set_logger()
    state, detail = read(frames=int(sys.argv[1]) if len(sys.argv) > 1 else FRAMES)
    print("roof (north camera): %s" % state.upper())
    for f in detail.get("per_frame", []):
        print("   %s" % f)
    if detail.get("why"):
        print("   %s" % detail["why"])
    return 0 if state in ("shut", "open") else 1


if __name__ == "__main__":
    sys.exit(main())
