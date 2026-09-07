"""
kasa_state.py
Scope and roof state from the inside Kasa camera, in one verdict.

    safe, closed, is_open, when = kasa_state.kasa_status(verify_pose=True, frames=3)

SINCE 2026-09-07 THIS IS THE SAFETY EYE. The scope-top webcam and its
microphone were retired that day (one USB device; they always failed
together, and a bump to it on 2026-09-06 cost a night -- see the webcam-bump
note in vision_safety). vision_safety.visual_status(), the single documented
entry point for every roof/mount decision, now answers from here. From
2026-08-20 to then this ran shadow-only beside the webcam.

Two independent read-outs from each frame, both POSITIVE readings:

SCOPE -- the AprilTag (id 0) on the OTA against the parked reference recorded
from a real PWI4 mount_park() (local/scope_marker_parked.json). Corners repeat
to 0.1-0.5 px against a 187 px tolerance, works day and night and survives
the flat panel lit beside it. Structurally better than mount telemetry: a
mount power cycle destroys PWI4's home reference and get_is_parked() then
returns False while the scope sits parked; the tag measures the physical
scope against a fixed camera and does not care. First off-park reading
2026-09-06 21:26, tracking squid: 1076 px out. Not subtle.

ROOF -- the AprilTag (id 1) on the roof panel, seen only when the roof is
over the aperture, against the shut reference (local/roof_marker_shut.json),
with the scope tag as WITNESS that the camera can see at all:

    roof tag at its shut position     -> SHUT     (positive)
    roof tag gone, scope tag seen     -> OPEN     (positive: the camera can
                                                  see, and the roof is not there)
    roof tag seen somewhere else      -> UNKNOWN  (unexplained; refuse)
    neither tag                       -> UNKNOWN  (blind: lens, power, pointing)

The old aperture-statistics rule (green excess by day, IR return by night)
is kept ONLY as a veto on OPEN: a tag that fails to DECODE is also "absent",
and in IR at night that happens on a real fraction of frames. A false OPEN
while the roof is shut is the collision case, so an OPEN that the aperture
contradicts is UNKNOWN, and a gating read takes several frames and needs
every one of them to say open.

POSE FIRST. Both references describe the scene from ONE camera pose, so a
gating read drives the camera there (cloud round trip) before it looks, and
refuses if it cannot. A record of where the camera was last put is not
knowledge of where it is (2026-09-05: record said (-123,363), camera sat at
(-666,306) after the phone app moved it).

ONE STREAM. The camera serves a single stream, and the roof-move audio
capture (kasa_audio) holds it for ~50 s from just before the relay fires. A
grab for a gating read waits for that capture to finish rather than reading
a busy camera as unknown.

Every ambiguous path returns unknown, and unknown always means "refuse to
move anything".

    python -m sentry.kasa_state              # one-shot verdict, printed
    python -m sentry.kasa_state --gate       # as the roof gates read it
"""
import contextlib
import io
import json
import os
import sys
import time
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
SHUT_PATH = "local/roof_marker_shut.json"
ROOF_ID = 1
SCOPE_ID = 0

# Frames per GATING read. Decode failures are per-frame noise; requiring every
# frame to agree on OPEN turns a 16 % single-frame miss (the measured IR rate)
# into ~0.4 %, before the aperture veto. SHUT needs only one decoded tag at
# the shut position: a decoded tag is positive evidence the others cannot undo.
GATE_FRAMES = 3
# A grab waits this long for a roof-move audio capture to release the stream.
STREAM_WAIT_S = 75.0

# The aperture: roof underside when shut, sky when open. Same box as
# roof_region_stats.py and iriscam_shadow_log.py so every log stays comparable.
REGION_Y = (100, 1400)
REGION_X = (0, 700)

# Aperture rule thresholds (now a veto only; see module docstring).
GREEN_OPEN = +1.0
GREEN_SHUT = -1.0
NIGHT_P99_OPEN, NIGHT_P99_SHUT = 170.0, 180.0
NIGHT_EDGE_OPEN, NIGHT_EDGE_SHUT = 1.8, 2.0
SUN_DAY_DEG = 0.0

RETRIES = 3

# Width of the annotated decision picture written to cfg["camera safety"]
# ["scope_view"] (what Pushover attaches; its cap is 2.5 MB, a full 2560 px
# frame sits near it).
VIEW_WIDTH = 1280

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


def _grab(retries=RETRIES, timeout=20, wait_stream=True):
    from scripts.probe_kasa_camera import probe_kc_stream
    from sentry.sky_camera import credentials
    if wait_stream:
        try:
            from sentry import kasa_audio
            if not kasa_audio.wait_stream_free(STREAM_WAIT_S):
                _logger.warning("kasa_state: stream still held by an audio capture "
                                "after %.0f s; grabbing anyway", STREAM_WAIT_S)
        except Exception:  # noqa: BLE001 -- the wait is a courtesy, never a gate
            pass
    user, pw = credentials(config.data())
    for _ in range(retries):
        with contextlib.redirect_stdout(io.StringIO()):
            ok = probe_kc_stream(HOST, user, pw, snapshot_path=SNAP_PATH, timeout=timeout)
        if ok:
            img = cv2.imread(SNAP_PATH)
            if img is not None and img.size:
                return img
    return None


# ------------------------------------------------------------------ references

def _parked_reference():
    from scripts.scope_marker_check import PARKED_PATH
    if not os.path.exists(PARKED_PATH):
        return None
    with open(PARKED_PATH) as fh:
        return json.load(fh)


def _shut_reference():
    try:
        with open(SHUT_PATH) as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


def _required_pose():
    """(camera name, (x, y)) the park reference was recorded at, or None."""
    try:
        st = _parked_reference()
        ptz = st.get("ptz") if st else None
        return (st.get("camera", "Iris cam"), tuple(ptz)) if ptz else None
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ per-frame

def _scope_verdict(found, parked_ref, pose):
    """'safe' / 'UNSAFE' / 'unknown' from the scope tag, plus detail."""
    from scripts.scope_marker_check import compare
    from scripts import kasa_pose
    if parked_ref is None:
        return "unknown", {"why": "no parked reference on file"}
    parked = {int(k): np.array(v) for k, v in parked_ref["markers"].items()}
    ref_pose = parked_ref.get("ptz")
    now_pose = pose or kasa_pose.last(parked_ref.get("camera", "Iris cam"))
    verdict, detail = compare(found, parked, ref_pose, now_pose)
    detail["pose_verified"] = pose is not None
    return verdict, detail


def roof_tag_verdict(found, ref):
    """('shut'|'open'|'unknown', detail) from the tags in one frame.

    Pure: the pose is established once per status call, before any frame is
    taken, so it is not re-checked here.
    """
    if ref is None:
        return "unknown", {"why": "no shut reference recorded; run "
                                  "scripts/roof_marker_check.py --set-shut"}
    roof_seen = ROOF_ID in found
    scope_seen = SCOPE_ID in found
    if roof_seen:
        ref_c = np.array(ref["markers"][str(ROOF_ID)])
        d = float(np.linalg.norm(np.array(found[ROOF_ID]) - ref_c, axis=1).max())
        tol = float(ref.get("tolerance_px", 200.0))
        if d <= tol:
            return "shut", {"worst_corner_px": round(d, 1), "witness": scope_seen}
        # Visible but not where the roof puts it when shut. Not "open" -- open
        # means the tag is GONE -- but something unexplained, so refuse.
        return "unknown", {"why": "roof tag found %.0f px from its shut position "
                                  "(tolerance %.0f)" % (d, tol),
                           "worst_corner_px": round(d, 1), "tag_elsewhere": True}
    if scope_seen:
        return "open", {"why": "roof tag absent while the scope tag is visible"}
    return "unknown", {"why": "neither tag visible; the camera is blind"}


def _aperture_hint(img):
    """The pre-tag aperture rule: 'open' / 'shut' / 'unknown' plus detail.
    A veto on OPEN only; it decides nothing on its own any more."""
    reg = img[REGION_Y[0]:REGION_Y[1], REGION_X[0]:REGION_X[1]]
    f = reg.astype(np.float64)
    grey = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY)
    chroma = float(np.mean(np.max(f, 2) - np.min(f, 2)))
    detail = {"chroma": round(chroma, 2)}
    try:
        sun = _sun_altitude()
        detail["sun_alt"] = round(sun, 1)
    except Exception:  # noqa: BLE001
        return "unknown", detail
    if sun >= SUN_DAY_DEG:
        b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
        ge = float((g - (b + r) / 2.0).mean())
        detail.update(regime="day", green_excess=round(ge, 2))
        return ("open" if ge >= GREEN_OPEN else "shut" if ge <= GREEN_SHUT
                else "unknown"), detail
    if chroma > 0.5:
        detail.update(regime="night-lit")
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
    return "unknown", detail


def roof_verdict_with_veto(tag_verdict, tag_detail, hint):
    """Apply the aperture veto: an OPEN the aperture calls SHUT is UNKNOWN."""
    detail = dict(tag_detail, aperture=hint)
    if tag_verdict == "open" and hint == "shut":
        detail["why"] = ("roof tag absent but the aperture reads shut -- a tag "
                         "that failed to decode looks the same as a roof that "
                         "is not there; refusing")
        return "unknown", detail
    return tag_verdict, detail


# ------------------------------------------------------------------ combining

def combine_scope(verdicts):
    """One scope verdict from several frames.

    UNSAFE anywhere is UNSAFE: an off-park scope decoded in one frame is real.
    Otherwise safe needs at least one positive read; frames where the tag did
    not decode abstain rather than contradict.
    """
    vs = [v for v, _ in verdicts]
    if "UNSAFE" in vs:
        return "UNSAFE"
    if "safe" in vs:
        return "safe"
    return "unknown"


def combine_roof(verdicts):
    """One roof verdict from several frames.

    SHUT: at least one frame decoded the roof tag at its shut position and no
    frame saw it elsewhere. OPEN: every frame read open (tag gone with the
    witness seen, aperture not objecting). Anything else is unknown.
    """
    vs = [v for v, _ in verdicts]
    if any(d.get("tag_elsewhere") for _, d in verdicts):
        return "unknown"
    if "shut" in vs and "open" not in vs:
        return "shut"
    if vs and all(v == "open" for v in vs):
        return "open"
    return "unknown"


# ------------------------------------------------------------------ picture

def _annotate(img, found, parked_ref, shut_ref, scope, roof):
    """The frame with the tags outlined and the verdict stamped on it."""
    out = img.copy()
    colour = {"safe": (80, 220, 80), "shut": (80, 220, 80),
              "UNSAFE": (40, 40, 240), "open": (240, 180, 40)}
    for tid, corners in found.items():
        c = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
        col = colour.get(scope if tid == SCOPE_ID else roof, (200, 200, 200))
        cv2.polylines(out, [c], True, col, 4)
        cx, cy = int(c[:, 0, 0].mean()), int(c[:, 0, 1].mean())
        label = "scope tag" if tid == SCOPE_ID else "roof tag" if tid == ROOF_ID else "tag %d" % tid
        cv2.putText(out, label, (cx - 60, cy - 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3)
    banner = "scope %s | roof %s | %s" % (scope, roof,
                                         datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    cv2.rectangle(out, (0, out.shape[0] - 70), (out.shape[1], out.shape[0]), (0, 0, 0), -1)
    cv2.putText(out, banner, (20, out.shape[0] - 22), cv2.FONT_HERSHEY_SIMPLEX,
                1.6, (255, 255, 255), 3)
    if out.shape[1] > VIEW_WIDTH:
        h = int(out.shape[0] * VIEW_WIDTH / out.shape[1])
        out = cv2.resize(out, (VIEW_WIDTH, h), interpolation=cv2.INTER_AREA)
    return out


def _write_view(img, found, parked_ref, shut_ref, scope, roof):
    """Write the annotated decision picture to the configured scope_view path."""
    try:
        path = config.data()["camera safety"]["scope_view"]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        cv2.imwrite(path, _annotate(img, found, parked_ref, shut_ref, scope, roof),
                    [cv2.IMWRITE_JPEG_QUALITY, 82])
    except Exception:  # noqa: BLE001 -- a picture must never cost a verdict
        _logger.exception("kasa_state: could not write the decision picture")


# ------------------------------------------------------------------ entry

def kasa_status(quick=False, verify_pose=False, frames=1):
    """(safe, closed, is_open, when) -- vision_safety.visual_status()'s shape.

    verify_pose=True DRIVES the camera to the pose the references were recorded
    at before grabbing, and refuses if it cannot get there. Every gating read
    passes it. frames>1 takes that many frames and combines them (see
    combine_scope / combine_roof); the roof gates use GATE_FRAMES.

    quick=True is ONE short grab with no stream wait, for shadow sampling.

    safe means "the scope is within tolerance of the recorded park pose". Any
    failure anywhere returns (False, False, False, when): unconfirmed on every
    axis, and last_detail says why.
    """
    global last_detail
    when = datetime.now().astimezone()

    verified = None
    if verify_pose:
        from scripts import kasa_pose
        need = _required_pose()
        if need is None:
            last_detail = {"camera": False,
                           "why": "park reference records no camera pose; "
                                  "re-run scope_marker_check.py --set-parked"}
            _logger.warning("kasa_status: %s", last_detail["why"])
            return False, False, False, when
        verified = kasa_pose.ensure_at(need[0], need[1])
        if verified is None:
            last_detail = {"camera": False,
                           "why": "could not place %r at %s; pose unknown"
                                  % (need[0], need[1])}
            _logger.warning("kasa_status: %s", last_detail["why"])
            return False, False, False, when

    from scripts.scope_marker_check import find_markers
    parked_ref = _parked_reference()
    shut_ref = _shut_reference()
    dict_name = (parked_ref or {}).get("dict", "APRILTAG_36h11")

    n = 1 if quick else max(1, int(frames))
    scope_v, roof_v, per_frame = [], [], []
    img = found = None
    for i in range(n):
        if i:
            time.sleep(0.5)
        img_i = _grab(retries=1, timeout=8, wait_stream=False) if quick else _grab()
        if img_i is None:
            per_frame.append({"camera": False})
            continue
        img = img_i
        found = find_markers(img, dict_name)
        sv, sd = _scope_verdict(found, parked_ref, verified)
        tv, td = roof_tag_verdict(found, shut_ref)
        hint, hd = _aperture_hint(img)
        rv, rd = roof_verdict_with_veto(tv, td, hint)
        rd.update({k: v for k, v in hd.items() if k in ("regime", "sun_alt")})
        scope_v.append((sv, sd))
        roof_v.append((rv, rd))
        per_frame.append({"tags": sorted(found), "scope": sv, "roof": rv,
                          "roof_tag": tv, "aperture": hint})

    if img is None:
        last_detail = {"camera": False, "frames": n, "per_frame": per_frame}
        _logger.warning("kasa_status: no frame from the camera (%d attempts)", n)
        return False, False, False, when

    scope = combine_scope(scope_v)
    roof = combine_roof(roof_v)
    # Detail from the last frame that decoded something, for the callers'
    # captions; the per-frame list carries the rest.
    sdet = next((d for v, d in reversed(scope_v) if v != "unknown"), scope_v[-1][1])
    rdet = next((d for v, d in reversed(roof_v) if v != "unknown"), roof_v[-1][1])
    last_detail = {"camera": True, "scope": scope, "scope_detail": sdet,
                   "roof": roof, "roof_detail": rdet,
                   "frames": n, "per_frame": per_frame,
                   "pose_verified": verified is not None}
    _write_view(img, found, parked_ref, shut_ref, scope, roof)
    # Wording coupled to _KASA_RE in iris/conductor/shadow.py; change together.
    _logger.info("kasa_status: scope=%s roof=%s (%s; %d/%d frames decoded a tag; pose %s)",
                 scope, roof, rdet.get("regime", "?"),
                 sum(1 for f in per_frame if f.get("tags")), n,
                 "verified" if verified is not None else "unverified")
    return scope == "safe", roof == "shut", roof == "open", when


def main():
    gate = "--gate" in sys.argv
    safe, closed, is_open, when = kasa_status(verify_pose=gate,
                                              frames=GATE_FRAMES if gate else 1)
    print("scope : %s" % ("SAFE (at park pose)" if safe else
                          last_detail.get("scope", "unknown")))
    print("roof  : %s" % ("CLOSED" if closed else ("OPEN" if is_open else "unknown")))
    for k, v in (last_detail.get("roof_detail") or {}).items():
        print("   %-12s %s" % (k, v))
    sd = last_detail.get("scope_detail") or {}
    if "worst_corner_px" in sd:
        print("   %-12s %.1f px" % ("scope_off", sd["worst_corner_px"]))
    for f in last_detail.get("per_frame") or []:
        print("   frame: %s" % f)
    return 0 if (safe and (closed or is_open)) else 1


if __name__ == "__main__":
    sys.exit(main())
