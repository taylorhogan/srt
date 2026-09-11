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
with the scope tag as WITNESS that the camera can see at all, and the gold
STAR on the wall (local/roof_star_open.json) as the positive marker for open:

    roof tag at its shut position     -> SHUT     (positive)
    roof tag gone, scope tag seen,
        open star seen at its place   -> OPEN     (positive: the star is only
                                                  in view when the roof is off)
    roof tag gone, scope tag seen,
        star not seen                 -> UNKNOWN  (consistent with open, but
                                                  nothing positive says so)
    roof tag seen somewhere else      -> UNKNOWN  (unexplained; refuse)
    neither tag                       -> UNKNOWN  (blind: lens, power, pointing)

Until 2026-09-11 the second line needed no star: "tag gone + witness" was
OPEN. The user rejected that on 2026-09-10 ("you need to see the open
star"): a tag that fails to DECODE, a smear over that part of the lens, or a
roof stopped half way all look like absence, and a false OPEN while the roof
is shut is the collision case. The star is the webcam-era gold star on the
dark wall left of the aperture: in view when open, hidden by the roof
structure when shut (8/8 lit open frames on file within 15 px of one spot;
0/9 shut frames, where the nearest yellow thing is pine 60+ px away). It is
found by colour (a star-sized yellow blob within tolerance of the golden
position) AND context (the recorded template -- a bright compact blob on the
dark wall -- correlates there; pine over the window, bright everywhere with
no dark surround, correlates at 0.33 or less), and it needs a COLOUR frame: a gating read switches the inside light on and waits for the
camera to leave greyscale IR before it looks. On an IR frame the star is
"unknown", so OPEN cannot be confirmed without the light.

The old aperture-statistics rule (green excess by day, IR return by night)
is kept ONLY as a veto on OPEN. A gating read takes several frames; OPEN
needs every one of them consistent with open (tag gone, witness seen,
aperture not objecting) and the star seen in at least one, which mirrors
SHUT: a positive read that the other frames cannot undo.

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
STAR_PATH = "local/roof_star_open.json"
STAR_TEMPLATE_PATH = "local/roof_star_template.png"
ROOF_ID = 1
SCOPE_ID = 0

# The open star (see module docstring; recorded by scripts/roof_star_check.py).
# Measured 2026-09-11 on every colour frame in sentry/roof_frames_kasa:
#   open: blob 312-519 px, 0-15 px from the golden position, template ncc
#         0.48-1.00 there (daylight ~0.5-0.6, night-lit 0.92-1.00)
#   shut: nearest yellow blob (pine) 60-83 px away, 487-1838 px, ncc <= 0.33
# Position and shape are each required; neither alone has the margin.
STAR_SEARCH_PX = 80            # half-width of the window searched around the golden spot
STAR_TOLERANCE_PX = 40         # blob centroid must be this close to it
STAR_AREA_PX = (150, 1200)     # star-sized: the pine strips run bigger
STAR_NCC_MIN = 0.40            # template correlation at the blob
STAR_TEMPLATE_MARGIN = 12      # px of wall kept around the star in the template
STAR_HSV_LO, STAR_HSV_HI = (8, 60, 110), (45, 255, 255)   # yellow, lit
# A frame whose channels differ by less than this on average is greyscale IR.
IR_CHROMA = 0.5
# After the inside light goes on the camera stays in IR for a few seconds
# (2026-09-10: three gating reads 7 s after the switch were still IR). A
# gating read waits up to this long for colour before it looks.
COLOUR_WAIT_S = 25.0

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


def _star_reference():
    try:
        with open(STAR_PATH) as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


_star_template_cache: dict = {}


def _star_template(ref):
    """The recorded star crop, or None if the reference names none / it is missing."""
    path = (ref or {}).get("template")
    if not path:
        return None
    mtime = os.path.getmtime(path) if os.path.exists(path) else None
    if mtime is None:
        return None
    hit = _star_template_cache.get(path)
    if hit is None or hit[0] != mtime:
        img = cv2.imread(path)
        if img is None or not img.size:
            return None
        _star_template_cache[path] = hit = (mtime, img)
    return hit[1]


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


def is_ir(img):
    """True when the frame is the camera's greyscale IR output (no colour at all)."""
    small = img[::8, ::8].astype(np.float32)
    return float(np.mean(small.max(axis=2) - small.min(axis=2))) <= IR_CHROMA


def find_star_blob(img, near, search_px, area=STAR_AREA_PX):
    """The star-sized yellow blob nearest *near* within the search window.

    Returns (distance_px, area_px, cx, cy, (x, y, w, h)) in full-frame
    coordinates, or None. Pure image work; no reference needed, so the
    recorder uses it too.
    """
    h, w = img.shape[:2]
    gx, gy = float(near[0]), float(near[1])
    x0, y0 = max(0, int(gx - search_px)), max(0, int(gy - search_px))
    x1, y1 = min(w, int(gx + search_px)), min(h, int(gy + search_px))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, STAR_HSV_LO, STAR_HSV_HI)
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    best = None
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if not (area[0] <= a <= area[1]):
            continue
        cx, cy = float(cent[i][0] + x0), float(cent[i][1] + y0)
        d = float(np.hypot(cx - gx, cy - gy))
        if best is None or d < best[0]:
            bx, by = int(stats[i, cv2.CC_STAT_LEFT] + x0), int(stats[i, cv2.CC_STAT_TOP] + y0)
            best = (d, a, cx, cy, (bx, by, int(stats[i, cv2.CC_STAT_WIDTH]),
                                   int(stats[i, cv2.CC_STAT_HEIGHT])))
    return best


def star_verdict(img, ref, template=None):
    """('seen'|'absent'|'unknown', detail): is the open star at its place?

    'seen' needs a star-sized yellow blob within tolerance of the golden
    position AND the recorded template correlating there. 'absent' is a
    colour frame that shows no such thing (the roof structure is over it, or
    the camera is not looking where it should). 'unknown' is a frame that
    cannot answer: IR, or no reference on file.
    """
    if ref is None:
        return "unknown", {"why": "no open-star reference recorded; run "
                                  "scripts/roof_star_check.py --set-open"}
    if is_ir(img):
        return "unknown", {"why": "camera in greyscale IR; the star needs colour",
                           "ir": True}
    gx, gy = ref["centre"]
    tol = float(ref.get("tolerance_px", STAR_TOLERANCE_PX))
    blob = find_star_blob(img, (gx, gy), int(ref.get("search_px", STAR_SEARCH_PX)))
    if blob is None:
        return "absent", {"why": "no star-sized yellow blob within %d px of (%.0f, %.0f)"
                                 % (int(ref.get("search_px", STAR_SEARCH_PX)), gx, gy)}
    d, a, cx, cy, _bbox = blob
    detail = {"star_px": round(d, 1), "star_area": a, "star_at": (round(cx), round(cy))}
    if d > tol:
        detail["why"] = ("nearest yellow blob is %.0f px from the star's golden "
                         "position (tolerance %.0f)" % (d, tol))
        return "absent", detail
    tmpl = template if template is not None else _star_template(ref)
    if tmpl is None:
        detail["why"] = "star template %s missing; re-run scripts/roof_star_check.py --set-open" \
                        % ref.get("template")
        return "unknown", detail
    th, tw = tmpl.shape[:2]
    h, w = img.shape[:2]
    sx0, sy0 = max(0, int(cx - tw / 2 - tol)), max(0, int(cy - th / 2 - tol))
    sx1, sy1 = min(w, int(cx + tw / 2 + tol)), min(h, int(cy + th / 2 + tol))
    region = img[sy0:sy1, sx0:sx1]
    if region.shape[0] < th or region.shape[1] < tw:
        detail["why"] = "star too close to the frame edge to match the template"
        return "unknown", detail
    ncc = float(cv2.minMaxLoc(cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED))[1])
    detail["star_ncc"] = round(ncc, 2)
    ncc_min = float(ref.get("ncc_min", STAR_NCC_MIN))
    if ncc < ncc_min:
        detail["why"] = ("yellow blob at the star's position but it does not correlate "
                         "with the recorded star (ncc %.2f < %.2f)" % (ncc, ncc_min))
        return "absent", detail
    return "seen", detail


def roof_tag_verdict(found, ref, star=None):
    """('shut'|'open'|'unknown', detail) from the tags in one frame.

    *star* is star_verdict()'s (verdict, detail) for the same frame; OPEN
    needs it to be 'seen'. A frame with the roof tag gone and the witness
    seen but no star is 'unknown' with detail["open_no_star"] set, so
    combine_roof can tell "consistent with open" from "contradicts open".

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
        sv, sd = star if star is not None else ("unknown", {"why": "open star not checked"})
        detail = {k: v for k, v in sd.items() if k != "why"}
        detail["star"] = sv
        if sv == "seen":
            detail["why"] = "roof tag absent, scope tag visible, open star at its place"
            return "open", detail
        detail["why"] = ("roof tag absent while the scope tag is visible, but the open "
                         "star was not seen (%s)" % sd.get("why", sv))
        detail["open_no_star"] = True
        return "unknown", detail
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
    frame read open or open-like (tag gone with the witness seen). OPEN:
    every frame is open-like with the aperture not objecting, and at least
    one of them SAW the star -- a positive read the other frames cannot
    undo, mirroring SHUT. Anything else is unknown.
    """
    vs = [v for v, _ in verdicts]
    if any(d.get("tag_elsewhere") for _, d in verdicts):
        return "unknown"
    open_like = [v == "open" or bool(d.get("open_no_star")) for v, d in verdicts]
    if "shut" in vs and not any(open_like):
        return "shut"
    if vs and all(open_like) and "open" in vs:
        return "open"
    return "unknown"


# ------------------------------------------------------------------ picture

def _annotate(img, found, parked_ref, shut_ref, scope, roof, star_ref=None, star=None):
    """The frame with the tags outlined and the verdict stamped on it."""
    out = img.copy()
    # The open star: golden position as a circle of the tolerance, and where
    # (if anywhere) this frame found it. The picture shows the positive open
    # evidence, or its absence, at a glance.
    if star_ref:
        try:
            gx, gy = (int(v) for v in star_ref["centre"])
            tol = int(star_ref.get("tolerance_px", STAR_TOLERANCE_PX))
            sv, sd = star if star is not None else ("unknown", {})
            col = {"seen": (80, 220, 80), "absent": (40, 40, 240)}.get(sv, (200, 200, 200))
            cv2.circle(out, (gx, gy), tol, col, 3)
            if sd.get("star_at"):
                ax, ay = (int(v) for v in sd["star_at"])
                cv2.drawMarker(out, (ax, ay), col, cv2.MARKER_CROSS, 40, 3)
            what = {"seen": "open star seen", "absent": "open star NOT seen"}.get(
                sv, "open star %s" % sv)
            cv2.putText(out, what, (gx - 150, gy - tol - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, col, 3)
        except Exception:  # noqa: BLE001 -- annotation must never cost a picture
            pass
    colour = {"safe": (80, 220, 80), "shut": (80, 220, 80),
              "UNSAFE": (40, 40, 240), "open": (240, 180, 40)}
    for tid, corners in found.items():
        c = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
        col = colour.get(scope if tid == SCOPE_ID else roof, (200, 200, 200))
        cv2.polylines(out, [c], True, col, 4)
        cx, cy = int(c[:, 0, 0].mean()), int(c[:, 0, 1].mean())
        label = "scope tag" if tid == SCOPE_ID else "roof tag" if tid == ROOF_ID else "tag %d" % tid
        cv2.putText(out, label, (cx - 60, cy - 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3)
    # The roof tag is ABSENT from its shut position when open; draw where it
    # would be, so the picture shows both halves of the open evidence: the
    # tag gone from here and the star seen over there.
    if shut_ref and ROOF_ID not in found:
        try:
            q = np.asarray(shut_ref["markers"][str(ROOF_ID)], dtype=np.int32).reshape(-1, 1, 2)
            col = colour.get(roof, (200, 200, 200))
            for k in range(4):
                a, b_ = tuple(q[k, 0]), tuple(q[(k + 1) % 4, 0])
                # dashed edge: 8 segments per side
                for t in range(0, 8, 2):
                    p0 = (int(a[0] + (b_[0] - a[0]) * t / 8), int(a[1] + (b_[1] - a[1]) * t / 8))
                    p1 = (int(a[0] + (b_[0] - a[0]) * (t + 1) / 8), int(a[1] + (b_[1] - a[1]) * (t + 1) / 8))
                    cv2.line(out, p0, p1, col, 3)
            cx, cy = int(q[:, 0, 0].mean()), int(q[:, 0, 1].mean())
            what = "roof tag absent" if roof == "open" else "roof tag not seen"
            cv2.putText(out, what, (cx - 150, cy - 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
        except Exception:  # noqa: BLE001 -- annotation must never cost a picture
            pass
    banner = "scope %s | roof %s | %s" % (scope, roof,
                                         datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    cv2.rectangle(out, (0, out.shape[0] - 70), (out.shape[1], out.shape[0]), (0, 0, 0), -1)
    cv2.putText(out, banner, (20, out.shape[0] - 22), cv2.FONT_HERSHEY_SIMPLEX,
                1.6, (255, 255, 255), 3)
    if out.shape[1] > VIEW_WIDTH:
        h = int(out.shape[0] * VIEW_WIDTH / out.shape[1])
        out = cv2.resize(out, (VIEW_WIDTH, h), interpolation=cv2.INTER_AREA)
    return out


def _write_view(img, found, parked_ref, shut_ref, scope, roof, star_ref=None, star=None):
    """Write the annotated decision picture to the configured scope_view path."""
    try:
        path = config.data()["camera safety"]["scope_view"]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        cv2.imwrite(path, _annotate(img, found, parked_ref, shut_ref, scope, roof,
                                    star_ref, star),
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
    star_ref = _star_reference()
    dict_name = (parked_ref or {}).get("dict", "APRILTAG_36h11")

    n = 1 if quick else max(1, int(frames))
    scope_v, roof_v, per_frame = [], [], []
    img = found = star = None
    for i in range(n):
        if i:
            time.sleep(0.5)
        img_i = _grab(retries=1, timeout=8, wait_stream=False) if quick else _grab()
        if img_i is not None and i == 0 and not quick and star_ref and is_ir(img_i):
            img_i = _await_colour(img_i)
        if img_i is None:
            per_frame.append({"camera": False})
            continue
        img = img_i
        found = find_markers(img, dict_name)
        sv, sd = _scope_verdict(found, parked_ref, verified)
        star = star_verdict(img, star_ref)
        tv, td = roof_tag_verdict(found, shut_ref, star)
        hint, hd = _aperture_hint(img)
        rv, rd = roof_verdict_with_veto(tv, td, hint)
        rd.update({k: v for k, v in hd.items() if k in ("regime", "sun_alt")})
        scope_v.append((sv, sd))
        roof_v.append((rv, rd))
        per_frame.append({"tags": sorted(found), "scope": sv, "roof": rv,
                          "roof_tag": tv, "aperture": hint, "star": star[0],
                          "star_px": star[1].get("star_px")})

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
    star_seen = sum(1 for f in per_frame if f.get("star") == "seen")
    last_detail = {"camera": True, "scope": scope, "scope_detail": sdet,
                   "roof": roof, "roof_detail": rdet,
                   "star": star[0] if star else "unknown", "star_seen": star_seen,
                   "frames": n, "per_frame": per_frame,
                   "pose_verified": verified is not None}
    _write_view(img, found, parked_ref, shut_ref, scope, roof, star_ref, star)
    # Wording coupled to _KASA_RE in iris/conductor/shadow.py; change together.
    # The offsets ride on the line so the stops' repeatability accumulates in
    # the log for free: the roof tag is 1.2 px/mm, so a shut-stop drift shows
    # here long before anyone would notice it by eye.
    star_word = ("seen %d/%d" % (star_seen, n) if star_seen
                 else "n/a (%s)" % (star[1].get("why", "unknown") if star else "unknown")
                 if (star and star[0] == "unknown") else "not seen")
    _logger.info("kasa_status: scope=%s roof=%s (%s; %d/%d frames decoded a tag; pose %s; "
                 "scope tag %.1f px off park; roof tag %s; open star %s)",
                 scope, roof, rdet.get("regime", "?"),
                 sum(1 for f in per_frame if f.get("tags")), n,
                 "verified" if verified is not None else "unverified",
                 float(sdet.get("worst_corner_px", 0.0) or 0.0),
                 ("%.1f px off shut" % rdet["worst_corner_px"]) if "worst_corner_px" in rdet else "not seen",
                 star_word)
    return scope == "safe", roof == "shut", roof == "open", when


def _await_colour(first):
    """Regrab until the camera leaves IR, up to COLOUR_WAIT_S; return the last frame.

    The inside light is already on when a gating read starts, but the camera
    takes a few seconds to switch out of greyscale, and the open star cannot
    be seen in greyscale. Returns an IR frame if the wait runs out (the tags
    still decode; OPEN will then be refused for want of the star).
    """
    deadline = time.time() + COLOUR_WAIT_S
    img = first
    t0 = time.time()
    while time.time() < deadline:
        time.sleep(2.0)
        g = _grab(retries=1, timeout=8, wait_stream=False)
        if g is None:
            continue
        img = g
        if not is_ir(g):
            _logger.info("kasa_status: camera switched to colour %.0f s after the light",
                         time.time() - t0)
            return g
    _logger.warning("kasa_status: camera still in IR %.0f s after the light; the open "
                    "star cannot be checked on a greyscale frame", COLOUR_WAIT_S)
    return img


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
