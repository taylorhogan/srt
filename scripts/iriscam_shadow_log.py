"""
iriscam_shadow_log.py
Log what the inside Kasa cam WOULD have said about roof and scope state, all
night, alongside nothing else. Read-only. Decides nothing.

The goal is to retire the safety webcam and its microphone and sense both roof
and scope state from the Kasa cam alone. Scope state is close: the AprilTag on
the printed plate detects 6/6 in daylight and 5/5 at night, repeats to 0.3 px,
and still decodes when the frame is shrunk 3.3x. Roof state is not close, and
the reason is specific rather than vague:

    the ONLY validated roof discriminator is green_excess = g - (b+r)/2 in the
    aperture, and at night this camera emits a true greyscale frame. Measured
    on five night captures: chroma identically 0.00, green_excess identically
    0.00. Open or shut, the number is the same number. It is not degraded at
    night, it is dead at night.

So a night discriminator has to be found, and there is no night data with the
roof open to find it in. Tonight is imaging, so the roof opens after dusk and
shuts before dawn -- the first open/shut pair this camera will ever have seen
in the dark. That is what this script is for. It is a recorder, not a detector.

WHAT IT MUST NOT DO, and why each one is a real hazard here:

  * It must never turn on the inside light. get_status_with_lights() does, by
    design, because roof moves happen when nobody is imaging. This runs DURING
    imaging. A light in the observatory ruins every sub being taken. The
    capture path used here (probe_kc_stream) never touches the light.
  * It must never move anything, and it imports nothing that can.
  * It must never be load-bearing. Nothing reads its output at runtime.
  * It must survive the camera being unreachable, the mount losing power at
    end.bat, and its own bugs -- a traceback in a logger must not be the reason
    a night is lost, so the sample loop swallows everything.

Roof ground truth is deliberately NOT collected here. Asking vision_safety
would drive the other camera on its own schedule and, worse, is exactly the
system being replaced -- circular. The roof's open and shut times come out of
iris.log afterwards, where the relay writes them.

    python scripts/iriscam_shadow_log.py                  # until dawn
    python scripts/iriscam_shadow_log.py --minutes 30      # a bounded test
"""
import argparse
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

LOG_PATH = "local/iriscam_shadow_log.jsonl"
FRAME_DIR = "local/iriscam_shadow"
HOST = "192.168.87.65"

# The aperture: roof underside when shut, sky when open. Same box as
# roof_region_stats.py -- the two logs have to be comparable.
REGION_Y = (100, 1400)
REGION_X = (0, 700)

# Keep a frame every so often so the numbers can be re-derived, or a surprise
# explained, without re-running the night. At ~250 kB a frame and one every 20
# samples, a 12-hour night is well under a gigabyte.
KEEP_EVERY = 20


def metrics(img):
    """Aperture statistics, including the ones that might work in the dark.

    green_excess is kept even though it is known to be identically zero at
    night: it is the daylight discriminator, and logging its collapse through
    dusk is the evidence that the collapse is real and not a one-off.
    """
    reg = img[REGION_Y[0]:REGION_Y[1], REGION_X[0]:REGION_X[1]]
    f = reg.astype(float)
    b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    grey = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(grey, (5, 5), 0), 40, 120)

    # Stars, if any are reachable through the aperture: a plausible night
    # discriminator that has nothing to do with colour. Threshold well above
    # the local background so IR-lit roof texture cannot masquerade as stars.
    bg = float(np.median(grey))
    sd = float(np.std(grey)) or 1.0
    n_lab, _, stats, _ = cv2.connectedComponentsWithStats(
        (grey > bg + 5 * sd).astype(np.uint8), 8)
    points = sum(1 for i in range(1, n_lab)
                 if 2 <= stats[i, cv2.CC_STAT_AREA] <= 60)

    return {
        "mean": round(float(grey.mean()), 2),
        "median": round(bg, 2),
        "sd": round(sd, 2),
        "p99": round(float(np.percentile(grey, 99)), 2),
        "edge_pct": round(100.0 * float((edges > 0).mean()), 3),
        "green_excess": round(float((g - (b + r) / 2.0).mean()), 3),
        "chroma": round(float(np.mean(np.max(f, 2) - np.min(f, 2))), 3),
        "points": points,
    }


def marker(img):
    """AprilTag corners, or why not. Corners, not a verdict -- the parked pose
    is not yet known, and tonight is where it comes from."""
    det = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
        cv2.aruco.DetectorParameters())
    corners, ids, rejected = det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if ids is None:
        return {"found": False, "rejected": len(rejected)}
    q = corners[0][0]
    sides = [float(np.linalg.norm(q[i] - q[(i + 1) % 4])) for i in range(4)]
    diags = [float(np.linalg.norm(q[0] - q[2])), float(np.linalg.norm(q[1] - q[3]))]
    return {
        "found": True,
        "id": int(ids.ravel()[0]),
        "corners": [[round(float(x), 2), round(float(y), 2)] for x, y in q],
        "centre": [round(float(q[:, 0].mean()), 2), round(float(q[:, 1].mean()), 2)],
        "side_px": round(float(np.mean(sides)), 2),
        # Obliquity shows up in the diagonals when the tilt axis runs near a
        # diagonal, which is this fixture's case.
        "diag_ratio": round(min(diags) / max(diags), 4),
    }


def mount_state():
    """PWI4 telemetry: the independent witness the tag gets compared against.

    Deliberately NOT pwi4_utils.get_is_parked(). That function returns False on
    any exception, including "PWI4 refused the connection because the mount is
    powered off". For a roof gate that is the right fail-safe -- unknown must
    read as not-parked and refuse the move. For a log it is a fabrication, and
    it would fabricate at the worst moment: end.bat cuts mount power, so every
    sample after it would claim the scope was NOT parked while the tag sat
    plainly in the parked pose, inventing a disagreement to explain tomorrow.

    Logs alt/az rather than a boolean. The whole reason the safe/unsafe
    threshold is still a placeholder is that PX_PER_DEGREE came from one
    hand-moved ~5 degree estimate. Tag corners against real mount angles all
    night measures it properly, and a boolean throws that away.
    """
    out = {"reachable": False}
    try:
        from hardware_control.pwi4_client import PWI4
        with contextlib.redirect_stdout(io.StringIO()):
            st = PWI4().status()
        out["reachable"] = True
        out["connected"] = bool(st.mount.is_connected)
        if st.mount.is_connected:
            out["alt"] = round(float(st.mount.altitude_degs), 4)
            out["az"] = round(float(st.mount.azimuth_degs), 4)
            out["slewing"] = bool(st.mount.is_slewing)
            out["tracking"] = bool(st.mount.is_tracking)
    except Exception as e:
        out["why"] = type(e).__name__
    return out


def sample(n):
    from configs import config
    from scripts.probe_kasa_camera import probe_kc_stream
    from sentry.sky_camera import credentials

    os.makedirs(FRAME_DIR, exist_ok=True)
    path = os.path.join(FRAME_DIR, "shadow_%05d.jpg" % n)
    user, pw = credentials(config.data())
    # probe_kc_stream narrates to stdout; a night of that is noise.
    with contextlib.redirect_stdout(io.StringIO()):
        ok = probe_kc_stream(HOST, user, pw, snapshot_path=path, timeout=20)
    if not ok or not os.path.exists(path):
        return {"when": datetime.now().astimezone().isoformat(timespec="seconds"),
                "n": n, "camera": False}
    img = cv2.imread(path)
    if img is None:
        return {"when": datetime.now().astimezone().isoformat(timespec="seconds"),
                "n": n, "camera": False, "unreadable": True}

    row = {"when": datetime.now().astimezone().isoformat(timespec="seconds"),
           "n": n, "camera": True,
           "frame_lum": round(float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()), 2),
           "aperture": metrics(img), "marker": marker(img),
           "mount": mount_state()}

    # probe_kc_stream leaves the raw .h264 it decoded from beside the .jpg.
    # Deleting only the jpg leaked 1029 h264 files (162 MB) in one night before
    # this was caught -- the jpg count looked exactly right the whole time,
    # which is why it went unnoticed. Both, or neither.
    if n % KEEP_EVERY:
        for junk in (path, os.path.splitext(path)[0] + ".h264"):
            with contextlib.suppress(OSError):
                os.remove(junk)
    else:
        row["frame"] = path
        with contextlib.suppress(OSError):
            os.remove(os.path.splitext(path)[0] + ".h264")   # keep the jpg only
    return row


def _mt(m):
    """One short field for the console line."""
    if not m or not m.get("reachable"):
        return "unreachable"
    if not m.get("connected"):
        return "disconnected"
    return "alt%.1f az%.1f%s" % (m.get("alt", 0), m.get("az", 0),
                                 " SLEW" if m.get("slewing") else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=int, default=30, help="seconds between samples")
    ap.add_argument("--minutes", type=float, default=None,
                    help="stop after this long; default runs until killed")
    args = ap.parse_args()

    os.makedirs("local", exist_ok=True)
    deadline = time.time() + args.minutes * 60 if args.minutes else None
    print("logging every %ds -> %s   (read-only; no lights, no hardware)"
          % (args.interval, LOG_PATH))

    n = 0
    while deadline is None or time.time() < deadline:
        started = time.time()
        try:
            row = sample(n)
        except Exception as e:                     # never let the logger stop the night
            row = {"when": datetime.now().astimezone().isoformat(timespec="seconds"),
                   "n": n, "error": repr(e)[:200]}
        with open(LOG_PATH, "a") as fh:
            fh.write(json.dumps(row) + "\n")

        m = row.get("marker") or {}
        a = row.get("aperture") or {}
        print("%s  cam=%-5s tag=%-5s side=%-6s ap_mean=%-7s green=%-7s pts=%-4s mount=%s"
              % (row["when"][11:19], row.get("camera"), m.get("found"),
                 m.get("side_px", "-"), a.get("mean", "-"), a.get("green_excess", "-"),
                 a.get("points", "-"), _mt(row.get("mount"))))
        n += 1
        time.sleep(max(0, args.interval - (time.time() - started)))
    print("done: %d samples -> %s" % (n, LOG_PATH))


if __name__ == "__main__":
    sys.exit(main() or 0)
