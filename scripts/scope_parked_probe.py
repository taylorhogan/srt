"""
scope_parked_probe.py
Capture one Iris cam frame and score how far the scope is from parked.

The roof question is "has a big region changed from wood to sky". The parked
question is not like that: "not parked" is a continuum of poses, so there is
nothing to threshold. This instead asks "prove it is parked" — how well does
the scope area match a stored parked reference, once the frame has been
registered against the fixed ceiling. Anything unrecognised scores low and
fails closed, which is the direction a safety gate must fail in.

    python scope_parked_probe.py --set-reference     # with the scope PARKED
    python scope_parked_probe.py --label up-45       # then each test pose

Every capture appends to local/scope_parked_log.jsonl so the parked and
not-parked populations can be compared afterwards.
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
from scripts.roof_region_stats import _reference, register
from sentry.sky_camera import credentials

HOST = "192.168.87.65"
REF_PATH = "local/scope_parked_reference.jpg"     # legacy single reference
REF_DIR = "local/parked_refs"

# "Parked" is one POSE under many lightings, not one appearance. A single
# reference conflates the two: measured 2026-08-16, a parked scope in bad light
# scored 0.699 while a genuinely MOVED scope in good light scored 0.609 -- 0.09
# of margin, which is not a safety gate. Scoring against a library of parked
# frames and taking the best match separates pose from illumination, because a
# moved scope matches NONE of them. Same eight frames: margin 0.090 -> 0.234.
#
# So add references as conditions appear -- roof open and shut, day, dusk,
# night IR, lights on. Each one raises the parked floor; none can lower it.
LOG_PATH = "local/scope_parked_log.jsonl"

# Where the scope lives in frame. Deliberately excludes two things: the roof
# aperture on the left (x < 620), which changes with roof state and would make
# a parked score depend on the roof, and the registration fiducial band on the
# right (x >= 1750), which must stay independent of what is being measured.
SCOPE_BOX = (slice(150, 1300), slice(620, 1750))


def grab(path):
    user, pw = credentials(config.data())
    if not probe_kc_stream(HOST, user, pw, snapshot_path=path, timeout=15):
        raise SystemExit("no frame from the camera")
    return cv2.imread(path)


def load_library():
    """{name: image} of every parked reference, newest naming wins."""
    out = {}
    if os.path.isdir(REF_DIR):
        for f in sorted(os.listdir(REF_DIR)):
            if f.lower().endswith((".jpg", ".png")):
                img = cv2.imread(os.path.join(REF_DIR, f))
                if img is not None:
                    out[os.path.splitext(f)[0]] = img
    if not out and os.path.exists(REF_PATH):
        img = cv2.imread(REF_PATH)
        if img is not None:
            out["legacy"] = img
    return out


def scope_score(img, ref_img, meta, templates):
    """(score, shift) — 1.0 is identical to the parked reference, 0 unrelated.

    Registered first, so a camera that has drifted does not read as a scope
    that has moved. Returns None for the score if the frame will not register:
    an unregistered frame cannot say anything about the scope's pose.
    """
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shift = register(grey, meta, templates)
    if shift is None:
        return None, None
    dx, dy, _ = shift
    ys, xs = SCOPE_BOX
    h, w = grey.shape
    y0, y1 = max(0, ys.start + dy), min(h, ys.stop + dy)
    x0, x1 = max(0, xs.start + dx), min(w, xs.stop + dx)
    cur = grey[y0:y1, x0:x1].astype(np.float32)
    base = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)[ys, xs].astype(np.float32)
    if cur.shape != base.shape:
        base = base[:cur.shape[0], :cur.shape[1]]
    # Zero-mean normalised correlation: survives the exposure and colour
    # changes between morning and midday, which a raw difference would not.
    a, b = cur - cur.mean(), base - base.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum())) + 1e-9
    return float((a * b).sum() / denom), (dx, dy)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set-reference", action="store_true",
                    help="ADD the current frame to the parked library (use --label "
                         "to name the condition, e.g. night-ir, roof-shut)")
    ap.add_argument("--label", default="probe")
    ap.add_argument("--parked", choices=["yes", "no"], default=None,
                    help="ground truth for this capture, if known")
    args = ap.parse_args()

    ref_img, meta, templates = _reference()
    if meta is None:
        raise SystemExit("no camera-framing reference; run roof_region_stats.py "
                         "--set-reference first")

    if args.set_reference:
        os.makedirs(REF_DIR, exist_ok=True)
        path = os.path.join(REF_DIR, "%s.jpg" % args.label)
        grab(path)
        print("added parked reference %r -> %s" % (args.label, path))
        print("library now holds %d reference(s): %s"
              % (len(load_library()), ", ".join(sorted(load_library()))))
        return 0

    library = load_library()
    if not library:
        raise SystemExit("no parked references yet; run --set-reference with "
                         "the scope parked")
    img = grab("local/scope_probe_%s.jpg" % args.label)
    per_ref = {}
    score, shift = None, None
    for name, ref_img in library.items():
        sc, sh = scope_score(img, ref_img, meta, templates)
        if sc is None:
            shift = None
            break
        per_ref[name] = sc
        shift = sh
        if score is None or sc > score:
            score = sc

    print("\nlabel            : %s" % args.label)
    if score is None:
        print("registration     : REFUSED -- cannot judge the scope")
        print("  This is itself a result: if the fiducial patches sit on the")
        print("  scope, moving it breaks the framing check as well.")
    else:
        print("registration     : dx=%+d dy=%+d" % shift)
        for name in sorted(per_ref, key=lambda k: -per_ref[k]):
            print("  vs %-22s %.4f" % (name, per_ref[name]))
        print("scope match      : %.4f   (best of %d references)"
              % (score, len(per_ref)))
    if args.parked:
        print("ground truth     : parked=%s" % args.parked)

    row = {"label": args.label, "per_ref": per_ref, "when": datetime.now().astimezone().isoformat(timespec="seconds"),
           "score": score, "shift": shift, "truth": args.parked}
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as fh:
        fh.write(json.dumps(row) + "\n")

    rows = [json.loads(l) for l in open(LOG_PATH) if l.strip()]
    scored = [r for r in rows if r.get("score") is not None and r.get("truth")]
    if len(scored) >= 2:
        yes = [r["score"] for r in scored if r["truth"] == "yes"]
        no = [r["score"] for r in scored if r["truth"] == "no"]
        print("\n  parked     n=%d  %s" % (len(yes), ["%.3f" % v for v in yes]))
        print("  NOT parked n=%d  %s" % (len(no), ["%.3f" % v for v in no]))
        if yes and no:
            print("  separation : parked min %.3f vs not-parked max %.3f  -> %s"
                  % (min(yes), max(no),
                     "SEPARATED" if min(yes) > max(no) else "*** OVERLAP ***"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
