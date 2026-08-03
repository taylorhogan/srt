"""A/B the two exposure scorers against the real captured ladders.

No camera. Replays every saved frame through best_exposure_score and
marker_match_score, reports which exposure each would have chosen, and what
parked/closed/open verdict that choice produces.

Ground truth for the two sets captured 2026-08-03:
  16-54-55  roof CLOSED -> want parked=True closed=True  open=False
  16-57-08  roof OPEN   -> want parked=True closed=False open=True
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, r"C:\Users\iriso\Documents\development\srt")
import cv2 as cv
import math
from sentry import vision_safety as vs
from sentry.inside_camera_server import best_exposure_score
from configs import config

cs = config.data()["camera safety"]
ACC, MINC = cs["accuracy"], cs["match_confidence"]
TRUTH = {
    "2026-08-03_16-54-55": ("roof CLOSED", (True, True, False)),
    "2026-08-03_16-57-08": ("roof OPEN", (True, False, True)),
    # captured by open_roof's retry loop, roof still open
    "2026-08-03_17-02-43": ("roof OPEN", (True, False, True)),
}


def verdict(frame):
    """parked/closed/open exactly as _visual_status_once decides them."""
    out = {}
    for name, tpl, pos in (("parked", "parked template", "parked pos"),
                           ("closed", "closed template", "closed pos"),
                           ("open", "open template", "open pos")):
        _, _, center, conf = vs.find_template_rectangle(frame, cs[tpl])
        err = math.dist(center, cs[pos])
        out[name] = (err < ACC and conf >= MINC, conf, err)
    parked = out["parked"][0]
    closed = out["closed"][0] if parked else False
    is_open = out["open"][0] if parked else False
    if closed and is_open:                      # ambiguous -> both dropped
        closed = is_open = False
    return (parked, closed, is_open), out


root = Path(r"C:\Users\iriso\Documents\development\srt\base_images\exposure_sets")
overall = True
for d in sorted(root.iterdir()):
    meta = json.loads((d / "meta.json").read_text())
    label, want = TRUTH.get(d.name, ("unknown", None))
    print(f"=== {d.name}   {label}   sun alt {meta['sun_altitude_deg']:.1f}")
    rows = []
    for f in meta["frames"]:
        frame = cv.imread(str(d / f["file"]), cv.IMREAD_COLOR)
        if frame is None:
            continue
        got, detail = verdict(frame)
        rows.append({
            "exp": f["exposure"],
            "whole": best_exposure_score(frame),
            "marker": vs.marker_match_score(frame),
            "verdict": got,
        })

    for tag, key in (("best_exposure_score (current)", "whole"),
                     ("marker_match_score  (new)", "marker")):
        pick = max(rows, key=lambda r: r[key])
        ok = (want is None) or (pick["verdict"] == want)
        if key == "marker":          # only the NEW scorer must pass
            overall &= ok
        p, c, o = pick["verdict"]
        print(f"  {tag:32} -> exp {pick['exp']:>3}  score {pick[key]:+.3f}  "
              f"verdict parked={p} closed={c} open={o}   {'PASS' if ok else 'FAIL'}")
    if want:
        print(f"  {'want':32}    {'':11}         "
              f"parked={want[0]} closed={want[1]} open={want[2]}")
    print()

print("A/B", "PASS — new scorer correct on both roof states" if overall else "FAIL")
