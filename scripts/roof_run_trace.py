"""
roof_run_trace.py
Score every frame of a recording for BOTH roof state and scope pose.

Built for an `image!! 3` run, which walks the whole state machine in one pass:
roof opens, the NINA prelude homes the scope (a large software-driven slew),
home_and_park parks it, then end.py closes the roof. A single continuous trace
through that is worth more than snapshots, because it shows the transitions and
lets frames be labelled against iris.log timestamps rather than by guesswork.

    python roof_run_trace.py local/iriscam_rec/imagerun_20260816_193000

Each frame gets:
  registration   - the framing check; an unregistered frame is scored as
                   unknown rather than silently measured against the wrong box
  green_excess   - roof: sign flips between shut and open IN DAYLIGHT. At night
                   the camera returns monochrome IR and this is ~0 for BOTH
                   states, which is exactly what a night run tests.
  edge_pct       - roof: the candidate that should survive monochrome. 12-15%
                   shut vs 21-26% open across every daylight run so far.
  scope_match    - parked: correlation against the stored parked reference.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from datetime import datetime

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import cv2
import numpy as np

from scripts.roof_region_stats import (_reference, register, REGION_Y, REGION_X)
from scripts.scope_parked_probe import REF_PATH as PARKED_REF, SCOPE_BOX


def trace(rec_dir):
    ref, meta, templates = _reference()
    if meta is None:
        raise SystemExit("no framing reference; run roof_region_stats.py "
                         "--set-reference first")
    parked_ref = cv2.imread(PARKED_REF)
    if parked_ref is None:
        raise SystemExit("no parked reference; run scope_parked_probe.py "
                         "--set-reference with the scope parked")
    parked_grey = cv2.cvtColor(parked_ref, cv2.COLOR_BGR2GRAY)[SCOPE_BOX].astype(np.float32)

    rows = []
    for path in sorted(glob.glob(os.path.join(rec_dir, "frame_*.jpg"))):
        m = re.search(r"frame_(\d+\.\d)s", path)
        if not m:
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shift = register(grey, meta, templates)
        row = {"t_s": float(m.group(1)),
               "registered": shift is not None,
               "dx": None if shift is None else shift[0],
               "dy": None if shift is None else shift[1]}
        if shift is None:
            rows.append(row)
            continue
        dx, dy, _ = shift
        h, w = grey.shape

        y0, y1 = max(0, REGION_Y[0] + dy), min(h, REGION_Y[1] + dy)
        x0, x1 = max(0, REGION_X[0] + dx), min(w, REGION_X[1] + dx)
        roi = img[y0:y1, x0:x1]
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        b, gr, r = (roi[:, :, i].astype(np.float64) for i in range(3))
        row["mean"] = float(g.mean())
        row["edge_pct"] = float(100.0 * cv2.Canny(g, 60, 160).mean() / 255.0)
        row["green_excess"] = float((gr - (b + r) / 2).mean())
        # Channel spread near zero means the camera is in monochrome IR, so
        # green_excess carries no information at all -- flagged, not guessed at.
        row["mono"] = bool(float(np.mean(np.max(roi, axis=2) - np.min(roi, axis=2))) < 3.0)

        ys, xs = SCOPE_BOX
        sy0, sy1 = max(0, ys.start + dy), min(h, ys.stop + dy)
        sx0, sx1 = max(0, xs.start + dx), min(w, xs.stop + dx)
        cur = grey[sy0:sy1, sx0:sx1].astype(np.float32)
        base = parked_grey[:cur.shape[0], :cur.shape[1]]
        a, bb = cur - cur.mean(), base - base.mean()
        row["scope_match"] = float((a * bb).sum() /
                                   (np.sqrt((a * a).sum() * (bb * bb).sum()) + 1e-9))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    args = ap.parse_args()

    rows = trace(args.recording)
    if not rows:
        raise SystemExit("no frames in %s" % args.recording)

    out = os.path.join(args.recording, "trace.csv")
    keys = ["t_s", "registered", "dx", "dy", "mean", "edge_pct", "green_excess",
            "mono", "scope_match"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})

    print("%7s %5s %7s %8s %9s %6s   %s"
          % ("t_s", "reg", "edge%", "greenX", "scope", "mono", "roof / scope"))
    print("-" * 74)
    for r in rows:
        if not r["registered"]:
            print("%7.1f %5s %s" % (r["t_s"], "NO", " " * 30 + "unregistered"))
            continue
        # Bars make a 25-minute run readable at a glance; the transitions are
        # what matter, not any single row.
        sm = r["scope_match"]
        bar = "#" * max(0, min(20, int(sm * 20)))
        roof = "OPEN " if r["edge_pct"] > 19 else "shut "
        print("%7.1f %5s %7.2f %8.2f %9.3f %6s   %s %-21s"
              % (r["t_s"], "ok", r["edge_pct"], r["green_excess"], sm,
                 "yes" if r["mono"] else "no", roof, bar))

    reg = [r for r in rows if r["registered"]]
    print("\n%d frames, %d registered (%d refused)"
          % (len(rows), len(reg), len(rows) - len(reg)))
    if reg:
        sm = [r["scope_match"] for r in reg]
        ed = [r["edge_pct"] for r in reg]
        mono = sum(r["mono"] for r in reg)
        print("  scope_match  min %.3f  max %.3f" % (min(sm), max(sm)))
        print("  edge_pct     min %.2f  max %.2f" % (min(ed), max(ed)))
        print("  monochrome (IR) frames: %d of %d" % (mono, len(reg)))
        if mono:
            print("  -> green excess is uninformative on those; edge density is"
                  " the metric under test tonight")
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
