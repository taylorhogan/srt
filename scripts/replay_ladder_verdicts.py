"""Replay every saved exposure ladder through the new voting verdict."""
import json
import os
import sys
from pathlib import Path

import cv2 as cv

sys.path.insert(0, os.path.abspath("."))
from sentry import vision_safety as vs

# Ground truth: mine (08-04, plus the 08-03 open sets) merged with the labels
# already recorded in scripts/ab_exposure_scorer.py.
TRUTH = {
    "2026-08-03_16-54-55": "closed",  # daylight, roof closed
    "2026-08-03_16-57-08": "open",    # daylight, roof open
    "2026-08-03_17-02-43": "open",    # open_roof retry loop, roof still open
    "2026-08-03_18-15-14": "open",    # daylight, roof open
    "2026-08-03_18-18-51": "open",    # daylight, roof open
    "2026-08-03_20-58-30": "open",    # night, roof open
    "2026-08-04_01-43-00": "closed",  # the misread: roof WAS closed
    "2026-08-04_01-44-12": "closed",  # 72s later, read closed
}

root = Path("base_images/exposure_sets")
fails = 0
print(f"{'ladder':22} {'sun':>6}  vote                          verdict      truth")
for d in sorted(root.iterdir()):
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    rungs = []
    for entry in meta["frames"]:
        frame = cv.imread(str(d / entry["file"]), cv.IMREAD_COLOR)
        rungs.append(vs._evaluate_rung(frame, entry["exposure"]))
    got = vs._decide_from_rungs(rungs)
    lm = got["last_match"]
    roof = "closed" if got["closed"] else ("open" if got["open"] else "UNKNOWN")
    park = "parked" if got["parked"] else "NOT-parked"
    want = TRUTH.get(d.name)
    mark = "" if want is None else ("PASS" if (roof == want and park == "parked") else "FAIL")
    fails += mark == "FAIL"
    v = lm["votes"]
    print(f"{d.name:22} {meta['sun_altitude_deg']:6.1f}  "
          f"parked {v['parked']}/{lm['lit_rungs']} closed {v['closed']} open {v['open']}"
          f"{'':<10} {park}/{roof:<8} {want or '-':<8} {mark}")

# What the OLD code did: decide from the one frame the exposure scorer picked.
print("\nsingle-frame verdict (the shipped behaviour), for the labelled sets:")
for name, want in TRUTH.items():
    d = root / name
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    best = max(meta["frames"], key=lambda f: f["score"])
    frame = cv.imread(str(d / best["file"]), cv.IMREAD_COLOR)
    one = vs._decide_from_rungs([vs._evaluate_rung(frame, best["exposure"])])
    roof = "closed" if one["closed"] else ("open" if one["open"] else "UNKNOWN")
    flag = "" if roof == want else f"  <-- WRONG (want {want})"
    print(f"  {name}  exp {best['exposure']:>3}  roof {roof}{flag}")

print(f"\n{'ALL PASS' if not fails else f'{fails} FAILURES'}")
sys.exit(1 if fails else 0)
