"""Replay every saved exposure ladder through the new voting verdict."""
import json
import os
import sys
from pathlib import Path

import cv2 as cv

sys.path.insert(0, os.path.abspath("."))
from sentry import vision_safety as vs

# Ground truth. The 08-03/08-04 labels this started with have rolled off the
# disk -- exposure_capture_keep caps the directory at 30 -- so they are replaced
# by the 2026-08-16 session, where the roof state is known from the commands
# that were run and, independently, from the Iris cam recordings, which show
# the aperture open continuously until 10:34:43.
#
# 10-35-43 and 10-36-25 are deliberately absent: a manual close happened in
# that window and cannot be placed to the second. An unlabelled ladder is
# reported but not scored.
TRUTH = {
    "2026-08-16_01-58-26": "closed",  # end sequence close, night
    "2026-08-16_01-59-35": "closed",
    "2026-08-16_07-40-55": "closed",  # pre-open check, low sun
    "2026-08-16_07-43-14": "open",    # post-open confirm
    "2026-08-16_07-48-01": "open",    # pre-close check
    "2026-08-16_07-49-17": "open",
    "2026-08-16_07-51-27": "closed",  # post-close confirm
    "2026-08-16_07-53-45": "closed",
    "2026-08-16_10-24-16": "closed",  # pre-open check, sun 46 deg
    "2026-08-16_10-26-28": "open",    # the ambiguous run begins here
    "2026-08-16_10-27-07": "open",
    "2026-08-16_10-27-45": "open",
    "2026-08-16_10-28-19": "open",
    "2026-08-16_10-28-57": "open",
    "2026-08-16_10-29-36": "open",
    "2026-08-16_10-33-24": "open",
    "2026-08-16_10-34-02": "open",
    "2026-08-16_10-34-40": "open",
}

root = Path("base_images/exposure_sets")
fails = 0
print(f"{'ladder':22} {'sun':>6}  vote                          verdict      truth")
for d in sorted(root.iterdir()):
    if not (d / "meta.json").exists():
        continue
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
