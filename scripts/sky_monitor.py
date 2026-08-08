#!/usr/bin/env python3
"""Every 15 minutes: photograph the sky, count the stars if it is dark, publish.

Runs off the Kasa all-sky camera (sentry/sky_camera.py), counts point sources
with sentry/star_count.py, and pushes the picture plus the count to the lab
site's live panel. Frames are also kept in a rolling local archive, because the
cloud/rain/snow detector this is groundwork for needs labelled examples and
those can only be collected before the fact.

Publishes to sky.json / sky.jpg, NOT to the status.json that live_skymap.py
owns. That generator rewrites its file wholesale every 5 minutes; sharing one
file would mean whichever job finished last silently deleted the other's
fields.

The count is published with its purity, and suppressed as a sky measurement
when purity is low. That is not defensive padding: on a rain frame this
detector returns several hundred "stars" of which most are false, so an
unqualified count would report the worst sky of the night as the best.

Usage:  python scripts/sky_monitor.py [--no-push] [--annotate]
"""
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

# astropy re-reports an unwritable IERS cache on every call. It falls back to
# the bundled table, which is fine to sub-arcsecond for a sun altitude, and the
# noise would otherwise be the bulk of this job's log.
warnings.filterwarnings("ignore")

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from configs import config
from iris_astronomy import sun
from sentry import sky_camera, star_count
from scripts import live_push

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    now = datetime.now(timezone.utc)
    status = {"generated": now.isoformat(timespec="seconds")}

    # --frame re-runs everything downstream of the camera against a file that
    # already exists. Needed to reprocess the archive when the detector
    # changes, which is the normal way a stored frame earns its keep.
    if "--frame" in sys.argv:
        frame = Path(sys.argv[sys.argv.index("--frame") + 1])
        if not frame.exists():
            print("no such frame:", frame)
            return 2
    else:
        frame = sky_camera.capture()
    if frame is None:
        status.update(camera="unavailable", night=None, stars=None)
        _publish(status, None)
        print("sky_monitor: camera unavailable")
        return 1

    status["camera"] = "ok"
    status["captured"] = datetime.fromtimestamp(
        frame.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")

    # The project already has one definition of night; a second one here could
    # disagree with the scheduler about whether the observatory is working.
    night, sun_alt = sun.is_night()
    status["night"] = bool(night)
    status["sun_alt_deg"] = round(float(sun_alt), 1)

    if not night:
        status["stars"] = None
        status["note"] = "daylight"
        print("sky_monitor: sun at %.1f deg, not counting" % sun_alt)
    else:
        ann = None
        if "--annotate" in sys.argv:
            ann = ROOT / "iris_astronomy" / "scratch" / "sky_marked.png"
            ann.parent.mkdir(parents=True, exist_ok=True)
        res = star_count.count_stars(frame, annotate=ann)
        res.pop("_stars", None)
        status.update({k: res[k] for k in (
            "stars", "false_positives", "purity", "trustworthy",
            "threshold_adu", "threshold_sigma", "noise_adu",
            "sky_median_adu", "masked_fraction", "median_fwhm_px",
            "brightest_peak_adu")})
        if not res["trustworthy"]:
            # Keep the raw number for the archive, but say plainly that it is
            # not a reading of the sky.
            status["note"] = ("count unreliable: %d of %d detections are false "
                              "by the negative-image control"
                              % (res["false_positives"], res["stars"]))
        print("sky_monitor: %d stars, purity %.0f%%, sun %.1f deg"
              % (res["stars"], 100 * res["purity"], sun_alt))

    _publish(status, frame)
    dropped = sky_camera.prune()
    if dropped:
        print("pruned %d old frames" % dropped)
    return 0


def _publish(status, frame):
    out_dir = ROOT / "iris_astronomy" / "scratch"
    out_dir.mkdir(parents=True, exist_ok=True)
    js = out_dir / "sky_status.json"
    js.write_text(json.dumps(status, indent=1))

    if "--no-push" in sys.argv:
        print("not pushing (--no-push):", json.dumps(status))
        return
    pairs = [(js, "sky.json")]
    if frame is not None:
        # Picture first: a viewer that catches the pair mid-push should see an
        # old count beside a new picture, never a new count beside an old one.
        pairs.insert(0, (frame, "sky.jpg"))
    live_push.push(pairs)
    print("pushed", len(pairs), "file(s) to", live_push.HOST + ":" + live_push.DEST)


if __name__ == "__main__":
    sys.exit(main())
