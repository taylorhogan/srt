#!/usr/bin/env python3
"""Every 5 minutes: photograph the sky, count the stars if it is dark, publish.

Runs off the Kasa all-sky camera (sentry/sky_camera.py), counts point sources
with sentry/star_count.py, and pushes the picture plus the count to the lab
site's live panel. Frames are archived locally, because the cloud/rain/snow
detector this is groundwork for needs labelled examples and those can only be
collected before the fact. Retention is event-aware rather than a flat rolling
cap -- see sentry/sky_archive.py for why keeping the newest N would delete the
storms and keep the four thousandth clear sky.

Publishes to sky.json / sky.jpg, NOT to the status.json that live_skymap.py
owns. That generator rewrites its file wholesale every 5 minutes; sharing one
file would mean whichever job finished last silently deleted the other's
fields.

The count is published with its purity, and suppressed as a sky measurement
when purity is low. That is not defensive padding: on a rain frame this
detector returns several hundred "stars" of which most are false, so an
unqualified count would report the worst sky of the night as the best.

Usage:  python scripts/sky_monitor.py [--no-push] [--annotate] [--frame PATH]
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
from sentry import (plate_solve, rain_detect, sky_annotate, sky_archive,
                    sky_camera, star_count)
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
        # Log the failure as well as publishing it. Returning here without a
        # row is what made local/sky_log.jsonl survivorship-filtered: it could
        # only ever contain successes, so on 2026-08-13 it read 132/132 "ok"
        # while the site was serving camera:unavailable, and camera reliability
        # was not measurable from it at all. The row carries no `captured` and
        # no `trustworthy`, so preserve_events ignores it.
        try:
            sky_archive.append_index(status)
        except Exception as exc:      # recording a failure must not sink the run
            print("index append failed (%s)" % type(exc).__name__)
        print("sky_monitor: camera unavailable")
        return 1

    status["camera"] = "ok"
    # The frame's own timestamp, not its mtime. For a live capture the two
    # agree, but a frame that has been copied or reprocessed carries the mtime
    # of the copy: the 02:00 rain frame came back as 09:03, which marked the
    # wrong hour as interesting and preserved two innocent frames. Everything
    # else here already dates the frame this way, and one function should not
    # hold two notions of when a picture was taken.
    status["captured"] = plate_solve.frame_time(frame).isoformat(
        timespec="seconds")

    # The project already has one definition of night; a second one here could
    # disagree with the scheduler about whether the observatory is working.
    # Judged at the frame's own timestamp, not at wall-clock now, so that
    # reprocessing an archived frame reports the sky that frame actually saw.
    night, sun_alt = sun.is_night(plate_solve.frame_time(frame))
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
        status.update({k: res[k] for k in (
            "stars", "false_positives", "purity", "trustworthy",
            "threshold_adu", "threshold_sigma", "noise_adu",
            "sky_median_adu", "masked_fraction", "median_fwhm_px",
            "brightest_peak_adu")})
        _add_photometry(status, frame, res)
        if not res["trustworthy"]:
            # Keep the raw number for the archive, but say plainly that it is
            # not a reading of the sky.
            status["note"] = ("count unreliable: %d of %d detections are false "
                              "by the negative-image control"
                              % (res["false_positives"], res["stars"]))
        print("sky_monitor: %d stars, purity %.0f%%, sun %.1f deg"
              % (res["stars"], 100 * res["purity"], sun_alt))

    # Measured on every capture, day or night, because the daylight burst
    # trigger has nothing else to compare against.
    try:
        status["frame_level_adu"] = round(_frame_level(frame), 1)
    except Exception:
        pass

    # Rain prediction/detection from adjacent video frames. Takes its own short
    # burst because a single still cannot show motion, and motion is the whole
    # signal. Night-gated and rate-limited inside rain_detect; never raises.
    if "--frame" not in sys.argv:
        rain_detect.step(status, frame)

    # Before publishing: a burst is time-critical in a way the push is not. If
    # rain is starting, the seconds spent uploading a JPEG are seconds of the
    # onset going unrecorded, and the onset does not come round again.
    if "--frame" not in sys.argv:
        try:
            _maybe_burst(status, config.data())
        except Exception as exc:
            print("burst step failed (%s)" % type(exc).__name__)

    _publish(status, frame)

    # Log first, then preserve, then prune -- in that order. The index is what
    # the preserve step reads to find frames the detector distrusted, so a
    # capture must be recorded before anything is allowed to delete it.
    try:
        sky_archive.append_index(status)
        sky_camera.prune_bursts()
        ev = sky_archive.preserve_events()
        if ev.get("moved"):
            print("preserved %d frame(s) %s" % (ev["moved"], ev["by"]))
    except Exception as exc:
        # Retention must never sink a capture, but a silent failure here means
        # the pruner below is running without the labels that protect frames.
        print("archive step failed (%s); pruning without event labels"
              % type(exc).__name__)
    dropped = sky_camera.prune()
    if dropped:
        print("pruned %d old frames" % dropped)
    return 0


def _frame_level(frame):
    """Median level of the whole frame. Cheap, and available in daylight.

    Deliberately not a sky brightness -- auto-exposure renormalises every frame,
    so the absolute number means little. It exists only so consecutive captures
    can be compared, which is the one daytime signal available when there are no
    stars to count.
    """
    import numpy as np
    from sentry import star_count
    return float(np.median(star_count._load(frame)))


def _maybe_burst(status, cfg):
    """Pull consecutive frames when this capture looks like weather.

    Three stills five minutes apart cannot describe a fifteen-minute shower, and
    the camera can give 15 fps. Bursting only on suspicion keeps that free on
    clear nights, and firing on the FIRST odd frame catches the onset -- the
    minutes when rain is starting are the ones a safety trigger would need, and
    exactly the ones a wet-hours-only filter throws away.

    Night only. Daytime rain is not wanted -- the roof is shut so it carries no
    safety information, and the camera behaves differently enough in daylight
    that those frames would not transfer to a night-time detector.
    """
    cam = cfg.get("sky camera", {})
    reasons = []
    purity = status.get("purity")
    if purity is not None and purity < float(cam.get("burst_purity_below", 0.85)):
        reasons.append("purity %.2f" % purity)

    if not reasons:
        return
    path, nparts = sky_camera.capture_burst(cfg=cfg)
    status["burst_reason"] = "; ".join(reasons)
    if path is None:
        status["burst"] = None
        print("burst wanted (%s) but capture failed" % status["burst_reason"])
        return
    status["burst"] = path.name
    status["burst_parts"] = nparts
    print("burst: %s (%d video parts, %.1f MB) because %s"
          % (path.name, nparts, path.stat().st_size / 1e6, status["burst_reason"]))


def _add_photometry(status, frame, res):
    """Limiting magnitude, if the camera's plate solution still fits.

    Limiting magnitude is the better cloud reading. A star count also falls when
    a branch grows into the field or the frame is cropped differently; how faint
    the sky lets us see is a property of the sky. It needs the plate solution to
    know which catalogue stars should have been visible.

    Never blind-solves here. That search takes minutes, and a job on a 5-minute
    timer must not sometimes take minutes -- run `python sentry/plate_solve.py
    <frame> --save` once, by hand, and this picks it up from then on. The camera
    is bolted down, so once is enough.
    """
    sol = plate_solve.load()
    if sol is None:
        status["note_solve"] = ("no plate solution stored; run "
                                "sentry/plate_solve.py <frame> --save once")
        return
    when = plate_solve.frame_time(frame)
    try:
        v = plate_solve.verify(sol, frame, when, res["_stars"])
        status["solve_matches"] = v["matches"]
        status["solve_residual_px"] = v["residual_px"]
        # A knocked or refocused camera shows up here first: the stored geometry
        # stops landing on the stars. Say so rather than publish a magnitude
        # derived from a solution that no longer describes this camera.
        #
        # Tested as a FRACTION of what was detected, never an absolute count.
        # Cloud drives the count down without moving the camera -- 187 stars and
        # 88 matches at 03:34 became 112 and 29 forty minutes later as twilight
        # came up -- so an absolute floor would cry "re-solve" every time the
        # sky got worse, which is exactly when the reading matters. A wrong
        # solution matches a percent or two; a right one on a bad night still
        # matches a quarter.
        #
        # Only ever judged on a frame the detector trusts. On a rain frame the
        # geometry is fine and the sky is simply opaque, but the count fills
        # with junk -- 521 detections, 286 of them false, 9 matching, a 2%
        # share -- which trips any share test and would send someone chasing a
        # camera fault during a storm. An untrustworthy frame says nothing
        # about where the camera points, so it gets no verdict either way.
        if not res["trustworthy"]:
            status["note_solve"] = ("sky too poor to check the plate solution "
                                    "on this frame")
            return
        share = v["matches"] / max(res["stars"], 1)
        if v["matches"] < 6 or share < 0.08:
            status["note_solve"] = (
                "plate solution no longer fits (%d matches, %.0f%% of "
                "detections); re-solve" % (v["matches"], 100 * share))
            return
        # Integer magnitude bins: the card shows this as a table, and half
        # magnitudes make it twice as long without telling anyone more.
        lm = plate_solve.limiting_magnitude(sol, frame, when, res["_stars"],
                                            res["_mask"], step=1.0)
        status["limiting_mag"] = lm["limiting_mag"]
        status["stars_expected"] = lm["stars_visible_area"]
        status["stars_matched"] = lm["stars_matched"]
        # Published so the picture and the table can be read together: the
        # circle drawn on sky.jpg is this radius, and it is the area the counts
        # beside it were measured in.
        status["measured_radius_px"] = lm["measured_radius_px"]
        status["completeness"] = lm["completeness"]
    except Exception as exc:            # never let photometry sink the capture
        status["note_solve"] = "photometry failed: %s" % type(exc).__name__


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
        # Compass bearings are drawn on a COPY, never on the archived frame.
        # The archive is training data for the weather detector, and a label
        # burnt into the same pixels every frame is exactly the sort of
        # constant a classifier learns in place of the sky. Falls back to the
        # raw frame if there is no plate solution to compute bearings from.
        shown = out_dir / "sky_published.jpg"
        try:
            shown = sky_annotate.annotate(frame, shown) or frame
        except Exception as exc:
            print("compass annotation failed (%s); publishing the raw frame"
                  % type(exc).__name__)
            shown = frame
        # Picture first: a viewer that catches the pair mid-push should see an
        # old count beside a new picture, never a new count beside an old one.
        pairs.insert(0, (shown, "sky.jpg"))
    live_push.push(pairs)
    print("pushed", len(pairs), "file(s) to", live_push.HOST + ":" + live_push.DEST)


if __name__ == "__main__":
    sys.exit(main())
