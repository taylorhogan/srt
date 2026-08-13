#!/usr/bin/env python3
"""Photograph the sky with the ASI all-sky camera, measure it, publish it --
but only when the roof is genuinely open.

This camera lives UNDER the roof. With the roof shut it takes a perfectly good
picture of the roof's own underside, and the star detector, pointed at that
textured surface, returns a few hundred "stars" of which nearly all are false.
Publishing that would put a photograph of a closed roof on the site captioned as
the sky. So the gate is not a nicety: nothing is published as sky unless the
roof is known open, and when it is not the panel is withdrawn from the page
rather than left showing the last good frame.

Two independent ways to know, because neither covers the whole night:

  * THE STARS THEMSELVES. If the stored plate solution puts catalogue stars on
    top of detections, the camera is looking at sky -- there is no other way for
    that to happen, and the underside of a roof cannot fake it. This is the
    stronger evidence and it costs nothing extra, since the solve is already
    being verified to compute the limiting magnitude.

  * THE SAFETY CAMERA. vision_safety is the observatory's authority on roof
    state, and it works in daylight and through cloud, when there are no stars
    to appeal to. But it reads the roof only when the scope is confirmed parked,
    so during an imaging run -- exactly when the roof is certainly open -- it
    cannot answer. Hence the pair.

Publishes to allsky.json / allsky.jpg, its own files. The other two generators
each rewrite theirs wholesale on a timer, so a shared file would mean whichever
job finished last had deleted the others' fields.

Usage:  python scripts/allsky_monitor.py [--no-push] [--frame PATH] [--annotate]
"""
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

# astropy re-reports an unwritable IERS cache on every call. It falls back to the
# bundled table, which is fine to sub-arcsecond for a sun altitude, and the noise
# would otherwise be the bulk of this job's log.
warnings.filterwarnings("ignore")

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from configs import config
from iris_astronomy import sun
from sentry import asi_allsky, plate_solve, sky_annotate, star_count
from scripts import live_push

ROOT = Path(__file__).resolve().parents[1]
PROFILE = asi_allsky.PROFILE

# Catalogue stars that must land on detections before the frame is accepted as
# a picture of the sky. Eight correct positions is not something a textured
# surface produces by accident -- the closed-roof frames measured on 2026-08-12
# gave hundreds of detections and would match essentially none of them, because
# a match requires the right brightness in the right place at the right time.
ROOF_OPEN_MIN_MATCHES = 8


def main() -> int:
    now = datetime.now(timezone.utc)
    status = {"generated": now.isoformat(timespec="seconds"), "camera": "ok"}
    cfg = config.data()

    frame, meta = _get_frame(status)
    if frame is None:
        _publish(status, None)
        return 1
    status.update({k: meta[k] for k in ("exposure_s", "gain", "clip_pct",
                                        "level_adu") if k in meta})
    if meta.get("warning"):
        status["note_camera"] = meta["warning"]
    status["captured"] = plate_solve.frame_time(frame).isoformat(timespec="seconds")

    night, sun_alt = sun.is_night(plate_solve.frame_time(frame))
    status["night"] = bool(night)
    status["sun_alt_deg"] = round(float(sun_alt), 1)

    res = dets = mask = None
    if night:
        ann = None
        if "--annotate" in sys.argv:
            ann = ROOT / "iris_astronomy" / "scratch" / "allsky_marked.png"
            ann.parent.mkdir(parents=True, exist_ok=True)
        res = star_count.count_stars(frame, annotate=ann, profile=PROFILE)
        dets, mask = res["_stars"], res["_mask"]
        status.update({k: res[k] for k in (
            "stars", "false_positives", "purity", "trustworthy",
            "threshold_adu", "threshold_sigma", "noise_adu",
            "sky_median_adu", "masked_fraction", "median_fwhm_px",
            "brightest_peak_adu")})
        if not res["trustworthy"]:
            status["note"] = ("count unreliable: %d of %d detections are false "
                              "by the negative-image control"
                              % (res["false_positives"], res["stars"]))
        _add_photometry(status, frame, res)
    else:
        status["stars"] = None
        status["note"] = "daylight"

    _add_roof(status, cfg)
    _publish(status, frame)
    _prune(cfg)

    print("allsky: roof %s (%s), %s stars, sun %.1f deg"
          % (status["roof"], status.get("roof_source", "-"),
             status.get("stars"), sun_alt))
    return 0


def _get_frame(status):
    """The frame to work on: a fresh capture, or one named with --frame.

    --frame re-runs everything downstream of the camera against a file that
    already exists, which is how a stored frame gets reprocessed after the
    detector changes.
    """
    if "--frame" in sys.argv:
        frame = Path(sys.argv[sys.argv.index("--frame") + 1])
        if not frame.exists():
            print("no such frame:", frame)
            status.update(camera="unavailable", roof="unknown",
                          note_camera="no such frame")
            return None, {}
        return frame, {}
    try:
        _a, meta = asi_allsky.capture(profile=PROFILE)
    except Exception as exc:
        # A camera that is down is a thing to report, not a crash: this runs on
        # a timer, and the panel must withdraw rather than freeze.
        status.update(camera="unavailable", roof="unknown",
                      note_camera="%s: %s" % (type(exc).__name__, str(exc)[:160]))
        print("allsky: capture failed --", type(exc).__name__, str(exc)[:160])
        return None, {}
    return Path(meta["path"]), meta


def _add_photometry(status, frame, res):
    """Limiting magnitude, if this camera's plate solution still fits.

    Never blind-solves. That search takes minutes and this runs on a timer --
    run `python sentry/plate_solve.py <frame.fits> --profile "allsky camera"
    --save` once by hand and this picks it up from then on.
    """
    sol = plate_solve.load(profile=PROFILE)
    if sol is None:
        status["note_solve"] = ('no plate solution stored for the all-sky '
                                'camera; run sentry/plate_solve.py <frame.fits> '
                                '--profile "allsky camera" --save once')
        return
    when = plate_solve.frame_time(frame)
    try:
        v = plate_solve.verify(sol, frame, when, res["_stars"], profile=PROFILE)
        status["solve_matches"] = v["matches"]
        status["solve_residual_px"] = v["residual_px"]
        # An untrustworthy frame says nothing about where the camera points --
        # on a rain or cloud frame the geometry is fine and the sky is simply
        # opaque -- so it gets no verdict on the solution either way.
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
        # Integer magnitude bins: this is shown as a table, and half magnitudes
        # make it twice as long without telling anyone more.
        lm = plate_solve.limiting_magnitude(sol, frame, when, res["_stars"],
                                            res["_mask"], step=1.0,
                                            profile=PROFILE)
        status["limiting_mag"] = lm["limiting_mag"]
        status["stars_expected"] = lm["stars_visible_area"]
        status["stars_matched"] = lm["stars_matched"]
        status["measured_radius_px"] = lm["measured_radius_px"]
        status["completeness"] = lm["completeness"]
    except Exception as exc:            # never let photometry sink the capture
        status["note_solve"] = "photometry failed: %s" % type(exc).__name__


def _add_roof(status, cfg):
    """Decide whether the roof is open, and record how we know.

    Order matters. The star evidence is tried first because it is free -- the
    solve has already been verified above -- and because it is the only one that
    works during an imaging run, when the scope is not parked and the safety
    camera declines to read the roof at all.

    Anything short of positive evidence is "unknown", never "open". The whole
    point of the gate is that a picture of a closed roof must not reach the site
    captioned as the sky, and defaulting to open on a failed check is exactly
    how that would happen.
    """
    matches = status.get("solve_matches")
    if matches is not None and matches >= ROOF_OPEN_MIN_MATCHES:
        status["roof"] = "open"
        status["roof_source"] = "stars"
        status["roof_reason"] = (
            "%d catalogue stars land on detections in this frame, which is only "
            "possible looking at open sky" % matches)
        return

    if cfg.get(PROFILE, {}).get("roof_check_vision", True):
        verdict = _vision_roof(cfg)
        if verdict is not None:
            is_open, reason = verdict
            status["roof"] = "open" if is_open else "closed"
            status["roof_source"] = "vision"
            status["roof_reason"] = reason
            return

    status["roof"] = "unknown"
    status["roof_source"] = "none"
    status["roof_reason"] = (
        "no stars matched (%s) and the safety camera could not read the roof"
        % ("none detected" if matches is None else "%d matches" % matches))


VISION_CACHE = "allsky_roof_vision.json"


def _vision_roof(cfg):
    """(is_open, reason) from the safety camera, or None if it cannot say.

    Returns None rather than False when the read is unresolved. "The camera
    could not tell" and "the roof is shut" are different facts, and collapsing
    them would report a confident closed on every flaky webcam frame.

    A read costs a Kasa discovery sweep and an exposure ladder on the indoor
    webcam, which is far too much to spend every five minutes for an answer that
    changes twice a day -- so verdicts are cached. But NOT the open one. A cached
    "open" would keep publishing pictures for the length of the cache after the
    roof had shut, which is the one failure this whole gate exists to prevent;
    the verdict that puts a photograph on a public page is always taken fresh.
    """
    ttl = float(cfg.get(PROFILE, {}).get("roof_vision_cache_s", 900))
    cached = _read_vision_cache(ttl)
    if cached is not _MISS:
        return cached

    try:
        from sentry import vision_safety
    except Exception as exc:
        print("allsky: vision unavailable (%s)" % type(exc).__name__)
        return None
    try:
        parked, closed, is_open, _mod = vision_safety.visual_status()
    except Exception as exc:
        print("allsky: vision read failed (%s)" % type(exc).__name__)
        return None

    if is_open:
        return True, "the safety camera sees the roof open"
    if closed:
        out = (False, "the safety camera sees the roof closed")
        _write_vision_cache(out)
        return out
    # Unresolved: either the scope is not parked, so vision declines to read the
    # roof at all, or the frame matched neither template. Not a failure worth
    # re-taking every cycle, so it is cached like a closed verdict.
    _write_vision_cache(None)
    return None


def _cache_path():
    return ROOT / "iris_astronomy" / "scratch" / VISION_CACHE


_MISS = object()


def _read_vision_cache(ttl):
    """The cached verdict if it is still fresh, else ``_MISS`` to force a read.

    A cache MISS and a cached "cannot tell" have to be distinguishable, which is
    why this does not just return None for both. "Cannot tell" is the state an
    imaging run sits in from dusk to dawn -- vision declines to read the roof
    while the scope is unparked -- so folding it into the miss would run a full
    Kasa discovery sweep and an exposure ladder every five minutes until
    daylight, which is the opposite of what the cache is for.
    """
    p = _cache_path()
    if not p.exists():
        return _MISS
    try:
        c = json.loads(p.read_text())
        age = (datetime.now(timezone.utc)
               - datetime.fromisoformat(c["at"])).total_seconds()
    except (ValueError, OSError, KeyError, TypeError):
        return _MISS
    if age > ttl:
        return _MISS
    v = c.get("verdict")
    if v is None:
        return None                     # cached "cannot tell" -- a real hit
    return bool(v[0]), v[1] + " (checked %d min ago)" % (age / 60)


def _write_vision_cache(verdict):
    p = _cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.write_text(json.dumps(
            {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "verdict": list(verdict) if verdict else None}))
    except OSError:
        pass                    # a cache that cannot be written is not an error


def _publish(status, frame):
    out_dir = ROOT / "iris_astronomy" / "scratch"
    out_dir.mkdir(parents=True, exist_ok=True)
    js = out_dir / "allsky_status.json"
    js.write_text(json.dumps(status, indent=1))

    if "--no-push" in sys.argv:
        print("not pushing (--no-push):", json.dumps(status, indent=1))
        return

    pairs = [(js, "allsky.json")]
    # The picture is pushed ONLY when the roof is known open. Not merely hidden
    # on the far side: a JPEG sitting at a public URL is published whatever the
    # page chooses to render, and a photograph of the inside of the observatory
    # is not something to put there by accident.
    if frame is not None and status.get("roof") == "open":
        shown = _display_copy(frame, out_dir)
        if shown is not None:
            # Picture first, so a viewer catching the pair mid-push sees an old
            # count beside a new picture, never a new count beside an old one.
            pairs.insert(0, (shown, "allsky.jpg"))
    live_push.push(pairs)
    print("pushed", len(pairs), "file(s) to", live_push.HOST + ":" + live_push.DEST)


def _display_copy(frame, out_dir):
    """A viewable JPEG of the frame, with compass bearings if they are known."""
    frame = Path(frame)
    jpg = frame.with_suffix(".jpg")
    if not jpg.exists():
        try:
            asi_allsky.save_jpeg(star_count._load(frame), jpg)
        except Exception as exc:
            print("allsky: could not render a JPEG (%s)" % type(exc).__name__)
            return None
    shown = out_dir / "allsky_published.jpg"
    try:
        # Annotates a COPY. The stored frame stays clean, because it is training
        # data for everything that comes later and burnt-in text in the same
        # pixels every frame is exactly the constant a classifier learns instead
        # of the sky.
        return sky_annotate.annotate(jpg, shown, profile=PROFILE) or jpg
    except Exception as exc:
        print("allsky: compass annotation failed (%s); publishing the plain frame"
              % type(exc).__name__)
        return jpg


def _prune(cfg):
    """Hold the frame archive to its configured cap, newest kept."""
    c = cfg.get(PROFILE, {})
    keep = int(c.get("keep_frames", 2000))
    d = ROOT / c.get("capture_dir", "local/allsky_frames")
    if not d.exists():
        return
    fits = sorted(d.glob("allsky_*.fits"), key=lambda p: p.name)
    dropped = 0
    for p in fits[:max(len(fits) - keep, 0)]:
        for q in (p, p.with_suffix(".jpg")):
            try:
                q.unlink()
                dropped += 1
            except OSError:
                pass
    if dropped:
        print("pruned %d old file(s)" % dropped)


if __name__ == "__main__":
    sys.exit(main())
