#!/usr/bin/env python3
"""Rain prediction and detection from the all-sky camera's own video.

The signal
----------
Rain is not subtle on this camera, because its infrared illuminator lights
raindrops a few centimetres from the lens. What arrives is near-field, wildly
defocused streaks, and every frame holds a fresh set of drops.

The measurement is the **fraction of pixels that change between adjacent video
frames**, and it has to be that rather than brightness: the camera has no manual
exposure, so auto-exposure renormalises every frame and absolute level carries
no weather information (it tracks the illuminator, which is why a rain frame
reads 89 ADU against 16.7 on a clear night -- that gap is the LED, not the
sky). Sidereal drift between adjacent frames is 0.0078 px, so stars are static
to a hundredth of a pixel and anything that moves is not a star, with a 100x
margin.

Measured on 7.5 days of the camera's own SD-card footage (2026-08-03..08-10):

    clear night   0.1 - 1.1 %        rain   16 - 100 %
    daylight rain 3 - 7 %            <- NOT detected, see the gate below

Only inside MEASURED_RADIUS_PX of the optical axis, which excludes the tree
line entirely so wind cannot fake it -- the same region the star completeness
is measured in, and for the same reason.

Why the gates are what they are
-------------------------------
**Night only.** In daylight there is no illuminator, so drops are not lit: an
hour with 23.4 mm falling read 7%. Gated on the scheduler's own definition of
night (sun below -10 deg), which also removes the one false-positive mode
found -- twilight, where the frame level drifted 8.4 ADU *within a single
burst* as dawn came up and auto-exposure hunted, registering the whole frame as
moving.

**Persistence, not a single sample.** The prediction threshold of 1% overlaps
the clear-night range, so one sample at 1% is worthless. Requiring N consecutive
samples is what makes a low threshold usable: on the 2026-08-05 storm it
correctly ignored an isolated 14.9% spike and several 2-3% blips that all
reverted immediately. A sample the camera failed to deliver is not a sample:
it neither joins nor breaks the run unless the outage runs past
`rain_gap_tolerance_s`, because a dead camera says nothing about the sky.

**Two thresholds, because onset is a ramp not a step.** Measured on two storms,
1% -> 10% took 26 and 38 minutes, which is the whole reason a prediction is
possible at all. But escalation past 10% is NOT gradual -- 10% -> 30% took 26
minutes on one storm and 2 minutes on the other -- so DETECT means "already
raining hard", never "getting worse slowly".

What it cannot do
-----------------
Nothing in daylight. Little or nothing for a storm arriving at dusk, because
the camera only becomes useful as the illuminator comes on. Cloud is not a
usable precursor either: on 2026-08-05 stars collapsed from 13 to near zero at
21:40 and rain did not start for another 2.25 hours.

Usage:  python sentry/rain_detect.py --status
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from configs import config

STATE_FILE = "local/rain_state.json"
LOG_FILE = "local/rain_log.jsonl"

# Defaults; every one is overridable from cfg["sky camera"].
DEFAULTS = {
    "rain_predict_pct": 1.0,     # 3x this = onset ramp has started
    "rain_detect_pct": 10.0,     # 3x this = raining hard
    "rain_persist": 3,           # consecutive samples required
    "rain_alert_gap_s": 6 * 3600,
    "rain_gap_tolerance_s": 1200,  # a gap shorter than this does not break a run
    "rain_moon_boost_pct": 3.0,  # added to the predict threshold at full moon
    "rain_diff_adu": 12.0,       # per-pixel change that counts as "moved"
    "rain_burst_seconds": 2.0,   # only needs a handful of adjacent frames
    "rain_min_frames": 4,
}


def _root():
    return Path(__file__).resolve().parents[1]


def _tunables(cfg=None):
    cam = (cfg or config.data()).get("sky camera", {})
    return {k: cam.get(k, v) for k, v in DEFAULTS.items()}


# ------------------------------------------------------------------ metric

def motion_fraction(frames, diff_adu=12.0, cx=None, cy=None, radius=None):
    """Fraction of central-sky pixels (%) changing by > diff_adu between frames.

    Takes the MAX change across the burst rather than the mean: a drop crosses
    a given pixel in one frame pair, so averaging over the burst would divide
    the signal by the number of pairs.
    """
    import numpy as np
    from sentry import plate_solve

    if frames is None or len(frames) < 2:
        return None
    a = np.asarray(frames, dtype=np.float32)
    h, w = a.shape[1], a.shape[2]
    sol = plate_solve.load()
    if cx is None:
        cx = sol["cx"] if sol else w / 2.0
    if cy is None:
        cy = sol["cy"] if sol else h / 2.0
    if radius is None:
        radius = plate_solve.MEASURED_RADIUS_PX
    # Scale the geometry if the frames are not full resolution.
    if sol is not None and w != int(sol["cx"] * 2):
        s = w / float(sol["cx"] * 2)
        cx, cy, radius = cx * s, cy * s, radius * s
    yy, xx = np.mgrid[0:h, 0:w]
    sky = ((xx - cx) ** 2 + (yy - cy) ** 2) < radius * radius
    if not sky.any():
        return None
    moved = np.abs(np.diff(a, axis=0)).max(0)[sky]
    return float(100.0 * (moved > diff_adu).mean())


def measure(cfg=None, keep_burst=False):
    """Capture a short burst and return {moving_pct, n_frames, burst}.

    Returns None if the camera did not give us enough frames -- an unreachable
    camera must read as "no measurement", never as "no rain".
    """
    from sentry import sky_camera
    cfg = cfg or config.data()
    t = _tunables(cfg)
    path, _n = sky_camera.capture_burst(seconds=t["rain_burst_seconds"], cfg=cfg)
    if path is None:
        return None
    try:
        frames = sky_camera.decode_burst(path, limit=16)
        if not frames or len(frames) < t["rain_min_frames"]:
            return None
        pct = motion_fraction(frames, diff_adu=t["rain_diff_adu"])
        if pct is None:
            return None
        return {"moving_pct": round(pct, 3), "n_frames": len(frames),
                "burst": str(path)}
    finally:
        if not keep_burst:
            try:
                Path(path).unlink()
            except OSError:
                pass


# ------------------------------------------------------------------- state

def _load_state():
    p = _root() / STATE_FILE
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"samples": [], "last_predict": 0, "last_detect": 0}


def _save_state(st):
    p = _root() / STATE_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
    tmp.replace(p)


def _log(entry):
    p = _root() / LOG_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def _run_over(samples, thr, n):
    """True when the last n samples all exceed thr. Requires n of them."""
    if len(samples) < n:
        return False
    return all(s["moving_pct"] > thr for s in samples[-n:])


def _consecutive_tail(samples, gap_s):
    """Drop everything before the last break in cadence.

    "Consecutive" has to mean consecutive in TIME, not merely adjacent in the
    list. Once a dropout is allowed to leave the run standing, two samples that
    sit side by side in the list can be an hour apart, and three of those are
    not evidence of a ramp.
    """
    cut = 0
    for i in range(1, len(samples)):
        if samples[i]["t"] - samples[i - 1]["t"] > gap_s:
            cut = i
    return samples[cut:]


def moon_onset_boost(when=None, cfg=None):
    """Percentage points added to the predict threshold while the moon is up.

    The full moon's glare gradient reads as sustained 1-3% central-sky motion
    -- measured across the 2026-08-25..29 waxing week, one marginal false
    "predict" per night at 1.0-1.4%, peaking at 2.9%, growing with lunar
    illumination -- while every real event in rain_log.jsonl reads 10-99%. So
    the predict threshold rises with the moon: boost = rain_moon_boost_pct x
    illuminated fraction when the moon is above 5 deg, zero otherwise. At full
    moon that puts onset at 4.0%, above the observed glare ceiling; a dark sky
    keeps the sensitive 1.0%. Detect (10%) is never adjusted -- moonlight
    cannot reach it.

    On any failure (no astropy, no plate solution for the site, bad clock)
    this returns 0.0: the failure direction is a false alert on a moonlit
    night, never a missed storm on a dark one.
    """
    t = _tunables(cfg)
    full = float(t["rain_moon_boost_pct"])
    if full <= 0:
        return 0.0
    try:
        import numpy as np
        from astropy.coordinates import AltAz, EarthLocation, get_body
        from astropy.time import Time

        loc = (cfg or config.data())["location"]
        site = EarthLocation.from_geodetic(loc["longitude"], loc["latitude"],
                                           loc.get("elevation", 0.0))
        ts = Time(when if when is not None else datetime.now(timezone.utc))
        moon = get_body("moon", ts, site)
        if moon.transform_to(AltAz(obstime=ts, location=site)).alt.deg <= 5.0:
            return 0.0
        sun = get_body("sun", ts, site)
        illum = (1.0 - np.cos(np.radians(sun.separation(moon).deg))) / 2.0
        return round(full * float(illum), 3)
    except Exception:
        return 0.0


def _allowed(last, now, gap):
    """Has the rate-limit window expired? A falsy `last` means never alerted.

    Deliberately not `now - last >= gap`: with a "never" sentinel of 0 that
    expression is only true because Unix timestamps are large, which works in
    production and silently blocks every alert under any other clock.
    """
    return (not last) or (now - float(last)) >= gap


def evaluate(moving_pct, night, sun_alt, now=None, cfg=None, state=None,
             onset_boost=0.0):
    """Fold one sample into the state machine. Returns (state, alert or None).

    alert is None, "predict" or "detect". Pure apart from the clock, so the
    thresholds and rate limiting are testable without a camera. `onset_boost`
    (from moon_onset_boost) raises the predict threshold only -- the caller
    computes it so this stays a pure function of its arguments.
    """
    t = _tunables(cfg)
    now = float(now if now is not None else time.time())
    st = state if state is not None else _load_state()
    st.setdefault("samples", [])
    st.setdefault("last_predict", 0)
    st.setdefault("last_detect", 0)

    # A daylight sample is not evidence either way. Drop the run rather than
    # let a day reading sit in the history and break a night sequence later.
    if not night:
        st["samples"] = []
        return st, None

    # A camera that returned no frames has told us nothing about the sky, so it
    # must not read as "no rain". Hold the run untouched and let the spacing
    # rule below decide whether it survives once a real sample arrives: a brief
    # hiccup keeps the ramp, a long outage breaks it. Clearing here instead --
    # as this did until 2026-08-13 -- means any dropout during an onset silently
    # discards the evidence the prediction is built from.
    if moving_pct is None:
        return st, None

    st["samples"].append({"t": now, "moving_pct": float(moving_pct)})
    # A gap means the run is broken -- 3 samples spanning last night and this
    # one are not 3 consecutive samples.
    st["samples"] = _consecutive_tail(
        st["samples"], float(t["rain_gap_tolerance_s"]))[-12:]

    gap = t["rain_alert_gap_s"]
    n = int(t["rain_persist"])
    alert = None
    if (_run_over(st["samples"], t["rain_detect_pct"], n)
            and _allowed(st.get("last_detect"), now, gap)):
        alert = "detect"
        st["last_detect"] = now
        # Stamp the prediction limiter too. Rain hard enough to detect always
        # satisfies the predict condition as well, so the next sample would
        # otherwise fall through to the elif and announce "rain likely within
        # ~25-40 min" while it is already pouring -- which is what happened at
        # 00:45 on 2026-08-13, five minutes after a 99.3% detect. The stamping
        # is deliberately ONE WAY: a prediction does not suppress the detection
        # that confirms it.
        st["last_predict"] = now
    elif (_run_over(st["samples"], t["rain_predict_pct"] + float(onset_boost), n)
            and _allowed(st.get("last_predict"), now, gap)):
        alert = "predict"
        st["last_predict"] = now
    return st, alert


# ------------------------------------------------------------------ notify

def _notify(kind, moving_pct, sun_alt, image, cfg, onset_boost=0.0):
    from utils import pushover
    t = _tunables(cfg)
    if kind == "detect":
        msg = ("RAIN DETECTED - %.1f%% of the sky frame is moving (threshold %.0f%%). "
               "Treat as already raining hard; escalation past this point has been "
               "as fast as 2 minutes." % (moving_pct, t["rain_detect_pct"]))
        prio = 1
    else:
        eff = t["rain_predict_pct"] + float(onset_boost)
        msg = ("Rain likely within ~25-40 min - %.1f%% of the sky frame is moving "
               "(threshold %.1f%%, %d consecutive samples). Onset ramp has started."
               % (moving_pct, eff, int(t["rain_persist"])))
        if onset_boost:
            msg += " Threshold is moon-raised; this cleared it anyway."
        prio = 0
    msg += " Sun %.0f deg." % sun_alt
    try:
        return bool(pushover.push_message(msg, image=str(image) if image else None,
                                          priority=prio))
    except Exception:
        return False


def step(status, frame, cfg=None):
    """One cycle. Called by sky_monitor after it has captured its still.

    `status` is the monitor's status dict (for night/sun_alt); `frame` is the
    still to attach to any alert. Never raises -- rain detection must not be
    able to sink a capture.
    """
    cfg = cfg or config.data()
    try:
        night = bool(status.get("night"))
        sun_alt = float(status.get("sun_alt_deg", 0.0))
        m = measure(cfg) if night else None
        pct = m["moving_pct"] if m else None
        boost = moon_onset_boost(cfg=cfg) if night else 0.0
        st, alert = evaluate(pct, night, sun_alt, cfg=cfg, onset_boost=boost)
        entry = {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "night": night, "sun_alt_deg": round(sun_alt, 1),
                 "moving_pct": pct, "alert": alert,
                 # The predict threshold this sample was judged against; the
                 # transparency chart shades "wet" from this rather than the
                 # static default, so a moonlit night is not painted yellow.
                 "onset": round(_tunables(cfg)["rain_predict_pct"] + boost, 3),
                 "run": len(st.get("samples", []))}
        if alert:
            entry["sent"] = _notify(alert, pct, sun_alt, frame, cfg,
                                    onset_boost=boost)
        _save_state(st)
        _log(entry)
        if pct is not None:
            status["rain_moving_pct"] = pct
        if alert:
            status["rain_alert"] = alert
        return entry
    except Exception as exc:
        try:
            _log({"t": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "error": "%s: %s" % (type(exc).__name__, exc)})
        except Exception:
            pass
        return None


if __name__ == "__main__":
    cfg = config.data()
    t = _tunables(cfg)
    st = _load_state()
    print("thresholds: predict >%.1f%%, detect >%.1f%%, %d consecutive, "
          "one alert of each per %.0f h"
          % (t["rain_predict_pct"], t["rain_detect_pct"], int(t["rain_persist"]),
             t["rain_alert_gap_s"] / 3600.0))
    print("run in progress: %d sample(s)" % len(st.get("samples", [])))
    for k in ("last_predict", "last_detect"):
        v = st.get(k, 0)
        print("  %-13s %s" % (k, datetime.fromtimestamp(v).isoformat(timespec="seconds")
                              if v else "never"))
    if "--measure" in sys.argv:
        print("measuring...", flush=True)
        print(measure(cfg))
