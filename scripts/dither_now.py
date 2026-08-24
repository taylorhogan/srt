"""
dither_now.py
Dither the scope by commanding PWI4 directly, because NINA cannot.

WHY THIS EXISTS. NINA's Direct Guider dithers with ASCOM PulseGuide, a
temporary RATE change. Measured 2026-08-23 over 46 subs: NINA issued RA pulses
on every dither (its log shows targets spanning +/-17.8 px) and the achieved RA
stayed within +/-0.23 px, while Dec achieved its full +/-23 px. So the dither
was one-dimensional, and a 1-D dither cannot average the banding running along
the un-dithered axis -- measured at 6-8x independent-pixel noise on frame
differences, which is noise that does NOT fall with sqrt(N).

The mount is not the problem. Measured 2026-08-24 with dither_ra_test.py:
PWI4's own mount_offset moves RA and HOLDS it, at 100% of the commanded amount
(30 arcsec of RA -> 16.4 arcsec on sky, exactly 30 x cos(dec) at dec 56.9).
A target-coordinate offset is tracked to and held; a rate nudge is nulled by
the servo returning to the model-predicted position. So dithering through
PWI4's offset API works where pulse guiding does not.

TWO THINGS THAT ARE EASY TO GET WRONG HERE:

  * RA offsets are in RA-COORDINATE arcsec, not great-circle arcsec on the sky.
    They differ by cos(dec) -- a factor of 2 at dec 60, and worse further north.
    Commanding the same number in both axes gives a dither that is far smaller
    in RA than in Dec and shrinks as the target climbs in declination. The
    conversion is applied below.

  * Offsets are set ABSOLUTELY (set_total_arcsec), not accumulated
    (add_arcsec). Adding deltas means any lost or crashed invocation leaves the
    accumulated offset somewhere unknown, and errors compound across a night
    into a slow walk off target. Setting the total makes every dither
    independent of every other and makes a missed one cost nothing.

SAFETY. This moves the mount, and it is a deliberate exception to the usual
"confirm the roof is open first" rule -- justified, not overlooked. The motion
is a handful of arcseconds, while a TRACKING mount is already moving 15
arcsec/second; this dither is under half a second of tracking. If the roof were
shut, tracking would already be the hazard and the dither would be irrelevant.
A roof check would also cost ~10 s of camera grab per sub and cannot even
answer during an imaging run, when the scope is not parked. Instead the script
refuses unless the mount is TRACKING, which is a cheap proxy for "an imaging
run is underway" and blocks casual misuse.

Usage (called from the NINA sequence after each exposure):
    python scripts/dither_now.py
    python scripts/dither_now.py --pixels 25
    python scripts/dither_now.py --reset     # clear offsets, e.g. at end of target
"""
import argparse
import json
import math
import os
import random
import sys
import time

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

STATE_PATH = "local/dither_state.json"

# Dither box half-width, in MAIN-CAMERA pixels. 20 px against a 6.38 px FWHM is
# ~3x, inside the 2x-5x band where dithering actually buys decorrelation;
# below ~2x consecutive subs land a defect on nearly the same sky, above ~5x
# just spends settle time. Matches the value already set in NINA.
DEFAULT_PIXELS = 20.0

# Never make a step smaller than this fraction of the box. A uniform draw
# occasionally lands almost on top of the previous position, and those subs
# are the ones where a hot pixel or a band survives rejection -- rejection
# works by a defect being a LONE outlier at a sky pixel.
MIN_STEP_FRAC = 0.5

SETTLE_TIMEOUT_S = 25.0
SETTLE_EXTRA_S = 1.5


def _state():
    try:
        return json.load(open(STATE_PATH))
    except Exception:
        return {}


def _save(d):
    try:
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        json.dump(d, open(STATE_PATH, "w"), indent=2)
    except Exception:
        pass          # the state file only tunes step size; losing it is benign


def _read_radec(pwi4, n=3, gap=0.25):
    ras, decs = [], []
    for _ in range(n):
        s = pwi4.status()
        ras.append(s.mount.ra_j2000_hours * 15.0)
        decs.append(s.mount.dec_j2000_degs)
        time.sleep(gap)
    return sum(ras) / len(ras), sum(decs) / len(decs)


def pick_target(half_px, last):
    """A random point in the box, at least MIN_STEP_FRAC*half_px from the last.

    Rejection-sampled rather than computed, because the constraint is on the
    STEP while the draw is on the POSITION, and a handful of tries is cheaper
    than the algebra. Falls back to the last draw after 40 attempts so a tight
    constraint can never spin.
    """
    lo = MIN_STEP_FRAC * half_px
    for _ in range(40):
        x = random.uniform(-half_px, half_px)
        y = random.uniform(-half_px, half_px)
        if last is None or math.hypot(x - last[0], y - last[1]) >= lo:
            return x, y
    return x, y


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pixels", type=float, default=None,
                    help="dither box half-width in main-camera pixels "
                         "(default %.0f)" % DEFAULT_PIXELS)
    ap.add_argument("--reset", action="store_true",
                    help="clear all offsets and exit (start/end of a target)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    from configs import config
    from hardware_control.pwi4_client import PWI4
    cfg = config.data()
    arcsec_px = float(cfg["nina"].get("arc_sec_per_pixel", 0.26))
    half_px = args.pixels if args.pixels is not None else DEFAULT_PIXELS

    # Every failure below is reported and returned, never raised. NINA runs
    # this unattended after every sub; a traceback on a dead socket would put a
    # stack trace in the sequence log once per exposure and tell the operator
    # nothing they can act on.
    pwi4 = PWI4()
    try:
        s = pwi4.status()
    except Exception as e:
        print("dither: PWI4 not reachable (%s) -- nothing done" % type(e).__name__)
        return 1
    if not s.mount.is_connected:
        print("dither: mount not connected -- nothing done")
        return 1

    if args.reset:
        pwi4.mount_offset(ra_reset=0, dec_reset=0)
        _save({})
        print("dither: offsets reset")
        return 0

    # Refuses when not tracking: an offset is only a dither while the mount is
    # following a target, and this is the cheap proxy for "a run is underway".
    if not s.mount.is_tracking:
        print("dither: mount is not tracking -- refusing to offset")
        return 1

    dec = s.mount.dec_j2000_degs
    st = _state()
    last = st.get("last_px")
    x_px, y_px = pick_target(half_px, last)

    # On-sky arcsec, then RA converted to RA-COORDINATE arcsec by /cos(dec).
    # Guarded near the pole, where RA coordinate offsets stop being a sensible
    # way to move and the division blows up.
    x_as, y_as = x_px * arcsec_px, y_px * arcsec_px
    cosd = math.cos(math.radians(dec))
    if abs(cosd) < 0.05:
        print("dither: dec %.1f is too close to the pole for an RA offset; "
              "dithering in Dec only for this frame" % dec)
        ra_cmd = 0.0
    else:
        ra_cmd = x_as / cosd

    pwi4.mount_offset(ra_set_total_arcsec=ra_cmd, dec_set_total_arcsec=y_as)

    # Wait for the mount to STOP MOVING, rather than sleeping a fixed guess --
    # the next exposure must not start mid-move.
    #
    # Settling is detected as position stability, not as arrival at a predicted
    # place. Offsets are set as an absolute TOTAL, so the distance this
    # particular dither has to travel depends on the previous total, which this
    # process does not reliably know (the state file is best-effort and a
    # missed invocation makes it wrong). Stability needs no such knowledge.
    STABLE_ARCSEC = 0.3
    t0 = time.time()
    prev = _read_radec(pwi4, n=2, gap=0.15)
    settled = False
    while time.time() - t0 < SETTLE_TIMEOUT_S:
        time.sleep(0.4)
        cur = _read_radec(pwi4, n=2, gap=0.15)
        moved = math.hypot((cur[0] - prev[0]) * cosd * 3600.0,
                           (cur[1] - prev[1]) * 3600.0)
        prev = cur
        if moved < STABLE_ARCSEC and not pwi4.status().mount.is_slewing:
            settled = True
            break
    if not settled:
        # Say so rather than pretend: an unsettled dither means the sub that
        # follows may be trailed, and that is worth seeing in the log.
        print("dither: WARNING did not settle within %.0fs" % SETTLE_TIMEOUT_S)
    time.sleep(SETTLE_EXTRA_S)

    _save({"last_px": [x_px, y_px], "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "ra_cmd_arcsec": round(ra_cmd, 2), "dec_cmd_arcsec": round(y_as, 2)})
    if not args.quiet:
        print("dither: total offset -> %+.1f, %+.1f px  (%+.2f, %+.2f arcsec on "
              "sky; RA commanded %+.2f arcsec of RA at dec %.1f), settled in %.1fs"
              % (x_px, y_px, x_as, y_as, ra_cmd, dec, time.time() - t0))
    return 0


if __name__ == "__main__":
    # A dither that fails must never take the night down with it: a missed
    # dither costs one sub's worth of decorrelation, whereas an aborted
    # sequence costs the night. Report and carry on.
    try:
        sys.exit(main())
    except Exception as exc:      # noqa: BLE001
        print("dither: FAILED (%s: %s) -- imaging continues" % (type(exc).__name__, exc))
        sys.exit(1)
