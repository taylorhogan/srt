"""station_track.py — find the space station in a recorded pass and measure it.

    python scripts/station_track.py local/iss/tiangong_20260905_202827.json

THE PROBLEM THIS SOLVES, AND WHY THE OBVIOUS APPROACH FAILS
"What is the brightest thing in each frame?" is the wrong question. A moonlit
cloud edge outshines a space station most of the time, so the per-frame answer
is usually weather, and fitting a line through those answers produces confident
nonsense -- a track, with a speed, that is not the spacecraft.

So the question is turned around: WHICH STRAIGHT, CONSTANT-SPEED PATH DO THE
MOST DETECTIONS AGREE WITH? Clouds cannot conspire to drift in a straight line
at a fixed rate, and a satellite in a short pass cannot do anything else. That
one change of question does the whole job of telling a spacecraft from weather,
and it is why this is a RANSAC over (x, y, t) rather than a brightest-pixel
tracker.

The consensus model is deliberately CONSTANT VELOCITY IN PIXELS, not in angle.
Over a 2-3 minute pass through a 104-degree fisheye the true angular rate is
not constant and the projection is not linear, so this is an approximation --
but it only has to be good enough to separate one coherent object from
incoherent weather, and a wrong model that is wrong SMOOTHLY still collects the
satellite's own detections and rejects clouds. The reported angular rate is
computed afterwards from the plate solution, on the inliers, where the geometry
is done properly.

Detection is on a frame differenced against a rolling median. A satellite is
the thing that is bright HERE and was not bright here a moment ago; stars are
fixed, and cloud is slow and diffuse. Nothing is thresholded on absolute
brightness, because the whole point is that the spacecraft is often not the
brightest thing present.

Outputs an annotated JPEG and a JSON summary beside the recording.
"""
import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import cv2
import numpy as np

# Detection
BG_FRAMES     = 9      # rolling median depth. Odd, and long enough that a
                       # fast mover never sits in the median of its own
                       # neighbourhood.
MAX_PER_FRAME = 4      # keep the few best candidates per frame, not one: the
                       # station is frequently not the brightest thing, and
                       # keeping only the winner throws it away before RANSAC
                       # ever sees it.
MIN_SIGMA     = 4.0    # above the difference frame's own noise
MIN_AREA      = 2      # px

# Consensus
TOL_PX        = 6.0    # how far a detection may sit from the fitted path
MIN_INLIERS   = 12
ITERS         = 4000
MIN_SPEED     = 2.0    # px/s. Below this it is drifting cloud, not a station.
MAX_SPEED     = 400.0

# How far the fitted track may sit from where the satellite actually was.
# The watcher computed that path to arm the recorder in the first place, so
# checking against it costs nothing and is far stronger evidence than the
# inlier count. It exists because both ISS recordings to date produced a
# confident straight track -- annotated image, quoted angular rate -- that was
# NOT the station: 2026-09-23 fitted 27 of 10588 candidates at 0.145 deg/s in
# the south-west while the ISS was in the north-west and then in eclipse. A
# real pass scores 281-484 inliers at 0.7-1.05 deg/s. 20 degrees is generous:
# a genuine track lands within a degree or two once the plate solution is good.
MAX_MISS_DEG  = 20.0
SAMPLES       = 7      # points along the track compared with the TLE
# A station in low orbit cannot crawl. Overhead the ISS sweeps ~1.1 deg/s and
# even down at 35 degrees elevation it manages ~0.4, so anything slower is an
# aircraft, a cloud edge or a line fitted through noise. Measured here: real
# tracks 0.70-1.05 deg/s, the three false ones 0.076, 0.117 and 0.145 -- a
# factor of five, which is a far cleaner separation than the positional miss
# (real 3-18 deg against false 26-62) because that one also carries the
# straight-line model's own error on a near-zenith pass.
MIN_DEG_PER_S = 0.30


def detect(video, fps, downscale=1, verbose=True):
    """[(t_s, x, y, peak)] candidate moving points, in FULL-frame pixels."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit("could not open %s" % video)
    buf, dets, idx = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if downscale > 1:
            g = cv2.resize(g, (g.shape[1] // downscale, g.shape[0] // downscale))
        buf.append(g.astype(np.float32))
        if len(buf) > BG_FRAMES:
            buf.pop(0)
        if len(buf) == BG_FRAMES:
            mid = BG_FRAMES // 2
            cur = buf[mid]
            bg = np.median(np.stack(buf[:mid] + buf[mid + 1:]), axis=0)
            d = cur - bg
            sd = float(np.std(d)) or 1.0
            m = (d > MIN_SIGMA * sd).astype(np.uint8)
            n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
            cand = []
            for i in range(1, n):
                if st[i, cv2.CC_STAT_AREA] < MIN_AREA:
                    continue
                y0, y1 = st[i, cv2.CC_STAT_TOP], st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT]
                x0, x1 = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_LEFT] + st[i, cv2.CC_STAT_WIDTH]
                cand.append((float(d[y0:y1, x0:x1].max()),
                             float(cen[i][0]) * downscale,
                             float(cen[i][1]) * downscale))
            cand.sort(reverse=True)
            # frame index of the CENTRE of the buffer, which is what `cur` is
            t = (idx - mid) / fps
            for peak, x, y in cand[:MAX_PER_FRAME]:
                dets.append((t, x, y, peak))
        idx += 1
        if verbose and idx % 400 == 0:
            print("  %d frames, %d candidates" % (idx, len(dets)))
    cap.release()
    if verbose:
        print("  %d frames, %d candidates total" % (idx, len(dets)))
    return dets, idx


def angular_sep(alt1, az1, alt2, az2):
    """Great-circle separation between two alt/az directions, in degrees. Pure."""
    a1, a2 = math.radians(alt1), math.radians(alt2)
    dz = math.radians(az1 - az2)
    c = (math.sin(a1) * math.sin(a2)
         + math.cos(a1) * math.cos(a2) * math.cos(dz))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def where_was_it(sat_name, times, tle=None):
    """[(alt, az, sunlit)] for *sat_name* at each datetime, or None.

    From the TLE stored in the recording's sidecar when there is one (the
    elements the watcher armed the pass with), else the current cache. None
    means the check could not be made (no TLE, no ephemeris, no network) and
    is treated as "unchecked" rather than as a failure -- an absent
    cross-check must not invent a verdict either way.
    """
    try:
        from skyfield.api import EarthSatellite
        from sentry import station_watch as sw
        site, eph, ts = sw._sky()
        if tle and tle.get("l1") and tle.get("l2"):
            l1, l2 = tle["l1"], tle["l2"]
        else:
            defn = next(d for d in sw.SATELLITES if d["name"] == sat_name)
            l1, l2 = sw.get_tle(defn)
        sat = EarthSatellite(l1, l2, sat_name, ts)
        out = []
        for when in times:
            tx = ts.from_datetime(when)
            alt, az, _ = (sat - site).at(tx).altaz()
            out.append((float(alt.degrees), float(az.degrees),
                        bool(sat.at(tx).is_sunlit(eph))))
        return out
    except Exception as exc:              # noqa: BLE001
        print("  (could not check against the TLE: %s: %s)" % (type(exc).__name__, exc))
        return None


def consensus(dets, iters=ITERS, tol=TOL_PX, seed=7):
    """The straight constant-speed path the most detections agree with."""
    if len(dets) < MIN_INLIERS:
        return None
    rng = np.random.default_rng(seed)
    A = np.array([(d[0], d[1], d[2]) for d in dets])   # t, x, y
    best = None
    n = len(A)
    for _ in range(iters):
        i, j = rng.integers(0, n, 2)
        dt = A[j, 0] - A[i, 0]
        if abs(dt) < 1.0:            # need a real time base for a velocity
            continue
        vx = (A[j, 1] - A[i, 1]) / dt
        vy = (A[j, 2] - A[i, 2]) / dt
        speed = float(np.hypot(vx, vy))
        if not (MIN_SPEED <= speed <= MAX_SPEED):
            continue
        px = A[i, 1] + vx * (A[:, 0] - A[i, 0])
        py = A[i, 2] + vy * (A[:, 0] - A[i, 0])
        r = np.hypot(A[:, 1] - px, A[:, 2] - py)
        inl = r < tol
        # One detection per frame at most: a bright cloud edge sitting still
        # can otherwise contribute the same blob for hundreds of frames and
        # win on raw count without describing any motion at all.
        cnt = len(np.unique(A[inl, 0]))
        if best is None or cnt > best[0]:
            best = (cnt, inl.copy(), (vx, vy))
    if best is None or best[0] < MIN_INLIERS:
        return None
    inl = best[1]
    # Refit properly on the inliers: least squares x(t) and y(t).
    t = A[inl, 0]
    cx = np.polyfit(t, A[inl, 1], 1)
    cy = np.polyfit(t, A[inl, 2], 1)
    return {"n_inliers": int(best[0]), "mask": inl,
            "vx": float(cx[0]), "vy": float(cy[0]),
            "x0": float(cx[1]), "y0": float(cy[1]),
            "t_first": float(t.min()), "t_last": float(t.max())}


def analyse(recording, downscale=2, tol=TOL_PX, refresh=False):
    """Measure one recording: the dict that goes into <base>_track.json.

    Always returns a dict with "is_satellite" -- False with a "reject_reason"
    when nothing coherent was found -- so a caller can post either answer.
    Private keys (_fit, _dets, _video, _fps, _nframes) carry what make_clip
    needs; they are never written to the JSON.
    """
    base = os.path.splitext(recording)[0]
    meta = json.load(open(base + ".json"))
    video = base + ".h264"
    t0 = datetime.fromisoformat(meta["capture_start"])
    t1 = datetime.fromisoformat(meta["capture_end"])
    p = meta["pass"]

    print("%s pass, peak alt %.1f deg at %s"
          % (p["sat"], p["peak_alt_deg"], p["peak"][11:19]))

    probe = cv2.VideoCapture(video)
    nframes = 0
    while probe.grab():
        nframes += 1
    probe.release()
    fps = nframes / (t1 - t0).total_seconds()
    print("  %d frames over %.0f s -> %.2f fps" % (nframes, (t1 - t0).total_seconds(), fps))

    # Detection is the expensive half (a full decode of every frame at
    # 2560x1440) and the fitting is the half worth iterating on, so the
    # candidates are cached beside the recording. Keyed on the downscale, and
    # invalidated if the video is newer than the cache.
    cache = "%s_dets_d%d.npz" % (base, downscale)
    if (not refresh and os.path.exists(cache)
            and os.path.getmtime(cache) >= os.path.getmtime(video)):
        dets = [tuple(r) for r in np.load(cache)["dets"]]
        print("re-using %d cached candidates (%s)" % (len(dets), cache))
    else:
        print("detecting movers...")
        dets, _ = detect(video, fps, downscale=downscale)
        np.savez_compressed(cache, dets=np.array(dets))
        print("cached %d candidates -> %s" % (len(dets), cache))
    print("finding the consensus track...")
    fit = consensus(dets, tol=tol)
    if fit is None:
        print("NO TRACK: no straight constant-speed path collected "
              "%d agreeing frames." % MIN_INLIERS)
        print("That is a real answer, not a failure -- an overcast pass "
              "leaves nothing coherent to find.")
        out = {"sat": p["sat"], "peak_alt_deg": p["peak_alt_deg"], "peak": p["peak"],
               "candidates": len(dets), "is_satellite": False,
               "reject_reason": ("no straight constant-speed path collected %d "
                                 "agreeing frames" % MIN_INLIERS)}
        with open(base + "_track.json", "w") as fh:
            json.dump(out, fh, indent=2)
        return out

    speed_px = float(np.hypot(fit["vx"], fit["vy"]))
    span = fit["t_last"] - fit["t_first"]
    print("  %d frames agree, over %.1f s" % (fit["n_inliers"], span))
    print("  %.1f px/s in the image" % speed_px)

    # Angular rate from the plate solution, on the fitted endpoints -- the
    # projection is a 104-degree fisheye, so pixels are not degrees and the
    # conversion has to be done with the real geometry, not a scale factor.
    out = {"sat": p["sat"], "peak_alt_deg": p["peak_alt_deg"],
           "peak": p["peak"], "frames_agreeing": fit["n_inliers"],
           "track_seconds": round(span, 1),
           "speed_px_s": round(speed_px, 2),
           "candidates": len(dets)}
    try:
        from sentry import plate_solve
        sol = plate_solve.load()
        if sol is not None:
            pts = []
            track_altaz = []            # sampled along the whole track
            for tt in (fit["t_first"], fit["t_last"]):
                x = fit["x0"] + fit["vx"] * tt
                y = fit["y0"] + fit["vy"] * tt
                alt, az = plate_solve.pixel_to_altaz(sol, x, y)
                pts.append((alt, az, x, y))
            (a1, z1, ex1, ey1), (a2, z2, ex2, ey2) = pts
            a1, z1, a2, z2 = (float(np.ravel(v)[0]) for v in (a1, z1, a2, z2))
            out.update(px_first=[round(ex1, 1), round(ey1, 1)],
                       px_last=[round(ex2, 1), round(ey2, 1)])
            # ravel: pixel_to_altaz hands back arrays, so unit_from_altaz
            # stacks on the last axis and returns (1,3). np.dot of two of
            # those is a shape error, not a dot product.
            v1 = np.ravel(plate_solve.unit_from_altaz(z1, a1))
            v2 = np.ravel(plate_solve.unit_from_altaz(z2, a2))
            sep = float(np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1, 1))))
            out.update(alt_first=round(a1, 1), az_first=round(z1, 1),
                       alt_last=round(a2, 1), az_last=round(z2, 1),
                       arc_deg=round(sep, 2),
                       deg_per_s=round(sep / span, 3) if span else None)
            print("  %.1f deg of sky in %.1f s -> %.3f deg/s"
                  % (sep, span, sep / span))
            print("  from alt %.1f az %.1f to alt %.1f az %.1f" % (a1, z1, a2, z2))
            # Sampled ALONG the track, not just its ends. The fit is a
            # straight line in pixels while the true path curves across a
            # 104-degree fisheye, so the ends are exactly where the two
            # disagree most: the near-zenith Tiangong passes miss by 16-19
            # deg at the endpoints and a couple of degrees in the middle.
            # Comparing the whole track and taking the median keeps the
            # cross-check about identity rather than about the model's shape.
            for k in range(SAMPLES):
                tt = fit["t_first"] + (fit["t_last"] - fit["t_first"]) * k / (SAMPLES - 1.0)
                x = fit["x0"] + fit["vx"] * tt
                y = fit["y0"] + fit["vy"] * tt
                alt, az = plate_solve.pixel_to_altaz(sol, x, y)
                track_altaz.append((tt, float(np.ravel(alt)[0]), float(np.ravel(az)[0])))
        else:
            print("  (no plate solution stored; pixel rate only)")
    except Exception as exc:                      # noqa: BLE001
        print("  (angular rate unavailable: %s: %s)" % (type(exc).__name__, exc))

    # Is this the satellite at all? The inlier count cannot answer that -- a
    # straight line through noise looks exactly like a short faint pass -- but
    # the TLE can, because we know where the thing was supposed to be.
    track_altaz = locals().get("track_altaz") or []
    ok, why = True, None
    if track_altaz:
        pred = where_was_it(p["sat"],
                            [t0 + timedelta(seconds=tt) for tt, _, _ in track_altaz],
                            tle=meta.get("tle"))
        if pred is None:
            out["checked_against_tle"] = False
        else:
            misses = [angular_sep(a, z, pa, pz)
                      for (_, a, z), (pa, pz, _) in zip(track_altaz, pred)]
            lit = [q[2] for q in pred]
            median_miss = float(np.median(misses))
            out.update(checked_against_tle=True,
                       miss_deg_median=round(median_miss, 1),
                       miss_deg_samples=[round(m, 1) for m in misses],
                       sunlit_during_track=lit,
                       predicted_first=[round(pred[0][0], 1), round(pred[0][1], 1)],
                       predicted_last=[round(pred[-1][0], 1), round(pred[-1][1], 1)])
            print("  %s was at alt %.1f az %.1f -> alt %.1f az %.1f (sunlit %s)"
                  % (p["sat"], pred[0][0], pred[0][1], pred[-1][0], pred[-1][1],
                     "yes" if any(lit) else "NO, in eclipse"))
            print("  track misses it by %.0f deg (median of %d samples: %s)"
                  % (median_miss, len(misses),
                     " ".join("%.0f" % m for m in misses)))
            if median_miss > MAX_MISS_DEG:
                ok = False
                why = ("track sits %.0f deg from where %s actually was"
                       % (median_miss, p["sat"]))
            elif out.get("deg_per_s") and out["deg_per_s"] < MIN_DEG_PER_S:
                ok = False
                why = ("%.3f deg/s is too slow for anything in low orbit"
                       % out["deg_per_s"])
            elif not any(lit):
                ok = False
                why = "%s was in Earth's shadow for the whole track" % p["sat"]
    out["is_satellite"] = ok
    if why:
        out["reject_reason"] = why
        print("  REJECTED: %s" % why)
        print("  The recording is fine; there was simply no %s in it to find."
              % p["sat"])

    # Annotate the frame nearest the peak, with the track drawn across it.
    cap = cv2.VideoCapture(video)
    want = int(((datetime.fromisoformat(p["peak"]) - t0).total_seconds()) * fps)
    frame = None
    for i in range(nframes):
        ok, f = cap.read()
        if not ok:
            break
        if i >= want:
            frame = f
            break
    cap.release()
    if frame is not None:
        x1 = int(fit["x0"] + fit["vx"] * fit["t_first"])
        y1 = int(fit["y0"] + fit["vy"] * fit["t_first"])
        x2 = int(fit["x0"] + fit["vx"] * fit["t_last"])
        y2 = int(fit["y0"] + fit["vy"] * fit["t_last"])
        colour = (0, 215, 255) if ok else (60, 60, 255)
        cv2.line(frame, (x1, y1), (x2, y2), colour, 3)
        cv2.circle(frame, (x1, y1), 26, colour, 3)
        cv2.circle(frame, (x2, y2), 26, colour, 3)
        A = np.array([(d[0], d[1], d[2]) for d in dets])
        for (tt, xx, yy) in A[fit["mask"]]:
            cv2.circle(frame, (int(xx), int(yy)), 7, (0, 255, 120), 2)
        cv2.putText(frame, "%s%s  %s  peak alt %.0f deg"
                    % ("" if ok else "NOT ", p["sat"],
                       p["peak"][:19].replace("T", " "), p["peak_alt_deg"]),
                    (40, 1330), cv2.FONT_HERSHEY_SIMPLEX, 1.6, colour, 4)
        cv2.putText(frame, "%d frames agree of %d candidates   %s"
                    % (fit["n_inliers"], len(dets),
                       ("%.3f deg/s" % out["deg_per_s"]) if out.get("deg_per_s")
                       else "%.0f px/s" % speed_px),
                    (40, 1390), cv2.FONT_HERSHEY_SIMPLEX, 1.3, colour, 3)
        if why:
            cv2.putText(frame, why, (40, 1440 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 3)
        img_out = base + "_track.jpg"
        cv2.imwrite(img_out, frame)
        out["image"] = img_out
        print("  wrote %s" % img_out)

    with open(base + "_track.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("  wrote %s" % (base + "_track.json"))
    out.update(_fit=fit, _dets=dets, _video=video, _fps=fps, _nframes=nframes)
    return out


# The clip: the seconds around the fitted track, cropped to where it happened,
# at real speed, with the agreeing detections drawn as a trail. Cut only for a
# confirmed track -- a line fitted through noise makes an equally convincing
# movie, and the annotated still has fooled the eye that way before.
CLIP_PAD_S    = 5.0    # seconds of context either side of the track
CLIP_MAX_S    = 60.0
CLIP_SIZE     = 800    # output square, px
CLIP_MARGIN   = 120    # px of frame kept around the track's ends
CLIP_STRIP    = 56     # caption strip height
CLIP_TRAIL_LAG_S = 1.5 # the trail stops this far behind the station, so the
                       # dots never sit on top of the thing being shown
CLIP_STRETCH  = (1.0, 99.7)   # percentiles of the first crop mapped to 0..255,
                              # fixed for the whole clip so it does not flicker


def _stretch_lut(crop, pcts=CLIP_STRETCH):
    """A 256-entry LUT that maps the crop's own dark range onto full scale."""
    lo, hi = np.percentile(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), pcts)
    hi = max(hi, lo + 8)
    lut = np.clip((np.arange(256) - lo) * 255.0 / (hi - lo), 0, 255)
    return lut.astype(np.uint8)


def _fit_caption(label, width, scale=0.85, thickness=2, pad=12):
    """Shrink the font until the caption fits the strip."""
    while scale > 0.45:
        (w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if w <= width - 2 * pad:
            break
        scale -= 0.05
    return scale


def clip_window(t_first, t_last, fps, nframes, pad_s=CLIP_PAD_S, max_s=CLIP_MAX_S):
    """(first, last) frame indices, last exclusive, around the track.

    Padded either side, clamped to the recording, and if that is longer than
    max_s, trimmed symmetrically about the track's middle.
    """
    a = int(math.floor((t_first - pad_s) * fps))
    b = int(math.ceil((t_last + pad_s) * fps)) + 1
    a, b = max(0, a), min(int(nframes), b)
    limit = int(max_s * fps)
    if b - a > limit:
        mid = int(round((t_first + t_last) / 2.0 * fps))
        a = max(0, mid - limit // 2)
        b = min(int(nframes), a + limit)
        a = max(0, b - limit)
    return a, b


def crop_box(x1, y1, x2, y2, frame_w, frame_h, margin=CLIP_MARGIN, min_size=CLIP_SIZE):
    """(x, y, side): a square holding both ends of the track plus a margin,
    at least min_size, never larger than the frame, clamped inside it."""
    side = int(math.ceil(max(abs(x2 - x1), abs(y2 - y1)) + 2 * margin))
    side = min(max(min_size, side), int(frame_w), int(frame_h))
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    x = int(round(cx - side / 2.0))
    y = int(round(cy - side / 2.0))
    x = min(max(0, x), int(frame_w) - side)
    y = min(max(0, y), int(frame_h) - side)
    return x, y, side


def summary_line(out):
    """One sentence for the chat and the push: the verdict and the numbers."""
    head = "%s pass %s, peak alt %.0f deg: " % (
        out["sat"], out["peak"][:16].replace("T", " "), out["peak_alt_deg"])
    if not out.get("is_satellite"):
        return head + "no %s found -- %s." % (out["sat"], out.get("reject_reason", "no track"))
    parts = ["%d frames agree" % out["frames_agreeing"]]
    if out.get("deg_per_s") and out.get("arc_deg") is not None:
        parts.append("%.2f deg/s over a %.0f deg arc" % (out["deg_per_s"], out["arc_deg"]))
    if out.get("alt_first") is not None:
        parts.append("alt %.0f az %.0f -> alt %.0f az %.0f"
                     % (out["alt_first"], out["az_first"], out["alt_last"], out["az_last"]))
    if out.get("checked_against_tle"):
        parts.append("within %.0f deg of the TLE" % out["miss_deg_median"])
    return head + "FOUND. " + ", ".join(parts) + "."


def make_clip(out, out_path=None):
    """Write <base>_clip.mp4 (H.264, plays inline in the chat) and return its path."""
    fit, dets, video, fps = out["_fit"], out["_dets"], out["_video"], out["_fps"]
    a, b = clip_window(fit["t_first"], fit["t_last"], fps, out["_nframes"])
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError("could not open %s" % video)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 2560
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1440
    ends = [(fit["x0"] + fit["vx"] * t, fit["y0"] + fit["vy"] * t)
            for t in (fit["t_first"], fit["t_last"])]
    x, y, side = crop_box(ends[0][0], ends[0][1], ends[1][0], ends[1][1], fw, fh)
    scale = CLIP_SIZE / float(side)
    A = np.array([(d[0], d[1], d[2]) for d in dets])[fit["mask"]]      # t, x, y agreeing
    out_path = out_path or os.path.splitext(video)[0] + "_clip.mp4"
    # Media Foundation is the backend that writes real H.264 on this machine;
    # the ffmpeg build lacks openh264 and would fall back to MPEG-4 part 2,
    # which browsers do not play.
    writer = cv2.VideoWriter(out_path, cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*"avc1"),
                             float(fps), (CLIP_SIZE, CLIP_SIZE + CLIP_STRIP))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("H.264 writer would not open for %s" % out_path)
    label = "%s  %s  peak alt %.0f deg" % (out["sat"], out["peak"][:16].replace("T", " "),
                                          out["peak_alt_deg"])
    if out.get("deg_per_s"):
        label += "  %.2f deg/s" % out["deg_per_s"]
    font_scale = _fit_caption(label, CLIP_SIZE)
    gold, green = (0, 215, 255), (0, 255, 120)
    lut = None
    i = 0
    while i < b:
        ok, frame = cap.read()
        if not ok:
            break
        if i >= a:
            t = i / fps
            crop = frame[y:y + side, x:x + side]
            if side != CLIP_SIZE:
                crop = cv2.resize(crop, (CLIP_SIZE, CLIP_SIZE), interpolation=cv2.INTER_AREA)
            if lut is None:
                lut = _stretch_lut(crop)
            crop = cv2.LUT(crop, lut)
            for (_, xx, yy) in A[A[:, 0] <= t - CLIP_TRAIL_LAG_S]:
                cv2.circle(crop, (int((xx - x) * scale), int((yy - y) * scale)), 3, green, -1)
            if fit["t_first"] <= t <= fit["t_last"]:
                px = fit["x0"] + fit["vx"] * t
                py = fit["y0"] + fit["vy"] * t
                cv2.circle(crop, (int((px - x) * scale), int((py - y) * scale)), 18, gold, 2)
            canvas = np.zeros((CLIP_SIZE + CLIP_STRIP, CLIP_SIZE, 3), np.uint8)
            canvas[CLIP_STRIP:] = crop
            cv2.putText(canvas, label, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, font_scale, gold, 2)
            writer.write(canvas)
        i += 1
    cap.release()
    writer.release()
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", help="the pass .json (or .h264) written by station_watch")
    ap.add_argument("--downscale", type=int, default=2,
                    help="detect on a reduced frame; centroids are scaled back up")
    ap.add_argument("--tol", type=float, default=TOL_PX)
    ap.add_argument("--refresh", action="store_true",
                    help="re-detect even if a candidate cache exists")
    ap.add_argument("--clip", action="store_true",
                    help="also cut <base>_clip.mp4 when the track is confirmed")
    args = ap.parse_args()
    out = analyse(args.recording, downscale=args.downscale, tol=args.tol, refresh=args.refresh)
    if args.clip and out.get("is_satellite"):
        print("  wrote %s" % make_clip(out))
    return 0 if out.get("is_satellite") else 1


if __name__ == "__main__":
    sys.exit(main())
