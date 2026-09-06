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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", help="the pass .json (or .h264) written by station_watch")
    ap.add_argument("--downscale", type=int, default=2,
                    help="detect on a reduced frame; centroids are scaled back up")
    ap.add_argument("--tol", type=float, default=TOL_PX)
    ap.add_argument("--refresh", action="store_true",
                    help="re-detect even if a candidate cache exists")
    args = ap.parse_args()

    base = os.path.splitext(args.recording)[0]
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
    cache = "%s_dets_d%d.npz" % (base, args.downscale)
    if (not args.refresh and os.path.exists(cache)
            and os.path.getmtime(cache) >= os.path.getmtime(video)):
        dets = [tuple(r) for r in np.load(cache)["dets"]]
        print("re-using %d cached candidates (%s)" % (len(dets), cache))
    else:
        print("detecting movers...")
        dets, _ = detect(video, fps, downscale=args.downscale)
        np.savez_compressed(cache, dets=np.array(dets))
        print("cached %d candidates -> %s" % (len(dets), cache))
    print("finding the consensus track...")
    fit = consensus(dets, tol=args.tol)
    if fit is None:
        print("NO TRACK: no straight constant-speed path collected "
              "%d agreeing frames." % MIN_INLIERS)
        print("That is a real answer, not a failure -- an overcast pass "
              "leaves nothing coherent to find.")
        return 1

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
        else:
            print("  (no plate solution stored; pixel rate only)")
    except Exception as exc:                      # noqa: BLE001
        print("  (angular rate unavailable: %s: %s)" % (type(exc).__name__, exc))

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
        cv2.line(frame, (x1, y1), (x2, y2), (0, 215, 255), 3)
        cv2.circle(frame, (x1, y1), 26, (0, 215, 255), 3)
        cv2.circle(frame, (x2, y2), 26, (0, 215, 255), 3)
        A = np.array([(d[0], d[1], d[2]) for d in dets])
        for (tt, xx, yy) in A[fit["mask"]]:
            cv2.circle(frame, (int(xx), int(yy)), 7, (0, 255, 120), 2)
        cv2.putText(frame, "%s  %s  peak alt %.0f deg"
                    % (p["sat"], p["peak"][:19].replace("T", " "), p["peak_alt_deg"]),
                    (40, 1330), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 215, 255), 4)
        cv2.putText(frame, "%d frames agree of %d candidates   %s"
                    % (fit["n_inliers"], len(dets),
                       ("%.3f deg/s" % out["deg_per_s"]) if out.get("deg_per_s")
                       else "%.0f px/s" % speed_px),
                    (40, 1390), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 215, 255), 3)
        img_out = base + "_track.jpg"
        cv2.imwrite(img_out, frame)
        out["image"] = img_out
        print("  wrote %s" % img_out)

    with open(base + "_track.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("  wrote %s" % (base + "_track.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
