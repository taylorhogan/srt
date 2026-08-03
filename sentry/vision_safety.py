# https://stackoverflow.com/questions/52509316/opencv-rectangle-filled
import os,sys
import time
import math
from pathlib import Path
import logging
import cv2 as cv
if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from sentry import  inside_camera_server
from configs import config
from utils import pushover

cfg = config.data()

_logger = logging.getLogger(__name__)

# Per-template confidence/error from the most recent visual_status() call, so
# callers (e.g. the web-chat `status` command) can report the raw match scores
# without changing visual_status()'s 4-tuple return signature.
last_match: dict | None = None


_TEMPLATE_CACHE: dict = {}


def _load_template(path):
    """Read a template once and keep it. The marker scorer runs three matches
    per swept exposure, so an uncached imread would be 30 disk reads a sweep."""
    tpl = _TEMPLATE_CACHE.get(path)
    if tpl is None:
        tpl = cv.imread(path, cv.IMREAD_COLOR)
        if tpl is not None:
            _TEMPLATE_CACHE[path] = tpl
    return tpl


def find_template_rectangle (image, template_image_path):
    # Load the main image and the template
    template = _load_template(template_image_path)
    main_image = image

    # Convert the images to grayscale for processing
    main_image_gray = cv.cvtColor(main_image, cv.COLOR_BGR2GRAY)
    template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

    # Get dimensions of the template
    template_height, template_width = template_gray.shape[:2]

    # Apply template matching (choose a method, e.g., TM_CCOEFF_NORMED)
    method = cv.TM_CCOEFF_NORMED
    result = cv.matchTemplate(main_image_gray, template_gray, method)

    # Find the minimum and maximum values with their locations
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

    # Decide the top-left corner of the best match based on the method
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        best_match_top_left = min_loc
    else:
        best_match_top_left = max_loc

    # Calculate the bottom-right corner using the top-left corner and template size
    best_match_bottom_right = (best_match_top_left[0] + template_width,
                               best_match_top_left[1] + template_height)


    center = ((best_match_top_left[0] + template_width) / 2, (best_match_top_left[1] + template_height) / 2)
    return best_match_top_left, best_match_bottom_right, center, max_val

def test_find_template(image, template_image_path):

    main_image = image

    best_match_top_left, best_match_bottom_right, center, max_val = find_template_rectangle(image, template_image_path)

    print (best_match_top_left, best_match_bottom_right, center)
    # Draw a rectangle around the matched region on the original image
    cv.rectangle(main_image, best_match_top_left, best_match_bottom_right, (0, 255, 0), 2)


    # Display the results
    cv.imshow("Matched Image", main_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def find_template(image, template_image_path):
    best_match_top_left, best_match_bottom_right, center, max_val = find_template_rectangle(image, template_image_path)









def marker_match_score(frame) -> float:
    """Exposure score for the vision-safety sweep: how readable are the markers?

    Replaces best_exposure_score for this one caller. That scorer grades the
    WHOLE frame — mean brightness, contrast, clipping — but the parked/closed/
    open verdict depends only on three small marker regions. With the roof open
    in daylight the two disagree badly. Measured 2026-08-03 on a real roof-open
    ladder: the open marker resolves at exposure -7 (0.66 confidence, 30 px from
    its expected position) and at NO other exposure, but -7 scores -2.343 on the
    whole-frame metric because the frame is 82% clipped, so the sweep chose -11
    and the roof could not be confirmed open.

    Score = sum of match confidence over templates that land near where they are
    expected. Summing is what makes it work: it prefers the frame where the most
    markers are simultaneously readable, rather than the one where any single
    marker is sharpest. Validated against both captured ladders — it selects
    exposure -7 for the roof-closed set (parked 0.88 + closed 0.91 = 1.79) and
    for the roof-open set (parked 0.91 + open 0.66 = 1.57), and -7 gives the
    correct verdict in both.

    Position is judged with find_template_rectangle's own centre convention, the
    same one the verdict uses. That convention is off by a factor of two (see
    the note on line 56), but scoring and gating must agree, so this must NOT be
    "fixed" here independently.
    """
    cs = cfg["camera safety"]
    accuracy = cs["accuracy"]
    total = 0.0
    for tpl_key, pos_key in (("parked template", "parked pos"),
                             ("closed template", "closed pos"),
                             ("open template", "open pos")):
        try:
            _, _, center, conf = find_template_rectangle(frame, cs[tpl_key])
            if math.dist(center, cs[pos_key]) < accuracy:
                total += float(conf)
        except Exception:
            _logger.debug("marker_match_score: %s failed", tpl_key, exc_info=True)
    return total


def _score_exposure_set(capture_dir, accuracy, min_conf):
    """Add per-template match confidence to the most recent saved ladder.

    The frames alone do not answer the question. What we need to know is which
    exposure the *matcher* wanted, versus the one best_exposure_score picked —
    that scorer grades the whole frame, so a blown-out sky patch drags it to an
    exposure that leaves the markers dark. Writing both into meta.json makes the
    disagreement measurable instead of theoretical.

    Best-effort throughout: this is diagnostics hanging off a safety path and
    must never affect the verdict or raise into it.
    """
    import json
    try:
        root = Path(capture_dir)
        sets = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.name)
        if not sets:
            return
        latest = sets[-1]
        meta_path = latest / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        templates = {
            "parked": (cfg["camera safety"]["parked template"], cfg["camera safety"]["parked pos"]),
            "closed": (cfg["camera safety"]["closed template"], cfg["camera safety"]["closed pos"]),
            "open":   (cfg["camera safety"]["open template"],   cfg["camera safety"]["open pos"]),
        }
        rows = []
        for entry in meta.get("frames", []):
            frame = cv.imread(str(latest / entry["file"]), cv.IMREAD_COLOR)
            if frame is None:
                continue
            for name, (tpl, expected) in templates.items():
                _, _, center, conf = find_template_rectangle(frame, tpl)
                err = math.dist(center, expected)
                entry[f"{name}_conf"] = float(conf)
                entry[f"{name}_err"] = float(err)
                entry[f"{name}_ok"] = bool(conf >= min_conf and abs(err) < accuracy)
            rows.append(entry)
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")

        # One compact table per capture: exposure vs what each gate saw.
        best_scorer = max(rows, key=lambda r: r["score"], default=None)
        best_parked = max(rows, key=lambda r: r["parked_conf"], default=None)
        _logger.info("exposure ladder %s: scorer chose exp %s (parked_conf %.2f), "
                     "best parked_conf %.2f at exp %s%s",
                     latest.name,
                     best_scorer["exposure"], best_scorer["parked_conf"],
                     best_parked["parked_conf"], best_parked["exposure"],
                     "  <-- DISAGREE" if best_scorer["exposure"] != best_parked["exposure"] else "")
        for r in rows:
            _logger.info("   exp %-4s luma %5.1f clip %4.1f%% score %+.3f  "
                         "parked %.2f/%3.0fpx  closed %.2f/%3.0fpx  open %.2f/%3.0fpx",
                         r["exposure"], r["luma"], r["clipped_pct"], r["score"],
                         r["parked_conf"], r["parked_err"],
                         r["closed_conf"], r["closed_err"],
                         r["open_conf"], r["open_err"])
    except Exception:
        _logger.warning("scoring the exposure ladder failed", exc_info=True)


def _visual_status_once():
    global last_match

    # Snapshot and read must be atomic: take_snapshot() clobbers scope_view.jpg
    # with a placeholder before capturing, so a concurrent visual_status (e.g.
    # the `update` and `status` commands each running in their own thread) could
    # otherwise overwrite the file between our snapshot and our imread — giving a
    # None/placeholder frame or a misclassified state. Hold the camera session
    # across both (the RLock lets take_snapshot re-acquire the lock).
    image_path = cfg["camera safety"]["scope_view"]
    # In daylight, keep the whole exposure ladder the sweep already takes. It is
    # the dataset needed to fix daytime roof detection and costs no extra camera
    # time; the camera server discards it unless the scene reads as daylight.
    capture_dir = (cfg["camera safety"].get("exposure_capture_dir")
                   if cfg["camera safety"].get("exposure_capture") else None)
    # Also keep the ladder at night when the roof last read OPEN. A real
    # roof-open frame is the missing piece for both the centre-of-match fix
    # (open pos has never been measured in corrected coordinates) and the
    # match_confidence question (the open template scores 0.684 against a
    # CLOSED scene, so the true-open value decides whether the threshold can
    # separate them at all). Neither needs daylight, and the roof is far more
    # often open at night. Uses the previous verdict — one call stale at worst.
    was_open = bool((last_match or {}).get("is_open"))
    scorer = (marker_match_score
              if cfg["camera safety"].get("marker_exposure_scorer", True) else None)
    with inside_camera_server.camera_session():
        print ("take snapshot")
        inside_camera_server.take_snapshot(capture_dir=capture_dir,
                                           capture_force=was_open,
                                           scorer=scorer)

        print("read snapshot")
        image_rgb = cv.imread(image_path, cv.IMREAD_COLOR)

    # A flaky USB webcam snapshot can yield no frame / a half-written file, so
    # cv.imread returns None. Fail safe instead of crashing in cvtColor below:
    # report an untrusted, all-False state (scope not confirmed parked, roof not
    # confirmed open or closed) so no caller reads an unreadable frame as
    # permission to move hardware.
    if image_rgb is None:
        _logger.warning("vision snapshot unreadable (%s) — reporting UNKNOWN/untrusted", image_path)
        last_match = {"error": "snapshot unreadable", "trusted": False}
        try:
            mod_date = time.ctime(os.path.getmtime(image_path))
        except OSError:
            mod_date = time.ctime()
        return False, False, False, mod_date

    mod_date = time.ctime(os.path.getmtime(image_path))

    print ("analysing image")
    parked_best_match_top_left, parked_best_match_bottom_right, parked_center, max_val_parked = find_template_rectangle(image_rgb, cfg['camera safety']['parked template'])
    closed_best_match_top_left, closed_best_match_bottom_right, closed_center, max_val_closed = find_template_rectangle(image_rgb, cfg['camera safety']['closed template'])
    open_best_match_top_left, open_best_match_bottom_right, open_center, max_val_open = find_template_rectangle(image_rgb, cfg['camera safety']['open template'])
    cv.rectangle(image_rgb, parked_best_match_top_left, parked_best_match_bottom_right, (0, 0, 255), 2)
    cv.imwrite(cfg["camera safety"]["scope_view"], image_rgb)
    cv.rectangle(image_rgb, closed_best_match_top_left, closed_best_match_bottom_right, (0, 255, 0), 2)
    cv.imwrite(cfg["camera safety"]["scope_view"], image_rgb)
    cv.rectangle(image_rgb, open_best_match_top_left, open_best_match_bottom_right, (255, 255, 255), 2)
    cv.imwrite(cfg["camera safety"]["scope_view"], image_rgb)

    accuracy = cfg["camera safety"]["accuracy"]
    print(accuracy)
    # A state is only trusted when the template match is both close to the
    # expected pixel position AND confident enough. cv.matchTemplate always
    # returns a best-match location somewhere, so confidence alone only says the
    # marker pattern appears in frame — it is *position* that determines state.
    min_conf = cfg["camera safety"]["match_confidence"]
    print("min match confidence", min_conf)

    gray = cv.cvtColor(image_rgb, cv.COLOR_BGR2GRAY)
    frame_luma = float(gray.mean())
    # Conservative default: a correctly exposed lit frame targets mean ~115, so a
    # floor of 25 only trips on genuinely dark frames. Tunable via config once a
    # real lit-vs-dark snapshot pair is measured (the logged value calibrates it).
    min_luma = cfg["camera safety"].get("min_trust_luma", 25.0)
    too_dark = frame_luma < min_luma

    def _verdict(error: float, conf: float) -> str:
        """Which gate(s) a template failed: 'ok', 'position', 'confidence', or both."""
        fails = []
        if abs(error) >= accuracy:
            fails.append("position")
        if conf < min_conf:
            fails.append("confidence")
        return "+".join(fails) if fails else "ok"

    # --- Scope parked? ---------------------------------------------------------
    # Parked is the gating state: the parked marker must sit near its expected
    # position with enough confidence, on a lit frame.
    parked_error = math.dist(parked_center, cfg["camera safety"]["parked pos"])
    print(parked_center)
    print(cfg["camera safety"]["parked pos"])
    print (parked_error)
    print("parked confidence", max_val_parked)
    parked = (not too_dark) and abs(parked_error) < accuracy and max_val_parked >= min_conf
    parked_verdict = "dark frame" if too_dark else _verdict(parked_error, max_val_parked)

    # --- Roof open / closed ----------------------------------------------------
    # The open/closed marker positions are only valid in the PARKED geometry: a
    # slewed scope changes what the camera sees at those pixels, so the roof state
    # cannot be read at all unless the scope is parked. When parked, position
    # decides state (confidence is only a sanity floor against a spurious hit).
    closed_error = math.dist(closed_center, cfg["camera safety"]["closed pos"])
    open_error = math.dist(open_center, cfg["camera safety"]["open pos"])
    print(closed_center, cfg["camera safety"]["closed pos"], closed_error, "closed conf", max_val_closed)
    print(open_center, cfg["camera safety"]["open pos"], open_error, "open conf", max_val_open)
    if parked:
        closed = abs(closed_error) < accuracy and max_val_closed >= min_conf
        open = abs(open_error) < accuracy and max_val_open >= min_conf
        closed_verdict = _verdict(closed_error, max_val_closed)
        open_verdict = _verdict(open_error, max_val_open)
        if closed and open:
            # Both markers landed at their (well-separated) positions — physically
            # impossible. The templates can't discriminate this frame, so report
            # the roof state as unknown rather than guess.
            _logger.warning(
                "vision ROOF STATE AMBIGUOUS (closed_conf=%.2f open_conf=%.2f) — roof unknown",
                max_val_closed, max_val_open,
            )
            closed = open = False
            closed_verdict = open_verdict = "ambiguous (both matched)"
    else:
        # Not parked (or too dark): the roof state is undeterminable by design.
        closed = open = False
        closed_verdict = open_verdict = "unreadable (scope not parked)"

    if capture_dir:
        _score_exposure_set(capture_dir, accuracy, min_conf)

    trusted = bool(parked)
    print ("parked, closed, open", str(parked), str(closed), str(open))
    print("frame luma", frame_luma, "min", min_luma, "trusted", trusted)
    _logger.info(
        "vision parked=%s closed=%s open=%s luma=%.1f (min %.1f)",
        parked, closed, open, frame_luma, min_luma,
    )

    last_match = {
        "min_conf": min_conf,
        "accuracy": accuracy,
        "frame_luma": frame_luma,
        "min_trust_luma": min_luma,
        "trusted": trusted,
        # The resolved booleans. The per-template dicts below carry the evidence;
        # these carry the verdict, so a caller cannot mistake "the key exists"
        # for "the roof is open".
        "is_parked": bool(parked),
        "is_closed": bool(closed),
        "is_open": bool(open),
        "parked": {"conf": float(max_val_parked), "error": float(parked_error), "verdict": parked_verdict},
        "closed": {"conf": float(max_val_closed), "error": float(closed_error), "verdict": closed_verdict},
        "open":   {"conf": float(max_val_open),   "error": float(open_error),   "verdict": open_verdict},
    }

    mod_date = time.ctime(os.path.getmtime(cfg["camera safety"]["scope_view"]))
    return parked,  closed, open, mod_date


def visual_status(retries: int = 2, delay: float = 2.0):
    """Report (parked, closed, open, mod_date) from the inside camera, retrying
    to ride out a garbage webcam frame.

    This is the single vision entry point for every mount/roof safety decision
    (roof open/close, end-of-night shutdown, pre-flats check, ``status``). A
    single torn/starved/unreadable snapshot reports ``parked=False`` with
    ``last_match["trusted"] == False`` even when the scope is really parked —
    which once wrongly blocked the end-of-night roof close (the mount was
    confirmed parked by PWI4, but one corrupt frame said otherwise, so the roof
    was left open all night). A fresh snapshot almost always comes back clean,
    so re-take it up to *retries* extra times whenever the read is untrusted,
    and return the first trusted result.

    ``trusted`` is exactly "scope confirmed parked", so this only re-tries when
    the frame gives us no usable state. A genuinely-unparked scope stays
    untrusted through every retry and the (still-safe) not-parked result is
    returned unchanged — callers fail safe exactly as before, just after having
    given a garbage frame a few more chances to resolve. Retrying never
    fabricates a "parked": each attempt is an independent snapshot subject to
    the same position/confidence/luma gates, so the roof-move preconditions are
    unchanged per frame — we simply stop acting on the first corrupt one.
    """
    parked, closed, open, mod_date = _visual_status_once()
    for attempt in range(1, retries + 1):
        if last_match and last_match.get("trusted"):
            break
        reason = (last_match or {}).get("error") or "scope not confirmed parked"
        _logger.warning(
            "vision read untrusted (%s) — retrying snapshot %d/%d",
            reason, attempt, retries,
        )
        time.sleep(delay)
        parked, closed, open, mod_date = _visual_status_once()
    return parked, closed, open, mod_date


if __name__ == '__main__':
    cfg = config.data()
    inside_view = cfg["camera safety"]["scope_view"]

    just_finding_template = False
    if just_finding_template:

        inside_camera_server.take_snapshot()
        image =  img_rgb = cv.imread(cfg["camera safety"]["scope_view"], cv.IMREAD_COLOR)
        test_find_template(image, cfg['camera safety']['open template'])
    else:
        parked, closed, open, mod_date = visual_status()
        print (parked, closed, open, mod_date)
        pushover.push_message("roof is not closed, stopping", inside_view)
