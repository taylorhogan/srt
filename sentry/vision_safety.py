# https://stackoverflow.com/questions/52509316/opencv-rectangle-filled
import os,sys
import time
import math
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


def find_template_rectangle (image, template_image_path):
    # Load the main image and the template
    template = cv.imread(template_image_path, cv.IMREAD_COLOR)  # Path to the template
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









def _visual_status_once():
    global last_match

    # Snapshot and read must be atomic: take_snapshot() clobbers scope_view.jpg
    # with a placeholder before capturing, so a concurrent visual_status (e.g.
    # the `update` and `status` commands each running in their own thread) could
    # otherwise overwrite the file between our snapshot and our imread — giving a
    # None/placeholder frame or a misclassified state. Hold the camera session
    # across both (the RLock lets take_snapshot re-acquire the lock).
    image_path = cfg["camera safety"]["scope_view"]
    with inside_camera_server.camera_session():
        print ("take snapshot")
        inside_camera_server.take_snapshot()

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
