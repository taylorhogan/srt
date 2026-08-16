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


# How many exposure rungs must agree before a roof state is believed. One is
# enough because the rule that actually protects us is that the OTHER state
# must score ZERO. Replayed over all 13 ladders captured so far, every vote is
# unanimous: the 5 closed sets scored closed 3-5 and open 0; the 8 open sets
# scored open 1-5 and closed 0. Open resolves on exactly ONE rung in 7 of those
# 8 — the open marker is readable in a far narrower exposure band than the
# closed one — so requiring two votes would fail almost every roof-open check.
_MIN_ROOF_VOTES = 1

# How many rungs must agree before the scope is called parked. NOT a majority:
# the sweep deliberately spans exposures that are unusable at both ends, and in
# daylight the whole bright half blows out — two real roof-open ladders
# (2026-08-03 17:02, 17:07) read parked on exactly 5 of 10 rungs and nothing at
# all on the top 5, so a majority rule called a parked scope unparked. Counting
# is the right test: 3 exposures independently putting the marker within
# `accuracy` px of the parked position is not a coincidence (wrong matches miss
# by 120-400px, never narrowly), and every ladder measured so far has 5 or more.
_MIN_PARKED_VOTES = 3

# How far ahead the winning roof state must be before an ambiguous ladder is
# resolved instead of refused.
#
# This was 0.15, chosen as mid-plateau on 18 ladders where 0.00-0.20 all scored
# alike. A 19th ladder — a genuinely open roof at 11:50, held out because it did
# not exist yet — then landed at a lead of 0.147 and was refused. It was not a
# wrong answer, but it was the answer the whole change exists to produce, and it
# showed the plateau's upper edge is between 0.12 and 0.15, not 0.20.
#
# 0.10 is mid-plateau on 19: 0.00-0.12 all give 15 correct, 0 unknown, 0 wrong.
# Set to a large number to restore the old refuse-on-any-conflict behaviour.
_ROOF_TIEBREAK_MARGIN = 0.10


def _match_score(match, accuracy):
    """Quality of one template match: confident AND close both count.

    Confidence alone cannot separate a true match from a phantom — measured
    2026-08-16, a phantom scored 0.63 against a true match's 0.65. Distance
    does separate them (77px vs 44px), but only relative to the other
    candidate, which is why this is a score to compare rather than a threshold.
    """
    return match["conf"] - abs(match["error"]) / accuracy


def _evaluate_rung(frame, exposure=None):
    """Match all three markers against one exposure of the sweep."""
    cs = cfg["camera safety"]
    rung = {
        "exposure": exposure,
        "luma": float(cv.cvtColor(frame, cv.COLOR_BGR2GRAY).mean()),
        "frame": frame,
    }
    for name, tpl_key, pos_key in (("parked", "parked template", "parked pos"),
                                   ("closed", "closed template", "closed pos"),
                                   ("open",   "open template",   "open pos")):
        top_left, bottom_right, center, conf = find_template_rectangle(frame, cs[tpl_key])
        rung[name] = {
            "conf": float(conf),
            "error": float(math.dist(center, cs[pos_key])),
            "rect": (top_left, bottom_right),
        }
    return rung


def _decide_from_rungs(rungs):
    """Vote the parked/roof verdict across every exposure of one sweep.

    The safety question is never "what does this frame say" — it is "what is
    the observatory doing". A single frame answers that only if its exposure
    happens to land where the markers are readable, and the readable band is
    narrow: on 2026-08-04 the closed marker sat at its expected position (0.88
    confidence, 19px) at exposures -6..-8 and was 610px away at -5, one rung
    brighter. Picking a frame and reading it therefore stakes a hardware-safety
    verdict on an exposure guess, and it has now failed in both directions —
    a closed roof reported NOT closed, and an open roof reported not open.

    The ladder does not have that problem. Replayed over all 13 ladders
    captured so far, the vote is unanimous every time — the losing state scores
    exactly zero rungs — while the single-frame verdict gets 3 of the 8
    labelled sets wrong, every one of them a roof-open case. So the verdict
    comes from counting rungs instead of choosing one:

    * only rungs bright enough to trust at all are counted (``min_trust_luma``);
    * ``parked`` needs ``_MIN_PARKED_VOTES`` of those rungs — a count, not a
      majority, because the ladder's bright end is unusable by design;
    * the roof is then read only from rungs that are themselves parked-ok (the
      open/closed marker positions are only meaningful in the parked geometry),
      and a state wins only if it has votes and the opposite state has NONE.

    Votes on both sides means the ladder cannot discriminate this scene, which
    is reported as unknown — not guessed. Unknown is safe: visual_status()
    re-takes the whole ladder, and callers refuse to move hardware on it.
    """
    cs = cfg["camera safety"]
    accuracy = cs["accuracy"]
    min_conf = cs["match_confidence"]
    min_luma = cs.get("min_trust_luma", 25.0)

    def ok(rung, state):
        m = rung[state]
        return m["conf"] >= min_conf and abs(m["error"]) < accuracy

    lit = [r for r in rungs if r["luma"] >= min_luma]
    parked_rungs = [r for r in lit if ok(r, "parked")]
    # Capped by the rungs available so the no-ladder fallback (a single frame)
    # still decides on that frame, exactly as before.
    need_parked = min(_MIN_PARKED_VOTES, len(lit)) if lit else 1
    parked = len(parked_rungs) >= need_parked

    closed_rungs: list = []
    open_rungs: list = []
    closed = is_open = False
    if parked:
        closed_rungs = [r for r in parked_rungs if ok(r, "closed")]
        open_rungs = [r for r in parked_rungs if ok(r, "open")]
        closed = len(closed_rungs) >= _MIN_ROOF_VOTES and not open_rungs
        is_open = len(open_rungs) >= _MIN_ROOF_VOTES and not closed_rungs
        if closed_rungs and open_rungs:
            # Both states have votes. The unanimity rule calls this unknown,
            # which is why a roof open in daylight goes unconfirmed: the closed
            # marker is GONE once the roof opens, so its region fills with
            # foliage against sky — dark-shape-on-bright-ground, the same
            # greyscale signature as a blue triangle on silver. The matcher is
            # handed an endless supply of near-matches for exactly the feature
            # it hunts, and one of them vetoes any number of correct votes.
            #
            # So compare the two states against EACH OTHER instead of letting
            # either veto. Score trades confidence against how far the match
            # missed; the winner must lead by `margin` or it stays unknown.
            # Measured on 18 labelled ladders: 10 correct -> 14, all four
            # ambiguous cases resolved, zero wrong at ANY margin from 0.00 to
            # 0.60, and a flat plateau from 0.00-0.20 so this is not tuned to
            # an edge. Above 0.30 it degrades gracefully back to the old
            # behaviour.
            best_closed = max(_match_score(r["closed"], accuracy) for r in closed_rungs)
            best_open = max(_match_score(r["open"], accuracy) for r in open_rungs)
            lead = abs(best_closed - best_open)
            if lead >= _ROOF_TIEBREAK_MARGIN:
                closed = best_closed > best_open
                is_open = not closed
                _logger.warning(
                    "vision roof ambiguous (%d closed / %d open) — resolved to %s "
                    "on match quality (closed %.2f vs open %.2f, lead %.2f)",
                    len(closed_rungs), len(open_rungs),
                    "closed" if closed else "open", best_closed, best_open, lead,
                )
            else:
                _logger.warning(
                    "vision ROOF STATE AMBIGUOUS — %d rung(s) read closed and %d read open; "
                    "neither leads by %.2f (closed %.2f vs open %.2f); roof unknown",
                    len(closed_rungs), len(open_rungs), _ROOF_TIEBREAK_MARGIN,
                    best_closed, best_open,
                )

    # Representative rung per state: the winning rung when the state won,
    # otherwise the lit rung that came closest, so a refusal message quotes the
    # best evidence against itself rather than an arbitrary frame.
    def representative(state, winners):
        pool = winners or lit or rungs
        return max(pool, key=lambda r: r[state]["conf"])

    def gate_fails(rung, state):
        fails = []
        if abs(rung[state]["error"]) >= accuracy:
            fails.append("position")
        if rung[state]["conf"] < min_conf:
            fails.append("confidence")
        return "+".join(fails) if fails else "ok"

    def describe(state, won, winners):
        rung = representative(state, winners)
        if won:
            return rung, "ok"
        if not lit:
            return rung, f"all {len(rungs)} rung(s) below min luma {min_luma:.0f}"
        if state != "parked" and not parked:
            return rung, "unreadable (scope not parked)"
        if state != "parked" and closed_rungs and open_rungs:
            return rung, (f"ambiguous (closed {len(closed_rungs)} rung(s), "
                          f"open {len(open_rungs)} rung(s))")
        return rung, f"{len(winners)}/{len(lit)} rungs: {gate_fails(rung, state)}"

    parked_rep, parked_verdict = describe("parked", parked, parked_rungs)
    closed_rep, closed_verdict = describe("closed", closed, closed_rungs)
    open_rep, open_verdict = describe("open", is_open, open_rungs)

    # The frame a human is shown is the one that best demonstrates the verdict.
    # This is the ONLY place a frame is singled out, and nothing depends on it.
    if closed:
        display = closed_rep
    elif is_open:
        display = open_rep
    else:
        display = parked_rep

    _logger.info(
        "vision parked=%s closed=%s open=%s — votes parked %d/%d lit (%d rungs), "
        "closed %d, open %d; showing exp %s luma %.1f",
        parked, closed, is_open, len(parked_rungs), len(lit), len(rungs),
        len(closed_rungs), len(open_rungs), display["exposure"], display["luma"],
    )
    for r in rungs:
        _logger.debug(
            "   exp %-4s luma %5.1f  parked %.2f/%4.0fpx  closed %.2f/%4.0fpx  open %.2f/%4.0fpx",
            r["exposure"], r["luma"],
            r["parked"]["conf"], r["parked"]["error"],
            r["closed"]["conf"], r["closed"]["error"],
            r["open"]["conf"], r["open"]["error"],
        )

    return {
        "parked": parked, "closed": closed, "open": is_open,
        "display": display,
        "last_match": {
            "min_conf": min_conf,
            "accuracy": accuracy,
            "frame_luma": display["luma"],
            "min_trust_luma": min_luma,
            "trusted": bool(parked),
            "rungs": len(rungs),
            "lit_rungs": len(lit),
            "votes": {"parked": len(parked_rungs),
                      "closed": len(closed_rungs),
                      "open": len(open_rungs)},
            # The resolved booleans. The per-template dicts below carry the
            # evidence; these carry the verdict, so a caller cannot mistake
            # "the key exists" for "the roof is open".
            "is_parked": bool(parked),
            "is_closed": bool(closed),
            "is_open": bool(is_open),
            "parked": {"conf": parked_rep["parked"]["conf"],
                       "error": parked_rep["parked"]["error"],
                       "verdict": parked_verdict},
            "closed": {"conf": closed_rep["closed"]["conf"],
                       "error": closed_rep["closed"]["error"],
                       "verdict": closed_verdict},
            "open": {"conf": open_rep["open"]["conf"],
                     "error": open_rep["open"]["error"],
                     "verdict": open_verdict},
        },
    }


def _write_annotated(rung, image_path):
    """Write the frame the verdict was read from, with each marker's best match
    boxed (parked red, closed green, open white), for the chat card / pushover.

    Copies first: the ladder frames are the evidence, and drawing on them would
    alter what a later re-read of the same sweep would see.
    """
    image = rung["frame"].copy()
    for name, color in (("parked", (0, 0, 255)),
                        ("closed", (0, 255, 0)),
                        ("open", (255, 255, 255))):
        top_left, bottom_right = rung[name]["rect"]
        cv.rectangle(image, top_left, bottom_right, color, 2)
    cv.imwrite(image_path, image)


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
        # Take the ladder while we still hold the camera session, so it can
        # only ever be the sweep belonging to OUR snapshot.
        ladder = inside_camera_server.take_ladder()

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

    print ("analysing image")
    # Every exposure the sweep took is evidence. The frame that happened to be
    # written to disk is only one of them, and not necessarily the readable one
    # — so it decides nothing here. Fall back to it alone only when there is no
    # ladder (a test_path copy, or a sweep that captured nothing), which
    # reduces exactly to the old single-frame behaviour.
    if ladder and ladder.get("frames"):
        rungs = [_evaluate_rung(frame, exposure) for frame, exposure
                 in zip(ladder["frames"], ladder["exposures"])]
    else:
        _logger.warning("vision: no exposure ladder — deciding on the single written frame")
        rungs = [_evaluate_rung(image_rgb)]

    decision = _decide_from_rungs(rungs)
    parked = decision["parked"]
    closed = decision["closed"]
    open = decision["open"]
    last_match = decision["last_match"]
    _write_annotated(decision["display"], image_path)

    # Diagnostics: annotate the ladder kept on disk with the same per-rung
    # measurements. Best-effort and after the verdict, so it can never affect it.
    if capture_dir:
        _score_exposure_set(capture_dir, last_match["accuracy"], last_match["min_conf"])

    mod_date = time.ctime(os.path.getmtime(image_path))
    print ("parked, closed, open", str(parked), str(closed), str(open))
    return parked, closed, open, mod_date


def _unresolved_reason():
    """Why the last read is not usable, or None when it resolved both states.

    Two ways a frame can fail to answer the question it was taken to answer:

    * ``trusted`` is False — the scope is not confirmed parked, so nothing
      (including the roof) can be read from the frame at all.
    * the scope IS parked but the roof matched neither ``closed`` nor ``open``.
      The roof is always in one of those two states when the scope is parked,
      so "neither" is not a state — it means the templates could not read this
      particular frame.
    """
    lm = last_match or {}
    if not lm.get("trusted"):
        return lm.get("error") or "scope not confirmed parked"
    if not lm.get("is_closed") and not lm.get("is_open"):
        closed_verdict = (lm.get("closed") or {}).get("verdict", "?")
        open_verdict = (lm.get("open") or {}).get("verdict", "?")
        return (f"roof neither closed nor open (closed: {closed_verdict}; "
                f"open: {open_verdict})")
    return None


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

    A parked scope whose roof reads neither closed NOR open is retried for the
    same reason (see :func:`_unresolved_reason`): the roof is always in one of
    those states, so "neither" is a failure to read the frame, not a finding.
    That case bit us on 2026-08-04 — the roof had just closed (audio and motor
    current both nominal) but the confirming frame came back ~35 luma brighter
    than the exposure the ladder scored, which puts the closed marker ~610px
    off its expected position. It reported "roof is NOT closed", and because
    the scope WAS parked the read counted as trusted and was never re-taken.
    The very next snapshot, 72s later, read closed at 0.89 confidence.

    A genuinely-unresolvable read stays unresolved through every retry and the
    (still-safe) all-False result is returned unchanged — callers fail safe
    exactly as before, just after having given a garbage frame a few more
    chances. Retrying never fabricates a state: each attempt is an independent
    snapshot subject to the same position/confidence/luma gates, so the
    roof-move preconditions are unchanged per frame — we simply stop acting on
    the first unreadable one.
    """
    parked, closed, open, mod_date = _visual_status_once()
    for attempt in range(1, retries + 1):
        reason = _unresolved_reason()
        if reason is None:
            break
        _logger.warning(
            "vision read unresolved (%s) — retrying snapshot %d/%d",
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
