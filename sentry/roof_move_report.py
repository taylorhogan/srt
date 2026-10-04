"""One line per roof move: how did it go?

    Roof open: 10.9 s, peak 362 W / running 344 W (typical), audio normal 0.70,
    north tag OPEN 11 px.

Owner, 2026-10-04: in auto mode the only roof report was the audio verdict;
the motor-current signature, its comparison against the good library and the
frozen golden set, and the post-move position were measured on every move
and shown nowhere unless something was wrong. This puts them in one line.

INFORMATIONAL ONLY (owner, 2026-09-16): nothing here gates a move, a night or
a verdict. The only thing that acts on motor current is the stall watchdog in
super_user_commands._wait_for_roof_travel. A flagged item gets a warning sign
in the line; the existing anomaly alerts (chat + phone) are unchanged.

Pure: the caller hands in what was measured, this only words it.
"""


def _current_part(feats, cur):
    if not feats or not feats.get("valid"):
        return "motor current not captured"
    dur = feats.get("move_duration_s")
    head = "%s peak %.0f W / running %.0f W" % (
        ("%.1f s," % dur) if dur is not None else "",
        feats.get("peak_w") or 0, feats.get("running_w") or 0)
    head = head.strip()
    cur = cur or {}
    flags = []
    if cur.get("is_anomaly"):
        flags.append("outside the recent good range: " + "; ".join(cur.get("reasons") or [])[:160])
    if cur.get("golden_ok") is False:
        flags.append("outside the healthy (golden) range: "
                     + "; ".join(cur.get("golden_reasons") or [])[:160])
    if not feats.get("returned_to_baseline", True):
        flags.append("current did not return to baseline")
    if flags:
        return "%s (⚠ %s)" % (head, " | ".join(flags))
    return "%s (typical)" % head


def _audio_part(audio):
    if not audio:
        return "audio not captured"
    v = audio.get("verdict")
    score = audio.get("best_score")
    if v == "good":
        return "audio normal %.2f" % score if score is not None else "audio normal"
    if v == "bad":
        return "⚠ audio does NOT match known-good %.2f" % score if score is not None \
            else "⚠ audio does NOT match known-good"
    return "audio not classified"


def _north_part(north):
    if not north:
        return "north tag not read yet"
    state = (north.get("state") or "unknown").upper()
    off = (north.get("off_open_px") if state == "OPEN"
           else north.get("off_shut_px") if state == "SHUT" else None)
    if off is not None:
        return "north tag %s %.0f px" % (state, off)
    return "north tag %s" % state


def line(direction, feats=None, cur=None, audio=None, north=None):
    """The one-line report. Pure.

    feats : roof_current_signature features (move_duration_s, peak_w, running_w,
            returned_to_baseline, valid)
    cur   : judge_and_record() result (is_anomaly, reasons, golden_ok, golden_reasons)
    audio : kasa_audio verdict (verdict, best_score)
    north : vision_safety.last_match["north"] from a read taken AFTER the move
    """
    word = {"open": "open", "close": "close"}.get(direction or "", "move")
    return "Roof %s: %s, %s, %s." % (word, _current_part(feats, cur),
                                     _audio_part(audio), _north_part(north))


def flagged(feats=None, cur=None, audio=None):
    """True when anything in the line carries a warning sign. Pure."""
    if feats and feats.get("valid"):
        if (cur or {}).get("is_anomaly") or (cur or {}).get("golden_ok") is False:
            return True
        if not feats.get("returned_to_baseline", True):
            return True
    return bool(audio and audio.get("verdict") == "bad")
