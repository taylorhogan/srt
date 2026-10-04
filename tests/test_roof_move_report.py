"""sentry/roof_move_report: the one line posted after every roof move."""
from sentry.roof_move_report import flagged, line

FEATS = {"valid": True, "move_duration_s": 10.93, "peak_w": 362.1, "running_w": 344.3,
         "returned_to_baseline": True}
OK = {"is_anomaly": False, "reasons": [], "golden_ok": True, "golden_reasons": []}
GOOD_AUDIO = {"verdict": "good", "best_score": 0.704}
NORTH_OPEN = {"state": "open", "off_open_px": 11.2, "off_shut_px": 640.0}


def test_a_normal_open():
    assert line("open", FEATS, OK, GOOD_AUDIO, NORTH_OPEN) == (
        "Roof open: 10.9 s, peak 362 W / running 344 W (typical), audio normal 0.70, "
        "north tag OPEN 11 px.")
    assert not flagged(FEATS, OK, GOOD_AUDIO)


def test_golden_drift_and_bad_audio_are_flagged_not_hidden():
    cur = dict(OK, golden_ok=False, golden_reasons=["peak_w 520 is high (good 350.0±20.0)"])
    text = line("close", FEATS, cur, {"verdict": "bad", "best_score": 0.12},
                {"state": "shut", "off_shut_px": 8.0})
    assert "⚠ outside the healthy (golden) range: peak_w 520" in text
    assert "⚠ audio does NOT match known-good 0.12" in text
    assert text.endswith("north tag SHUT 8 px.")
    assert flagged(FEATS, cur, None)


def test_missing_pieces_are_said_plainly():
    text = line("open", None, None, None, None)
    assert text == ("Roof open: motor current not captured, audio not captured, "
                    "north tag not read yet.")
    assert line("open", FEATS, OK, GOOD_AUDIO, {"state": "unknown"}).endswith(
        "north tag UNKNOWN.")


def test_a_stuck_motor_reads_as_a_warning():
    feats = dict(FEATS, returned_to_baseline=False)
    assert "current did not return to baseline" in line("open", feats, OK, GOOD_AUDIO, NORTH_OPEN)
    assert flagged(feats, OK, GOOD_AUDIO)
