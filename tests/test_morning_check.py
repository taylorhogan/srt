from scripts.morning_check import evaluate, KASA_DEVICES

GOOD_VISION = {"camera": True, "closed": True, "parked": True, "scope_verdict": "safe",
               "frames": 3, "scope_tag_frames": 3}
GOOD_KASA = {"resolved": {n: 0 for n in KASA_DEVICES}, "error": None}
GOOD_PEG = {1: 0, 2: 0, 3: 0, 4: 100}


def test_all_clear():
    assert evaluate(GOOD_VISION, GOOD_KASA, GOOD_PEG) == ([], [])


def test_inside_light_on_is_not_a_problem():
    kasa = {"resolved": dict(GOOD_KASA["resolved"], **{"Iris inside light": 1}), "error": None}
    assert evaluate(GOOD_VISION, kasa, GOOD_PEG) == ([], [])


def test_mount_on():
    kasa = {"resolved": dict(GOOD_KASA["resolved"], **{"Telescope mount": 1}), "error": None}
    problems, _ = evaluate(GOOD_VISION, kasa, GOOD_PEG)
    assert problems == ["telescope mount is powered ON"]


def test_unreadable_is_a_problem():
    kasa = {"resolved": {"Telescope mount": None}, "error": None}
    problems, _ = evaluate({"camera": False, "why": "cloud down"}, kasa, None)
    text = " ".join(problems)
    assert "vision read failed" in text
    assert "'Telescope mount' found but did not answer" in text
    assert "'Roof motor' not found" in text
    assert "Pegasus could not be read" in text


def test_roof_open_scope_moved_pegasus_on():
    vision = dict(GOOD_VISION, closed=False, parked=False, scope_verdict="UNSAFE")
    problems, _ = evaluate(vision, GOOD_KASA, {1: 0, 2: 100, 3: 0})
    assert "roof NOT confirmed closed" in problems
    assert "scope NOT confirmed parked (scope tag UNSAFE)" in problems
    assert "Pegasus port(s) not off: 2=100" in problems


def test_missed_tag_frame_warns_without_failing():
    problems, warnings = evaluate(dict(GOOD_VISION, scope_tag_frames=2), GOOD_KASA, GOOD_PEG)
    assert problems == []
    assert warnings and "2/3" in warnings[0]


# --------------------------------------------------- roof plug + speed (2026-09-19)

from scripts.morning_check import speed_warning

SPEED = {"download_mbps": 300.0, "upload_mbps": 20.0, "ping_ms": 12.0, "server": "x"}


def test_roof_motor_powered_is_a_problem():
    """Left on after the 09-17 limit-switch visit; two mornings passed it."""
    kasa = {"resolved": dict(GOOD_KASA["resolved"], **{"Roof motor": 1}), "error": None}
    problems, _ = evaluate(GOOD_VISION, kasa, GOOD_PEG)
    assert problems == ["roof motor is powered ON (its relay is live)"]


def test_speed_is_reported_not_judged_without_history():
    assert speed_warning(SPEED, []) is None
    assert speed_warning(SPEED, [300.0] * 4) is None          # fewer than 5 samples


def test_speed_warns_only_when_it_collapses_against_this_line():
    past = [300.0, 280.0, 310.0, 295.0, 305.0]
    assert speed_warning(SPEED, past) is None
    assert speed_warning(dict(SPEED, download_mbps=200.0), past) is None
    w = speed_warning(dict(SPEED, download_mbps=90.0), past)
    assert w and "below half" in w
    # A slow line is not a fault: the same 90 Mbps on a 100 Mbps history is fine.
    assert speed_warning(dict(SPEED, download_mbps=90.0), [100.0] * 5) is None


def test_failed_speed_test_warns_but_does_not_fail_the_check():
    problems, warnings = evaluate(GOOD_VISION, GOOD_KASA, GOOD_PEG, None, [300.0] * 5)
    assert problems == []
    assert warnings == ["internet speed test failed (no result)"]


# ------------------------------------------- suspect speed results (2026-09-21)

from scripts.morning_check import read_speed, suspect_reason

GOOD_SPEED = {"download_mbps": 91.7, "upload_mbps": 38.2, "ping_ms": 47.2, "server": "B"}
# What 2026-09-21 09:01 actually returned: half an hour of ping, 7 Mbps, odd server.
BOGUS = {"download_mbps": 7.0, "upload_mbps": 7.6, "ping_ms": 1800000.0, "server": "Nitel"}
HIST = [91.7, 88.0, 95.0, 90.0, 93.0]


def test_suspect_reason_catches_the_2026_09_21_result():
    assert suspect_reason(GOOD_SPEED, HIST) is None
    assert "implausible ping" in suspect_reason(BOGUS, HIST)
    assert "no result" in suspect_reason(None, HIST)
    assert "non-positive" in suspect_reason(dict(GOOD_SPEED, download_mbps=0), HIST)
    # A collapsed download with a sane ping is suspect too — usually a bad server.
    assert "below half" in suspect_reason(dict(GOOD_SPEED, download_mbps=9.0), HIST)
    # ... but not before there is history to judge against.
    assert suspect_reason(dict(GOOD_SPEED, download_mbps=9.0), []) is None


def test_retries_until_believable_and_records_every_attempt():
    seq = iter([BOGUS, BOGUS, GOOD_SPEED])
    speed, tries = read_speed(HIST, pause_s=0, run=lambda: next(seq))
    assert speed == GOOD_SPEED
    assert [t["attempt"] for t in tries] == [1, 2, 3]
    assert [t["suspect"] is None for t in tries] == [False, False, True]


def test_stops_at_the_first_believable_result():
    calls = []
    speed, tries = read_speed(HIST, pause_s=0,
                              run=lambda: calls.append(1) or GOOD_SPEED)
    assert speed == GOOD_SPEED and len(calls) == 1 and len(tries) == 1


def test_all_suspect_keeps_the_last_numbers_rather_than_discarding_them():
    speed, tries = read_speed(HIST, pause_s=0, run=lambda: BOGUS)
    assert speed == BOGUS and len(tries) == 3
    assert all(t["suspect"] for t in tries)


def test_implausible_records_are_kept_out_of_the_baseline(tmp_path, monkeypatch):
    """The 09-21 record (7 Mbps, half-hour ping) must not become the norm."""
    import json as _json
    import scripts.morning_check as mc
    p = tmp_path / "log.jsonl"
    p.write_text("\n".join(_json.dumps({"speed": s}) for s in
                           (GOOD_SPEED, BOGUS, dict(GOOD_SPEED, download_mbps=88.0))) + "\n")
    monkeypatch.setattr(mc, "LOG_PATH", str(p))
    assert mc._speed_history() == [91.7, 88.0]
