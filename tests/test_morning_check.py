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
