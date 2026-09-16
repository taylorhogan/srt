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
