"""scripts/north_roof_record.check: what may become a north roof reference."""
from scripts.north_roof_record import check, median_corners

SHUT = [[1047.5, 818.0], [1047.0, 891.0], [975.0, 889.0], [975.0, 816.0]]
OPEN = [[1175.0, 185.0], [1167.0, 528.5], [797.0, 523.5], [828.0, 182.5]]
REF = {"tag_id": 2, "shut": {"markers": {"2": SHUT}}, "open": {"markers": {"2": OPEN}}}


def _shift(corners, dx, dy):
    return [[x + dx, y + dy] for x, y in corners]


def test_reaim_of_the_2026_10_01_size_is_accepted():
    """Iris North's view moved ~(16, 95) px; that must be recordable."""
    frames = [_shift(OPEN, 15.2 + 0.3 * (i % 3), 94.9) for i in range(10)]
    new, problems, facts = check("open", frames, REF, 10)
    assert problems == [] and facts["frames"] == 10
    assert new == median_corners(frames)


def test_somewhere_else_is_not_a_reaim():
    _, problems, _ = check("open", [_shift(OPEN, 200, 0)] * 10, REF, 10)
    assert any("not a re-aim" in p for p in problems)


def test_too_close_to_the_other_state_refuses():
    # a part-way roof: the tag between the two references
    mid = [[(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] for a, b in zip(SHUT, OPEN)]
    _, problems, _ = check("shut", [mid] * 10, REF, 10)
    assert problems


def test_a_moving_tag_refuses():
    frames = [_shift(OPEN, 0, 3 * i) for i in range(10)]
    _, problems, _ = check("open", frames, REF, 10)
    assert any("not still" in p for p in problems)


def test_too_few_decodes_refuses():
    _, problems, _ = check("open", [OPEN] * 4, REF, 10)
    assert problems == ["tag decoded in only 4/10 frames"]
