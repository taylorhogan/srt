"""The north camera's roof verdict: both states positive, absence never a state.

Pure logic only, so this runs on a bare CI runner -- no cv2, no numpy, no
config. The measured geometry it is built on (2026-09-22): shut 72.6 px at
(1011, 854), open 350.7 px at (992, 355), 650 px apart on the worst matched
corner, frame scatter <= 3 px, and the shut reading reproduced to 1-2 px
across a full open/close cycle.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentry.north_roof import classify, combine, worst_corner

TOL = 100.0
# Squares standing in for the real references, 650 px apart as measured.
SHUT = [[1011, 854], [1083, 854], [1083, 926], [1011, 926]]
OPEN = [[992, 355], [1342, 355], [1342, 705], [992, 705]]


def shifted(ref, dx, dy):
    return [[x + dx, y + dy] for x, y in ref]


# ---------------------------------------------------------------- per frame

def test_the_two_recorded_positions_classify_as_themselves():
    assert classify(SHUT, SHUT, OPEN, TOL) == "shut"
    assert classify(OPEN, SHUT, OPEN, TOL) == "open"


def test_small_drift_still_reads_the_same_state():
    """Roof stops repeated to 1-2 px across a cycle; 15 px is Iris cam's worst."""
    assert classify(shifted(SHUT, 2, -2), SHUT, OPEN, TOL) == "shut"
    assert classify(shifted(SHUT, 15, 15), SHUT, OPEN, TOL) == "shut"
    assert classify(shifted(OPEN, -15, 10), SHUT, OPEN, TOL) == "open"


def test_a_roof_part_way_is_elsewhere_not_the_nearer_end():
    half = [[(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] for a, b in zip(SHUT, OPEN)]
    assert classify(half, SHUT, OPEN, TOL) == "elsewhere"
    # and just outside tolerance of shut, still not shut
    assert classify(shifted(SHUT, 0, 140), SHUT, OPEN, TOL) == "elsewhere"


def test_a_rotated_tag_in_the_right_place_is_not_accepted():
    """Matched corners, not centre-and-size: a tag knocked askew must not pass."""
    rotated = [SHUT[2], SHUT[3], SHUT[0], SHUT[1]]     # same square, corners rolled
    assert classify(rotated, SHUT, OPEN, TOL) == "elsewhere"


def test_worst_corner_is_the_largest_matched_distance():
    assert worst_corner(SHUT, SHUT) == 0
    assert worst_corner(shifted(SHUT, 3, 4), SHUT) == 5


# ---------------------------------------------------------------- combining

def test_open_needs_every_frame():
    assert combine(["open", "open", "open"]) == "open"
    assert combine(["open", "open", "absent"]) == "unknown"
    assert combine(["open", "open", "blind"]) == "unknown"


def test_shut_tolerates_a_frame_that_did_not_decode():
    assert combine(["shut", "absent", "shut"]) == "shut"
    assert combine(["absent", "absent", "shut"]) == "shut"
    assert combine(["shut", "blind", "blind"]) == "shut"


def test_contradiction_and_part_way_are_unknown():
    assert combine(["shut", "open", "shut"]) == "unknown"
    assert combine(["elsewhere", "shut", "shut"]) == "unknown"
    assert combine(["elsewhere", "elsewhere", "elsewhere"]) == "unknown"


def test_a_blind_read_is_never_a_state():
    assert combine(["blind", "blind", "blind"]) == "unknown"
    assert combine(["absent", "absent", "absent"]) == "unknown"
    assert combine([]) == "unknown"


def test_absence_alone_never_means_open():
    """The rule this whole module exists to replace."""
    assert combine(["absent", "absent", "absent"]) != "open"
    assert combine(["blind", "absent", "blind"]) != "open"
