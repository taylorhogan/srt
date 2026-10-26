"""Queue coordinates are written only when two sources agree (2026-10-26)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.fill_queue_coords import verdict  # noqa: E402


def test_agreement_with_either_second_source_writes():
    assert verdict(0.0, 0.3, None)[0]
    assert verdict(0.0, None, 4.0)[0]                       # our frames, with a framing offset
    assert verdict(0.0, 0.3, 4.0) == (True, "NED 0.3', frames 4.0'")


def test_no_second_source_or_a_disagreement_is_left_for_review():
    assert verdict(0.0, None, None) == (False, "no second source to check against")
    assert verdict(0.0, 9.0, None)[0] is False               # NED 9' away
    assert verdict(0.0, 9.0, 20.0)[0] is False
    assert verdict(None, 0.1, 0.1) == (False, "SIMBAD has no position")
