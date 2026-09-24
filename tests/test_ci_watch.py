"""The CI watcher's transitions: green once, stuck once, nothing on a quiet poll."""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.ci_watch import assess

T0 = datetime(2026, 9, 24, 8, 0, tzinfo=timezone.utc)
A, B = "a" * 40, "b" * 40


def poll(state, main, release, minutes):
    return assess(state, main, release, T0 + timedelta(minutes=minutes), stuck_after_s=12 * 60)


def test_a_push_then_a_green_run_announces_green_exactly_once():
    st, ev = poll({}, A, A, 0)                 # baseline: quiet, nothing to say
    assert ev == []
    st, ev = poll(st, B, A, 1)                 # pushed; CI in progress
    assert ev == []
    st, ev = poll(st, B, B, 4)                 # release caught up
    assert ev == [("green", B)]
    st, ev = poll(st, B, B, 9)                 # same tips again: silence
    assert ev == []


def test_main_ahead_too_long_announces_stuck_once():
    st, _ = poll({}, A, A, 0)
    st, ev = poll(st, B, A, 1)
    assert ev == []
    st, ev = poll(st, B, A, 6)                 # under the threshold
    assert ev == []
    st, ev = poll(st, B, A, 14)
    assert ev == [("stuck", B, A, 13)]
    st, ev = poll(st, B, A, 19)                # still stuck: not repeated
    assert ev == []


def test_a_fix_after_a_stuck_run_announces_green():
    st, _ = poll({}, A, A, 0)
    st, _ = poll(st, B, A, 1)
    st, _ = poll(st, B, A, 14)                 # stuck announced
    C = "c" * 40
    st, ev = poll(st, C, A, 15)                # fix pushed: new sha resets the clock
    assert ev == []
    st, ev = poll(st, C, C, 18)
    assert ev == [("green", C)]


def test_first_ever_poll_with_equal_tips_says_nothing():
    """A fresh state file must not announce 'green' for a release that has
    been sitting there since yesterday."""
    st, ev = poll({}, A, A, 0)
    assert ev == []
