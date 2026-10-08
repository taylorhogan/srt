"""Fitting two targets into one night's dark hours (control/slot_plan).

Rows are shaped exactly like rank_targets_tonight's output. Hours are the
night of 2026-09-06/07 in spirit: dark 21:00-05:00, the first target sets
around 02:00, and the question is who gets the tail.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from control.slot_plan import plan_slots, window_for, Slot  # noqa: E402

T0 = datetime(2026, 9, 6, 21, 0)
HOURS = [T0 + timedelta(hours=i) for i in range(8)]          # 21..04
WX_ALL = {h.hour: True for h in HOURS}


def row(name, symbols, priority=5, good=None, alt=60.0):
    good = good if good is not None else symbols.count("+")
    return (name, good, alt, "nebula", None, list(symbols), priority)


def test_one_target_fills_its_window_only():
    rows = [row("squid", "++++++--")]
    s = plan_slots(rows, HOURS, WX_ALL)
    assert [x.name for x in s] == ["squid"]
    assert s[0].start == T0 and s[0].end == T0 + timedelta(hours=6) and s[0].good_hours == 6


def test_second_target_takes_the_tail():
    rows = [row("squid", "++++++--"), row("wizard", "--++++++")]
    s = plan_slots(rows, HOURS, WX_ALL)
    assert [x.name for x in s] == ["squid", "wizard"]
    assert s[1].start == T0 + timedelta(hours=6) and s[1].end == T0 + timedelta(hours=8)
    assert s[1].good_hours == 2


def test_too_short_a_tail_is_no_second_slot():
    rows = [row("squid", "+++++++-"), row("wizard", "--++++++")]
    s = plan_slots(rows, HOURS, WX_ALL, min_slot_hours=2)
    assert [x.name for x in s] == ["squid"]


def test_weather_counts_against_both_slots():
    wx = dict(WX_ALL); wx[3] = False; wx[4] = False      # 03 and 04 clouded
    rows = [row("squid", "++++++--"), row("wizard", "--++++++")]
    s = plan_slots(rows, HOURS, wx, min_slot_hours=2)
    assert [x.name for x in s] == ["squid"]             # only 02 is left for it


def test_a_pinned_early_setter_may_go_first():
    # 'early' is only up 21-23; the pick 'late' is up 21-05. Late-first would
    # leave early nothing; early-first costs late two hours but gains two.
    # Allowed only because early is PINNED: it is the operator's call.
    rows = [row("late", "++++++++", priority=9), row("early", "++------", priority=8)]
    s = plan_slots(rows, HOURS, WX_ALL, min_slot_hours=2)
    assert [x.name for x in s] == ["early", "late"]
    assert s[0].end == T0 + timedelta(hours=2) and s[1].start == T0 + timedelta(hours=2)
    assert s[0].good_hours + s[1].good_hours == 8


def test_an_unpinned_early_setter_does_not_displace_the_pick():
    rows = [row("late", "++++++++", priority=9), row("early", "++------")]
    assert [x.name for x in plan_slots(rows, HOURS, WX_ALL)] == ["late"]


def test_pinned_candidate_beats_a_longer_unpinned_one():
    rows = [row("squid", "++++++--"),
            row("long", "----++++"),
            row("pinned", "------++", priority=9)]
    s = plan_slots(rows, HOURS, WX_ALL, min_slot_hours=2)
    assert [x.name for x in s] == ["squid", "pinned"]


def test_an_unpinned_candidate_never_takes_hours_the_pick_can_use():
    rows = [row("a", "++++++++"), row("b", "----++++")]
    assert [x.name for x in plan_slots(rows, HOURS, WX_ALL)] == ["a"]


def test_two_pins_that_overlap_split_the_night():
    rows = [row("a", "++++++++", priority=9), row("b", "----++++", priority=8)]
    s = plan_slots(rows, HOURS, WX_ALL, min_slot_hours=2)
    assert [x.name for x in s] == ["a", "b"]
    assert s[0].end <= s[1].start
    assert s[0].good_hours == 4 and s[1].good_hours == 4       # 21-01 / 01-05
    # the split never leaves the first pin under half its hours
    rows = [row("a", "++++++++", priority=9), row("b", "-+++++++", priority=8)]
    s = plan_slots(rows, HOURS, WX_ALL, min_slot_hours=2)
    assert s[0].good_hours == 4 and s[1].good_hours == 4


def test_slots_hard_end_never_overlaps():
    for rows in ([row("squid", "++++++--"), row("wizard", "--++++++")],
                 [row("a", "++++++++", priority=9), row("b", "----++++", priority=8)]):
        s = plan_slots(rows, HOURS, WX_ALL)
        assert len(s) == 2 and s[0].end <= s[1].start


def test_max_slots_one_is_the_old_behaviour():
    rows = [row("squid", "++++++--"), row("wizard", "--++++++")]
    assert [x.name for x in plan_slots(rows, HOURS, WX_ALL, max_slots=1)] == ["squid"]


def test_slot_seconds_are_good_hours():
    assert Slot("x", T0, T0 + timedelta(hours=3), 3).seconds == 3 * 3600.0


def test_two_pins_split_in_whichever_order_yields_more_hours():
    # pick 'wizard' up 22-03, 'squid' up 21-02, both pinned. Wizard-first
    # gives 22-01 + 01-03 = 5 h; squid-first gives 21-00 + 00-04 = 7 h.
    rows = [row("wizard", "-++++++-", priority=100), row("squid", "++++++--", priority=100)]
    s = plan_slots(rows, HOURS, WX_ALL, min_slot_hours=2)
    assert [x.name for x in s] == ["squid", "wizard"]
    assert s[0].good_hours + s[1].good_hours == 7
    assert s[0].end <= s[1].start


def test_window_for_ends_at_the_earliest_of_horizon_dawn_and_weather():
    """The hard end every sequence gets for its last target."""
    # horizon: sets after hour 5 -> end 03:00
    assert window_for(row("a", "++++++--"), HOURS, WX_ALL).end == T0 + timedelta(hours=6)
    # dawn: up all night -> end = end of dark hours
    assert window_for(row("b", "++++++++"), HOURS, WX_ALL).end == T0 + timedelta(hours=8)
    # weather: cloud from 01:00 (the 09-10 night) -> end 01:00 though up till 04
    wx = {**WX_ALL, 1: False, 2: False, 3: False, 4: False}
    w = window_for(row("ngc7380", "-++++++-"), HOURS, wx)
    assert w.start == T0 + timedelta(hours=1) and w.end == T0 + timedelta(hours=4) and w.good_hours == 3
    # nothing usable
    assert window_for(row("m33", "----++++"), HOURS, wx) is None


def test_signature_changes_when_a_slot_appears_or_a_window_moves():
    from control.slot_plan import signature
    t0 = datetime(2026, 9, 13, 22, 0)
    a = [Slot("ngc7380", t0, t0 + timedelta(hours=3), 3, 100)]
    b = [Slot("ngc7380", t0, t0 + timedelta(hours=3), 3, 100),
         Slot("m33", t0 + timedelta(hours=5), t0 + timedelta(hours=7), 2, 100)]
    c = [Slot("ngc7380", t0, t0 + timedelta(hours=4), 4, 100)]
    assert signature(a) == signature(list(a))
    assert signature(a) != signature(b)          # second slot gained (2026-09-13)
    assert signature(a) != signature(c)          # same target, longer window
    assert signature([]) == ()


# --- a third slot (2026-10-08) --------------------------------------------

def test_third_slot_fills_the_tail_after_two():
    rows = [row("a", "++++----"), row("b", "----++--"), row("c", "------++")]
    s = plan_slots(rows, HOURS, WX_ALL, min_slot_hours=2, max_slots=3)
    assert [x.name for x in s] == ["a", "b", "c"]
    assert s[2].start == HOURS[6] and s[2].end == HOURS[7] + timedelta(hours=1)
    assert s[2].good_hours == 2


def test_third_slot_may_go_first_in_the_gap_before_slot_one():
    # the pick rises late; a short early target fits before it
    rows = [row("late", "---+++--"), row("mid", "------++"), row("early", "++------")]
    s = plan_slots(rows, HOURS, WX_ALL, min_slot_hours=2, max_slots=3)
    assert [x.name for x in s] == ["early", "late", "mid"]
    assert s[0].end <= s[1].start and s[1].end <= s[2].start


def test_third_slot_never_takes_hours_from_the_first_two():
    rows = [row("a", "++++----"), row("b", "----++++"), row("c", "++++++++")]
    s = plan_slots(rows, HOURS, WX_ALL, min_slot_hours=2, max_slots=3)
    assert [x.name for x in s] == ["a", "b"]


def test_third_slot_needs_min_hours():
    rows = [row("a", "++++----"), row("b", "----+++-"), row("c", "-------+")]
    s = plan_slots(rows, HOURS, WX_ALL, min_slot_hours=2, max_slots=3)
    assert [x.name for x in s] == ["a", "b"]


def test_pinned_third_candidate_beats_a_longer_unpinned_one():
    rows = [row("a", "+++-----"), row("b", "---++---"), row("long", "-----+++"),
            row("pin", "-----++-", priority=9)]
    s = plan_slots(rows, HOURS, WX_ALL, min_slot_hours=2, max_slots=3)
    assert [x.name for x in s] == ["a", "b", "pin"]


def test_max_slots_two_is_unchanged_by_the_third_slot_rule():
    rows = [row("a", "++++----"), row("b", "----++--"), row("c", "------++")]
    assert [x.name for x in plan_slots(rows, HOURS, WX_ALL, min_slot_hours=2, max_slots=2)] == ["a", "b"]


def test_three_slots_never_overlap_and_run_in_order():
    rows = [row("a", "++++----"), row("b", "----++--"), row("c", "------++")]
    s = plan_slots(rows, HOURS, WX_ALL, min_slot_hours=2, max_slots=3)
    for x, y in zip(s, s[1:]):
        assert x.end <= y.start


# --- max_hours: a short monitoring block (2026-10-08, the M31 Cepheid) -----

def test_capped_target_fills_a_gap_trimmed_to_its_cap():
    rows = [row("a", "++++----"), row("b", "------++"), row("v1", "++++++++", priority=50)]
    s = plan_slots(rows, HOURS, WX_ALL, min_slot_hours=2, max_slots=3, max_hours={"v1": 1})
    assert [x.name for x in s] == ["a", "v1", "b"]
    v1 = s[1]
    assert v1.good_hours == 1 and v1.end - v1.start == timedelta(hours=1)
    assert HOURS[4] <= v1.start and v1.end <= HOURS[6]


def test_pinned_capped_target_with_no_gap_is_reserved_at_an_edge():
    rows = [row("v1", "++++++++", priority=50), row("a", "++++----"), row("b", "----++++")]
    s = plan_slots(rows, HOURS, WX_ALL, min_slot_hours=2, max_slots=3, max_hours={"v1": 1})
    # a and b would fill the night; v1 takes the dusk hour and they shift
    assert [x.name for x in s] == ["v1", "a", "b"]
    assert s[0].start == HOURS[0] and s[0].good_hours == 1
    assert s[1].start == HOURS[1] and s[1].good_hours == 3 and s[2].good_hours == 4
    for x, y in zip(s, s[1:]):
        assert x.end <= y.start


def test_edge_reservation_prefers_the_side_where_the_run_is_longer():
    # v1 rises late: not good at dusk, good through dawn -> the dawn hour
    rows = [row("v1", "---+++++", priority=50), row("a", "++++----"), row("b", "----++++")]
    s = plan_slots(rows, HOURS, WX_ALL, min_slot_hours=2, max_slots=3, max_hours={"v1": 1})
    assert [x.name for x in s] == ["a", "b", "v1"] and s[2].start == HOURS[7]


def test_unpinned_capped_target_with_no_gap_is_skipped():
    rows = [row("a", "++++----"), row("b", "----++++"), row("v1", "++++++++")]
    s = plan_slots(rows, HOURS, WX_ALL, min_slot_hours=2, max_slots=3, max_hours={"v1": 1})
    assert [x.name for x in s] == ["a", "b"]


def test_cap_stands_in_for_min_slot_hours():
    rows = [row("a", "+++++++-"), row("v1", "++++++++", priority=50)]
    s = plan_slots(rows, HOURS, WX_ALL, min_slot_hours=2, max_slots=3, max_hours={"v1": 1})
    assert [x.name for x in s] == ["a", "v1"] and s[1].start == HOURS[7]


def test_capped_block_is_centred_in_the_gap():
    rows = [row("a", "++------"), row("v1", "++++++++", priority=50)]
    s = plan_slots(rows, HOURS, WX_ALL, min_slot_hours=2, max_slots=3, max_hours={"v1": 2})
    v1 = s[1]
    assert v1.good_hours == 2 and v1.start == HOURS[4]   # hours 2..7 free, middle two


def test_capped_pinned_target_takes_the_gap_before_an_unpinned_filler():
    rows = [row("a", "++++----"), row("b", "------++"), row("c", "----++++"), row("v1", "++++++++", priority=50)]
    s = plan_slots(rows, HOURS, WX_ALL, min_slot_hours=2, max_slots=3, max_hours={"v1": 1})
    # c is a main-slot candidate and takes the tail (a 21-01, c 01-05); no gap is
    # left, so v1 is reserved at dusk and the mains shift: v1, a, c
    assert [x.name for x in s] == ["v1", "a", "c"]


def test_no_caps_is_unchanged():
    rows = [row("a", "++++----"), row("b", "----++--"), row("c", "------++")]
    assert [x.name for x in plan_slots(rows, HOURS, WX_ALL, 2, 3)] == ["a", "b", "c"]
