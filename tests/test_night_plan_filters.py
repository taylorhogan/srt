"""C1 of docs/AUTOMATION_PLAN.md (2026-10-30): the noon plan decides the filters.

Each slot of the NIGHT_PLAN record carries "filters" (the ratio), "filter_source"
(explicit > monitor > advisor > type) and "filter_reasons"; "filter_mode" says
whether the scheduler writes it. The decision is pure given the queue rows and
the advisor's answer, so it is tested here without a forecast or a disk.
"""
from datetime import datetime, timedelta, timezone

from control import night_plan as NP
from control import slot_plan

TZ = timezone(timedelta(hours=-4))
T0 = datetime(2026, 10, 30, 21, 0, tzinfo=TZ)

SLOTS = [{"dso": "ngc1961", "start": "2026-10-30T21:00-04:00", "end": "2026-10-31T01:00-04:00", "hours": 4},
         {"dso": "hubblev1", "start": "2026-10-31T01:00-04:00", "end": "2026-10-31T02:00-04:00", "hours": 1}]
ROWS = [{"dso": "ngc1961", "status": "waiting"},
        {"dso": "hubblev1", "status": "waiting", "lightcurve": True, "filter_plan": {"G": 20}},
        {"dso": "ic1396", "status": "waiting", "filter_plan": {"Ha": 2, "O-III": 1}},
        {"dso": "m31", "status": "completed", "filter_plan": {"L": 9}},
        {"dso": "m31", "status": "waiting"}]

ADVICE = {"ngc1961": {"action": "plan", "plan": {"L": 4, "R": 1, "G": 1, "B": 1},
                      "reasons": ["need (goal - have): L 30, R 4, G 4, B 4", "seeing forecast good: L x1.5"]}}


def _plan(slots=SLOTS, will=True):
    return {"best": slots[0]["dso"], "good_hours": 4, "will_image": will, "slots": list(slots)}


def _advise(dso, start, end):
    assert isinstance(start, datetime) and isinstance(end, datetime)
    return ADVICE.get(dso) or {"action": "plan", "plan": None, "reasons": ["no frames yet"]}


def test_filter_mode_reads_the_config_and_defaults_to_shadow():
    assert NP.filter_mode({}) == "shadow"
    assert NP.filter_mode({"nina": {"filter_advisor": "DEFAULT"}}) == "default"
    assert NP.filter_mode({"nina": {"filter_advisor": "off"}}) == "off"
    assert NP.filter_mode({"nina": {"filter_advisor": "banana"}}) == "shadow"


def test_queue_row_is_the_first_non_completed_match():
    assert NP.queue_row(ROWS, "M 31") == {"dso": "m31", "status": "waiting"}
    assert NP.queue_row(ROWS, "nope") is None


def test_an_explicit_plan_always_wins():
    out = NP.slot_filters("ic1396", NP.queue_row(ROWS, "ic1396"), ADVICE["ngc1961"])
    assert out["filters"] == {"Ha": 2, "O-III": 1} and out["filter_source"] == "explicit"


def test_a_monitor_keeps_its_own_plan():
    out = NP.slot_filters("hubblev1", NP.queue_row(ROWS, "hubblev1"), None)
    assert out["filters"] == {"G": 20} and out["filter_source"] == "monitor"


def test_the_advisor_decides_when_the_queue_says_nothing():
    out = NP.slot_filters("ngc1961", NP.queue_row(ROWS, "ngc1961"), ADVICE["ngc1961"])
    assert out["filters"] == {"L": 4, "R": 1, "G": 1, "B": 1}
    assert out["filter_source"] == "advisor"
    assert out["filter_reasons"] == ADVICE["ngc1961"]["reasons"]


def test_a_done_target_still_gets_the_advisors_deepening_split():
    adv = {"action": "done", "plan": {"L": 2, "R": 1, "G": 1, "B": 1}, "reasons": ["every goal met"]}
    assert NP.slot_filters("x", None, adv)["filter_source"] == "advisor"


def test_no_advice_falls_back_to_the_type_split_and_says_why():
    out = NP.slot_filters("ngc1961", {}, {"action": "error", "plan": None,
                                           "reasons": ["advisor failed: KeyError: 'x'"]})
    assert out["filters"] is None and out["filter_source"] == "type"
    assert out["filter_reasons"][0].startswith("advisor failed") and "object-type" in out["filter_reasons"][-1]
    # a malformed queue plan is no plan
    assert NP.slot_filters("x", {"filter_plan": {"Ha": "lots"}}, None)["filter_source"] == "type"
    assert NP.slot_filters("x", {"filter_plan": {"Ha": 0}}, None)["filter_source"] == "type"


def test_apply_filters_records_every_slot_and_the_mode():
    out = NP.apply_filters(_plan(), ROWS, _advise, "shadow")
    assert out["filter_mode"] == "shadow"
    a, b = out["slots"]
    assert a["filters"] == {"L": 4, "R": 1, "G": 1, "B": 1} and a["filter_source"] == "advisor"
    assert b["filters"] == {"G": 20} and b["filter_source"] == "monitor"
    # the slot's own fields are untouched
    assert (a["dso"], a["start"], a["end"], a["hours"]) == tuple(SLOTS[0].values())


def test_apply_filters_does_not_ask_the_advisor_about_an_explicit_target():
    asked = []

    def advise(dso, start, end):
        asked.append(dso)
        return None
    NP.apply_filters(_plan(), ROWS, advise, "default")
    assert asked == ["ngc1961"]


def test_a_crashing_advisor_is_a_reason_not_a_failed_plan():
    def advise(dso, start, end):
        raise RuntimeError("forecast down")
    out = NP.apply_filters(_plan(), ROWS, advise, "default")
    a = out["slots"][0]
    assert "filters" not in a and a["filter_source"] == "type"
    assert any("forecast down" in r for r in a["filter_reasons"])


def test_mode_off_records_only_the_mode():
    out = NP.apply_filters(_plan(), ROWS, _advise, "off")
    assert out["filter_mode"] == "off" and "filters" not in out["slots"][0]


def test_the_slots_round_trip_with_their_filters():
    rec = NP.apply_filters(_plan(), ROWS, _advise, "default")["slots"]
    slots = NP.slots_from_record(rec)
    assert slots[0].filters == {"L": 4, "R": 1, "G": 1, "B": 1} and slots[1].filters == {"G": 20}
    assert isinstance(slots[0], slot_plan.Slot) and slots[0].good_hours == 4
    back = NP.slots_record(slots)
    assert back[0]["filters"] == rec[0]["filters"] and back[1]["filters"] == rec[1]["filters"]
    # a record without filters gives a Slot without, as before
    plain = NP.slots_from_record([{k: v for k, v in rec[0].items() if k != "filters"}])
    assert plain[0].filters is None and "filters" not in NP.slots_record(plain)[0]


def test_slot_equality_and_signature_still_hold_without_filters():
    s = slot_plan.Slot("x", T0, T0 + timedelta(hours=3), 3)
    assert s == slot_plan.Slot("x", T0, T0 + timedelta(hours=3), 3, 5, None)
    assert slot_plan.signature([s]) == slot_plan.signature([slot_plan.Slot("x", T0, T0 + timedelta(hours=3), 3,
                                                                           filters={"L": 4})])


def test_describe_filters_shadow_and_default():
    plan = NP.apply_filters(_plan(), ROWS, _advise, "shadow")
    written = {"ngc1961": {"L": 20, "R": 5, "G": 5, "B": 5}}
    text = NP.describe_filters(plan, written)
    assert text.startswith("Filter advisor (shadow, nothing changed):")
    assert "ngc1961 21:00-01:00  advised L=4 R=1 G=1 B=1 (advisor)  |  sequence L\u00d720 R\u00d75 G\u00d75 B\u00d75" in text
    assert "    seeing forecast good: L x1.5" in text
    assert "hubblev1 01:00-02:00  advised G=20 (monitor)  |  sequence -" in text
    assert "nina.filter_advisor = default" in text
    plan["filter_mode"] = "default"
    text = NP.describe_filters(plan, written)
    assert text.startswith("Filters tonight:") and "filters L=4 R=1 G=1 B=1 (advisor)" in text
    assert "nina.filter_advisor = default" not in text
    assert NP.describe_filters(NP.apply_filters(_plan(), ROWS, _advise, "off")) == ""
    assert NP.describe_filters({"slots": SLOTS, "filter_mode": "shadow"}) == ""   # an old record


def test_read_filters_skips_a_night_that_is_off(tmp_path):
    out = NP.read_filters(_plan(will=False), tmp_path / "q.json", {"nina": {"filter_advisor": "default"}})
    assert out["filter_mode"] == "default" and "filter_source" not in out["slots"][0]
