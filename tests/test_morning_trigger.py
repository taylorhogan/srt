"""scripts/morning_trigger: when the flats poll starts (pure; no ssh, no clock)."""
import datetime as dt

from scripts import morning_trigger as mt

TZ = dt.timezone(dt.timedelta(hours=-5))


def _t(day, hh, mm):
    return dt.datetime(2026, 11, day, hh, mm, tzinfo=TZ)


def _timeline(written, *slots):
    return {"written": written.isoformat(),
            "slots": [{"dso": d, "start": a.isoformat(), "end": b.isoformat()} for d, a, b in slots]}


TONIGHT = _timeline(_t(20, 12, 1), ("ross248", _t(20, 20, 0), _t(20, 21, 0)),
                    ("ic405", _t(20, 23, 0), _t(21, 4, 0)))


def test_the_morning_a_run_serves_is_the_same_either_side_of_midnight():
    assert mt.morning_date(_t(20, 20, 0)) == dt.date(2026, 11, 21)
    assert mt.morning_date(_t(21, 2, 0)) == dt.date(2026, 11, 21)
    assert mt.morning_date(_t(21, 6, 30)) == dt.date(2026, 11, 21)
    assert mt.morning_date(_t(21, 9, 30)) == dt.date(2026, 11, 21)


def test_planned_end_is_tonights_last_slot_and_an_old_timeline_is_ignored():
    assert mt.planned_end(TONIGHT, _t(20, 20, 0)) == _t(21, 4, 0)
    assert mt.planned_end(TONIGHT, _t(21, 3, 0)) == _t(21, 4, 0)
    yesterday = _timeline(_t(19, 16, 17), ("ic405", _t(19, 22, 0), _t(20, 1, 0)))
    assert mt.planned_end(yesterday, _t(20, 20, 0)) is None
    assert mt.planned_end(None, _t(20, 20, 0)) is None
    assert mt.planned_end({"written": "junk", "slots": []}, _t(20, 20, 0)) is None


def test_with_a_plan_the_wait_runs_to_the_last_end_plus_the_allowance_in_short_steps():
    v, until, why = mt.wait_verdict(_t(20, 20, 0), TONIGHT, [], "06:30", 85, 15)
    assert v == "sleep" and until == _t(20, 20, 15) and why.startswith("last slot ends 04:00 + 85")
    v, until, _ = mt.wait_verdict(_t(21, 5, 20), TONIGHT, [], "06:30", 85, 15)
    assert v == "sleep" and until == _t(21, 5, 25)          # the expected moment itself
    v, until, _ = mt.wait_verdict(_t(21, 5, 25), TONIGHT, [], "06:30", 85, 15)
    assert v == "poll" and until is None


def test_an_end_in_the_journal_starts_the_poll_at_once():
    for name in ("NINA_FLATS_DONE", "SHUTDOWN_DONE"):
        v, _, why = mt.wait_verdict(_t(20, 22, 30), TONIGHT, [(_t(20, 22, 20), name)], "06:30", 85, 15)
        assert v == "poll" and name in why


def test_lights_ending_early_pull_the_expected_time_in():
    ev = [(_t(21, 1, 0), "NIGHT_END_REQUESTED")]
    v, until, why = mt.wait_verdict(_t(21, 1, 5), TONIGHT, ev, "06:30", 85, 15)
    assert v == "sleep" and until == _t(21, 1, 20) and why.startswith("lights ended 01:00 + 85")
    v, _, _ = mt.wait_verdict(_t(21, 2, 25), TONIGHT, ev, "06:30", 85, 15)
    assert v == "poll"
    # A late end request after the plan's end never pushes the time out.
    late = [(_t(21, 4, 30), "NIGHT_END_REQUESTED")]
    v, _, why = mt.wait_verdict(_t(21, 5, 30), TONIGHT, late, "06:30", 85, 15)
    assert v == "poll" and why.startswith("last slot ends")


def test_without_a_timeline_the_wait_runs_to_the_usual_start():
    v, until, why = mt.wait_verdict(_t(20, 20, 0), None, [], "06:30", 85, 15)
    assert v == "sleep" and until == _t(20, 20, 15) and "06:30" in why
    v, until, _ = mt.wait_verdict(_t(21, 6, 20), None, [], "06:30", 85, 15)
    assert v == "sleep" and until == _t(21, 6, 30)
    assert mt.wait_verdict(_t(21, 6, 30), None, [], "06:30", 85, 15)[0] == "poll"


def test_journal_lines_are_parsed_from_the_select_string_output(monkeypatch):
    lines = ['C:\\x\\2026-11-21.jsonl:12:{"seq":1,"ts":"2026-11-21T00:59:47-05:00","kind":"transition",'
             '"event":"NIGHT_END_REQUESTED"}',
             '{"seq":2,"ts":"2026-11-21T02:18:46-05:00","kind":"transition","event":"NINA_FLATS_DONE"}',
             '{"seq":3,"ts":"2026-11-21T02:20:00-05:00","kind":"note","event":"DAY_TICK"}']
    monkeypatch.setattr(mt, "_ps", lambda script, timeout=60: "\n".join(lines))
    ev = mt.journal_events(_t(21, 3, 0))
    assert [n for _t_, n in ev] == ["NINA_FLATS_DONE"]      # a prefixed line is skipped, DAY_TICK unwatched
    assert mt.flats_done_in_journal() is True
    monkeypatch.setattr(mt, "_ps", lambda script, timeout=60: None)
    assert mt.journal_events(_t(21, 3, 0)) == [] and mt.read_timeline() is None
