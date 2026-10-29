from datetime import date

from scripts.noon_watch import evaluate

TODAY = date(2026, 10, 15)
WAITING = "10/15/2026 06:04:18 AM Waiting for noon at 2026-10-15 12:00:00"
START = "10/15/2026 12:00:07 PM State: NOON_CHECK"


def test_finished_with_imaging_planned():
    lines = [WAITING, START, "10/15/2026 12:02:26 PM State: WAITING_FOR_PRE_SUNSET"]
    assert evaluate(lines, TODAY, True) == []


def test_finished_with_a_skip():
    lines = [WAITING, START, "10/15/2026 12:01:00 PM Skipping tonight — not enough good hours (1)"]
    assert evaluate(lines, TODAY, True) == []


def test_the_2026_10_15_hang():
    lines = [WAITING, START, "10/15/2026 12:00:09 PM HTTP Request: POST http://127.0.0.1:8947/api/logs/"]
    p = evaluate(lines, TODAY, False, "TimeoutError: timed out")
    assert len(p) == 2
    assert p[0].startswith("noon check STARTED (12:00:07 PM) but never finished")
    assert "api/logs" in p[0]
    assert "web chat server not answering" in p[1]


def test_yesterdays_finish_does_not_count():
    lines = ["10/14/2026 12:02:26 PM State: WAITING_FOR_PRE_SUNSET", WAITING]
    p = evaluate(lines, TODAY, True)
    assert p == ["noon check never started today (scheduler not waking at noon?)"]


def test_traceback_continuation_lines_are_ignored():
    lines = [START, "  File \"x.py\", line 1, State: WAITING_FOR_PRE_SUNSET"]
    assert len(evaluate(lines, TODAY, True)) == 1


def test_unhandled_scheduler_exception_is_reported():
    lines = [START, "10/15/2026 12:00:30 PM Unhandled exception in scheduler loop",
             "10/15/2026 12:02:26 PM State: WAITING_FOR_PRE_SUNSET"]
    assert evaluate(lines, TODAY, True) == ["scheduler logged an unhandled exception today"]


def test_a_hold_skip_is_a_problem_but_a_weather_skip_is_not():
    hold = [WAITING, START, "10/15/2026 12:00:30 PM Skipping tonight — conductor's noon plan: skipped"]
    p = evaluate(hold, TODAY, True)
    assert len(p) == 1 and "hold" in p[0] and "resolve!" in p[0]
    weather = [WAITING, START, "10/15/2026 12:00:30 PM Skipping tonight — conductor's noon plan: bad"]
    assert evaluate(weather, TODAY, True) == []
