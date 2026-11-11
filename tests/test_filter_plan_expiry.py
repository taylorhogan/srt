"""C2 of docs/AUTOMATION_PLAN.md (2026-10-30): `filters <dso> ... for N` ends by itself.

The date rules and the plan reader are pure (control/filter_plan) and run on
CI; the queue file round trip needs control.instructions (astropy, the chat
server) and runs in the venv.
"""
import json
from datetime import date, datetime, timedelta, timezone

import pytest

from control import filter_plan as FP
from control import night_plan as NP

EDT = timezone(timedelta(hours=-4))
EST = timezone(timedelta(hours=-5))


def test_a_night_is_named_after_its_evening():
    assert FP.night_for(datetime(2026, 10, 30, 14, 0)) == date(2026, 10, 30)
    assert FP.night_for(datetime(2026, 10, 30, 23, 59)) == date(2026, 10, 30)
    assert FP.night_for(datetime(2026, 10, 31, 2, 0)) == date(2026, 10, 30)      # the night in progress
    assert FP.night_for(datetime(2026, 10, 31, 6, 0)) == date(2026, 10, 31)
    # the DST change (2026-11-01 01:30 happens twice) is nothing: date and hour only
    assert FP.night_for(datetime(2026, 11, 1, 1, 30, tzinfo=EDT)) == date(2026, 10, 31)
    assert FP.night_for(datetime(2026, 11, 1, 1, 30, tzinfo=EST)) == date(2026, 10, 31)


def test_a_plan_set_at_any_hour_starts_tonight():
    assert FP.until_for(datetime(2026, 10, 30, 14, 0), 1) == "2026-10-30"
    assert FP.until_for(datetime(2026, 10, 30, 14, 0), 2) == "2026-10-31"
    assert FP.until_for(datetime(2026, 10, 31, 2, 0), 1) == "2026-10-31"      # the running night cannot change
    assert FP.until_for(datetime(2026, 10, 31, 10, 0), 1) == "2026-10-31"     # before noon: tonight
    assert FP.until_for(datetime(2026, 10, 30, 14, 0), 0) == "2026-10-30"     # never less than tonight


def test_active_plan_honours_the_last_night():
    row = {"filter_plan": {"L": 2, "R": 1}, "filter_plan_until": "2026-10-31"}
    assert FP.active_plan(row, date(2026, 10, 30)) == {"L": 2, "R": 1}
    assert FP.active_plan(row, date(2026, 10, 31)) == {"L": 2, "R": 1}        # the last night still applies
    assert FP.active_plan(row, date(2026, 11, 1)) is None
    assert FP.is_expired(row, date(2026, 11, 1)) and not FP.is_expired(row, date(2026, 10, 31))
    # no limit, no row, no plan, a plan of zeros, a malformed date
    assert FP.active_plan({"filter_plan": {"L": 2}}, date(2030, 1, 1)) == {"L": 2}
    assert FP.active_plan(None, date(2026, 10, 30)) is None
    assert FP.active_plan({}, date(2026, 10, 30)) is None
    assert FP.active_plan({"filter_plan": {"L": 0}}, date(2026, 10, 30)) is None
    assert FP.active_plan({"filter_plan": {"L": 1}, "filter_plan_until": "soon"}, date(2026, 10, 30)) == {"L": 1}


def test_nights_left_and_the_display():
    row = {"filter_plan": {"L": 2}, "filter_plan_until": "2026-10-31"}
    now = datetime(2026, 10, 30, 14, 0)
    assert FP.nights_left(row, now) == 2
    assert FP.describe_until(row, now) == " (until 2026-10-31, 2 more nights)"
    assert FP.describe_until(row, datetime(2026, 10, 31, 14, 0)) == " (until 2026-10-31, tonight only)"
    assert FP.describe_until(row, datetime(2026, 11, 1, 9, 0)) == " (expired 2026-10-31)"
    assert FP.nights_left(row, datetime(2026, 11, 3, 9, 0)) == 0
    assert FP.describe_until({"filter_plan": {"L": 2}}, now) == "" and FP.nights_left({}, now) is None


def test_split_for_nights():
    assert FP.split_for_nights(["L=2", "R=1", "for", "2"]) == (["L=2", "R=1"], 2, None)
    assert FP.split_for_nights(["L=2", "R=1", "FOR", "1"]) == (["L=2", "R=1"], 1, None)
    assert FP.split_for_nights(["L=2", "R=1"]) == (["L=2", "R=1"], None, None)
    toks, n, err = FP.split_for_nights(["L=2", "for"])
    assert toks == ["L=2"] and n is None and "needs a number" in err
    assert "whole number" in FP.split_for_nights(["L=2", "for", "0"])[2]
    assert "whole number" in FP.split_for_nights(["L=2", "for", "two"])[2]
    assert "at most" in FP.split_for_nights(["L=2", "for", "999"])[2]
    assert FP.split_for_nights([]) == ([], None, None)


def test_expired_rows_and_the_chat_line():
    rows = [{"dso": "a", "filter_plan": {"L": 2}, "filter_plan_until": "2026-10-30"},
            {"dso": "b", "filter_plan": {"Ha": 1}, "filter_plan_until": "2026-10-31"},
            {"dso": "c", "filter_plan": {"Ha": 1}},
            {"dso": "d", "filter_plan_until": "2026-10-01"}]                    # a limit with no plan: nothing
    gone = FP.expired_rows(rows, date(2026, 10, 31))
    assert [(r["dso"], p, u) for r, p, u in gone] == [("a", {"L": 2}, "2026-10-30")]
    text = FP.describe_expired([{"dso": "a", "plan": {"L": 2, "R": 1}, "until": "2026-10-30"}])
    assert text.startswith("Filter plans expired") and "  a -- L=2 R=1, last night 2026-10-30" in text
    assert FP.describe_expired([]) == ""


# ------------------------------------------------------------ imaged nights (2026-11-11)

def _counted(n, imaged=(), until="2026-12-09"):
    return {"filter_plan": {"Ha": 3, "O-III": 2}, "filter_plan_until": until,
            "filter_plan_count": {"nights": n, "from": "2026-11-11", "imaged": list(imaged)}}


def test_limit_for_caps_at_30_nights_or_twice_n():
    now = datetime(2026, 11, 11, 9, 0)
    assert FP.limit_for(now, 5) == ("2026-12-10", {"nights": 5, "from": "2026-11-11", "imaged": []})
    assert FP.limit_for(now, 20)[0] == "2026-12-20"                    # 2 N = 40 nights beats 30


def test_cloudy_nights_do_not_count():
    row = _counted(2)
    frames = {date(2026, 11, 10), date(2026, 11, 11), date(2026, 11, 14)}   # 11-10 is before the plan
    # planning 11-12: 11-11 imaged; 11-12 not yet; one used
    assert FP.count_imaged(row, frames, date(2026, 11, 12)) == ["2026-11-11"]
    # 11-12 and 11-13 cloudy: still one used on 11-14, the plan stands
    row["filter_plan_count"]["imaged"] = FP.count_imaged(row, frames, date(2026, 11, 14))
    assert row["filter_plan_count"]["imaged"] == ["2026-11-11"]
    assert FP.active_plan(row, date(2026, 11, 14)) == {"Ha": 3, "O-III": 2}
    assert FP.describe_until(row, datetime(2026, 11, 14, 12)) == \
        " (1 of 2 imaged nights used; ends by 2026-12-09 regardless)"
    # 11-14 imaged: two used by 11-15, out
    row["filter_plan_count"]["imaged"] = FP.count_imaged(row, frames, date(2026, 11, 15))
    assert FP.is_expired(row, date(2026, 11, 15)) and FP.active_plan(row, date(2026, 11, 15)) is None
    assert FP.describe_until(row, datetime(2026, 11, 15, 12)) == " (expired: 2 imaged nights)"
    gone = FP.expired_rows([dict(row, dso="ic405")], date(2026, 11, 15))
    assert gone[0][2] == "2 imaged nights, 2026-11-11 2026-11-14"
    text = FP.describe_expired([{"dso": "ic405", "plan": {"Ha": 3}, "until": gone[0][2]}])
    assert "ic405 -- Ha=3, after 2 imaged nights, 2026-11-11 2026-11-14" in text


def test_the_cap_still_ends_a_plan_that_never_images():
    row = _counted(3, until="2026-12-10")
    assert FP.active_plan(row, date(2026, 12, 10)) is not None
    assert FP.is_expired(row, date(2026, 12, 11))
    assert FP.count_imaged({"filter_plan": {"L": 1}}, {date(2026, 11, 12)}, date(2026, 11, 13)) is None
    # an unreadable count is no count: the date rule alone
    bad = dict(_counted(1), filter_plan_count={"nights": "x"})
    assert FP.imaged_used(bad) is None and not FP.is_expired(bad, date(2026, 11, 20))


def test_nights_with_frames_reads_the_light_folders(tmp_path):
    for night, name in (("2026-11-10", "a.fits"), ("2026-11-11", "b.fits.bad"), ("2026-11-12", "c.FITS")):
        d = tmp_path / "cdk17" / night / "LIGHT"
        d.mkdir(parents=True)
        (d / name).write_text("")
    (tmp_path / "cdk17" / "2026-11-13" / "FLAT").mkdir(parents=True)
    (tmp_path / "cdk17" / "notadate" / "LIGHT").mkdir(parents=True)
    (tmp_path / "frame_stats.json").write_text("[]")
    assert FP.nights_with_frames(tmp_path) == {date(2026, 11, 10), date(2026, 11, 12)}
    assert FP.nights_with_frames(tmp_path / "missing") == set()


def test_the_noon_reason_names_the_imaged_night():
    out = NP.slot_filters("ic405", _counted(5, ["2026-11-11"]), None, night=date(2026, 11, 12))
    assert out["filter_source"] == "explicit" and out["filter_reasons"][0].endswith("imaged night 2 of 5")


# ------------------------------------------------------------ the noon plan's decision

SLOTS = [{"dso": "ngc1961", "start": "2026-11-01T21:00-04:00", "end": "2026-11-02T01:00-04:00", "hours": 4}]
ADVICE = {"action": "plan", "plan": {"L": 4, "R": 1, "G": 1, "B": 1}, "reasons": ["need"]}


def _advise(dso, start, end):
    return ADVICE


def test_the_noon_decision_treats_an_expired_plan_as_no_plan():
    plan = {"best": "ngc1961", "will_image": True, "slots": SLOTS}
    live = [{"dso": "ngc1961", "status": "waiting", "filter_plan": {"L": 1, "R": 1},
             "filter_plan_until": "2026-11-01"}]
    out = NP.apply_filters(plan, live, _advise, "default")                     # night = the slot's: 11-01
    assert out["slots"][0]["filter_source"] == "explicit"
    assert out["slots"][0]["filters"] == {"L": 1, "R": 1}
    assert "last night 2026-11-01" in out["slots"][0]["filter_reasons"][0]
    dead = [dict(live[0], filter_plan_until="2026-10-31")]
    out = NP.apply_filters(plan, dead, _advise, "default")
    assert out["slots"][0]["filter_source"] == "advisor" and out["slots"][0]["filters"] == ADVICE["plan"]
    # an explicit night overrides the slot's
    out = NP.apply_filters(plan, live, _advise, "default", night=date(2026, 11, 2))
    assert out["slots"][0]["filter_source"] == "advisor"
    # no limit: as before
    out = NP.apply_filters(plan, [{"dso": "ngc1961", "status": "waiting", "filter_plan": {"L": 1}}], _advise, "default")
    assert out["slots"][0]["filter_source"] == "explicit"


# ------------------------------------------------------------ the queue file (venv)

@pytest.fixture
def queue(tmp_path, monkeypatch):
    instr = pytest.importorskip("control.instructions")
    p = tmp_path / "q.json"
    p.write_text(json.dumps([{"dso": "ngc1961", "status": "waiting"},
                             {"dso": "m31", "status": "completed", "filter_plan": {"L": 9}},
                             {"dso": "m31", "status": "waiting"}]))
    monkeypatch.setattr(instr, "_INSTRUCTIONS_PATH", str(p))
    return instr, p


def test_set_get_and_expire_round_trip(queue):
    instr, p = queue
    assert instr.set_filter_plan_db("NGC 1961", {"L": 2, "R": 1}, until="2026-10-31")
    row = json.loads(p.read_text())[0]
    assert row["filter_plan"] == {"L": 2, "R": 1} and row["filter_plan_until"] == "2026-10-31"
    assert instr.get_filter_plan("ngc1961", night=date(2026, 10, 31)) == {"L": 2, "R": 1}
    assert instr.get_filter_plan("ngc1961", night=date(2026, 11, 1)) is None
    # a new plan without a limit lifts the old limit
    assert instr.set_filter_plan_db("ngc1961", {"L": 3})
    assert "filter_plan_until" not in json.loads(p.read_text())[0]
    # limit an existing plan; refuse where there is none
    assert instr.set_filter_plan_until_db("ngc1961", "2026-10-30")
    assert not instr.set_filter_plan_until_db("m31", "2026-10-30")          # the waiting m31 has no plan
    # the noon refresh drops what has run out and says what it dropped
    assert instr.expire_filter_plans(night=date(2026, 10, 30)) == []
    gone = instr.expire_filter_plans(night=date(2026, 10, 31))
    assert gone == [{"dso": "ngc1961", "plan": {"L": 3}, "until": "2026-10-30"}]
    rows = json.loads(p.read_text())
    assert "filter_plan" not in rows[0] and "filter_plan_until" not in rows[0]
    assert rows[0]["status_history"][-1]["filter_plan_expired"] == {"L": 3}
    assert rows[1]["filter_plan"] == {"L": 9}                                  # completed rows are left alone
    # clearing clears the limit too
    assert instr.set_filter_plan_db("m31", {"L": 1}, until="2026-12-01") and instr.set_filter_plan_db("m31", None)
    assert all(k not in json.loads(p.read_text())[2] for k in ("filter_plan", "filter_plan_until"))


def test_the_noon_refresh_counts_then_expires(queue):
    instr, p = queue
    until, count = FP.limit_for(datetime(2026, 11, 11, 9), 2)
    assert instr.set_filter_plan_db("ngc1961", {"Ha": 3}, until=until, count=count)
    # 11-11 cloudy: nothing counted, nothing written
    assert instr.expire_filter_plans(night=date(2026, 11, 12), frame_nights={}) == []
    assert json.loads(p.read_text())[0]["filter_plan_count"]["imaged"] == []
    # 11-12 imaged: one counted, still standing on 11-13
    assert instr.expire_filter_plans(night=date(2026, 11, 13), frame_nights={"ngc1961": {date(2026, 11, 12)}}) == []
    assert json.loads(p.read_text())[0]["filter_plan_count"]["imaged"] == ["2026-11-12"]
    assert instr.get_filter_plan("ngc1961", night=date(2026, 11, 13)) == {"Ha": 3}
    # 11-13 imaged too: out at the next noon, every key with it
    gone = instr.expire_filter_plans(night=date(2026, 11, 14),
                                     frame_nights={"ngc1961": {date(2026, 11, 12), date(2026, 11, 13)}})
    assert gone == [{"dso": "ngc1961", "plan": {"Ha": 3}, "until": "2 imaged nights, 2026-11-12 2026-11-13"}]
    row = json.loads(p.read_text())[0]
    assert not any(k in row for k in ("filter_plan", "filter_plan_until", "filter_plan_count"))


# ------------------------------------------------------------ the chat command (venv)

def test_filters_cmd_for_n(monkeypatch):
    suc = pytest.importorskip("cmd_processing.super_user_commands")
    posted, saved, limited = [], [], []
    rows = {"ngc1961": {"dso": "ngc1961", "status": "waiting"}}
    monkeypatch.setattr(suc.config, "data", lambda: {"nina": {"filter_wheel": {"L": 1, "R": 2, "G": 3, "B": 4}}})
    monkeypatch.setattr(suc.social_server, "post_social_message", lambda m: posted.append(m))
    monkeypatch.setattr(suc.instructions, "get_instruction_by_dso", lambda d: rows.get(d.lower().replace(" ", "")))
    monkeypatch.setattr(suc.instructions, "get_filter_plan", lambda d, night=None: rows.get(d, {}).get("filter_plan"))
    monkeypatch.setattr(suc.instructions, "set_filter_plan_db",
                        lambda d, p, until=None, count=None: saved.append((d, p, until, count)) or True)
    monkeypatch.setattr(suc.instructions, "set_filter_plan_until_db",
                        lambda d, u, count=None: limited.append((d, u, count)) or bool(rows.get(d, {}).get("filter_plan")))
    monkeypatch.setattr(suc, "datetime", type("D", (), {"now": staticmethod(lambda: datetime(2026, 10, 30, 14, 0))}))

    # `for N` counts imaged nights from tonight, capped at 30 calendar nights
    suc.filters_cmd(["", "filters", "ngc1961", "L=2", "R=1", "for", "2"], "iris")
    assert saved == [("ngc1961", {"L": 2, "R": 1}, "2026-11-28",
                      {"nights": 2, "from": "2026-10-30", "imaged": []})]
    assert "0 of 2 imaged nights used; ends by 2026-11-28 regardless" in posted[-1]
    assert "cloudy nights do not count" in posted[-1] and "then the advisor decides" in posted[-1]
    suc.filters_cmd(["", "filters", "ngc1961", "L=2", "R=1"], "iris")
    assert saved[-1] == ("ngc1961", {"L": 2, "R": 1}, None, None) and "until `filters ngc1961 clear`" in posted[-1]
    suc.filters_cmd(["", "filters", "ngc1961", "L=2", "for", "x"], "iris")
    assert "whole number" in posted[-1] and len(saved) == 2
    # limit an existing plan; refuse where there is none
    suc.filters_cmd(["", "filters", "ngc1961", "for", "3"], "iris")
    assert limited == [("ngc1961", "2026-11-28", {"nights": 3, "from": "2026-10-30", "imaged": []})]
    assert "no filter plan to limit" in posted[-1]
    rows["ngc1961"]["filter_plan"] = {"L": 2, "R": 1}
    suc.filters_cmd(["", "filters", "ngc1961", "for", "3"], "iris")
    assert "L=2, R=1 (0 of 3 imaged nights used; ends by 2026-11-28 regardless)" in posted[-1]
    # show: past the cap
    rows["ngc1961"]["filter_plan_until"] = "2026-10-29"
    monkeypatch.setattr(suc.instructions, "get_filter_plan", lambda d, night=None: None)
    suc.filters_cmd(["", "filters", "ngc1961"], "iris")
    assert "(expired 2026-10-29, 0 of 3 imaged nights used) -- no longer applies" in posted[-1]
