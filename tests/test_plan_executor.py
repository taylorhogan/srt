"""The scheduler carrying out the conductor's plan (scheduler_server.driven_cycle, ADR 0015).

Needs the scheduler's whole import stack (Prefect, the chat server), so it
skips on the pytest-only CI runner and runs in the observatory's .venv.
GET /v1/plan is scripted; every side effect is recorded, nothing posts or
generates for real. What must hold: the scheduler acts on the conductor's
slots and never ranks to decide, a replan rewrites the sequence, a restart
never launches twice, and no plan is no night.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("prefect")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from end_points import scheduler_server as S  # noqa: E402

TZ = timezone(timedelta(hours=-4))
SUNSET = datetime(2026, 10, 16, 18, 10, tzinfo=TZ)
PLAN = {"best": "ngc7790", "good_hours": 6, "best_start": "2026-10-16T19:00-04:00", "will_image": True,
        "slots": [{"dso": "ngc7790", "start": "2026-10-16T19:00-04:00",
                   "end": "2026-10-17T01:00-04:00", "hours": 6, "priority": 50}]}


def rec(check, outcome, seq, plan=PLAN, ts="2026-10-16T12:00:30-04:00", why=None):
    r = {"check": check, "outcome": outcome, "plan": plan, "seq": seq, "ts": ts}
    if why:
        r["why"] = why
    return r


class World:
    def __init__(self, monkeypatch, replies, now=datetime(2026, 10, 16, 12, 0, 40, tzinfo=TZ)):
        self.replies = list(replies)
        self.now = now
        self.log = []
        from iris import client
        monkeypatch.setattr(client, "get_plan", self.get_plan)
        monkeypatch.setattr(S, "_local_now", lambda: self.now)
        monkeypatch.setattr(S.weather, "get_sunrise_sunset", lambda: (None, SUNSET))
        monkeypatch.setattr(S.time, "sleep", self.sleep)
        monkeypatch.setattr(S, "set_state", lambda st, *a, **k: self.log.append(("state", st.name)))
        monkeypatch.setattr(S, "_cross_check", lambda *a, **k: self.log.append(("cross", a[0])))
        monkeypatch.setattr(S, "_post_filters", lambda plan: self.log.append(("filters", plan.get("filter_mode"))))
        monkeypatch.setattr(S, "_publish_skip_to_site", lambda r: self.log.append(("skip_site", r)))
        monkeypatch.setattr(S, "_publish_plan_to_site", lambda *a, **k: self.log.append(("restamp",)))
        monkeypatch.setattr(S.obs_calendar, "set_today_stat", lambda *a: self.log.append(("cal",) + a))
        for name in ("post_social_message", "post_html_message", "tonight_cmd",
                     "post_dso_preview", "_post_station_passes"):
            monkeypatch.setattr(S.social_server, name,
                                lambda *a, _n=name, **k: self.log.append(("chat", _n, a[0] if a else None)))
        monkeypatch.setattr(S.pushover, "push_message", lambda m, **k: self.log.append(("push", m)))
        monkeypatch.setattr(S, "_get_best_object",
                            lambda: pytest.fail("the executor must not rank to decide"))

        def gen(dso, hours=0.0, notify=True, slots=None, plan_filters=False):
            self.log.append(("generate", dso, notify, [s.name for s in slots]))
            self.log.append(("plan_filters", plan_filters, [s.filters for s in slots]))
            S._PLAN_ON_DISK["sig"] = S.slot_plan.signature(slots, S._filter_plans_for(slots, plan_filters))
        monkeypatch.setattr(S, "generate_sequence_task", gen)
        monkeypatch.setattr(S, "imaging_task", lambda: self.log.append(("launch",)))
        monkeypatch.setattr(S.instructions, "get_filter_plan", lambda name: None)
        S._PLAN_ON_DISK.clear()

    def get_plan(self, day=None, timeout=8.0):
        r = self.replies[0] if len(self.replies) == 1 else self.replies.pop(0)
        return r

    def sleep(self, s):
        self.now += timedelta(seconds=s)

    def kinds(self, k):
        return [e for e in self.log if e[0] == k]


def reply(noon=None, sunset=None):
    return {"date": "2026-10-16", "noon": noon, "sunset": sunset}


def test_a_good_night_is_carried_out_from_the_conductors_slots(monkeypatch):
    w = World(monkeypatch, [reply(rec("noon", "good", 5)), reply(rec("noon", "good", 5)),
                            reply(rec("noon", "good", 5)),
                            reply(rec("noon", "good", 5), rec("sunset", "good", 9))])
    S.driven_cycle()
    assert w.kinds("generate") == [("generate", "ngc7790", True, ["ngc7790"])]
    assert w.kinds("launch") == [("launch",)]
    assert ("restamp",) in w.log                      # same plan at sunset: no regeneration
    assert [c[1] for c in w.kinds("cross")] == ["noon", "sunset"]


def test_a_changed_sunset_plan_regenerates_before_the_launch(monkeypatch):
    two = dict(PLAN, slots=PLAN["slots"] + [{"dso": "m33", "start": "2026-10-17T01:00-04:00",
                                             "end": "2026-10-17T05:00-04:00", "hours": 4}])
    w = World(monkeypatch, [reply(rec("noon", "good", 5)), reply(rec("noon", "good", 5)),
                            reply(rec("noon", "good", 5), rec("sunset", "good", 9, plan=two))])
    S.driven_cycle()
    gens = w.kinds("generate")
    assert len(gens) == 2 and gens[1][3] == ["ngc7790", "m33"]
    assert w.log.index(gens[1]) < w.log.index(("launch",))


def test_a_replan_rewrites_the_sequence(monkeypatch):
    other = dict(PLAN, best="m33", slots=[dict(PLAN["slots"][0], dso="m33")])
    w = World(monkeypatch, [reply(rec("noon", "good", 5)), reply(rec("noon", "good", 5)),
                            reply(rec("noon", "good", 12, plan=other)),
                            reply(rec("noon", "good", 12, plan=other), rec("sunset", "good", 20, plan=other))])
    S.driven_cycle()
    assert [g[1] for g in w.kinds("generate")] == ["ngc7790", "m33"]
    assert w.kinds("launch") == [("launch",)]


def test_the_plans_filters_are_written_only_when_its_mode_is_default(monkeypatch):
    """C1: the NIGHT_PLAN record carries each slot's ratio; the scheduler hands
    it to the generator when the record says filter_mode = default, and the
    filters post follows every generation."""
    slot = dict(PLAN["slots"][0], filters={"L": 4, "R": 1, "G": 1, "B": 1}, filter_source="advisor",
                filter_reasons=["need (goal - have): L 30, R 4, G 4, B 4"])
    for mode, want in (("shadow", False), ("default", True)):
        plan = dict(PLAN, slots=[slot], filter_mode=mode)
        w = World(monkeypatch, [reply(rec("noon", "good", 5, plan=plan)),
                                reply(rec("noon", "good", 5, plan=plan)),
                                reply(rec("noon", "good", 5, plan=plan)),
                                reply(rec("noon", "good", 5, plan=plan), rec("sunset", "good", 9, plan=plan))])
        S.driven_cycle()
        assert w.kinds("plan_filters") == [("plan_filters", want, [{"L": 4, "R": 1, "G": 1, "B": 1}])], mode
        assert w.kinds("filters") == [("filters", mode)]
        assert ("restamp",) in w.log                  # the same ratio at sunset: no regeneration


def test_a_ratio_that_changes_by_sunset_regenerates_when_live(monkeypatch):
    noon = dict(PLAN, filter_mode="default",
                slots=[dict(PLAN["slots"][0], filters={"L": 4, "R": 1, "G": 1, "B": 1}, filter_source="advisor")])
    sunset = dict(noon, slots=[dict(noon["slots"][0], filters={"L": 1, "R": 1, "G": 1, "B": 1})])
    w = World(monkeypatch, [reply(rec("noon", "good", 5, plan=noon)), reply(rec("noon", "good", 5, plan=noon)),
                            reply(rec("noon", "good", 5, plan=noon), rec("sunset", "good", 9, plan=sunset))])
    S.driven_cycle()
    assert [g[2] for g in w.kinds("plan_filters")] == [[{"L": 4, "R": 1, "G": 1, "B": 1}],
                                                        [{"L": 1, "R": 1, "G": 1, "B": 1}]]
    assert ("restamp",) not in w.log
    # in shadow the ratio is not the sequence's, so its change is not a reason to regenerate
    w = World(monkeypatch, [reply(rec("noon", "good", 5, plan=dict(noon, filter_mode="shadow"))),
                            reply(rec("noon", "good", 5, plan=dict(noon, filter_mode="shadow"))),
                            reply(rec("noon", "good", 5, plan=dict(noon, filter_mode="shadow")),
                                  rec("sunset", "good", 9, plan=dict(sunset, filter_mode="shadow")))])
    S.driven_cycle()
    assert len(w.kinds("plan_filters")) == 1 and ("restamp",) in w.log


def test_a_bad_noon_skips_without_generating(monkeypatch):
    w = World(monkeypatch, [reply(rec("noon", "bad", 5, plan=dict(PLAN, will_image=False, good_hours=1)))])
    S.driven_cycle()
    assert not w.kinds("generate") and not w.kinds("launch")
    assert ("cal", "weather", "ngc7790") in w.log and w.kinds("skip_site")


def test_a_failed_noon_says_why(monkeypatch):
    w = World(monkeypatch, [reply(rec("noon", "failed", 5, plan=None, why="TimeoutError: 600 s"))])
    S.driven_cycle()
    posts = [c[2] for c in w.kinds("chat") if c[1] == "post_social_message"]
    assert any("failed" in p and "TimeoutError" in p for p in posts)
    assert not w.kinds("generate")


def test_an_unreachable_conductor_is_no_night_and_one_push(monkeypatch):
    w = World(monkeypatch, [None])
    S.driven_cycle()
    assert len(w.kinds("push")) == 1 and not w.kinds("generate") and not w.kinds("launch")
    assert w.now >= SUNSET                            # waited until the sunset check, no longer


def test_a_restart_in_the_afternoon_regenerates_quietly_and_still_launches(monkeypatch):
    w = World(monkeypatch, [reply(rec("noon", "good", 5)), reply(rec("noon", "good", 5)),
                            reply(rec("noon", "good", 5)),
                            reply(rec("noon", "good", 5), rec("sunset", "good", 9))],
              now=datetime(2026, 10, 16, 15, 0, tzinfo=TZ))
    S.driven_cycle()
    assert w.kinds("generate") == [("generate", "ngc7790", False, ["ngc7790"])]
    assert not [c for c in w.kinds("chat") if c[1] == "tonight_cmd"]
    assert w.kinds("launch") == [("launch",)]


def test_a_restart_after_the_sunset_check_never_launches(monkeypatch):
    w = World(monkeypatch, [reply(rec("noon", "good", 5), rec("sunset", "good", 9))],
              now=datetime(2026, 10, 16, 22, 0, tzinfo=TZ))
    S.driven_cycle()
    assert not w.kinds("launch") and not w.kinds("generate")


def test_a_restart_on_a_skipped_day_says_nothing_again(monkeypatch):
    w = World(monkeypatch, [reply(rec("noon", "bad", 5, plan=dict(PLAN, will_image=False)))],
              now=datetime(2026, 10, 16, 15, 0, tzinfo=TZ))
    S.driven_cycle()
    assert not w.kinds("chat") and not w.kinds("cal")


def test_a_bad_sunset_skips_the_launch(monkeypatch):
    w = World(monkeypatch, [reply(rec("noon", "good", 5)), reply(rec("noon", "good", 5)),
                            reply(rec("noon", "good", 5), rec("sunset", "bad", 9,
                                  plan=dict(PLAN, will_image=False, good_hours=2)))])
    S.driven_cycle()
    assert not w.kinds("launch") and w.kinds("skip_site")


def test_replan_reply_text():
    from cmd_processing.super_user_commands import replan_reply_text
    assert "not answering" in replan_reply_text(None)
    assert "ranking the queue again" in replan_reply_text({"accepted": True})
    assert "PRE_FLIGHT" in replan_reply_text({"accepted": False, "state": "PRE_FLIGHT",
                                              "guard": "no transition"})


def test_one_slot_that_is_not_the_best_images_the_slot(monkeypatch):
    # 2026-11-03 sunset: best hubblev1, one slot ngc2276 20-02; the sequence
    # was hubblev1 until dawn and the slot never ran
    got = []
    monkeypatch.setattr(S, "_generate_nina_sequence",
                        lambda name, hours, end=None, filters=None: got.append((name, hours, end)) or None)
    end = datetime(2026, 11, 4, 2, 0, tzinfo=TZ)
    slot = S.slot_plan.Slot("ngc2276", end - timedelta(hours=6), end, 5, 5)
    S.generate_sequence_task.fn("hubblev1", 4, notify=False, slots=[slot])
    assert got == [("ngc2276", 5.0, end)]


def test_sunset_change_text_names_what_was_dropped():
    t0 = datetime(2026, 11, 3, 19, 0, tzinfo=TZ)
    noon = [S.slot_plan.Slot("hubblev1", t0, t0, 1, 50), S.slot_plan.Slot("ngc7790", t0, t0, 1, 50),
            S.slot_plan.Slot("ngc2276", t0, t0, 8, 5)]
    assert S.sunset_change_text(noon, noon) is None
    txt = S.sunset_change_text(noon, [noon[2]])
    assert "dropped hubblev1, ngc7790" in txt and "now ngc2276" in txt


def test_a_file_that_does_not_say_what_was_announced_is_refused_and_moved_aside(monkeypatch, tmp_path):
    # 2026-11-03: the announced slot was ngc2276, the generator wrote hubblev1
    # and every check passed. Now the file is compared with the slots.
    import json
    from nina_gen.sequence_lint import SequenceLintError
    out = tmp_path / "full_for_tonight.json"
    posts, pushes = [], []
    monkeypatch.setattr(S.social_server, "post_social_message", lambda m, *a, **k: posts.append(m))
    monkeypatch.setattr(S.pushover, "push_message", lambda m, **k: pushes.append((m, k.get("priority"))))
    monkeypatch.setattr(S, "_local_now", lambda: datetime(2026, 11, 3, 16, 33, tzinfo=TZ))
    monkeypatch.setattr(S.nina_sequence_gen, "announced_vs_written",
                        lambda slots, path: ["slot 1: announced ngc2276, the file has hubblev1"])
    S._REFUSED.update(day=None, key=None)

    def fake_generate(name, hours, end=None, filters=None):
        out.write_text(json.dumps({"target": name}))
        return ({"L": 8}, out)
    monkeypatch.setattr(S, "_generate_nina_sequence", fake_generate)
    end = datetime(2026, 11, 4, 2, 0, tzinfo=TZ)
    slot = S.slot_plan.Slot("ngc2276", end - timedelta(hours=6), end, 5, 5)
    with pytest.raises(SequenceLintError):
        S.generate_sequence_task.fn("ngc2276", 5, notify=False, slots=[slot])
    assert not out.exists() and (tmp_path / "full_for_tonight.refused.json").exists()
    assert posts and posts[-1].startswith("⛔ Sequence REFUSED for ngc2276")
    assert pushes == [(posts[-1][2:], 1)]
    # the same refusal again the same day (the loop retries every minute): said once
    with pytest.raises(SequenceLintError):
        S.generate_sequence_task.fn("ngc2276", 5, notify=False, slots=[slot])
    assert len(pushes) == 1 and len(posts) == 1


def test_a_file_that_says_what_was_announced_is_kept(monkeypatch, tmp_path):
    import json
    out = tmp_path / "full_for_tonight.json"
    monkeypatch.setattr(S.nina_sequence_gen, "announced_vs_written", lambda slots, path: [])
    monkeypatch.setattr(S, "_generate_nina_sequence",
                        lambda name, hours, end=None, filters=None: (out.write_text(json.dumps({})), out)[1:]
                        and ({"L": 4}, out))
    end = datetime(2026, 11, 4, 2, 0, tzinfo=TZ)
    slot = S.slot_plan.Slot("ngc2276", end - timedelta(hours=6), end, 5, 5)
    S.generate_sequence_task.fn("ngc2276", 5, notify=False, slots=[slot])
    assert out.exists()
