"""end_points/supervise -- the start_srt policy for a dead child (docs/AUTOMATION_PLAN.md B7)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from end_points import supervise as S  # noqa: E402


def test_the_social_servers_exit_codes_keep_their_meanings():
    assert S.decide(S.SOCIAL, 42, [], 0.0)["action"] == "restart_all"
    assert S.decide(S.SOCIAL, 0, [], 0.0)["action"] == "stop"
    v = S.decide(S.SOCIAL, 1, [], 0.0)
    assert v["action"] == "relaunch_all" and v["backoff_s"] == 5 and "everything" in v["push"]


def test_a_dead_sibling_relaunches_alone_with_a_push():
    v = S.decide("conductor", 1, [], 100.0)
    assert v["action"] == "relaunch" and v["priority"] == 1
    assert "conductor exited unexpectedly" in v["push"] and "it alone" in v["push"]
    assert S.decide("scheduler", -9, [], 100.0)["action"] == "relaunch"


def test_each_child_has_its_own_crash_loop_guard():
    times = []
    for i in range(4):
        v = S.decide("conductor", 1, times, 100.0 + i)
        assert v["action"] == "relaunch" and v["backoff_s"] == min(30, 5 * (i + 1))
    v = S.decide("conductor", 1, times, 104.0)
    assert v["action"] == "give_up" and v["priority"] == 2 and "giving up" in v["push"]
    # crashes outside the window are forgotten
    old = [0.0, 1.0, 2.0, 3.0]
    assert S.decide("scheduler", 1, old, 10_000.0)["action"] == "relaunch" and len(old) == 1
    # a deliberate restart clears the social server's tally
    tally = [1.0, 2.0]
    S.decide(S.SOCIAL, 42, tally, 3.0)
    assert tally == []


def test_dead_children_are_named_with_the_social_server_first():
    assert S.dead_children({"conductor": False, "scheduler": True, S.SOCIAL: False}) == [S.SOCIAL, "conductor"]
    assert S.dead_children({"conductor": True}) == []
