"""The refusal drill's pass/fail rules. CI-safe: the script keeps hardware imports in functions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.conductor_drill import nothing_moved_problems, stage1_problems

REFUSED = {"accepted": False, "kind": "ignored", "state": "ESTOP",
           "guard": "no transition for ROOF_OPEN_REQUESTED in state ESTOP"}
ADVISORY = (True, "conductor would have refused: no transition for ROOF_OPEN_REQUESTED in state ESTOP")


def test_stage1_passes_on_the_expected_refusal():
    assert stage1_problems("ESTOP", REFUSED, ADVISORY, "IDLE_DAY") == []


def test_stage1_fails_if_the_conductor_accepts_or_leaves_the_hold():
    p = stage1_problems("ESTOP", dict(REFUSED, accepted=True, state="MANUAL_OPENING"),
                        (True, None), "IDLE_DAY")
    assert any("ACCEPTED" in x for x in p)
    assert any("out of ESTOP" in x for x in p)
    assert any("no refusal text" in x for x in p)


def test_stage1_fails_without_hold_or_resolve():
    p = stage1_problems("IDLE_DAY", None, (True, "conductor unreachable"), "ESTOP")
    assert any("hold not entered" in x for x in p)
    assert any("unreachable" in x for x in p)
    assert any("resolve did not return" in x for x in p)


EV = {"motion_possible": "2026-09-17T09:30:26-04:00", "open_confirmed": "2026-09-17T09:30:15-04:00"}


def test_nothing_moved_passes_on_a_quiet_log():
    log = "Roof will not open: conductor refused — no transition\n"
    assert nothing_moved_problems(log, EV, dict(EV), 0, roof_shut=True) == []


def test_nothing_moved_catches_any_sign_of_motion():
    log = "kasa_do: 'Roof motor' switched on (verified)\nroof relay fire: direction=open\n"
    p = nothing_moved_problems(log, EV, dict(EV, motion_possible="2026-09-17T10:00:00-04:00"),
                               1, roof_shut=False)
    assert len(p) == 5
