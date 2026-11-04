"""The announced slots against the file N.I.N.A loads (2026-11-03).

_write_verified compares the file with the plan the generator was given.
On 2026-11-03 the generator was given the wrong target (the ranking's best,
hubblev1, for a plan whose one slot was ngc2276), wrote a self-consistent
file for it, and every check passed. announced_vs_written compares the
file with the conductor's slots: names, order, exposures, ends.
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from control.slot_plan import Slot  # noqa: E402
from nina_gen import nina_sequence_gen as g  # noqa: E402

NS = "NINA.Sequencer"
TZ = timezone(timedelta(hours=-5))


def _se(idn, name, iters):
    return {"$id": str(idn), "$type": f"{NS}.SequenceItem.Imaging.SmartExposure, {NS}",
            "Conditions": {"$id": str(idn + 1), "$type": "x", "$values": [
                {"$id": str(idn + 2), "$type": f"{NS}.Conditions.LoopCondition, {NS}",
                 "Iterations": iters, "CompletedIterations": 0}]},
            "Items": {"$id": str(idn + 3), "$type": "x", "$values": [
                {"$id": str(idn + 4), "$type": f"{NS}.SequenceItem.FilterWheel.SwitchFilter, {NS}",
                 "Filter": {"$id": str(idn + 5), "$type": "NINA.Core.Model.Equipment.FilterInfo, NINA.Core",
                            "_name": name, "_position": 1}},
                {"$id": str(idn + 6), "$type": f"{NS}.SequenceItem.Imaging.TakeExposure, {NS}",
                 "ExposureTime": 300.0}]}}


def _target(idn, name, end, counts, provider="time"):
    """A DeepSkyObjectContainer whose imaging loop ends at *end* (h, m) on
    the fixed Time provider, written inline or as a $ref to it."""
    if provider == "time":
        prov = {"$id": str(idn + 50), "$type": g.TIME_PROVIDER_TYPE}
    elif provider == "ref":
        prov = {"$ref": "900"}
    else:
        prov = {"$id": str(idn + 50), "$type": f"{NS}.Utility.DateTimeProvider.DawnProvider, {NS}"}
    cond = {"$id": str(idn + 40), "$type": f"{NS}.Conditions.TimeCondition, {NS}",
            "Hours": end[0], "Minutes": end[1], "Seconds": 0, "SelectedProvider": prov}
    blocks = [_se(idn + 100 + 10 * k, f, n) for k, (f, n) in enumerate(counts.items())]
    return {"$id": str(idn), "$type": f"{NS}.Container.DeepSkyObjectContainer, {NS}",
            "Name": name, "Target": {"$id": str(idn + 1), "TargetName": name},
            "Items": {"$id": str(idn + 2), "$type": "x", "$values": [
                {"$id": str(idn + 3), "$type": f"{NS}.Container.SequentialContainer, {NS}",
                 "Name": "LRGB_IMAGING",
                 "Conditions": {"$id": str(idn + 4), "$type": "x", "$values": [cond]},
                 "Items": {"$id": str(idn + 5), "$type": "x", "$values": blocks}}]}}


def _file(tmp_path, targets):
    seq = {"$id": "1", "$type": f"{NS}.Container.SequenceRootContainer, {NS}",
           "Items": {"$id": "2", "$type": "x", "$values": [
               {"$id": "900", "$type": g.TIME_PROVIDER_TYPE},     # the singleton a $ref points at
               {"$id": "3", "$type": f"{NS}.Container.TargetAreaContainer, {NS}", "Name": "Targets",
                "Items": {"$id": "4", "$type": "x", "$values": targets}}]}}
    p = tmp_path / "full_for_tonight.json"
    p.write_text(json.dumps(seq))
    return p


def _slot(name, start_h, end_h, day=3):
    start = datetime(2026, 11, day, start_h, 0, tzinfo=TZ)
    end = start + timedelta(hours=(end_h - start_h) % 24)
    return Slot(name, start, end, int((end - start).total_seconds() // 3600), 50)


def test_read_back_ends_reads_the_fixed_end_inline_or_by_ref_and_none_for_dawn(tmp_path):
    p = _file(tmp_path, [_target(100, "a", (2, 0), {"L": 4}),
                         _target(200, "b", (5, 30), {"L": 4}, provider="ref"),
                         _target(300, "c", (6, 0), {"L": 4}, provider="dawn")])
    assert g.read_back_ends(p) == [("a", (2, 0)), ("b", (5, 30)), ("c", None)]


def test_the_11_03_file_is_refused_against_the_announced_slot(tmp_path):
    # announced: ngc2276 20:00-02:00; the file: hubblev1 L x8 ending at dawn
    p = _file(tmp_path, [_target(100, "hubblev1", (4, 3), {"L": 8}, provider="dawn")])
    problems = g.announced_vs_written([_slot("ngc2276", 20, 2)], p)
    assert problems == ["slot 1: announced ngc2276, the file has hubblev1"]


def test_a_file_that_says_what_was_announced_passes(tmp_path):
    p = _file(tmp_path, [_target(100, "ngc2276", (2, 0), {"L": 4, "R": 1})])
    assert g.announced_vs_written([_slot("ngc2276", 20, 2)], p) == []


def test_names_compare_without_case_or_spaces(tmp_path):
    p = _file(tmp_path, [_target(100, "NGC 2276", (2, 0), {"L": 4})])
    assert g.announced_vs_written([_slot("ngc2276", 20, 2)], p) == []


def test_count_order_exposures_and_end_are_each_a_problem(tmp_path):
    p = _file(tmp_path, [_target(100, "a", (20, 0), {"L": 4}),
                         _target(200, "b", (23, 0), {}),
                         _target(300, "c", (2, 0), {"L": 4})])
    slots = [_slot("a", 19, 20), _slot("b", 20, 21), _slot("c", 21, 2), _slot("d", 2, 4)]
    problems = g.announced_vs_written(slots, p)
    assert any(p_.startswith("announced 4 slot(s)") for p_ in problems)
    assert "b: no exposure blocks in the file" in problems
    assert "b: announced to end 21:00, the file stops at 23:00" in problems
    # swapped order
    (tmp_path / "x").mkdir()
    swapped = _file(tmp_path / "x", [_target(100, "a", (20, 0), {"L": 4}),
                                     _target(200, "c", (23, 0), {"L": 4})])
    assert g.announced_vs_written([_slot("c", 19, 20), _slot("a", 20, 23)], swapped)[0] \
        == "slot 1: announced c, the file has a"


def test_no_slots_and_an_unreadable_file(tmp_path):
    assert g.announced_vs_written([], tmp_path / "missing.json") == []
    got = g.announced_vs_written([_slot("a", 19, 20)], tmp_path / "missing.json")
    assert len(got) == 1 and got[0].startswith("the written sequence could not be read back")
