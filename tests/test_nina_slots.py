"""Cloning the template's target container once per slot (nina_sequence_gen).

A synthetic template with the same shape N.I.N.A writes -- Newtonsoft $id /
$ref reference tracking, a TargetAreaContainer holding one
DeepSkyObjectContainer, WaitForTime / TimeCondition nodes, an ExternalScript
prototype -- because the real one lives outside the repo. What must hold:
every $id unique, every $ref resolvable, each clone patched for its own
target and window, and the hand-over scripts on the second slot only.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nina_gen import nina_sequence_gen as g  # noqa: E402

NS = "NINA.Sequencer"


def _se(idn, filt_id, iters):
    return {"$id": str(idn), "$type": f"{NS}.SequenceItem.Imaging.SmartExposure, {NS}",
            "Conditions": {"$id": str(idn + 1), "$type": "x", "$values": [
                {"$id": str(idn + 2), "$type": f"{NS}.Conditions.LoopCondition, {NS}",
                 "Iterations": iters, "CompletedIterations": 0}]},
            "Items": {"$id": str(idn + 3), "$type": "x", "$values": [
                {"$id": str(idn + 4), "$type": f"{NS}.SequenceItem.FilterWheel.SwitchFilter, {NS}",
                 "Filter": {"$id": str(idn + 5), "$type": "NINA.Core.Model.Equipment.FilterInfo, NINA.Core",
                            "_name": filt_id, "_position": 1}},
                {"$id": str(idn + 6), "$type": f"{NS}.SequenceItem.Imaging.TakeExposure, {NS}",
                 "ExposureTime": 300.0}]},
            "Parent": {"$ref": "20"}}


def _template():
    dso = {"$id": "20", "$type": f"{NS}.Container.DeepSkyObjectContainer, {NS}",
           "Name": "M 94",
           "Target": {"$id": "21", "$type": "x", "TargetName": "M 94",
                      "InputCoordinates": {"$id": "22", "$type": "NINA.Astrometry.InputCoordinates, NINA.Astrometry",
                                           "RAHours": 12, "RAMinutes": 50, "RASeconds": 0,
                                           "NegativeDec": False, "DecDegrees": 41, "DecMinutes": 0, "DecSeconds": 0}},
           "Conditions": {"$id": "23", "$type": "x", "$values": [
               {"$id": "24", "$type": f"{NS}.Conditions.TimeCondition, {NS}", "Hours": 4, "Minutes": 3, "Seconds": 57, "MinutesOffset": 0}]},
           "Items": {"$id": "25", "$type": "x", "$values": [
               {"$id": "26", "$type": f"{NS}.Container.SequentialContainer, {NS}", "Name": "Wait",
                "Items": {"$id": "27", "$type": "x", "$values": [
                    {"$id": "28", "$type": f"{NS}.SequenceItem.Utility.WaitForTime, {NS}",
                     "Hours": 21, "Minutes": 13, "Seconds": 0, "MinutesOffset": -10,
                     "SelectedProvider": {"$id": "60", "$type": f"{NS}.Utility.DateTimeProvider.NauticalDuskProvider, {NS}"},
                     "Parent": {"$ref": "26"}}]},
                "Parent": {"$ref": "20"}},
               {"$id": "29", "$type": f"{NS}.Container.SequentialContainer, {NS}", "Name": "IMAGING",
                "Items": {"$id": "30", "$type": "x", "$values": [_se(31, "Ha", 20), _se(40, "O-III", 30)]},
                "Parent": {"$ref": "20"}}]},
           "Parent": {"$ref": "10"}}
    end = {"$id": "50", "$type": f"{NS}.Container.SequentialContainer, {NS}", "Name": "end",
           "Items": {"$id": "51", "$type": "x", "$values": [
               {"$id": "52", "$type": "WhenPlugin.When.ExternalScript, WhenPlugin",
                "Script": '"C:\\\\x\\\\smessage.bat" "done"', "Parent": {"$ref": "50"}}]},
           "Parent": {"$ref": "10"}}
    return {"$id": "1", "$type": f"{NS}.Container.SequenceRootContainer, {NS}",
            "Items": {"$id": "2", "$type": "x", "$values": [
                {"$id": "10", "$type": f"{NS}.Container.TargetAreaContainer, {NS}", "Name": "Targets",
                 "Items": {"$id": "11", "$type": "x", "$values": [dso, end]}, "Parent": {"$ref": "1"}}]}}


def _ids_refs(node, ids, refs):
    if isinstance(node, dict):
        if "$id" in node:
            ids.append(node["$id"])
        if "$ref" in node:
            refs.append(node["$ref"])
        for v in node.values():
            _ids_refs(v, ids, refs)
    elif isinstance(node, list):
        for v in node:
            _ids_refs(v, ids, refs)


def _gen(tmp_path, slots, script="C:\\\\srt\\\\set_imaging_state.bat"):
    tpl = tmp_path / "tpl.json"
    tpl.write_text(json.dumps(_template()))
    out = tmp_path / "out.json"
    plans = g.generate_slots_sequence(tpl, slots, out, state_script=script)
    return json.loads(out.read_text()), plans


SLOTS = [
    {"name": "squid", "ra_hours": 21.196, "dec_degrees": 59.95, "seconds": None,
     "start": datetime(2026, 9, 7, 21, 0), "end": datetime(2026, 9, 8, 2, 15)},
    {"name": "ngc7380", "ra_hours": 22.79, "dec_degrees": 58.1, "seconds": None,
     "start": datetime(2026, 9, 8, 2, 15), "end": datetime(2026, 9, 8, 4, 40)},
]


def test_two_slots_give_two_target_containers_with_sound_references(tmp_path):
    seq, plans = _gen(tmp_path, SLOTS)
    ids, refs = [], []
    _ids_refs(seq, ids, refs)
    assert len(ids) == len(set(ids)), "duplicate $id would alias two objects in N.I.N.A"
    assert set(refs) <= set(ids), sorted(set(refs) - set(ids))
    area = g._find_target_area(seq)
    dsos = [it for it in g._items_of(area) if g._short_type(it) == "DeepSkyObjectContainer"]
    assert [d["Name"] for d in dsos] == ["squid", "ngc7380"]
    assert [d["Target"]["TargetName"] for d in dsos] == ["squid", "ngc7380"]
    # the end container is still there, after both targets
    assert g._short_type(g._items_of(area)[-1]) == "SequentialContainer"
    assert len(plans) == 2


def test_only_non_last_slots_get_a_hard_end_and_starts_are_untouched(tmp_path):
    seq, _ = _gen(tmp_path, SLOTS)
    dsos = [it for it in g._items_of(g._find_target_area(seq))
            if g._short_type(it) == "DeepSkyObjectContainer"]
    first, last = dsos
    # first slot: hard end at its boundary through the fixed Time provider
    tc = g._find_first(first, "TimeCondition")
    assert (tc["Hours"], tc["Minutes"]) == (SLOTS[0]["end"].hour, SLOTS[0]["end"].minute)
    assert "TimeProvider" in tc["SelectedProvider"]["$type"]
    # last slot: the template's own end condition, untouched (dawn)
    tc2 = g._find_first(last, "TimeCondition")
    assert (tc2["Hours"], tc2["Minutes"]) == (4, 3)
    assert "TimeProvider" not in tc2.get("SelectedProvider", {}).get("$type", "")
    # WaitForTime keeps the template's provider and offset in both slots
    for d in dsos:
        wt = g._find_first(d, "WaitForTime")
        assert wt["MinutesOffset"] == -10
    for d, slot in zip(dsos, SLOTS):
        ic = g._find_first(d, "InputCoordinates")
        assert ic["RAHours"] == int(slot["ra_hours"])
        assert ic["DecDegrees"] == int(slot["dec_degrees"])


def test_hand_over_scripts_bracket_the_second_slots_setup_only(tmp_path):
    seq, _ = _gen(tmp_path, SLOTS)
    dsos = [it for it in g._items_of(g._find_target_area(seq))
            if g._short_type(it) == "DeepSkyObjectContainer"]
    first_setup = g._items_of(g._items_of(dsos[0])[0])
    second_setup = g._items_of(g._items_of(dsos[1])[0])
    assert not [i for i in first_setup if g._short_type(i) == "ExternalScript"]
    scripts = [i["Script"] for i in second_setup if g._short_type(i) == "ExternalScript"]
    assert scripts[0].endswith("DONE_MAIN") and scripts[-1].endswith("IN_MAIN")
    assert g._short_type(second_setup[0]) == "ExternalScript"
    assert g._short_type(second_setup[-1]) == "ExternalScript"
    # the injected items point at their own container as parent
    pid = g._items_of(dsos[1])[0]["$id"]
    assert all(i["Parent"]["$ref"] == pid for i in second_setup if g._short_type(i) == "ExternalScript")


def test_one_slot_is_the_plain_template_shape(tmp_path):
    seq, plans = _gen(tmp_path, SLOTS[:1])
    dsos = [it for it in g._items_of(g._find_target_area(seq))
            if g._short_type(it) == "DeepSkyObjectContainer"]
    assert len(dsos) == 1 and dsos[0]["Name"] == "squid" and len(plans) == 1
    assert not [i for i in g._items_of(g._items_of(dsos[0])[0]) if g._short_type(i) == "ExternalScript"]


def test_singleton_providers_are_referenced_not_duplicated(tmp_path):
    """N.I.N.A hands back ONE instance of each date-time provider; a second
    $id for it fails the load ("A different Id has already been assigned",
    2026-09-08). The clone must point back at the original."""
    seq, _ = _gen(tmp_path, SLOTS)
    providers, refs_to_60 = [], 0
    def walk(n):
        nonlocal refs_to_60
        if isinstance(n, dict):
            if "$id" in n and "DateTimeProvider" in n.get("$type", ""):
                providers.append(n["$id"])
            if n.get("$ref") == "60":
                refs_to_60 += 1
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)
    walk(seq)
    # one NauticalDuskProvider definition (the template's), the clone refers to it;
    # plus exactly one TimeProvider definition for the first slot's hard end
    assert providers.count("60") == 1 and len(providers) == 2
    assert refs_to_60 == 1


def test_no_reference_points_forward(tmp_path):
    """Newtonsoft resolves $ref against objects already read."""
    seq, _ = _gen(tmp_path, SLOTS)
    seen = set()
    def walk(n):
        if isinstance(n, dict):
            if "$id" in n:
                seen.add(n["$id"])
            if "$ref" in n:
                assert n["$ref"] in seen, "forward reference " + n["$ref"]
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)
    walk(seq)
