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
            "Triggers": {"$id": str(idn + 7), "$type": "x", "$values": [
                {"$id": str(idn + 8), "$type": f"{NS}.Trigger.Guider.DitherAfterExposures, {NS}",
                 "AfterExposures": 1}]},
            "Parent": {"$ref": "29"}}          # the IMAGING container


def _template():
    dso = {"$id": "20", "$type": f"{NS}.Container.DeepSkyObjectContainer, {NS}",
           "Name": "M 94",
           "Target": {"$id": "21", "$type": "NINA.Astrometry.InputTarget, NINA.Astrometry",
                      "TargetName": "M 94", "PositionAngle": 0.0,
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
                     "Parent": {"$ref": "26"}},
                    {"$id": "62", "$type": f"{NS}.SequenceItem.Platesolving.Center, {NS}",
                     "Inherited": True, "Parent": {"$ref": "26"}},
                    {"$id": "61", "$type": f"{NS}.SequenceItem.Autofocus.RunAutofocus, {NS}",
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
    plans = g.generate_slots_sequence(tpl, slots, out, state_script=script, lint_scripts=False)
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


def test_every_slot_gets_a_hard_end_and_starts_are_untouched(tmp_path):
    """The last slot ends at its window end too (2026-09-10). The window end
    is the planner's: earliest of horizon, dawn and weather. Before this the
    last container kept the template's dawn and looped past the plan."""
    seq, _ = _gen(tmp_path, SLOTS)
    dsos = [it for it in g._items_of(g._find_target_area(seq))
            if g._short_type(it) == "DeepSkyObjectContainer"]
    prov_ids = set()
    for d, slot in zip(dsos, SLOTS):
        tc = g._find_first(d, "TimeCondition")
        assert (tc["Hours"], tc["Minutes"]) == (slot["end"].hour, slot["end"].minute)
        prov = tc["SelectedProvider"]
        # the fixed Time provider: defined once, the second slot $refs it
        if "$type" in prov:
            assert "TimeProvider" in prov["$type"]
            prov_ids.add(prov["$id"])
        else:
            assert prov["$ref"] in prov_ids
    # WaitForTime keeps the template's provider and offset in both slots
    for d in dsos:
        wt = g._find_first(d, "WaitForTime")
        assert wt["MinutesOffset"] == -10
    for d, slot in zip(dsos, SLOTS):
        ic = g._find_first(d, "InputCoordinates")
        assert ic["RAHours"] == int(slot["ra_hours"])
        assert ic["DecDegrees"] == int(slot["dec_degrees"])


def test_a_slot_without_an_end_keeps_the_templates_dawn(tmp_path):
    slots = [dict(SLOTS[0], end=None)]
    seq, _ = _gen(tmp_path, slots)
    d = g._find_first(g._find_target_area(seq), "DeepSkyObjectContainer")
    tc = g._find_first(d, "TimeCondition")
    assert (tc["Hours"], tc["Minutes"]) == (4, 3)
    assert "TimeProvider" not in tc.get("SelectedProvider", {}).get("$type", "")


def test_single_target_generate_sequence_takes_an_end(tmp_path):
    """The one-target path (scheduler single-slot night, webchat `sequence`)
    pins the same hard end; without end= it is the template's dawn."""
    tpl = tmp_path / "tpl.json"
    tpl.write_text(json.dumps(_template()))
    out = tmp_path / "out.json"
    end = datetime(2026, 9, 11, 1, 0)
    g.generate_sequence(tpl, "ngc7380", 22.79, 58.1, out, end=end, lint_scripts=False)
    seq = json.loads(out.read_text())
    d = g._find_first(g._find_target_area(seq), "DeepSkyObjectContainer")
    tc = g._find_first(d, "TimeCondition")
    assert (tc["Hours"], tc["Minutes"]) == (1, 0)
    assert "TimeProvider" in tc["SelectedProvider"]["$type"]
    ids, refs = [], []
    _ids_refs(seq, ids, refs)
    assert len(ids) == len(set(ids)) and set(refs) <= set(ids)
    g.generate_sequence(tpl, "ngc7380", 22.79, 58.1, out, lint_scripts=False)
    seq = json.loads(out.read_text())
    tc = g._find_first(g._find_first(g._find_target_area(seq), "DeepSkyObjectContainer"), "TimeCondition")
    assert (tc["Hours"], tc["Minutes"]) == (4, 3)


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
    # plus exactly one TimeProvider definition shared by both slots' hard ends
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


def test_explicit_autofocus_only_in_the_first_slot(tmp_path):
    seq, _ = _gen(tmp_path, SLOTS)
    dsos = [it for it in g._items_of(g._find_target_area(seq))
            if g._short_type(it) == "DeepSkyObjectContainer"]
    def types(d):
        return [g._short_type(i) for i in g._items_of(g._items_of(d)[0])]
    assert "RunAutofocus" in types(dsos[0])
    assert "RunAutofocus" not in types(dsos[1])
    # the hand-over scripts still bracket the second slot's setup
    assert types(dsos[1])[0] == "ExternalScript" and types(dsos[1])[-1] == "ExternalScript"


def test_rotation_turns_center_into_center_and_rotate_for_that_slot_only(tmp_path):
    slots = [dict(SLOTS[0]), dict(SLOTS[1], rotation=90)]
    seq, _ = _gen(tmp_path, slots)
    dsos = [it for it in g._items_of(g._find_target_area(seq))
            if g._short_type(it) == "DeepSkyObjectContainer"]
    first, second = dsos
    assert g._find_first(first, "Center") is not None
    assert g._find_first(first, "CenterAndRotate") is None
    assert first["Target"]["PositionAngle"] == 0
    car = g._find_first(second, "CenterAndRotate")
    assert car is not None and car["PositionAngle"] == 90 and car["Inherited"] is True
    assert g._find_first(second, "Center") is None
    assert second["Target"]["PositionAngle"] == 90

def _setup_types(container):
    setup = next(it for it in container["Items"]["$values"]
                 if g._short_type(it) == "SequentialContainer")
    return setup, [g._short_type(it) for it in setup["Items"]["$values"]]


def test_every_slot_switches_to_l_before_centering(tmp_path):
    """2026-09-15 03:00: the second slot centred through the H-alpha filter the
    previous block left in the wheel and ASTAP failed every solve. Every
    slot's Center now follows a SwitchFilter L, parented to its setup
    container, with ids of its own."""
    seq, _ = _gen(tmp_path, SLOTS)
    targets = [it for it in seq["Items"]["$values"][0]["Items"]["$values"]
               if g._short_type(it) == "DeepSkyObjectContainer"]
    assert len(targets) == 2
    for t in targets:
        setup, types = _setup_types(t)
        i = next(i for i, ty in enumerate(types) if ty in ("Center", "CenterAndRotate"))
        assert types[i - 1] == "SwitchFilter"
        sw = setup["Items"]["$values"][i - 1]
        assert sw["Filter"]["_name"] == "L"
        assert sw["Parent"] == {"$ref": setup["$id"]}
    ids, refs = [], []
    _ids_refs(seq, ids, refs)
    assert len(ids) == len(set(ids))
    assert set(refs) <= set(ids)


def test_center_filter_copies_the_templates_own_l_and_is_idempotent(tmp_path):
    tpl = _template()
    # a prelude-style SwitchFilter L with the wheel's real settings
    tpl["Items"]["$values"].insert(0, {
        "$id": "90", "$type": f"{NS}.Container.StartAreaContainer, {NS}",
        "Items": {"$id": "91", "$type": "x", "$values": [
            {"$id": "92", "$type": f"{NS}.SequenceItem.FilterWheel.SwitchFilter, {NS}",
             "Filter": {"$id": "93", "$type": "NINA.Core.Model.Equipment.FilterInfo, NINA.Core",
                        "_name": "L", "_position": 0, "_autoFocusExposureTime": 12.0,
                        "_autoFocusBinning": {"$id": "94", "$type": "NINA.Core.Model.Equipment.BinningMode, NINA.Core",
                                              "X": 2, "Y": 2}},
             "Parent": {"$ref": "90"}}]},
        "Parent": {"$ref": "1"}})
    next_id = [g._max_id(tpl) + 1]
    assert g.ensure_center_filter(tpl, tpl, next_id) == 1
    assert g.ensure_center_filter(tpl, tpl, next_id) == 0          # already there
    dso = next(it for it in tpl["Items"]["$values"][1]["Items"]["$values"]
               if g._short_type(it) == "DeepSkyObjectContainer")
    setup, types = _setup_types(dso)
    sw = setup["Items"]["$values"][types.index("Center") - 1]
    f = sw["Filter"]
    assert f["_name"] == "L" and f["_position"] == 0
    assert f["_autoFocusExposureTime"] == 12.0                       # the wheel's own settings
    assert f["$id"] != "93" and f["_autoFocusBinning"]["$id"] != "94"   # fresh ids
    ids, refs = [], []
    _ids_refs(tpl, ids, refs)
    assert len(ids) == len(set(ids))


def test_single_target_generate_sequence_switches_to_l_before_centering(tmp_path):
    tpl = tmp_path / "tpl.json"
    tpl.write_text(json.dumps(_template()))
    out = tmp_path / "out.json"
    g.generate_sequence(tpl, "ngc7380", 22.79, 58.1, out, lint_scripts=False)
    seq = json.loads(out.read_text())
    dso = next(it for it in seq["Items"]["$values"][0]["Items"]["$values"]
               if g._short_type(it) == "DeepSkyObjectContainer")
    _, types = _setup_types(dso)
    i = next(i for i, ty in enumerate(types) if ty in ("Center", "CenterAndRotate"))
    assert types[i - 1] == "SwitchFilter"
