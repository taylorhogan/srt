"""sequence_lint -- what N.I.N.A refuses, caught before the file is written.

Every rule gets a synthetic sequence that breaks it and one that does not.
The last test is the regression: a generator run whose template carries an
extra item inside a SmartExposure (the 2026-09-14 shape) raises instead of
writing. Pure Python; nothing here needs config, numpy or N.I.N.A.
"""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nina_gen import sequence_lint as L  # noqa: E402
from nina_gen import nina_sequence_gen as g  # noqa: E402
from test_nina_slots import _template, _gen, SLOTS, NS  # noqa: E402


def _seq():
    """A generated two-slot sequence from the slot tests' template: what the
    generator writes tonight, minus the script check."""
    return _template()


def _first(node, short):
    for n in L._walk(node):
        if L._short(n.get("$type")) == short:
            return n
    raise KeyError(short)


def _generated(tmp_path):
    seq, _ = _gen(tmp_path, SLOTS)
    return seq


def test_the_generators_output_is_clean(tmp_path):
    seq = _generated(tmp_path)
    assert L.lint(seq, template=_template()) == []


def test_ids_must_be_unique_and_refs_resolve(tmp_path):
    seq = _generated(tmp_path)
    _first(seq, "WaitForTime")["$id"] = _first(seq, "SmartExposure")["$id"]
    assert any(p.startswith("IDS:") and "more than once" in p for p in L.lint(seq))
    seq = _generated(tmp_path)
    _first(seq, "SmartExposure")["Parent"] = {"$ref": "99999"}
    assert any(p.startswith("IDS:") and "does not exist" in p for p in L.lint(seq))


def test_an_item_must_point_at_the_container_it_sits_in(tmp_path):
    seq = _generated(tmp_path)
    wft = _first(seq, "WaitForTime")
    wft["Parent"] = {"$ref": _first(seq, "SequenceRootContainer")["$id"]}
    assert any(p.startswith("PARENT:") for p in L.lint(seq))


def test_smart_exposure_shape_is_exactly_switch_filter_then_take_exposure(tmp_path):
    seq = _generated(tmp_path)
    se = _first(seq, "SmartExposure")
    se["Items"]["$values"].insert(1, {
        "$id": "77777", "$type": f"{NS}.SequenceItem.Focuser.MoveFocuserByTemperature, {NS}",
        "Parent": {"$ref": se["$id"]}})
    probs = L.lint(seq)
    assert any(p.startswith("SMART:") and "MoveFocuserByTemperature" in p for p in probs)
    seq = _generated(tmp_path)
    se = _first(seq, "SmartExposure")
    se["Triggers"]["$values"].clear()
    assert any(p.startswith("SMART:") and "Trigger" in p for p in L.lint(seq))
    seq = _generated(tmp_path)
    se = _first(seq, "SmartExposure")
    se["Conditions"]["$values"].clear()
    assert any(p.startswith("SMART:") and "Condition" in p for p in L.lint(seq))


def test_switch_filter_must_name_a_filter(tmp_path):
    seq = _generated(tmp_path)
    _first(seq, "SwitchFilter")["Filter"] = {"$ref": "no-such-id"}
    probs = L.lint(seq)
    assert any(p.startswith("FILTER:") for p in probs)


def test_center_must_follow_a_switch_filter(tmp_path):
    seq = _generated(tmp_path)
    for n in L._walk(seq):
        vals = L._items(n)
        if any(L._short(v.get("$type")) in ("Center", "CenterAndRotate") for v in vals):
            vals[:] = [v for v in vals if L._short(v.get("$type")) != "SwitchFilter"]
    probs = L.lint(seq)
    assert sum(p.startswith("CENTER:") for p in probs) == 2      # one per slot


def test_target_container_needs_a_target(tmp_path):
    seq = _generated(tmp_path)
    _first(seq, "DeepSkyObjectContainer")["Target"] = {}
    assert any(p.startswith("TARGET:") for p in L.lint(seq))


def test_types_outside_the_template_are_flagged_unless_allowed(tmp_path):
    seq = _generated(tmp_path)
    tpl = _template()
    assert not [p for p in L.lint(seq, template=tpl) if p.startswith("TYPES:")]
    _first(seq, "WaitForTime")["$type"] = "Somebody.Sequencer.NewThing, Somebody"
    probs = L.lint(seq, template=tpl)
    assert any(p.startswith("TYPES:") and "NewThing" in p for p in probs)
    assert not [p for p in L.lint(seq, template=tpl, extra_types=["Somebody.Sequencer.NewThing, Somebody"])
                if p.startswith("TYPES:")]


def test_scripts_must_exist_when_asked(tmp_path):
    seq = _generated(tmp_path)
    probs = L.lint(seq, check_scripts=True)
    assert any(p.startswith("SCRIPT:") and "does not exist" in p for p in probs)
    real = tmp_path / "set_imaging_state.bat"
    real.write_text("@echo off")
    for n in L._walk(seq):
        if L._short(n.get("$type")) == "ExternalScript":
            n["Script"] = '"%s" IN_MAIN' % str(real).replace("\\", "\\\\")
    assert not [p for p in L.lint(seq, check_scripts=True) if p.startswith("SCRIPT:")]
    assert not [p for p in L.lint(seq) if p.startswith("SCRIPT:")]     # off by default


def test_root_must_be_a_sequence_root_container():
    assert L.lint({"$type": "x"}) == ["ROOT: document is not a SequenceRootContainer"]
    assert L.lint([]) and L.lint([])[0].startswith("ROOT:")


def test_lint_file_and_cli(tmp_path, capsys):
    p = tmp_path / "seq.json"
    p.write_text(json.dumps(_generated(tmp_path)))
    assert L.lint_file(p) == []
    assert L.main([str(p)]) == 0
    assert "clean" in capsys.readouterr().out
    bad = _generated(tmp_path)
    _first(bad, "DeepSkyObjectContainer")["Target"] = {}
    p.write_text(json.dumps(bad))
    assert L.main([str(p)]) == 1
    assert "TARGET:" in capsys.readouterr().out


def test_generator_refuses_to_write_the_2026_09_14_shape(tmp_path):
    """A template whose SmartExposure already carries an extra item -- the
    shape that skipped every block that night -- must raise and leave no
    file behind, in both generators."""
    tpl = _template()
    se = _first(tpl, "SmartExposure")
    se["Items"]["$values"].insert(1, {
        "$id": "77777", "$type": f"{NS}.SequenceItem.Focuser.MoveFocuserByTemperature, {NS}",
        "Absolute": False, "Parent": {"$ref": se["$id"]}})
    tp = tmp_path / "tpl.json"
    tp.write_text(json.dumps(tpl))
    out = tmp_path / "out.json"
    with pytest.raises(L.SequenceLintError) as ei:
        g.generate_sequence(tp, "ngc7380", 22.79, 58.1, out, lint_scripts=False)
    assert any(p.startswith("SMART:") for p in ei.value.problems)
    assert not out.exists()
    with pytest.raises(L.SequenceLintError):
        g.generate_slots_sequence(tp, SLOTS, out, state_script="C:\\\\x\\\\s.bat", lint_scripts=False)
    assert not out.exists()
