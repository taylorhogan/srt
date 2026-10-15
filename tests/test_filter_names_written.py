"""The filters written into the sequence are the filters announced (2026-10-14).

cdk_full_sequence_4block.json (2026-08-31) names its four exposure blocks
Ha / O-III / O-III / O-III. The default galaxy split set only the counts, so
ngc6946 and ngc1961 -- the first targets in weeks without a `filters` plan --
were announced as LRGB and shot as Ha and O-III. Every block is now named,
and _verify_written refuses a target whose blocks differ from its plan.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nina_gen import nina_sequence_gen as g      # noqa: E402
from nina_gen.sequence_lint import SequenceLintError  # noqa: E402

NS = "NINA.Sequencer"
WHEEL = dict(g.DEFAULT_WHEEL)


def _se(idn, name, pos, iters=10):
    return {"$id": str(idn), "$type": f"{NS}.SequenceItem.Imaging.SmartExposure, {NS}",
            "Conditions": {"$id": str(idn + 1), "$type": "x", "$values": [
                {"$id": str(idn + 2), "$type": f"{NS}.Conditions.LoopCondition, {NS}",
                 "Iterations": iters, "CompletedIterations": 0}]},
            "Items": {"$id": str(idn + 3), "$type": "x", "$values": [
                {"$id": str(idn + 4), "$type": f"{NS}.SequenceItem.FilterWheel.SwitchFilter, {NS}",
                 "Filter": {"$id": str(idn + 5), "$type": "NINA.Core.Model.Equipment.FilterInfo, NINA.Core",
                            "_name": name, "_position": pos}},
                {"$id": str(idn + 6), "$type": f"{NS}.SequenceItem.Imaging.TakeExposure, {NS}",
                 "ExposureTime": 300.0}]}}


def _tonights_template_blocks():
    """The four blocks exactly as the 2026-08-31 template carries them."""
    return {"$id": "1", "$type": f"{NS}.Container.SequentialContainer, {NS}", "Name": "LRGB_IMAGING",
            "Items": {"$id": "2", "$type": "x", "$values": [
                _se(10, "Ha", 6), _se(20, "O-III", 5), _se(30, "O-III", 5), _se(40, "O-III", 5)]}}


def _positions(container):
    return [g._filter_node(se)["_position"] for se in g._collect_smart_exposures(container)]


@pytest.fixture(autouse=True)
def _wheel(monkeypatch):
    monkeypatch.setattr(g, "_wheel", lambda: dict(WHEEL))


def test_the_galaxy_default_names_every_block_lrgb():
    c = _tonights_template_blocks()
    plan = g._apply_filter_plan(c, 4 * 3600.0, "galaxy")
    assert set(plan) == {"L", "R", "G", "B"}
    assert g.written_plan(c) == {f: n for f, n in plan.items() if n > 0}
    assert _positions(c) == [0, 1, 2, 3]


def test_plan_for_a_target_without_a_plan_writes_what_it_announces(monkeypatch):
    instructions = pytest.importorskip("control.instructions")    # needs astropy
    monkeypatch.setattr(instructions, "get_filter_plan", lambda name: None)
    monkeypatch.setattr(g, "_classify_object_type", lambda name: "galaxy")
    monkeypatch.setattr(g, "_need_plan", lambda *a, **k: None)
    c = _tonights_template_blocks()
    plan = g._plan_for(c, "ngc6946", 4 * 3600.0)
    assert g.written_plan(c) == {f: n for f, n in plan.items() if n > 0}
    assert "Ha" not in g.written_plan(c)


def test_the_nebula_split_is_named_too():
    c = _tonights_template_blocks()
    plan = g._apply_filter_plan(c, 3 * 3600.0, "nebula")
    assert g.written_plan(c) == {f: n for f, n in plan.items() if n > 0}
    assert _positions(c)[1:] == [6, 5, 4]


def test_a_mismatch_is_refused():
    c = _tonights_template_blocks()
    with pytest.raises(SequenceLintError, match="plan is"):
        g._verify_written(c, {"L": 10, "R": 10, "G": 10, "B": 10}, "ngc6946")


def test_a_wrong_wheel_position_is_refused():
    c = {"$id": "1", "$type": "x", "Items": {"$id": "2", "$type": "x", "$values": [_se(10, "L", 6)]}}
    with pytest.raises(SequenceLintError, match="wheel position"):
        g._verify_written(c, {"L": 10}, "ngc6946")


def test_no_plan_is_not_checked():
    g._verify_written(_tonights_template_blocks(), {}, "x")      # the template's own blocks


# ------------------------------------------------------------ one classifier (2026-10-15)

import sys      # noqa: E402
import types    # noqa: E402


def _grid(answers):
    return types.SimpleNamespace(get_dso_type=lambda name: answers.get(name, "?"))


def test_the_generator_follows_the_grids_classifier(monkeypatch):
    """ic405, ngc2237 and m20 are N in the `tonight` table (its override list)
    but were 'unknown' -> LRGB here."""
    monkeypatch.setattr(g, "_queued_object_type", lambda name: None)
    monkeypatch.setitem(sys.modules, "iris_astronomy.astro_dso_visibility",
                        _grid({"ic405": "N", "ngc6946": "G", "m3": "C"}))
    assert g._classify_object_type("ic405") == "nebula"
    assert g._classify_object_type("ngc6946") == "galaxy"
    assert g._classify_object_type("m3") == "unknown"            # clusters stay broadband


def test_the_queued_type_still_wins(monkeypatch):
    monkeypatch.setattr(g, "_queued_object_type", lambda name: "nebula")
    monkeypatch.setitem(sys.modules, "iris_astronomy.astro_dso_visibility", _grid({"x": "G"}))
    assert g._classify_object_type("x") == "nebula"


def test_an_unclassified_name_without_simbad_is_unknown(monkeypatch):
    monkeypatch.setattr(g, "_queued_object_type", lambda name: None)
    monkeypatch.setitem(sys.modules, "iris_astronomy.astro_dso_visibility", _grid({}))
    monkeypatch.setitem(sys.modules, "astroquery.simbad", None)   # import fails
    assert g._classify_object_type("gravwav") == "unknown"
