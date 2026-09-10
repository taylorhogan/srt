"""Filter counts consume hours * efficiency, one pass per window.

An explicit `filters` plan is a ratio over the target's good hours (since
2026-09-10), not a literal count: ngc7380's 36/24/12 at 300 s is six hours,
and with the blocks running in sequence a three-hour window spent all of it
on Ha. Only when the hours are unknown are the counts literal. The
efficiency comes from cfg["nina"]["efficiency"] (default 0.75).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nina_gen import nina_sequence_gen as g  # noqa: E402

NS = "NINA.Sequencer"


def _se(idn, filt, exposure=300.0, iters=30):
    return {"$id": str(idn), "$type": f"{NS}.SequenceItem.Imaging.SmartExposure, {NS}",
            "Conditions": {"$values": [
                {"$type": f"{NS}.Conditions.LoopCondition, {NS}", "Iterations": iters}]},
            "Items": {"$values": [
                {"$type": f"{NS}.SequenceItem.FilterWheel.SwitchFilter, {NS}",
                 "Filter": {"$type": "NINA.Core.Model.Equipment.FilterInfo, NINA.Core",
                            "_name": filt, "_position": 1}},
                {"$type": f"{NS}.SequenceItem.Imaging.TakeExposure, {NS}",
                 "ExposureTime": exposure}]}}


def _blocks():
    # block 0 is the L block; the other three carry inline filters
    return [_se(1, "L"), _se(2, "R"), _se(3, "G"), _se(4, "B")]


def _iters(se):
    return se["Conditions"]["$values"][0]["Iterations"]


def test_explicit_plan_is_a_ratio_over_the_usable_seconds(monkeypatch):
    monkeypatch.setattr(g, "_wheel", lambda: {})
    blocks = _blocks()
    # three good hours at 0.75 = 8100 s = 27 frames of 300 s, split 3:2:1
    applied = g._apply_explicit_plan(blocks[1:], {"Ha": 36, "O-III": 24, "S-II": 12},
                                     usable_seconds=3 * 3600 * 0.75)
    assert applied == {"Ha": 13, "O-III": 9, "S-II": 4}
    assert [_iters(b) for b in blocks[1:]] == [13, 9, 4]
    assert sum(applied.values()) * 300 <= 3 * 3600 * 0.75


def test_explicit_plan_is_literal_without_hours(monkeypatch):
    monkeypatch.setattr(g, "_wheel", lambda: {})
    blocks = _blocks()
    applied = g._apply_explicit_plan(blocks[1:], {"Ha": 36, "O-III": 24, "S-II": 12})
    assert applied == {"Ha": 36, "O-III": 24, "S-II": 12}


def test_every_filter_with_a_share_gets_at_least_one_frame(monkeypatch):
    monkeypatch.setattr(g, "_wheel", lambda: {})
    blocks = _blocks()
    applied = g._apply_explicit_plan(blocks[1:], {"Ha": 40, "O-III": 1}, usable_seconds=2000)
    assert applied["O-III"] == 1 and applied["Ha"] >= 1


def test_zero_in_the_plan_stays_zero(monkeypatch):
    monkeypatch.setattr(g, "_wheel", lambda: {})
    blocks = _blocks()
    applied = g._apply_explicit_plan(blocks[1:], {"Ha": 10, "O-III": 0}, usable_seconds=6000)
    assert applied == {"Ha": 20, "O-III": 0}


def test_efficiency_comes_from_config_and_is_bounded(monkeypatch):
    import configs.config as config
    monkeypatch.setattr(config, "data", lambda: {"nina": {"efficiency": 0.5}})
    assert g._efficiency() == 0.5
    monkeypatch.setattr(config, "data", lambda: {"nina": {}})
    assert g._efficiency() == g.DEFAULT_EFFICIENCY
    monkeypatch.setattr(config, "data", lambda: {"nina": {"efficiency": 3}})
    assert g._efficiency() == g.DEFAULT_EFFICIENCY


def test_automatic_nebula_split_uses_the_efficiency(monkeypatch):
    monkeypatch.setattr(g, "_wheel", lambda: {})
    monkeypatch.setattr(g, "_efficiency", lambda: 0.75)
    blocks = _blocks()
    seq = {"Items": {"$values": blocks}}
    plan = g._apply_filter_plan(seq, 3 * 3600, "nebula")
    # 8100 s / 3 = 2700 s each = 9 frames of 300 s
    assert plan == {"Ha": 9, "O-III": 9, "S-II": 9}
    assert _iters(blocks[0]) == 0
