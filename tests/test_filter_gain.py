"""Per-filter camera gain (2026-10-24): every exposure block is written at its
filter's gain from cfg["nina"]["gain"], every other block at the profile's
-1/-1, and a block whose gain disagrees with the map is refused.

N.I.N.A keeps gain on the exposure instruction, not the filter, and the
generator renames blocks per plan, so the gain has to be rewritten on every
block every time: a 26 left on a block renamed from Ha to L would shoot L at
the narrowband gain."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nina_gen import nina_sequence_gen as g      # noqa: E402
from nina_gen.sequence_lint import SequenceLintError  # noqa: E402

NS = "NINA.Sequencer"
GAIN = {"Ha": (26, 30), "O-III": (26, 30)}
REAL_GAIN_MAP = g._gain_map          # captured before the fixture replaces it


def _se(idn, name, pos, iters=10, gain=-1, offset=-1, ref=None):
    filt = ({"$ref": str(ref)} if ref else
            {"$id": str(idn + 5), "$type": "NINA.Core.Model.Equipment.FilterInfo, NINA.Core",
             "_name": name, "_position": pos})
    return {"$id": str(idn), "$type": f"{NS}.SequenceItem.Imaging.SmartExposure, {NS}",
            "Conditions": {"$id": str(idn + 1), "$type": "x", "$values": [
                {"$id": str(idn + 2), "$type": f"{NS}.Conditions.LoopCondition, {NS}",
                 "Iterations": iters, "CompletedIterations": 0}]},
            "Items": {"$id": str(idn + 3), "$type": "x", "$values": [
                {"$id": str(idn + 4), "$type": f"{NS}.SequenceItem.FilterWheel.SwitchFilter, {NS}",
                 "Filter": filt},
                {"$id": str(idn + 6), "$type": f"{NS}.SequenceItem.Imaging.TakeExposure, {NS}",
                 "ExposureTime": 300.0, "Gain": gain, "Offset": offset}]}}


def _container(*blocks):
    return {"$id": "1", "$type": f"{NS}.Container.SequentialContainer, {NS}", "Name": "IMAGING",
            "Items": {"$id": "2", "$type": "x", "$values": list(blocks)}}


def _gains(container):
    return [(g._block_filter_name(se), g._block_gain(se)) for se in g._collect_smart_exposures(container)]


@pytest.fixture(autouse=True)
def _cfg(monkeypatch):
    monkeypatch.setattr(g, "_wheel", lambda: dict(g.DEFAULT_WHEEL))
    monkeypatch.setattr(g, "_gain_map", lambda: dict(GAIN))


def test_mapped_filters_get_their_gain_and_the_rest_get_the_profile():
    c = _container(_se(10, "L", 0), _se(20, "Ha", 6), _se(30, "O-III", 5), _se(40, "R", 1))
    written = g.apply_filter_gain(c)
    assert _gains(c) == [("L", (-1, -1)), ("Ha", (26, 30)), ("O-III", (26, 30)), ("R", (-1, -1))]
    assert written == {"L": (-1, -1), "Ha": (26, 30), "O-III": (26, 30), "R": (-1, -1)}


def test_a_stale_gain_does_not_survive_a_rename():
    # the template (or last night's plan) left 26 on a block that is L tonight
    c = _container(_se(10, "L", 0, gain=26, offset=30), _se(20, "Ha", 6))
    g.apply_filter_gain(c)
    assert _gains(c) == [("L", (-1, -1)), ("Ha", (26, 30))]


def test_a_ref_block_resolves_its_filter_through_the_id_index():
    start = {"$id": "100", "$type": "NINA.Core.Model.Equipment.FilterInfo, NINA.Core",
             "_name": "Ha", "_position": 6}
    seq = {"$id": "0", "Start": start, "Main": _container(_se(10, None, None, ref=100), _se(20, "L", 0))}
    g.apply_filter_gain(seq["Main"], ids=g._index_ids(seq))
    assert _gains(seq["Main"])[0] == (None, (26, 30)) or g._block_gain(g._collect_smart_exposures(seq["Main"])[0]) == (26, 30)
    assert g._block_gain(g._collect_smart_exposures(seq["Main"])[1]) == (-1, -1)


def test_an_unresolvable_block_keeps_the_profile_gain():
    c = _container(_se(10, None, None, ref=999))
    g.apply_filter_gain(c, ids={})
    assert g._block_gain(g._collect_smart_exposures(c)[0]) == (-1, -1)


def test_an_empty_map_writes_the_profile_everywhere(monkeypatch):
    monkeypatch.setattr(g, "_gain_map", lambda: {})
    c = _container(_se(10, "Ha", 6, gain=26, offset=30))
    g.apply_filter_gain(c)
    assert _gains(c) == [("Ha", (-1, -1))]


def test_verify_refuses_a_block_at_the_wrong_gain():
    c = _container(_se(10, "Ha", 6, iters=5), _se(20, "L", 0, iters=5))
    with pytest.raises(SequenceLintError) as e:
        g._verify_written(c, {"Ha": 5, "L": 5}, "m76")
    assert "gain" in str(e.value)
    g.apply_filter_gain(c)
    g._verify_written(c, {"Ha": 5, "L": 5}, "m76")       # written as the map says: accepted


def test_gain_map_reads_pairs_and_ignores_junk(monkeypatch):
    from configs import config
    monkeypatch.setattr(config, "data", lambda: {"nina": {"gain": {"Ha": [26, 30], "O-III": 26, "B": "x"}}})
    assert REAL_GAIN_MAP() == {"Ha": (26, 30), "O-III": (26, -1)}
