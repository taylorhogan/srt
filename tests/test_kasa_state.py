"""The roof and scope verdict logic of the inside Kasa camera, in isolation.

Since 2026-09-07 kasa_state is the only safety eye behind every roof move, so
the pure parts -- what one frame's tags mean, how several frames combine,
where the aperture veto applies -- get enumerated here. The dangerous edge is
a FALSE OPEN with the roof shut: a roof tag that fails to decode is "absent"
exactly like a roof that is not there, so OPEN must need every frame and must
yield to the aperture reading shut.

Runs on the observatory (needs cv2/numpy and the private config to import the
module); skipped on a bare CI runner like the other hardware-adjacent tests.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ks = pytest.importorskip("sentry.kasa_state")
np = pytest.importorskip("numpy")

REF = {"tolerance_px": 200.0,
       "markers": {"1": [[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]]}}
ROOF_AT_REF = np.array(REF["markers"]["1"])
ROOF_ELSEWHERE = ROOF_AT_REF + 500.0
SCOPE = np.array([[900.0, 900.0], [960.0, 900.0], [960.0, 960.0], [900.0, 960.0]])


def test_roof_tag_at_shut_position_is_shut():
    v, d = ks.roof_tag_verdict({1: ROOF_AT_REF + 3.0, 0: SCOPE}, REF)
    assert v == "shut" and d["witness"] is True and d["worst_corner_px"] < 5


def test_roof_tag_elsewhere_is_unknown_not_open():
    v, d = ks.roof_tag_verdict({1: ROOF_ELSEWHERE, 0: SCOPE}, REF)
    assert v == "unknown" and d.get("tag_elsewhere")


def test_roof_tag_gone_with_witness_is_open():
    v, _ = ks.roof_tag_verdict({0: SCOPE}, REF)
    assert v == "open"


def test_no_tags_is_blind_not_open():
    v, _ = ks.roof_tag_verdict({}, REF)
    assert v == "unknown"


def test_no_shut_reference_never_answers():
    for found in ({}, {0: SCOPE}, {1: ROOF_AT_REF, 0: SCOPE}):
        assert ks.roof_tag_verdict(found, None)[0] == "unknown"


def test_aperture_vetoes_open_only():
    assert ks.roof_verdict_with_veto("open", {}, "shut")[0] == "unknown"
    assert ks.roof_verdict_with_veto("open", {}, "open")[0] == "open"
    assert ks.roof_verdict_with_veto("open", {}, "unknown")[0] == "open"
    # a decoded tag at the shut position is positive evidence; no veto
    assert ks.roof_verdict_with_veto("shut", {}, "open")[0] == "shut"
    assert ks.roof_verdict_with_veto("unknown", {}, "open")[0] == "unknown"


def test_open_needs_every_frame():
    o, u, s = ("open", {}), ("unknown", {}), ("shut", {})
    assert ks.combine_roof([o, o, o]) == "open"
    assert ks.combine_roof([o, o, u]) == "unknown"
    assert ks.combine_roof([o, s, o]) == "unknown"
    assert ks.combine_roof([]) == "unknown"


def test_shut_needs_one_decoded_tag_and_no_contradiction():
    s, u, o = ("shut", {}), ("unknown", {}), ("open", {})
    elsewhere = ("unknown", {"tag_elsewhere": True})
    assert ks.combine_roof([u, s, u]) == "shut"
    assert ks.combine_roof([s, s, s]) == "shut"
    assert ks.combine_roof([s, o, u]) == "unknown"
    assert ks.combine_roof([s, elsewhere, s]) == "unknown"


def test_scope_unsafe_anywhere_wins_and_safe_needs_a_positive_read():
    safe, unsafe, unk = ("safe", {}), ("UNSAFE", {}), ("unknown", {})
    assert ks.combine_scope([safe, safe, unsafe]) == "UNSAFE"
    assert ks.combine_scope([safe, unk, unk]) == "safe"
    assert ks.combine_scope([unk, unk, unk]) == "unknown"
    assert ks.combine_scope([]) == "unknown"
