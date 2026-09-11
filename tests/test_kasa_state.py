"""The roof and scope verdict logic of the inside Kasa camera, in isolation.

Since 2026-09-07 kasa_state is the only safety eye behind every roof move, so
the pure parts -- what one frame's tags mean, how several frames combine,
where the aperture veto applies, what the open star adds -- get enumerated
here. The dangerous edge is a FALSE OPEN with the roof shut: a roof tag that
fails to decode is "absent" exactly like a roof that is not there, so since
2026-09-11 OPEN also needs the gold star on the wall SEEN (positive evidence,
like SHUT's decoded tag), in at least one frame, with every frame consistent
with open and none vetoed by the aperture.

Runs on the observatory (needs cv2/numpy and the private config to import the
module); skipped on a bare CI runner like the other hardware-adjacent tests.
The frame-replay tests at the end need the archived roof-move frames and the
recorded star reference, both gitignored, and skip without them.
"""
import glob
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ks = pytest.importorskip("sentry.kasa_state")
np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

REF = {"tolerance_px": 200.0,
       "markers": {"1": [[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]]}}
ROOF_AT_REF = np.array(REF["markers"]["1"])
ROOF_ELSEWHERE = ROOF_AT_REF + 500.0
SCOPE = np.array([[900.0, 900.0], [960.0, 900.0], [960.0, 960.0], [900.0, 960.0]])
STAR_SEEN = ("seen", {"star_px": 3.0, "star_ncc": 0.9})
STAR_ABSENT = ("absent", {"why": "no star-sized yellow blob"})
STAR_IR = ("unknown", {"why": "camera in greyscale IR", "ir": True})


def test_roof_tag_at_shut_position_is_shut():
    v, d = ks.roof_tag_verdict({1: ROOF_AT_REF + 3.0, 0: SCOPE}, REF, STAR_ABSENT)
    assert v == "shut" and d["witness"] is True and d["worst_corner_px"] < 5


def test_roof_tag_elsewhere_is_unknown_not_open():
    v, d = ks.roof_tag_verdict({1: ROOF_ELSEWHERE, 0: SCOPE}, REF, STAR_SEEN)
    assert v == "unknown" and d.get("tag_elsewhere")


def test_roof_tag_gone_with_witness_and_star_is_open():
    v, d = ks.roof_tag_verdict({0: SCOPE}, REF, STAR_SEEN)
    assert v == "open" and d["star"] == "seen" and d["star_px"] == 3.0


def test_roof_tag_gone_without_the_star_is_not_open():
    # The 2026-09-10 rule ("tag gone + witness = open") is gone: absence alone
    # is what a decode failure looks like. Open-like, but not open.
    for star in (STAR_ABSENT, STAR_IR, None):
        v, d = ks.roof_tag_verdict({0: SCOPE}, REF, star)
        assert v == "unknown" and d.get("open_no_star") is True


def test_no_tags_is_blind_not_open():
    v, d = ks.roof_tag_verdict({}, REF, STAR_SEEN)
    assert v == "unknown" and not d.get("open_no_star")


def test_no_shut_reference_never_answers():
    for found in ({}, {0: SCOPE}, {1: ROOF_AT_REF, 0: SCOPE}):
        assert ks.roof_tag_verdict(found, None, STAR_SEEN)[0] == "unknown"


def test_aperture_vetoes_open_only():
    assert ks.roof_verdict_with_veto("open", {}, "shut")[0] == "unknown"
    assert ks.roof_verdict_with_veto("open", {}, "open")[0] == "open"
    assert ks.roof_verdict_with_veto("open", {}, "unknown")[0] == "open"
    # a decoded tag at the shut position is positive evidence; no veto
    assert ks.roof_verdict_with_veto("shut", {}, "open")[0] == "shut"
    assert ks.roof_verdict_with_veto("unknown", {}, "open")[0] == "unknown"


def test_aperture_veto_drops_the_open_like_flag():
    # A vetoed frame contradicts open outright; it must not count as
    # "consistent with open" when the frames combine.
    v, d = ks.roof_verdict_with_veto("open", {"star": "seen"}, "shut")
    assert v == "unknown" and not d.get("open_no_star")


O = ("open", {"star": "seen"})
NS = ("unknown", {"open_no_star": True, "star": "absent"})
U = ("unknown", {})
S = ("shut", {})


def test_open_needs_every_frame_open_like_and_one_star():
    assert ks.combine_roof([O, O, O]) == "open"
    assert ks.combine_roof([O, NS, NS]) == "open"      # star seen once is positive
    assert ks.combine_roof([NS, NS, O]) == "open"
    assert ks.combine_roof([NS, NS, NS]) == "unknown"  # consistent, never positive
    assert ks.combine_roof([O, O, U]) == "unknown"     # a blind/vetoed frame
    assert ks.combine_roof([O, S, O]) == "unknown"
    assert ks.combine_roof([]) == "unknown"


def test_shut_needs_one_decoded_tag_and_no_contradiction():
    elsewhere = ("unknown", {"tag_elsewhere": True})
    assert ks.combine_roof([U, S, U]) == "shut"
    assert ks.combine_roof([S, S, S]) == "shut"
    assert ks.combine_roof([S, O, U]) == "unknown"
    assert ks.combine_roof([S, NS, S]) == "unknown"    # tag gone with witness contradicts shut
    assert ks.combine_roof([S, elsewhere, S]) == "unknown"


def test_scope_unsafe_anywhere_wins_and_safe_needs_a_positive_read():
    safe, unsafe, unk = ("safe", {}), ("UNSAFE", {}), ("unknown", {})
    assert ks.combine_scope([safe, safe, unsafe]) == "UNSAFE"
    assert ks.combine_scope([safe, unk, unk]) == "safe"
    assert ks.combine_scope([unk, unk, unk]) == "unknown"
    assert ks.combine_scope([]) == "unknown"


# ---------------------------------------------------------------- star, pure

def test_star_needs_a_reference_and_a_colour_frame():
    grey = np.full((1440, 2560, 3), 90, np.uint8)
    assert ks.star_verdict(grey, None)[0] == "unknown"
    ref = {"centre": [638, 718], "tolerance_px": 40, "search_px": 80, "template": None}
    v, d = ks.star_verdict(grey, ref)
    assert v == "unknown" and d.get("ir")


def _synthetic(star_at, template_from=None):
    """A dark-wall frame with a yellow star at *star_at* (BGR)."""
    img = np.zeros((1440, 2560, 3), np.uint8)
    img[:] = (60, 30, 20)                       # dark navy wall, clearly colour
    pts = []
    for k in range(10):
        r = 22 if k % 2 == 0 else 9
        a = np.pi / 2 + k * np.pi / 5
        pts.append((star_at[0] + r * np.cos(a), star_at[1] - r * np.sin(a)))
    cv2.fillPoly(img, [np.array(pts, np.int32)], (40, 210, 240))   # yellow
    return img


def test_star_seen_at_its_place_and_absent_elsewhere(tmp_path):
    ref_img = _synthetic((638, 718))
    tmpl = ref_img[718 - 34:718 + 34, 638 - 34:638 + 34]
    ref = {"centre": [638, 718], "tolerance_px": 40, "search_px": 80,
           "ncc_min": 0.4, "template": "unused"}
    v, d = ks.star_verdict(ref_img, ref, template=tmpl)
    assert v == "seen" and d["star_px"] < 2 and d["star_ncc"] > 0.9
    # within tolerance still counts (camera pose repeats to ~15 px)
    v, d = ks.star_verdict(_synthetic((650, 730)), ref, template=tmpl)
    assert v == "seen" and 15 < d["star_px"] < 20
    # a star-sized yellow blob 60 px away (where the pine sits when shut) does not
    v, d = ks.star_verdict(_synthetic((698, 718)), ref, template=tmpl)
    assert v == "absent" and d["star_px"] > 40
    # nothing yellow at all
    v, _ = ks.star_verdict(_synthetic((2000, 200)), ref, template=tmpl)
    assert v == "absent"


def test_pine_over_the_star_reads_absent():
    """When the roof is shut its pine underside fills the star's window: a
    bright tan field with no star-sized blob in it. And a yellow star painted
    onto pine (no dark surround) merges into that field -- the detector needs
    the star ON THE DARK WALL, which is the only place it is ever seen."""
    ref_img = _synthetic((638, 718))
    tmpl = ref_img[718 - 34:718 + 34, 638 - 34:638 + 34]
    ref = {"centre": [638, 718], "tolerance_px": 40, "search_px": 80,
           "ncc_min": 0.4, "template": "unused"}
    pine = np.zeros((1440, 2560, 3), np.uint8)
    pine[:] = (150, 200, 240)                   # tan: hue ~15, sat ~95, bright
    v, d = ks.star_verdict(pine, ref, template=tmpl)
    assert v == "absent", d
    painted = pine.copy()
    cv2.circle(painted, (638, 718), 14, (40, 210, 240), -1)
    v, d = ks.star_verdict(painted, ref, template=tmpl)
    assert v == "absent", d


def test_gating_read_waits_for_colour_after_the_light(monkeypatch):
    """2026-09-10: three gating reads 7 s after the light switch were still IR
    and could never have seen the star. The read regrabs until colour."""
    ir = np.full((64, 64, 3), 90, np.uint8)
    colour = np.zeros((64, 64, 3), np.uint8)
    colour[:] = (60, 30, 20)
    frames = iter([ir, ir, colour, colour])
    monkeypatch.setattr(ks, "_grab", lambda **kw: next(frames))
    monkeypatch.setattr(ks.time, "sleep", lambda s: None)
    monkeypatch.setattr(ks, "COLOUR_WAIT_S", 60.0)
    got = ks._await_colour(ir)
    assert not ks.is_ir(got)


def test_gating_read_gives_up_on_ir_but_returns_a_frame(monkeypatch):
    ir = np.full((64, 64, 3), 90, np.uint8)
    monkeypatch.setattr(ks, "_grab", lambda **kw: ir)
    monkeypatch.setattr(ks.time, "sleep", lambda s: None)
    monkeypatch.setattr(ks, "COLOUR_WAIT_S", 0.05)
    got = ks._await_colour(ir)
    assert ks.is_ir(got)


# ---------------------------------------------------------------- replay

FRAMES = sorted(glob.glob(str(ROOT / "sentry" / "roof_frames_kasa" / "open" / "*.jpg"))
                + glob.glob(str(ROOT / "sentry" / "roof_frames_kasa" / "shut" / "*.jpg")))


@pytest.mark.skipif(not FRAMES or not os.path.exists(str(ROOT / ks.STAR_PATH)),
                    reason="archived roof-move frames and the star reference are gitignored")
def test_star_on_every_archived_colour_frame():
    """Every lit/daylight OPEN frame on file shows the star; every SHUT frame
    does not; IR frames abstain. The one known exception is a dusk frame the
    camera took mid-switch (purple, blurred), which reads absent -- the safe
    direction."""
    os.chdir(ROOT)
    ref = ks._star_reference()
    for p in FRAMES:
        img = cv2.imread(p)
        if img is None:
            continue
        state = Path(p).parent.name            # open / shut
        v, d = ks.star_verdict(img, ref)
        if ks.is_ir(img):
            assert v == "unknown", p
        elif state == "shut":
            assert v == "absent", (p, d)
        elif "19-45-30" in p:                  # the dusk mid-switch frame
            assert v != "seen", (p, d)
        else:
            assert v == "seen", (p, d)
            assert d["star_px"] <= 20, (p, d)
