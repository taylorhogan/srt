"""stacking.stacker.background_structure_scores: the tree-limb gate (2026-10-07)."""
import pytest

# CI installs pytest only; the stacker needs numpy + astropy.
np = pytest.importorskip("numpy")
pytest.importorskip("astropy")
stacker = pytest.importorskip("stacking.stacker")
BG_STRUCTURE_MAX = stacker.BG_STRUCTURE_MAX
background_structure_scores = stacker.background_structure_scores

NY, NX = 8, 12


def _flat(level=1.0):
    return np.full((NY, NX), level)


def _galaxy():
    m = _flat()
    m[3:5, 5:7] += 0.4          # bright target, the same in every frame
    return m


def test_target_common_to_every_frame_cancels():
    maps = [_galaxy() for _ in range(8)]
    assert max(background_structure_scores(maps)) < 1e-9


def test_smooth_gradient_is_not_structure():
    y, x = np.mgrid[0:NY, 0:NX]
    maps = [_galaxy() for _ in range(7)] + [_galaxy() + 0.02 * x]   # moon/twilight ramp
    assert max(background_structure_scores(maps)) < BG_STRUCTURE_MAX


def test_tree_blotch_is_rejected_and_clean_frames_kept():
    clean = [_galaxy() + np.random.default_rng(i).normal(0, 0.003, (NY, NX)) for i in range(7)]
    tree = _galaxy()
    tree[0:4, 0:5] += 0.25       # soft out-of-focus branch, ~0.25 of the frame level
    s = background_structure_scores(clean + [tree])
    assert s[-1] > BG_STRUCTURE_MAX
    assert max(s[:-1]) < BG_STRUCTURE_MAX


def test_a_moonlit_frame_of_the_same_field_is_not_structure():
    # 2026-10-22/23: V1 inside M31's disk. A sky three times brighter dilutes the
    # target's share of the frame level; the plain difference read that as 0.17-0.21.
    dark = [_galaxy() + np.random.default_rng(i).normal(0, 0.003, (NY, NX)) for i in range(6)]
    moonlit = []
    for i in range(4):
        m = _galaxy() + 2.0 + np.random.default_rng(10 + i).normal(0, 0.003, (NY, NX))   # +2x sky
        moonlit.append(m / np.median(m))                                                    # as _background_map
    assert max(background_structure_scores(dark + moonlit)) < BG_STRUCTURE_MAX


def _raw(sky, seed, pedestal=10.0):
    """pedestal + sky x vignetting + target, normalised as _background_map does."""
    y, x = np.mgrid[0:NY, 0:NX]
    vig = 1.0 - 0.3 * (((x - NX / 2) / NX) ** 2 + ((y - NY / 2) / NY) ** 2) * 4
    gal = np.zeros((NY, NX)); gal[3:5, 5:7] = 0.4
    m = pedestal + sky * vig + gal + np.random.default_rng(seed).normal(0, 0.003, (NY, NX))
    return m / np.median(m), vig


def test_dark_and_moonlit_nights_both_pass_once_the_vignetting_is_known():
    # 2026-10-25: with a median alone, whichever kind of night was the minority
    # failed (V1's dark 10-08 once the moonlit frames outnumbered it, real
    # optics + M31's light: 8 of 12 over 0.08). A smooth synthetic falloff is
    # mostly taken by the plane, so this pins the mechanism -- the flat removes
    # what the median cannot -- rather than the real-data magnitude.
    dark = [_raw(0.5, i)[0] for i in range(4)]
    bright = [_raw(6.0, 10 + i)[0] for i in range(8)]
    vig = _raw(1.0, 0)[1]
    without = max(background_structure_scores(dark + bright))
    with_flat = max(background_structure_scores(dark + bright, vignette=vig))
    assert with_flat < without / 10 and with_flat < BG_STRUCTURE_MAX


def test_a_tree_is_still_caught_with_the_vignetting_in_the_fit():
    maps = [_raw(2.0, i)[0] for i in range(7)]
    tree = maps[0].copy()
    tree[0:4, 0:5] += 0.25
    s = background_structure_scores(maps + [tree], vignette=_raw(1.0, 0)[1])
    assert s[-1] > BG_STRUCTURE_MAX and max(s[:-1]) < BG_STRUCTURE_MAX
