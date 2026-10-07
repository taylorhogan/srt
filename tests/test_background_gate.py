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
