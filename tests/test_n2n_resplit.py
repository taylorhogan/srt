"""nn/resplit: a fresh disjoint half-split per patch, stacked with the stacker's combine."""
import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("astropy")
pytest.importorskip("sep")
torch = pytest.importorskip("torch")

from nn import resplit  # noqa: E402
from stacking import stacker  # noqa: E402


def test_split_halves_are_disjoint_equal_and_drop_the_odd_frame():
    rng = np.random.default_rng(1)
    for k in (4, 7, 30):
        a, b = resplit.split_halves(k, rng)
        assert len(a) == len(b) == k // 2
        assert not set(a) & set(b)
        assert set(a) | set(b) <= set(range(k))


def test_split_changes_between_draws():
    rng = np.random.default_rng(2)
    draws = {tuple(sorted(resplit.split_halves(20, rng)[0])) for _ in range(20)}
    assert len(draws) > 10


def test_combine_half_is_the_stackers_combine_with_the_sky_restored():
    rng = np.random.default_rng(3)
    cube = rng.normal(0.0, 5.0, (6, 16, 16)).astype(np.float32)
    cube[2, 4, 4] = 5000.0                      # a cosmic ray the clip must drop
    cube[1, :2, :] = np.nan                     # a frame that does not cover the top rows
    w = np.array([1, 2, 1, 1, 2, 1], dtype=np.float64)
    levels = np.array([100, 110, 90, 100, 105, 95], dtype=np.float64)
    out = resplit.combine_half(cube, w, levels)
    ref = stacker._combine_tile(cube, stacker.StackMethod.SIGMA_CLIP_FWHM, w / w.sum(), resplit.SIGMA)
    ref = ref + np.average(levels, weights=w / w.sum())
    assert np.allclose(out, ref, equal_nan=True)
    assert abs(out[4, 4] - np.average(levels, weights=w / w.sum())) < 30   # the ray is gone
    assert np.isfinite(out[0, 0])                                           # five frames still cover it
    fixed = resplit.combine_half(cube, w, levels, restore=250.0)
    assert np.allclose(fixed - out, 250.0 - np.average(levels, weights=w / w.sum()), equal_nan=True)


def _group(tmp_path, n=8, h=96, w=96, seed=0):
    rng = np.random.default_rng(seed)
    d = tmp_path / "test__Ha"
    (d / "frames").mkdir(parents=True)
    scene = np.zeros((h, w), np.float32)
    scene[40:44, 50:54] = 300.0
    files = []
    for i in range(n):
        f = (scene + rng.normal(0, 4.0, (h, w))).astype(np.float32)   # levelled: sky at 0
        p = d / "frames" / f"f{i:04d}.npy"
        np.save(p, f)
        files.append(f"frames/f{i:04d}.npy")
    full = scene + 200.0 + rng.normal(0, 4.0 / np.sqrt(n), (h, w)).astype(np.float32)
    np.save(d / resplit.ALL_STACK, full)
    np.save(d / resplit.BACKGROUND, np.full((h, w), 200.0, np.float32))
    (d / resplit.GROUP_FILE).write_text(json.dumps({
        "filter": "Ha", "n_frames": n, "files": files, "paths": files,
        "levels": [200.0] * n, "weights": [1.0 / n] * n, "restored": 200.0,
        "shape": [h, w], "coverage_box": [0, h, 0, w], "scale": 2.0, "sky": 200.0}))
    return d


def test_dataset_yields_two_normalised_patches_from_disjoint_halves(tmp_path, monkeypatch):
    d = _group(tmp_path)
    ds = resplit.ResplitDataset([d], patch_size=32, pairs_per_epoch=5, seed=0)
    assert len(ds) == 5
    seen = []
    orig = resplit.split_halves

    def spy(k, rng):
        a, b = orig(k, rng)
        seen.append((set(a.tolist()), set(b.tolist())))
        return a, b
    monkeypatch.setattr(resplit, "split_halves", spy)
    a, b = ds[0]
    assert a.shape == b.shape == (1, 32, 32)
    assert a.dtype == torch.float32
    assert seen and not (seen[0][0] & seen[0][1]) and len(seen[0][0]) == 4
    # Sky at 0 after the background, in asinh units: both halves near zero, not identical noise.
    assert abs(float(a.mean())) < 1.0 and abs(float(b.mean())) < 1.0
    assert not torch.allclose(a, b)


def test_dataset_pair_is_the_combined_halves_through_the_normalisation(tmp_path):
    d = _group(tmp_path, n=6, h=64, w=64)
    ds = resplit.ResplitDataset([d], patch_size=32, pairs_per_epoch=1, seed=0)
    g = ds.groups[0]
    rng = np.random.default_rng(5)
    pa, pb = ds.pair(g, 16, 24, rng)
    rng = np.random.default_rng(5)
    a, b = resplit.split_halves(6, rng)
    cube = np.stack([np.asarray(g["mm"][i][16:48, 24:56]) for i in a])
    half = resplit.combine_half(cube, g["weights"][a], g["levels"][a], g["restored"])
    want = np.arcsinh((half - 200.0) / 2.0).astype(np.float32)
    assert np.allclose(pa, want, atol=1e-5)
    assert pa.shape == pb.shape == (32, 32)


def test_origin_is_clamped_into_the_coverage_box(tmp_path):
    d = _group(tmp_path, h=96, w=96)
    rec = json.loads((d / resplit.GROUP_FILE).read_text())
    rec["coverage_box"] = [10, 60, 20, 70]
    (d / resplit.GROUP_FILE).write_text(json.dumps(rec))
    ds = resplit.ResplitDataset([d], patch_size=32, pairs_per_epoch=1, seed=0)
    for _ in range(50):
        y0, x0 = ds.origin(ds.groups[0])
        assert 10 <= y0 <= 60 - 32 and 20 <= x0 <= 70 - 32


def test_original_frames_drops_frames_that_arrived_after_the_stacks():
    """The ladder's frames stage reproduces the fixed halves from a group that
    has since grown (NGC 6946 Ha: 26 -> 49 frames after the stacks)."""
    import importlib
    ladder = importlib.import_module("scripts.n2n_lrgb_ladder")
    old = [f"2026-10-14/LIGHT/f{i}.fits" for i in range(18)] + [f"2026-10-18/LIGHT/f{i}.fits" for i in range(8)]
    new = [f"2026-10-19/LIGHT/f{i}.fits" for i in range(12)] + [f"2026-10-20/LIGHT/f{i}.fits" for i in range(11)]
    built = 1000.0
    mt = lambda p: 500.0 if p in old else 2000.0
    assert ladder.original_frames(old, 26, built, mt) == old
    assert ladder.original_frames(old + new, 26, built, mt) == old
    assert ladder.original_frames(old[:20], 26, built, mt) is None            # frames went missing
    assert ladder.original_frames(old + new, 30, built, mt) is None           # the tail does not account for it
