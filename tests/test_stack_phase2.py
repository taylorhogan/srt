"""Phase 2 of the stacker speed-up (2026-11-05): the registration pool, the
GPU sigma-clip combine and the machine-sized worker count. Every fast path
must give the slow path's answers.
"""
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("astropy")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stacking import stacker  # noqa: E402


def test_pool_workers_follow_ram_then_cores():
    assert stacker._machine_pool_workers(ram_gb=29, cpus=16) == 1     # the PC: serial
    assert stacker._machine_pool_workers(ram_gb=122, cpus=20) == 6    # the Spark
    assert stacker._machine_pool_workers(ram_gb=64, cpus=8) == 2
    assert stacker._machine_pool_workers(ram_gb=64, cpus=2) == 1
    assert stacker._machine_pool_workers(cap=3, ram_gb=122, cpus=20) == 3
    assert 1 <= stacker._STACK_REG_WORKERS <= 6


def test_threaded_combine_is_the_serial_combine(tmp_path):
    rng = np.random.default_rng(0)
    n, h, w = 9, 70, 130
    frames = []
    for i in range(n):
        f = rng.normal(5.0, 3.0, (h, w)).astype(np.float32)
        f[rng.integers(0, h, 30), rng.integers(0, w, 30)] += 80       # outliers to clip
        f[:, : 3 * i] = np.nan                                          # staggered footprints
        p = tmp_path / f"f{i:04d}.npy"
        np.save(p, f)
        frames.append(np.load(p, mmap_mode="r"))
    weights = (np.linspace(0.5, 1.5, n) / np.linspace(0.5, 1.5, n).sum()).astype(np.float32)
    outs = []
    for threads in (1, 4):
        result = np.empty((h, w), dtype=np.float32)
        cover = np.zeros((h, w), dtype=np.int32)
        msgs = []
        stacker._combine_tiles(frames, stacker.StackMethod.SIGMA_CLIP_FWHM, weights, 3.0, 32,
                               result, cover, progress_cb=msgs.append, threads=threads)
        outs.append((result, cover))
        assert msgs == ["combining 25%…", "combining 50%…", "combining 75%…", "combining 100%…"]
    np.testing.assert_array_equal(outs[0][0], outs[1][0])
    np.testing.assert_array_equal(outs[0][1], outs[1][1])
    assert outs[0][1].max() == n and outs[0][1][:, 0].max() == 1
    assert 1 <= stacker._COMBINE_THREADS <= 6


def _star_field(rng, shape, stars, dx, dy, sky=200.0):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    img = np.full(shape, sky, dtype=np.float64)
    for (x, y, a) in stars:
        img += a * np.exp(-(((xx - x - dx) ** 2 + (yy - y - dy) ** 2) / (2 * 1.6 ** 2)))
    return (img + rng.normal(0, 2.0, shape)).astype(np.float32)


def test_register_many_pool_gives_the_serial_answers(tmp_path):
    pytest.importorskip("astroalign")
    from astropy.io import fits
    rng = np.random.default_rng(3)
    shape = (240, 300)
    stars = [(rng.uniform(20, 280), rng.uniform(20, 220), rng.uniform(300, 4000)) for _ in range(70)]
    shifts = [(0, 0), (3, -2), (-4, 5), (6, 1), (-2, -6), (1, 7)]
    paths = []
    for i, (dx, dy) in enumerate(shifts):
        p = tmp_path / f"f{i}.fits"
        fits.PrimaryHDU(_star_field(rng, shape, stars, dx, dy)).writeto(p)
        paths.append(p)
    ref = stacker._load_fits_2d(paths[0])
    det = stacker._reference_control_points(stacker._despike(ref))
    assert det is not None and len(det) >= 10
    pending = [(i, p) for i, p in enumerate(paths) if i != 0]
    d_serial, d_pool = tmp_path / "serial", tmp_path / "pool"
    d_serial.mkdir(), d_pool.mkdir()
    serial = stacker._register_many(pending, det, shape, (None, None, None), d_serial, workers=1)
    pooled = stacker._register_many(pending, det, shape, (None, None, None), d_pool, workers=3)
    assert set(serial) == set(pooled) == {1, 2, 3, 4, 5}
    aligned = 0
    for i in serial:
        out_s, lev_s, why_s = serial[i]
        out_p, lev_p, why_p = pooled[i]
        assert (out_s is None) == (out_p is None) and why_s == why_p
        if out_s is not None:
            aligned += 1
            assert lev_s == lev_p and Path(out_p).name == f"f{i:04d}.npy"
            np.testing.assert_array_equal(np.load(out_s), np.load(out_p))
    assert aligned >= 4


def test_register_many_falls_back_to_serial_when_the_pool_fails(monkeypatch, tmp_path):
    import concurrent.futures as cf

    class Broken:
        def __init__(self, *a, **k):
            raise OSError("no pool")

    monkeypatch.setattr(cf, "ProcessPoolExecutor", Broken)
    monkeypatch.setattr(stacker, "_register_spill_worker",
                        lambda i, p, *a: (i, str(tmp_path / f"f{i:04d}.npy"), 1.5, ""))
    got = stacker._register_many([(i, Path(f"x{i}")) for i in range(5)], None, (4, 4),
                                 (None, None, None), tmp_path, workers=4)
    assert got == {i: (str(tmp_path / f"f{i:04d}.npy"), 1.5, "") for i in range(5)}


def test_no_pool_is_started_inside_a_worker(monkeypatch):
    import multiprocessing
    monkeypatch.setattr(multiprocessing, "parent_process", lambda: object())
    monkeypatch.setattr(stacker, "_register_spill_worker",
                        lambda i, p, *a: (i, f"f{i:04d}.npy", 0.5, ""))
    import concurrent.futures as cf

    class Never:
        def __init__(self, *a, **k):
            raise AssertionError("a pool was started inside a worker")

    monkeypatch.setattr(cf, "ProcessPoolExecutor", Never)
    got = stacker._register_many([(i, Path(f"x{i}")) for i in range(6)], None, (4, 4),
                                 (None, None, None), Path("."), workers=6)
    assert len(got) == 6
    monkeypatch.setattr(stacker, "_measure_fwhm_and_stars", lambda p: (1.0, 1))
    assert len(stacker._measure_many([Path(f"x{i}") for i in range(6)], workers=6)) == 6
