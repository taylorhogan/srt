"""The stacker's quality pass: cv2 despike, and FWHM measured in a process
pool (2026-11-05). Both must give the serial, scipy answers: the gate cuts
on these numbers, so a faster path that changed them would change which
frames a stack keeps.
"""
import multiprocessing
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stacking import stacker  # noqa: E402


@pytest.mark.parametrize("dtype", [np.uint16, np.float32, np.float64])
def test_despike_is_the_scipy_median_inside_and_keeps_the_dtype(dtype):
    from scipy.ndimage import median_filter
    rng = np.random.default_rng(1)
    frame = rng.integers(100, 400, (64, 80)).astype(dtype)
    frame[10, 10] = 60000      # a hot pixel
    frame[30:33, 40:43] = 5000  # a star survives
    out = stacker._despike(frame)
    assert out.dtype == frame.dtype and out.shape == frame.shape
    ref = median_filter(frame, size=3)
    np.testing.assert_array_equal(out[1:-1, 1:-1], ref[1:-1, 1:-1])
    assert out[10, 10] < 1000 and out[31, 41] == 5000


def test_measure_many_serial_below_the_pool_threshold(monkeypatch):
    calls = []
    monkeypatch.setattr(stacker, "_measure_fwhm_and_stars",
                        lambda p: calls.append(p) or (float(len(p.name)), 7))
    paths = [Path(f"f{i}.fits") for i in range(3)]
    got = stacker._measure_many(paths, workers=6)
    assert got == {p: (float(len(p.name)), 7) for p in paths} and calls == paths
    got1 = stacker._measure_many([Path("abcd.fits")] * 1 + paths, workers=1)
    assert got1[Path("abcd.fits")] == (9.0, 7)


@pytest.mark.skipif(multiprocessing.get_start_method() != "fork",
                    reason="the fake measurer only reaches a forked worker")
def test_measure_many_pool_gives_the_serial_answers(monkeypatch):
    monkeypatch.setattr(stacker, "_measure_fwhm_and_stars",
                        lambda p: (float(int(p.stem[1:])) / 10.0, int(p.stem[1:]) * 3))
    paths = [Path(f"f{i}.fits") for i in range(9)]
    serial = stacker._measure_many(paths, workers=1)
    pooled = stacker._measure_many(paths, workers=3)
    assert pooled == serial == {p: (i / 10.0, i * 3) for i, p in enumerate(paths)}


def test_measure_many_falls_back_to_serial_when_the_pool_fails(monkeypatch):
    import concurrent.futures as cf

    class Broken:
        def __init__(self, *a, **k):
            raise OSError("no pool")

    monkeypatch.setattr(cf, "ProcessPoolExecutor", Broken)
    monkeypatch.setattr(stacker, "_measure_fwhm_and_stars", lambda p: (2.5, 11))
    paths = [Path(f"f{i}.fits") for i in range(6)]
    assert stacker._measure_many(paths, workers=4) == {p: (2.5, 11) for p in paths}


def test_pool_size_follows_the_cores():
    assert 1 <= stacker._FWHM_POOL_WORKERS <= 6
    assert stacker._FWHM_POOL_MIN_FRAMES >= 2
