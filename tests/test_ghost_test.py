"""The ghost test's two scripts: the five positions, and the reader on a planted ghost."""
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
from scripts import ghost_test_sequence as gts  # noqa: E402  (stdlib only at import)
from scripts import ghost_test_measure as gtm   # noqa: E402  (numpy at import)


def test_positions_are_centre_twice_then_a_cross_in_arcminutes():
    pos = gts.positions("star", 2.0, 30.0, 8.0, 6.0)
    names = [p[0] for p in pos]
    assert names == ["star_c", "star_c2", "star_e", "star_w", "star_n", "star_s"]
    by = {p[0]: p for p in pos}
    assert by["star_c"][3] == ["L", "R", "G"] and by["star_c2"][3] == ["L", "B"]
    # 8' of RA at dec 30: 8/60 deg / (15 deg/h * cos 30)
    dra = (8.0 / 60.0) / (15.0 * np.cos(np.radians(30.0)))
    assert by["star_e"][1] == pytest.approx(2.0 + dra) and by["star_w"][1] == pytest.approx(2.0 - dra)
    assert by["star_n"][2] == pytest.approx(30.1) and by["star_s"][2] == pytest.approx(29.9)
    assert all(p[3] == ["L"] for p in pos[2:])


def _put(img, x, y, flux, sig=3.0):
    h, w = img.shape
    x0, y0, r = int(x), int(y), 16
    ya, yb, xa, xb = max(y0 - r, 0), min(y0 + r, h), max(x0 - r, 0), min(x0 + r, w)
    if yb <= ya or xb <= xa:
        return
    ys, xs = np.mgrid[ya:yb, xa:xb]
    img[ya:yb, xa:xb] += flux / (2 * np.pi * sig ** 2) * np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sig ** 2))


def test_reader_recovers_a_planted_inversion(tmp_path: Path):
    fits = pytest.importorskip("astropy.io.fits")
    pytest.importorskip("cv2")
    rng = np.random.default_rng(5)
    H, W = 1200, 1600
    field = np.c_[rng.uniform(0, W, 60), rng.uniform(0, H, 60), 10 ** rng.uniform(4, 5.5, 60)]
    C, m = np.array([820.0, 560.0]), 0.8
    star_sky = np.array([500.0, 420.0])
    for k, (px, py) in enumerate([(0, 0), (260, 0), (-260, 0), (0, 220), (0, -220)]):
        img = rng.normal(900, 20, (H, W)).astype(np.float32)
        for x, y, fl in field:
            _put(img, x - px, y - py, fl)
        s = star_sky - np.array([px, py])
        _put(img, s[0], s[1], 4e8, sig=4)                 # saturates widely
        g = C - m * (s - C)
        _put(img, g[0], g[1], 4e8 * 1e-4)                 # the ghost
        img[100, 100] = 65000                             # a hot pixel
        hdu = fits.PrimaryHDU(np.clip(img, 0, 65535).astype(np.float32))
        hdu.header["FILTER"] = "L"
        hdu.header["EXPTIME"] = 30.0
        hdu.writeto(tmp_path / ("synth_%d.fits" % k), overwrite=True)
    out = tmp_path / "fit.json"
    assert gtm.main([str(tmp_path), "--pattern", "synth_*.fits", "--min-move", "150",
                     "--json", str(out)]) == 0
    fits_found = json.loads(out.read_text())["fits"]
    assert fits_found, "no ghost fitted"
    best = fits_found[0]
    M = np.array(best["M"])
    assert best["rms"] < 3.0
    assert np.allclose(M, -m * np.eye(2), atol=0.05)
    assert np.allclose(best["centre"], C, atol=6)
