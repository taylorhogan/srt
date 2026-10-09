"""deblur=: Richardson-Lucy on the luminance that does not ring.

A synthetic field — stars of a known Gaussian width on a flat sky with a
gradient and a faint extended blob — pins the four rules the module exists
for: the kernel is recovered from the stars; stars come out sharper; no pixel
ends below its input (the dark deringing clamp, the thing every earlier
attempt lacked); the sky is left alone; and the option reaches compose() and
the options line. Skips without numpy/scipy/skimage.
"""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("skimage")

from stacking import color_process as cp  # noqa: E402
from stacking import deblur as db  # noqa: E402


def _field(sigma_px=2.0, seed=3, n=30, h=640, w=640, sky=300.0, noise=3.0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    img = np.full((h, w), sky, np.float64) + 0.05 * xx          # sky + gradient
    img += 40.0 * np.exp(-((yy - 320) ** 2 + (xx - 320) ** 2) / (2 * 60.0 ** 2))   # galaxy
    pos = []
    while len(pos) < n:
        y, x = rng.uniform(40, h - 40), rng.uniform(40, w - 40)
        if all(np.hypot(y - py, x - px) > 2.5 * db.PSF_RADIUS + 10 for py, px in pos):
            pos.append((y, x))
    amps = rng.uniform(300, 3000, n)
    for (y, x), a in zip(pos, amps):
        img += a * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma_px ** 2))
    img += rng.normal(0, noise, img.shape)
    return img.astype(np.float32), np.array(pos), amps


def test_kernel_is_measured_from_the_stars():
    img, pos, _ = _field(sigma_px=2.0)
    psf, info = db.measure_psf(img)
    assert psf is not None, info
    assert info["stars"] >= db.MIN_STARS
    assert abs(info["fwhm_px"] - 2.3548 * 2.0) < 0.3
    assert psf.shape == (2 * db.PSF_RADIUS + 1, 2 * db.PSF_RADIUS + 1)
    assert abs(psf.sum() - 1.0) < 1e-6
    r = db.PSF_RADIUS
    assert psf[r, r] == psf.max()                      # centred
    assert psf[0].max() == 0.0 and psf[:, 0].max() == 0.0   # tapered to zero


def test_sky_sigma_ignores_gradient_and_stars():
    img, _, _ = _field(noise=3.0)
    assert abs(db.sky_sigma(img) - 3.0) < 0.6


def test_stars_sharpen_and_nothing_goes_below_the_input():
    img, pos, _ = _field()
    out, info = db.deblur(img, iters=8)
    assert "why" not in info, info
    sig = info["sky_sigma"]
    # Dark deringing: the clamp is the invariant every earlier attempt lacked.
    assert float((img - out).max()) <= db.CLAMP_SIGMA * sig + 1e-3
    # Stars peak higher ...
    yi, xi = np.round(pos).astype(int).T
    gain = np.median(out[yi, xi] / img[yi, xi])
    assert gain > 1.3, gain
    # ... and nothing sits in a ring around them: the darkest pixel 4-8 px
    # out is no lower than it was.
    yy, xx = np.mgrid[-8:9, -8:9]
    ring = (np.hypot(yy, xx) > 4) & (np.hypot(yy, xx) <= 8)
    for y, x in zip(yi, xi):
        a, b = img[y - 8:y + 9, x - 8:x + 9][ring], out[y - 8:y + 9, x - 8:x + 9][ring]
        assert b.min() >= a.min() - db.CLAMP_SIGMA * sig - 1e-3


def test_sky_is_left_alone_and_flux_roughly_kept():
    img, pos, _ = _field()
    out, info = db.deblur(img, iters=8)
    far = np.ones(img.shape, bool)
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    for y, x in pos:
        far &= np.hypot(yy - y, xx - x) > 14
    far[:20] = far[-20:] = far[:, :20] = far[:, -20:] = False
    assert np.array_equal(out[far], img[far])          # blend is exactly 0 on sky
    assert abs(out.sum() / img.sum() - 1.0) < 0.01


def test_too_few_stars_returns_the_input():
    img, _, _ = _field(n=5)
    out, info = db.deblur(img, iters=8)
    assert out is img and "why" in info
    out, info = db.deblur(img, iters=0)
    assert out is img and info["iters"] == 0


def test_only_the_luminance_is_deblurred_when_there_is_one():
    img, _, _ = _field()
    chans = {"L": img, "R": img.copy(), "G": img.copy(), "B": img.copy()}
    out, infos = db.deblur_channels(chans, 4)
    assert set(infos) == {"L"}
    assert out["R"] is chans["R"] and not np.array_equal(out["L"], img)
    # No L: every distinct plane once, shared planes shared.
    o3 = img.copy()
    out, infos = db.deblur_channels({"R": img, "G": o3, "B": o3}, 4)
    assert set(infos) == {"R", "G", "B"} and out["G"] is out["B"]


def test_option_reaches_compose_and_the_options_line():
    img, pos, _ = _field()
    chans = {"L": img, "R": img * 0.8, "G": img * 0.9, "B": img}
    plain = cp.compose(chans, deblur=0)
    sharp = cp.compose(chans, deblur=4)
    assert plain.shape == sharp.shape and not np.array_equal(plain, sharp)
    assert cp.effective_options()["deblur"] == 0
    assert "deblur" not in cp.describe_options(cp.effective_options())
    assert "deblur=8" in cp.describe_options(cp.effective_options(deblur=8))
