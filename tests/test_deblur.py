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


def _field(sigma_px=2.0, seed=3, n=30, h=640, w=640, sky=300.0, noise=3.0, parts=False):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    bg = np.full((h, w), sky, np.float64) + 0.05 * xx          # sky + gradient
    bg += 40.0 * np.exp(-((yy - 320) ** 2 + (xx - 320) ** 2) / (2 * 60.0 ** 2))   # galaxy
    pos = []
    while len(pos) < n:
        y, x = rng.uniform(40, h - 40), rng.uniform(40, w - 40)
        if all(np.hypot(y - py, x - px) > 2.5 * db.PSF_RADIUS + 10 for py, px in pos):
            pos.append((y, x))
    amps = rng.uniform(300, 3000, n)
    st = np.zeros((h, w))
    for (y, x), a in zip(pos, amps):
        st += a * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma_px ** 2))
    img = bg + st + rng.normal(0, noise, (h, w))
    if parts:
        return img.astype(np.float32), np.array(pos), amps, bg, st
    return img.astype(np.float32), np.array(pos), amps


def test_kernel_is_measured_from_the_stars():
    img, pos, _ = _field(sigma_px=2.0)
    psf, info = db.measure_psf(img)
    assert psf is not None, info
    assert info["stars"] >= db.MIN_STARS
    assert abs(info["fwhm_px"] - 2.3548 * 2.0) < 0.3
    # Gaussian stars have no wing, so none is grafted and the core kernel is used.
    assert psf.shape == (2 * db.PSF_RADIUS + 1, 2 * db.PSF_RADIUS + 1)
    assert abs(psf.sum() - 1.0) < 1e-6
    r = db.PSF_RADIUS
    assert psf[r, r] == psf.max()                      # centred
    assert psf[0].max() == 0.0 and psf[:, 0].max() == 0.0   # tapered to zero


def test_wing_is_measured_from_bright_stars():
    """Stars with a real wing (Moffat beta=2.5) on a field wide enough to hold
    isolated bright ones: the kernel extends past the core radius and carries
    the wing flux the core stack alone cannot see."""
    rng = np.random.default_rng(7)
    h = w = 1400
    yy, xx = np.mgrid[:h, :w]
    img = np.full((h, w), 300.0)
    pos = []
    while len(pos) < 40:
        y, x = rng.uniform(100, h - 100), rng.uniform(100, w - 100)
        if all(np.hypot(y - py, x - px) > 2 * db.WING_RADIUS + 10 for py, px in pos):
            pos.append((y, x))
    alpha, beta = 3.5, 2.5
    for (y, x), a in zip(pos, rng.uniform(4000, 15000, 40)):
        r2 = (yy - y) ** 2 + (xx - x) ** 2
        img += a * (1 + r2 / alpha ** 2) ** (-beta)
    img += rng.normal(0, 3.0, img.shape)
    psf, info = db.measure_psf(img.astype(np.float32))
    assert psf is not None, info
    assert info["wing_stars"] >= db.WING_MIN_STARS, info
    assert psf.shape == (2 * db.WING_RADIUS + 1, 2 * db.WING_RADIUS + 1)
    R = db.WING_RADIUS
    r = np.hypot(*np.mgrid[-R:R + 1, -R:R + 1])
    truth = (1 + r ** 2 / alpha ** 2) ** (-beta); truth /= truth[r <= 60].sum()
    # wing flux between 24 and 60 px, kernel against the Moffat it was built from
    assert abs(psf[(r > 24) & (r <= 60)].sum() / truth[(r > 24) & (r <= 60)].sum() - 1) < 0.25


def test_sky_sigma_ignores_gradient_and_stars():
    img, _, _ = _field(noise=3.0)
    assert abs(db.sky_sigma(img) - 3.0) < 0.6


def test_stars_sharpen_and_nothing_goes_below_the_input():
    """The unsplit path with the input clamp: the pixelwise invariant."""
    img, pos, _ = _field()
    out, info = db.deblur(img, iters=8, split=False)
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


def test_local_sky_follows_slopes_and_ignores_stars():
    img, pos, _, truth, st = _field(parts=True)
    ref = db.local_sky(img)
    assert np.abs(ref - truth).mean() < 1.0            # follows gradient and bump
    yi, xi = np.round(pos).astype(int).T
    assert np.abs(ref[yi, xi] - truth[yi, xi]).max() < 6.0   # stars barely lift it


def test_sky_is_left_alone_and_flux_roughly_kept():
    img, pos, _ = _field()
    out, info = db.deblur(img, iters=8)
    far = np.ones(img.shape, bool)
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    for y, x in pos:
        far &= np.hypot(yy - y, xx - x) > 40       # the floor model's window around a patched star
    far &= np.hypot(yy - 320, xx - 320) > 24       # the galaxy bump's peak counts as a source
    far[:20] = far[-20:] = far[:, :20] = far[:, -20:] = False
    assert np.array_equal(out[far], img[far])          # blend is exactly 0 on sky
    assert abs(out.sum() / img.sum() - 1.0) < 0.01


def test_sky_clamp_drains_halos_but_not_below_the_sky():
    img, pos, _ = _field()
    held, _ = db.deblur(img, iters=8, clamp_to="input", split=False)
    drained, info = db.deblur(img, iters=8, clamp_to="sky", split=False)
    sig = info["sky_sigma"]
    # Halos may fall under the sky clamp, so flux is conserved more tightly...
    assert abs(drained.sum() / img.sum() - 1.0) < abs(held.sum() / img.sum() - 1.0)
    # ...and the bound is pixelwise: never below min(input, local sky) minus
    # one sigma, anywhere. The true sky is known here (the field is built
    # from it); the module's minimum-filter estimate of it runs up to a sigma
    # low on the curved galaxy bump, so the margin is the clamp plus that.
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    truth = 300.0 + 0.05 * xx + 40.0 * np.exp(-((yy - 320) ** 2 + (xx - 320) ** 2) / (2 * 60.0 ** 2))
    assert np.all(drained >= np.minimum(img, truth) - 3 * sig)
    # and halos do drain: the ring 4-8 px out sits lower than under the input clamp
    for y, x in pos:
        r = np.hypot(yy - y, xx - x); ring = (r > 4) & (r <= 8)
        assert drained[ring].mean() <= held[ring].mean() + 1e-3
    with pytest.raises(ValueError):
        db.deblur(img, iters=8, clamp_to="halo")


def test_split_stars_sharpen_on_a_zero_background_without_rings():
    """The default path: stars lifted off a fill, deconvolved apart, added
    back. No ring (nothing under the true sky), flux kept, cores higher,
    wings lower, and the starless image is what the galaxy path sees."""
    img, pos, _, truth, st = _field(parts=True)
    out, info = db.deblur(img, iters=8, split=True)
    assert info["split"] and info["split_stars"] >= len(pos) - 2, info
    sig = info["sky_sigma"]
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    # No ring: nothing ends below what the pixel is without its star (the
    # subtraction exposes the noise that was under the star) minus 3 sigma.
    assert np.all(out >= np.minimum(img - st, truth) - 3 * sig)
    assert abs(out.sum() / img.sum() - 1.0) < 0.01
    yi, xi = np.round(pos).astype(int).T
    assert np.median(out[yi, xi] / img[yi, xi]) > 1.3
    for y, x in pos:
        r = np.hypot(yy - y, xx - x); ring = (r > 4) & (r <= 8)
        assert out[ring].mean() < img[ring].mean()           # the wing drained
    # the star layer itself: amplitude x kernel per star, nothing else
    from scipy.ndimage import gaussian_filter, minimum_filter
    model = gaussian_filter(minimum_filter(img, db.FLOOR_MIN_SIZE), db.FLOOR_SMOOTH)
    psf, _ = db.measure_psf(img)
    starless, inner, mask, n, fits, wing = db.star_layer(img, model, sig, psf, None)
    assert fits.shape == (n, 5)
    assert np.array_equal(starless[~mask], img[~mask])          # untouched outside the footprint
    assert np.abs(starless - truth)[mask].mean() < 2 * sig       # the galaxy survives under the stars
    stars = inner
    assert n >= len(pos) - 2
    assert stars.min() >= 0 and 0.001 < mask.mean() < 0.2
    # the galaxy bump's peak is broad and must NOT be in the layer
    assert stars[300:341, 300:341].max() < 3 * sig
    # the layer reproduces each star's peak to 15%
    yi, xi = np.round(pos).astype(int).T
    truth_peak = img[yi, xi] - truth[yi, xi]
    assert np.median(np.abs(stars[yi, xi] / truth_peak - 1)) < 0.15


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
    line = cp.describe_options(cp.effective_options(deblur=15, dering="input"))
    assert "deblur=15" in line and "dering=input" in line
    assert "dering" not in cp.describe_options(cp.effective_options(deblur=8))       # sky is the default
    assert "dering" not in cp.describe_options(cp.effective_options(dering="input"))   # meaningless without deblur
