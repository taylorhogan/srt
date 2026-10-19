"""stacking/starless: RL on the picture with its stars taken off, the stars
rendered back as points at each channel's own flux.

A synthetic field pins what the module promises: the stars come back as
points (sharper than imaged) with their flux, a colour channel's stars carry
that channel's flux ratio rather than L's, a broad knot is left to RL rather
than rendered as a star, the galaxy sharpens, the sky is left alone, NaN
borders survive, and a starless field is refused with a reason. The field
is the one tests/test_deblur.py uses, with a known star list.
"""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("skimage")

from stacking import deblur as db  # noqa: E402
from stacking import starless  # noqa: E402

from tests.test_deblur import _field  # noqa: E402


def _flux_in(img, y, x, r=8):
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    m = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
    return float(img[m].sum())


def test_stars_come_back_as_points_with_their_flux_and_the_galaxy_sharpens():
    img, pos, amps, bg, st = _field(sigma_px=2.0, noise=1.0, parts=True)
    out, infos = starless.deblur_set({"L": img}, iters=8)
    assert set(infos) == {"L"} and "why" not in infos["L"], infos
    info = infos["L"]
    assert info["method"] == "starless" and info["iters"] == 8
    assert info["stars"] >= db.MIN_STARS and info["non_stellar"] == 0
    L = out["L"]
    # Each star: tighter than imaged, flux within 25%, measured above the local sky.
    sky_in = np.median(img)
    sky_out = np.median(L)
    tighter, kept = 0, 0
    for (y, x), a in zip(pos, amps):
        y, x = int(round(y)), int(round(x))
        fin = _flux_in(img - sky_in, y, x) - 0.0
        fout = _flux_in(L - sky_out, y, x)
        if (L[y - 1:y + 2, x - 1:x + 2].max() - sky_out) > (img[y - 1:y + 2, x - 1:x + 2].max() - sky_in):
            tighter += 1
        if abs(fout / fin - 1.0) < 0.25:
            kept += 1
    assert tighter >= 0.8 * len(pos), tighter
    assert kept >= 0.8 * len(pos), kept
    # The galaxy (a 60 px Gaussian blob of 40 ADU) is at least as peaked after RL.
    assert L[320, 320] - sky_out >= 0.95 * (img[320, 320] - sky_in)
    # The sky is left alone: the empty corner's median moves by less than a sigma.
    assert abs(np.median(L[:80, :80]) - np.median(img[:80, :80])) < 1.0


def test_a_colour_channel_keeps_its_own_star_flux():
    """Half the stars are twice as bright in R as in L: their rendered R flux
    says so, the others' says 1:1, and the field ratio is not forced on them."""
    img, pos, amps, bg, st = _field(sigma_px=2.0, noise=1.0, parts=True)
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    red = np.zeros_like(st)
    for i, ((y, x), a) in enumerate(zip(pos, amps)):
        k = 2.0 if i % 2 == 0 else 1.0
        red += k * a * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 2.0 ** 2))
    R = (bg + red + np.random.default_rng(5).normal(0, 1.0, img.shape)).astype(np.float32)
    out, infos = starless.deblur_set({"L": img, "R": R}, iters=4)
    assert "why" not in infos["R"], infos
    assert infos["R"]["iters"] == 0                      # the colour plane is not RL'd
    skyL, skyR = np.median(out["L"]), np.median(out["R"])
    ratios = []
    for i, ((y, x), a) in enumerate(zip(pos, amps)):
        if a < 1000:
            continue                                     # the bright ones, where the aperture is clean
        y, x = int(round(y)), int(round(x))
        ratios.append((i % 2 == 0, _flux_in(out["R"] - skyR, y, x) / _flux_in(out["L"] - skyL, y, x)))
    doubled = [r for d, r in ratios if d]
    same = [r for d, r in ratios if not d]
    assert len(doubled) >= 3 and len(same) >= 3
    assert 1.6 < np.median(doubled) < 2.4, doubled
    assert 0.8 < np.median(same) < 1.25, same


def test_a_broad_knot_is_not_rendered_as_a_star():
    """A knot 1.5x the stars' width passes star_layer's compactness test (a
    3x one never gets that far); the ring test must catch it. Either way it
    is not collapsed into a 2 px point."""
    img, pos, amps = _field(sigma_px=2.0, noise=1.0)
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    for (y, x, s_) in ((150, 480, 3.0), (480, 150, 6.0)):
        img = (img + 600.0 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * s_ ** 2))).astype(np.float32)
    out, infos = starless.deblur_set({"L": img}, iters=4)
    assert infos["L"]["non_stellar"] >= 1, infos
    # RL is allowed to sharpen a knot (the 3 px one deconvolves to 2.2 px,
    # 8% of its peak at r = 5); a rendered point would be under 1% there.
    sky = np.median(out["L"])
    for (y, x) in ((150, 480), (480, 150)):
        pk = out["L"][y, x] - sky
        assert out["L"][y, x + 5] - sky > 0.04 * pk, f"the knot at {(y, x)} was collapsed into a point"


def test_nan_border_is_kept_and_a_starless_field_is_refused():
    img, pos, amps = _field(noise=1.0)
    img = img.copy()
    img[:12, :] = np.nan
    out, infos = starless.deblur_set({"L": img}, iters=2)
    assert "why" not in infos["L"]
    assert np.isnan(out["L"][:12]).all() and not np.isnan(out["L"][12:]).any()
    few, _, _ = _field(n=5, noise=1.0)
    chans = {"L": few}
    out, infos = starless.deblur_set(chans, iters=2)
    assert out is chans and infos["L"]["why"]
    out, infos = starless.deblur_set(chans, iters=0)
    assert out is chans and infos == {}


def test_normalised_fill_follows_the_gradient():
    """A hole on a slope is filled along the slope, not at the frame's level:
    a weighted mean of the surround, so the rim leans a few pixels outward,
    but nowhere near the 24 ADU a flat fill at the median would be off by."""
    yy, xx = np.mgrid[:200, :200]
    img = (100.0 + 2.0 * xx).astype(np.float32)
    hole = (yy - 100) ** 2 + (xx - 100) ** 2 <= 15 ** 2
    filled = starless.normalised_fill(img, hole)
    assert np.array_equal(filled[~hole], img[~hole])
    err = np.abs(filled - img)[hole]
    flat = np.abs(np.median(img) - img)[hole]
    assert abs(filled[100, 100] - img[100, 100]) < 4.0
    assert err.mean() < 8.0 and err.max() < 20.0
    assert err.mean() < 0.6 * flat.mean()


def test_deblur_channels_default_is_starless_and_shares_planes():
    img, _, _ = _field(noise=1.0)
    o3 = img.copy()
    out, infos = db.deblur_channels({"R": img, "G": o3, "B": o3}, 2)
    assert set(infos) == {"R", "G", "B"}
    assert all(i.get("method") == "starless" for i in infos.values())
    assert out["G"] is out["B"] and infos["R"]["iters"] == 2 and infos["G"]["iters"] == 2
