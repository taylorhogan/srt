"""Palette gain: `gain=hubble` / `gain=Ha:0.4` on HOO/SHO/HSO (2026-10-28).

compose holds one brightness scale across channels so a palette shows the
real line ratios; on an HII region that makes SHO green (Ha is 3-5x S-II and
O-III). The palette gain is the one deliberate, named exception: a scaling of
the subtracted planes before the shared white point. These tests pin the
spelling, the recipe's role mapping (HOO's O-III is G and B), the measured
"hubble" factors, that a series pins one measurement, that colour recipes
refuse it, and that an unasked-for gain changes nothing.

Skips without numpy (the CI interpreter is pytest-only).
"""
import pytest

np = pytest.importorskip("numpy")

from stacking import color_process as cp  # noqa: E402


def test_parse_spellings():
    assert cp.parse_palette_gain("") is None and cp.parse_palette_gain(None) is None
    assert cp.parse_palette_gain("hubble") == "hubble"
    assert cp.parse_palette_gain("HUBBLE") == "hubble"
    assert cp.parse_palette_gain("ha:0.4") == {"Ha": 0.4}
    assert cp.parse_palette_gain("Ha:0.4/o-iii:1.5/s2:2") == {"Ha": 0.4, "O-III": 1.5, "S-II": 2.0}
    assert cp.parse_palette_gain({"ha": "0.5"}) == {"Ha": 0.5}
    for bad in ("ha", "ha:x", "ha:0", "ha:-1", "blue:2", "0.4"):
        with pytest.raises(ValueError):
            cp.parse_palette_gain(bad)


def test_roles_follow_the_recipe():
    assert cp.palette_gain_roles("SHO", "ha:0.4") == {"G": 0.4}
    assert cp.palette_gain_roles("HSO", "ha:0.4") == {"R": 0.4}
    assert cp.palette_gain_roles("HOO", "o-iii:1.5") == {"G": 1.5, "B": 1.5}     # one filter, two roles
    assert cp.palette_gain_roles("sho", "hubble") == "hubble"
    assert cp.palette_gain_roles("SHO", "") is None
    with pytest.raises(ValueError):
        cp.palette_gain_roles("HOO", "s-ii:2")                 # no S-II in HOO
    for colour in ("LRGB", "HALRGB"):
        with pytest.raises(ValueError):
            cp.palette_gain_roles(colour, "hubble")


def _planes(seed=3):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:120, 0:160]
    neb = np.exp(-(((yy - 60) / 25.0) ** 2 + ((xx - 80) / 40.0) ** 2))
    ha = 20.0 * neb + rng.normal(0, 0.3, neb.shape)
    s2 = 5.0 * neb + rng.normal(0, 0.3, neb.shape)
    o3 = 8.0 * neb + rng.normal(0, 0.3, neb.shape)
    return {"R": s2.astype(np.float32), "G": ha.astype(np.float32), "B": o3.astype(np.float32)}


def test_hubble_equalises_the_nebula_levels():
    sub = _planes()
    out, factors = cp.apply_palette_gain(sub, "hubble")
    assert factors["G"] == pytest.approx(1.0)                  # Ha is the strongest: untouched
    assert factors["R"] == pytest.approx(4.0, rel=0.05)         # S-II at 1/4 of Ha -> x4
    assert factors["B"] == pytest.approx(2.5, rel=0.05)
    lv = {c: cp.nebula_level(out[c]) for c in out}
    assert max(lv.values()) / min(lv.values()) < 1.02
    assert out["G"] is sub["G"]                                 # factor 1 leaves the plane alone


def test_hubble_level_is_the_nebula_not_the_stars():
    # Continuum stars are the same in every narrowband filter; a pixel
    # percentile high enough to be "the bright nebula" is really them, and
    # equalising it leaves a 1:4 nebula ratio at 1:4 (IC 410, 2026-10-28).
    rng = np.random.default_rng(5)
    sub = _planes()
    stars = np.zeros_like(sub["G"])
    yy, xx = np.mgrid[0:120, 0:160]
    for _ in range(60):
        y, x = rng.integers(5, 115), rng.integers(5, 155)
        stars += 300.0 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 4.0)
    sub = {c: v + stars for c, v in sub.items()}
    assert np.nanpercentile(sub["R"], 99.5) > 0.5 * np.nanpercentile(sub["G"], 99.5)  # pixels: stars
    _, factors = cp.apply_palette_gain(sub, "hubble")
    assert factors["R"] == pytest.approx(4.0, rel=0.15)        # blocks: the nebula
    assert factors["B"] == pytest.approx(2.5, rel=0.15)


def test_explicit_factor_scales_only_its_role():
    sub = _planes()
    out, factors = cp.apply_palette_gain(sub, {"G": 0.4}, protect_stars=False)
    assert factors == {"G": 0.4}
    assert np.allclose(out["G"], sub["G"] * 0.4)
    assert out["R"] is sub["R"] and out["B"] is sub["B"]


def _starry(seed=5, n=25, peak=300.0):
    rng = np.random.default_rng(seed)
    sub = _planes()
    stars = np.zeros_like(sub["G"])
    yy, xx = np.mgrid[0:120, 0:160]
    pos = []
    for _ in range(n):
        y, x = int(rng.integers(8, 112)), int(rng.integers(8, 152))
        stars += peak * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 3.0)
        pos.append((y, x))
    return {c: v + stars for c, v in sub.items()}, pos


def test_stars_are_neutral_under_the_gain():
    # Scaling R and B turned every star on IC 410 magenta; put back "as
    # imaged" they stayed magenta, because a star is fainter through Ha
    # (its own absorption line) than through S-II and O-III. "neutral":
    # the star light averaged over the planes, the same in every channel.
    pytest.importorskip("sep")
    pytest.importorskip("scipy")
    sub, pos = _starry(n=3, peak=30.0)   # few and faint enough that their discs leave bare nebula
    sub["G"] = sub["G"] * 0.7            # the Ha plane: a fainter star AND a fainter nebula
    out, factors = cp.apply_palette_gain(sub, {"R": 4.0, "B": 2.0})
    y, x = pos[0]
    mask, _ = cp.star_mask(sub["R"] + sub["G"] + sub["B"],
                           thresh_sigma=cp.PALETTE_STAR_THRESH_SIGMA,
                           radius_a=cp.PALETTE_STAR_RADIUS_A,
                           max_radius=cp.PALETTE_STAR_MAX_RADIUS_PX,
                           min_peak_flux=cp.PALETTE_STAR_MIN_PEAK_FLUX, radius_from="peak")
    assert mask[y, x] == 1.0
    star = {c: float(out[c][y, x] - cp.smooth_fill(sub[c], mask=mask)[y, x] * factors.get(c, 1.0))
            for c in out}
    assert star["R"] == pytest.approx(star["G"], rel=1e-3)      # the same light in every channel
    assert star["B"] == pytest.approx(star["G"], rel=1e-3)
    imaged, _ = cp.apply_palette_gain(sub, {"R": 4.0, "B": 2.0}, star_mode="imaged")
    si = {c: float(imaged[c][y, x] - cp.smooth_fill(sub[c], mask=mask)[y, x] * factors.get(c, 1.0))
          for c in imaged}
    assert si["G"] == pytest.approx(0.7 * si["R"], rel=0.05)    # as imaged keeps the Ha deficit
    # where the mask is zero the nebula took the full gain
    mask, _ = cp.star_mask(sub["R"] + sub["G"] + sub["B"],
                           thresh_sigma=cp.PALETTE_STAR_THRESH_SIGMA,
                           radius_a=cp.PALETTE_STAR_RADIUS_A,
                           max_radius=cp.PALETTE_STAR_MAX_RADIUS_PX,
                           min_peak_flux=cp.PALETTE_STAR_MIN_PEAK_FLUX, radius_from="peak")
    far = (mask < 1e-3)                  # the feather's tail: at most 3e-3 of the residual leaks
    assert far.any()
    assert np.allclose(out["R"][far], sub["R"][far] * 4.0, rtol=0.02, atol=0.1)


def test_star_discs_are_found_once_and_pinned_on_a_series():
    # On a denoised plane the noise is a fraction of the raw's, a bright
    # star's 3-sigma isophote outgrows the size limit, and the star falls
    # out of a mask found there; the raw set's discs are used instead.
    pytest.importorskip("sep")
    pytest.importorskip("scipy")
    sub, pos = _starry(n=6)
    chans = {c: v + 100.0 for c, v in sub.items()}
    info = {}
    cp._prepare(chans, False, 4, 99.9, ha_gain=None, info=info, palette_gain={"R": 4.0})
    discs = info["palette_stars"]
    assert len(discs) >= 5 and all(len(d) == 3 for d in discs)
    quiet = {c: v * 0.5 + 100.0 for c, v in sub.items()}          # a "denoised" twin
    info2 = {}
    out, _ = cp._prepare(quiet, False, 4, 99.9, ha_gain=None, info=info2,
                         palette_gain={"R": 4.0}, palette_stars=discs)
    assert info2["palette_stars"] == discs
    m1 = cp.mask_from_discs(sub["R"].shape, discs)
    m2, _ = cp.star_mask(sub["R"] + sub["G"] + sub["B"], thresh_sigma=3.0,
                         max_npix=cp.PALETTE_STAR_MAX_NPIX, radius_a=5.0, max_radius=90.0,
                         min_peak_flux=cp.PALETTE_STAR_MIN_PEAK_FLUX, radius_from="peak")
    assert np.allclose(m1, m2)


def test_a_nebula_blob_is_not_a_star():
    pytest.importorskip("sep")
    sub = _planes()                                               # one 25x40 px blob, no stars
    _, info = cp.star_mask(sub["R"] + sub["G"] + sub["B"], thresh_sigma=3.0,
                           max_npix=cp.PALETTE_STAR_MAX_NPIX, radius_a=5.0, max_radius=90.0,
                           min_peak_flux=cp.PALETTE_STAR_MIN_PEAK_FLUX)
    assert info["stars"] == 0
    out, _ = cp.apply_palette_gain(sub, "hubble")                 # and the gain is untouched by it
    assert np.isfinite(out["R"]).all()


def test_peak_sized_discs_do_not_grow_with_the_nebula_under_the_star():
    pytest.importorskip("sep")
    sub, pos = _starry(n=6)
    _, iso = cp.star_mask(sub["R"] + sub["G"] + sub["B"], thresh_sigma=3.0,
                          max_npix=cp.PALETTE_STAR_MAX_NPIX, radius_a=5.0, max_radius=90.0,
                          min_peak_flux=cp.PALETTE_STAR_MIN_PEAK_FLUX)
    _, pk = cp.star_mask(sub["R"] + sub["G"] + sub["B"], thresh_sigma=3.0,
                         max_npix=cp.PALETTE_STAR_MAX_NPIX, radius_a=5.0, max_radius=90.0,
                         min_peak_flux=cp.PALETTE_STAR_MIN_PEAK_FLUX, radius_from="peak")
    assert max(r for _, _, r in iso["discs"]) >= 60                # the star on the blob: the blob's size
    assert max(r for _, _, r in pk["discs"]) < 45                  # by peak: a star's size
    assert len(pk["discs"]) == len(iso["discs"])


def test_smooth_fill_ignores_stars_and_keeps_the_nebula():
    pytest.importorskip("scipy")
    sub, pos = _starry()
    fill = cp.smooth_fill(sub["G"])
    y, x = pos[0]
    assert fill[y, x] < 0.25 * sub["G"][y, x]
    assert fill[60, 80] == pytest.approx(20.0, rel=0.2)


def test_prepare_measures_once_and_a_series_pins_it():
    chans = {c: v + 100.0 for c, v in _planes().items()}      # with a sky
    info = {}
    subbed, white = cp._prepare(chans, False, 4, 99.9, ha_gain=None, info=info,
                                palette_gain="hubble")
    f = info["palette_gain"]
    assert f["G"] == pytest.approx(1.0) and f["R"] > 3 and f["B"] > 2
    # A second set (the denoised one) given the factors applies them as is.
    chans2 = {c: v * 0.5 + 100.0 for c, v in _planes(seed=4).items()}
    info2 = {}
    cp._prepare(chans2, False, 4, 99.9, ha_gain=None, info=info2, palette_gain=f)
    assert info2["palette_gain"] == f
    # Without a gain nothing is scaled and info says so.
    info3 = {}
    plain, _ = cp._prepare(chans, False, 4, 99.9, ha_gain=None, info=info3)
    assert info3["palette_gain"] is None
    assert np.allclose(plain["R"], chans["R"] - np.nanmedian(chans["R"]))


def test_white_point_is_taken_before_the_gain():
    # The p99.95 white is the stars; a filter scaled up carries its stars up,
    # and a white taken after the gain rose with it and darkened the whole
    # picture (IC 410: 124 -> 266 ADU). The gain is a tone, not an exposure.
    chans = {c: v + 100.0 for c, v in _planes().items()}
    _, w0 = cp._prepare(chans, False, 4, 99.9, ha_gain=None)
    _, w1 = cp._prepare(chans, False, 4, 99.9, ha_gain=None, palette_gain={"R": 4.0})
    assert w1 == pytest.approx(w0)


def test_hubble_caps_a_filter_that_has_no_nebula():
    sub = _planes()
    sub["R"] = np.random.default_rng(7).normal(0, 0.3, sub["R"].shape).astype(np.float32)  # noise only
    _, factors = cp.apply_palette_gain(sub, "hubble")
    assert factors["R"] <= cp.PALETTE_GAIN_MAX


def test_sky_black_offset_ignores_the_gain():
    # The shared black offset is 0.5 x the largest channel sigma; a channel
    # scaled x4 would otherwise set it four times larger and lift the sky
    # from 9 % to 30 % grey (IC 410, S-II x3.9).
    pytest.importorskip("sep")
    rng = np.random.default_rng(11)
    sub = {c: rng.normal(0, 1.0, (160, 160)).astype(np.float32) for c in ("R", "G", "B")}
    plain = cp.sky_anchored_blacks(sub, 0.5, 40.0)
    gained, factors = cp.apply_palette_gain(sub, {"R": 4.0})
    naive = cp.sky_anchored_blacks(gained, 0.5, 40.0)
    fixed = cp.sky_anchored_blacks(gained, 0.5, 40.0, gains=factors)
    assert naive["G"] < 3 * plain["G"]                           # the x4 noise drove the offset
    assert fixed["G"] == pytest.approx(plain["G"], abs=0.05)     # pre-gain sigma: unchanged


def test_compose_with_gain_turns_the_green_region_gold():
    chans = {c: v + 100.0 for c, v in _planes().items()}
    plain = cp.compose(chans, deblur=0, subtract_background=False, ha_gain=None)
    gold = cp.compose(chans, deblur=0, subtract_background=False, ha_gain=None,
                      palette_gain="hubble")
    for y, x in ((60, 80), (45, 60)):                           # the peak and the flank
        r0, g0, b0 = plain[y, x]
        r1, g1, b1 = gold[y, x]
        assert g0 > r0 * 1.3 and g0 > b0 * 1.15                 # true ratio: green
        assert abs(r1 - g1) < 0.05 and abs(b1 - g1) < 0.05      # equalised: neutral


def test_options_spelling():
    assert "gain" not in cp.describe_options(cp.effective_options())
    assert "gain=hubble" in cp.describe_options(cp.effective_options(palette_gain="hubble"))
    assert "gain=G:0.4" in cp.describe_options(cp.effective_options(palette_gain={"G": 0.4}))
    assert cp.describe_palette_gain({"Ha": 0.4, "O-III": 1.5}) == "Ha:0.4/O-III:1.5"
