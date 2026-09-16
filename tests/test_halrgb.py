"""HALRGB: LRGB plus the Ha EXCESS folded into R and L.

The trap this recipe exists to avoid is blending the raw Ha stack: an Ha
filter still passes ~5% of the continuum, so raw Ha into R reddens every
star and the whole galaxy disk rather than the HII regions. These tests build
a field with continuum stars (present in R and, at the continuum ratio, in
Ha) and one emission knot (present in Ha only) and pin: the ratio is
recovered from the stars; the excess map holds the knot and not the stars;
the blend raises R and L at the knot only; gain 0 is exactly LRGB.

Skips without numpy (the CI interpreter is pytest-only); the Spark
recipe-picker rule for HALRGB lives with the other picker tests.
"""
import pytest

np = pytest.importorskip("numpy")

from stacking import color_process as cp  # noqa: E402


def test_recipe_table_names_ha_as_a_fifth_role():
    m = cp.RECIPES["HALRGB"]
    assert m["HA"] == "Ha"
    assert {m[c] for c in ("R", "G", "B", "L")} == {"R", "G", "B", "L"}
    # The N2N/movie/process loops all enumerate roles explicitly; a new role
    # that is not one of these would be silently dropped by every one of them.
    assert set(m) <= {"R", "G", "B", "L", "HA"}


def _field(ratio=0.05, knot=40.0, seed=1):
    """R with stars; Ha = ratio*R plus one emission knot; L = R-ish."""
    rng = np.random.default_rng(seed)
    h = w = 96
    yy, xx = np.mgrid[:h, :w]
    r = np.zeros((h, w), np.float32)
    for cy, cx, amp in ((20, 20, 300.0), (70, 30, 500.0), (40, 75, 400.0),
                        (80, 80, 250.0), (15, 60, 350.0)):
        r += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 4.0)
    ha = ratio * r
    knot_mask = ((yy - 50) ** 2 + (xx - 48) ** 2) < 9
    ha = ha + knot * knot_mask
    l = r * 1.2
    g = r * 0.9
    b = r * 0.8
    noise = lambda: rng.normal(0.0, 0.3, (h, w)).astype(np.float32)
    return ({"R": r + noise(), "G": g + noise(), "B": b + noise(),
             "L": l + noise(), "HA": ha.astype(np.float32) + noise()},
            knot_mask)


def test_continuum_ratio_is_recovered_from_the_stars():
    ch, _ = _field(ratio=0.05)
    got = cp.ha_continuum_ratio(ch["HA"], ch["R"], min_pix=20)
    assert got is not None
    assert abs(got - 0.05) < 0.01


def test_excess_holds_the_knot_and_not_the_stars():
    ch, knot = _field(ratio=0.05, knot=40.0)
    out, ratio = cp.apply_ha_blend(ch, gain=1.0)
    ex = out["HA_EXCESS"]
    assert ex[knot].mean() > 30.0
    # Star cores: the continuum share is subtracted, so the excess there is
    # noise-sized, not 5% of a 500 ADU star (= 25 ADU).
    stars = ch["R"] > 100.0
    assert not knot[stars].any()
    assert abs(ex[stars].mean()) < 2.0


def test_blend_raises_r_and_l_at_the_knot_only():
    ch, knot = _field()
    out, _ = cp.apply_ha_blend(ch, gain=1.0)
    for c in ("R", "L"):
        delta = out[c] - ch[c]
        assert delta[knot].mean() > 30.0
        assert abs(delta[~knot].mean()) < 1.0
    for c in ("G", "B", "HA"):
        assert np.array_equal(out[c], ch[c])


def test_gain_zero_is_plain_lrgb():
    ch, _ = _field()
    out, _ = cp.apply_ha_blend(ch, gain=0.0)
    for c in ("R", "G", "B", "L"):
        assert np.array_equal(out[c], ch[c])
    assert "HA_EXCESS" in out          # still exported for inspection
    lrgb = {c: ch[c] for c in ("R", "G", "B", "L")}
    a = cp.compose(ch, subtract_background=False, ha_gain=0.0)
    b = cp.compose(lrgb, subtract_background=False)
    assert np.allclose(a, b)


def test_pinned_ratio_is_used_as_given():
    ch, knot = _field(ratio=0.05)
    out, used = cp.apply_ha_blend(ch, gain=1.0, ratio=0.5)
    assert used == 0.5
    # With the ratio pinned far too high, stars subtract to nothing and only
    # the knot (which has no continuum) survives.
    assert out["HA_EXCESS"][knot].mean() > 30.0


def test_without_ha_the_channels_pass_through():
    ch, _ = _field()
    lrgb = {c: ch[c] for c in ("R", "G", "B", "L")}
    out, ratio = cp.apply_ha_blend(lrgb, gain=1.0)
    assert ratio is None and out is lrgb


def test_compose_makes_the_knot_redder_and_brighter():
    ch, knot = _field()
    plain = cp.compose({c: ch[c] for c in ("R", "G", "B", "L")},
                       subtract_background=False)
    ha = cp.compose(ch, subtract_background=False, ha_gain=1.0)
    lum = lambda rgb: rgb @ np.array([0.299, 0.587, 0.114])
    assert lum(ha)[knot].mean() > lum(plain)[knot].mean()
    red_share = lambda rgb: rgb[..., 0] / np.maximum(rgb.sum(-1), 1e-6)
    assert red_share(ha)[knot].mean() > red_share(plain)[knot].mean()


def test_describe_options_only_names_a_non_default_gain():
    assert "ha=" not in cp.describe_options(cp.effective_options())
    assert "ha=1.5" in cp.describe_options(cp.effective_options(ha_gain=1.5))


def test_excess_is_zero_on_sky_noise():
    # No knot at all: the map is pure noise, and one-sided clipping would keep
    # half of it. The floor keeps almost nothing.
    ch, _ = _field(knot=0.0)
    out, _ = cp.apply_ha_blend(ch, gain=1.0)
    ex = out["HA_EXCESS"]
    assert float(np.mean(ex > 0)) < 0.05
    assert abs(float((out["R"] - ch["R"]).mean())) < 0.05


def test_black_point_pickers_skip_the_excess_plane():
    ch, _ = _field()
    out, _ = cp.apply_ha_blend(ch, gain=1.0)
    auto = cp.auto_stretch(out, white=100.0, lum="L")
    assert "HA_EXCESS" not in auto["blacks"]
    assert {"R", "G", "B", "L", "HA"} <= set(auto["blacks"])
    pytest.importorskip("sep")
    anchored = cp.sky_anchored_blacks(out, black_sigma=1.0)
    assert "HA_EXCESS" not in anchored and "R" in anchored


def test_n2n_render_picks_the_model_per_filter():
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "scripts"))
    import n2n_lrgb_render as n
    assert n.domain_for("Ha") == "narrowband"
    assert all(n.domain_for(f) == "broadband" for f in ("L", "R", "G", "B"))
