"""tone=hubble: the finish on a composed palette, and the gain's noise knee
(2026-11-18). The owner put another imager's IC 405 beside the morning
SHO: ours was green on a purple sky at full saturation, theirs gold on a
quiet grey. The data were not that different; the finish was. These tests
pin the spelling, that the tone touches nothing unless asked, what each
step does on a synthetic field (green removed, yellow-green pulled to gold,
grey left grey, the sky lifted to the target and no further), that a colour
recipe refuses it, that the knee leaves the smooth sky's scatter at its
native amplitude while the nebula takes the gain, and that a series pins
the knee in ADU.

Skips without numpy (the CI interpreter is pytest-only).
"""
import pytest

np = pytest.importorskip("numpy")

from stacking import color_process as cp  # noqa: E402


def test_parse_spellings():
    assert cp.parse_tone("") is None and cp.parse_tone(None) is None
    assert cp.parse_tone("none") is None and cp.parse_tone("off") is None
    assert cp.parse_tone("hubble") == "hubble" and cp.parse_tone("HUBBLE") == "hubble"
    with pytest.raises(ValueError):
        cp.parse_tone("warm")
    assert cp.tone_knee(None) == cp.PALETTE_GAIN_KNEE_SIGMA
    assert cp.tone_knee("hubble") == cp.TONE_HUBBLE["knee_sigma"] > 0
    assert cp.tone_saturation(None, 2.5) == 2.5
    assert cp.tone_saturation("hubble", 2.5) == cp.TONE_HUBBLE["saturation"]


def test_no_tone_changes_nothing():
    rng = np.random.default_rng(1)
    rgb = rng.uniform(0, 1, (20, 30, 3)).astype(np.float32)
    out = cp.apply_tone(rgb, None)
    assert out is rgb


def test_hue_pull_moves_green_to_gold_and_leaves_red_blue_grey():
    rgb = np.array([[[0.2, 0.8, 0.2],      # green
                     [0.8, 0.2, 0.2],      # red
                     [0.2, 0.2, 0.8],      # blue
                     [0.5, 0.5, 0.5]]], dtype=np.float32)   # grey
    out = cp.hue_pull(rgb)
    g, r, b, grey = out[0]
    assert g[0] > g[2] and g[1] > g[2]            # green went warm: red up, still no blue
    assert g[0] > 0.5                              # well toward gold, not still green
    assert np.allclose(r, rgb[0, 1], atol=1e-6)
    assert np.allclose(b, rgb[0, 2], atol=1e-6)
    assert np.allclose(grey, rgb[0, 3], atol=1e-6)
    # value and chroma are kept: brightness does not change with the hue
    assert out.max(axis=2) == pytest.approx(rgb.max(axis=2), abs=1e-5)
    assert (out.max(axis=2) - out.min(axis=2)) == pytest.approx(rgb.max(axis=2) - rgb.min(axis=2), abs=1e-5)


def test_lift_sky_puts_the_sky_percentile_at_the_target_and_never_lowers():
    rng = np.random.default_rng(2)
    rgb = np.clip(rng.normal(0.03, 0.01, (60, 80, 3)), 0, 1).astype(np.float32)
    rgb[..., 0] += 0.02                                   # a red-tinted sky
    out, rec = cp.lift_sky(rgb, 0.12)
    for i in range(3):                                    # every channel's sky at the target: neutral
        assert np.percentile(out[..., i], cp.TONE_SKY_PCT) == pytest.approx(0.12, abs=0.01)
    assert all(p > 0 for p in rec["pedestal"]) and rec["pedestal"][0] < rec["pedestal"][1]
    bright = np.clip(rng.normal(0.3, 0.01, (60, 80, 3)), 0, 1).astype(np.float32)
    same, rec2 = cp.lift_sky(bright, 0.12)
    assert rec2["pedestal"] == [0.0, 0.0, 0.0] and np.allclose(same, bright)


def test_apply_tone_takes_the_green_out_and_records_what_it_did():
    rgb = np.zeros((40, 40, 3), dtype=np.float32)
    rgb[..., 1] = 0.6                                   # an Ha-only field: pure green
    rgb[..., 0] = 0.15
    rgb[..., 2] = 0.1
    info = {}
    out = cp.apply_tone(rgb, "hubble", info=info)
    r, g, b = out[20, 20]
    assert g <= max(r, b) + 1e-6                        # no longer green-dominant
    assert r > b                                        # warm: gold, not teal
    assert info["tone"]["tone"] == "hubble" and "pedestal" in info["tone"]
    # the midtone lift: a dim field's median luminance is brought to the target, a bright one is left
    dim = np.full((40, 40, 3), 0.1, dtype=np.float32)
    d_info = {}
    lifted = cp.apply_tone(dim, "hubble", info=d_info)
    assert d_info["tone"]["gamma"] < 1.0
    assert float(np.median(lifted @ np.array([0.299, 0.587, 0.114], dtype=np.float32))) == pytest.approx(cp.TONE_HUBBLE["mid"], abs=0.01)
    bright = np.full((40, 40, 3), 0.5, dtype=np.float32)
    b_info = {}
    cp.apply_tone(bright, "hubble", info=b_info)
    assert b_info["tone"]["gamma"] == 1.0


def test_options_spelling_and_colour_recipes_refuse():
    assert "tone" not in cp.describe_options(cp.effective_options())
    assert "tone=hubble" in cp.describe_options(cp.effective_options(tone="hubble"))
    chans = {c: np.random.default_rng(3).normal(100, 1, (40, 40)).astype(np.float32) for c in ("L", "R", "G", "B")}
    with pytest.raises(ValueError):
        cp.compose(chans, deblur=0, subtract_background=False, ha_gain=None, tone="hubble")


def _planes(seed=3):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:160, 0:200]
    neb = np.exp(-(((yy - 80) / 20.0) ** 2 + ((xx - 100) / 30.0) ** 2))
    ha = 20.0 * neb + rng.normal(0, 0.3, neb.shape)
    s2 = 5.0 * neb + rng.normal(0, 0.3, neb.shape)
    return {"R": s2.astype(np.float32), "G": ha.astype(np.float32), "B": (8.0 * neb + rng.normal(0, 0.3, neb.shape)).astype(np.float32)}


def _smooth_sky_scatter(plane):
    sm = plane - cp.fine_residual(plane, cp.PALETTE_GAIN_SMOOTH_PX)
    sky = sm[:30, :]                                     # far from the nebula
    return 1.4826 * float(np.median(np.abs(sky - np.median(sky))))


def test_knee_leaves_the_smooth_sky_alone_and_the_nebula_takes_the_gain():
    sub = _planes()
    plain, f0 = cp.apply_palette_gain(sub, {"R": 4.0}, protect_stars=False)
    info = {}
    kneed, f1 = cp.apply_palette_gain(sub, {"R": 4.0}, protect_stars=False, knee=4.0, info=info)
    assert f0 == f1 == {"R": 4.0}
    s_in = _smooth_sky_scatter(sub["R"])
    assert _smooth_sky_scatter(plain["R"]) > 2.5 * s_in     # today's gain scales it (x4 before the re-split)
    assert _smooth_sky_scatter(kneed["R"]) < 1.5 * s_in     # the knee does not
    peak = kneed["R"][80, 100]
    assert peak == pytest.approx(plain["R"][80, 100], rel=0.05)                     # the nebula: the whole gain
    # the knee is in ADU at the 16 px scale, from the sky side of the
    # distribution: positive, and far under the nebula it must not mute
    assert 0 < info["palette_knee"]["R"] < 0.5 * cp.nebula_level(sub["R"])
    assert "G" not in info["palette_knee"]                                          # unscaled plane: no knee


def test_knee_off_is_bit_for_bit_the_old_gain():
    sub = _planes()
    a, _ = cp.apply_palette_gain(sub, {"R": 4.0, "B": 2.0}, protect_stars=False, knee=0.0)
    b, _ = cp.apply_palette_gain(sub, {"R": 4.0, "B": 2.0}, protect_stars=False, knee=None)
    assert np.array_equal(a["R"], b["R"]) and np.array_equal(a["B"], b["B"])


def test_a_series_pins_the_knee_in_adu():
    sub = _planes()
    info = {}
    cp.apply_palette_gain(sub, {"R": 4.0}, protect_stars=False, knee=2.0, info=info)
    quiet = {c: v * 0.1 for c, v in sub.items()}          # a "denoised" set: a tenth of the scatter
    own = {}
    cp.apply_palette_gain(quiet, {"R": 4.0}, protect_stars=False, knee=2.0, info=own)
    pinned = {}
    cp.apply_palette_gain(quiet, {"R": 4.0}, protect_stars=False, knee=info["palette_knee"], info=pinned)
    assert own["palette_knee"]["R"] < 0.2 * info["palette_knee"]["R"]
    assert pinned["palette_knee"]["R"] == info["palette_knee"]["R"]
    # and through _prepare
    chans = {c: v + 100.0 for c, v in sub.items()}
    p_info = {}
    cp._prepare(chans, False, 4, 99.9, ha_gain=None, palette_gain={"R": 4.0}, palette_knee=2.0, info=p_info)
    assert p_info["palette_knee"]["R"] > 0
    q_info = {}
    cp._prepare(chans, False, 4, 99.9, ha_gain=None, palette_gain={"R": 4.0},
                palette_knee=p_info["palette_knee"], info=q_info)
    assert q_info["palette_knee"] == p_info["palette_knee"]
