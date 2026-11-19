"""stacking/color_process.match_colour_widths (2026-11-19): a colour recipe's R, G
and B are brought to one star width before the balance and the stretch, so a
star's colour does not change with radius when one filter was shot broader
than the others (IC 405: B at 5.0 px half-width against R and G at 4.0 gave
yellow cores in cyan halos; the registration was within 0.1 px). Synthetic
fields: Gaussian stars of known width on a flat sky, no scipy beyond ndimage."""
import pytest

pytest.importorskip("numpy")          # the CI interpreter is pytest-only
pytest.importorskip("scipy")

import numpy as np  # noqa: E402

from stacking import color_process as cp  # noqa: E402

HALF = float(np.sqrt(2.0 * np.log(2.0)))     # half-max radius of a Gaussian in sigmas


def _field(sigma, seed=0, n=60, shape=(600, 600), peak=3000.0, noise=3.0):
    rng = np.random.default_rng(seed)
    img = np.zeros(shape, dtype=np.float32)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    pos = np.stack([rng.uniform(40, shape[0] - 40, n), rng.uniform(40, shape[1] - 40, n)], axis=1)
    for y, x in pos:
        img += peak * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2))
    return img + rng.normal(0, noise, shape).astype(np.float32), pos


def _planes(sig_r=2.0, sig_g=2.0, sig_b=3.0, sig_l=2.0):
    seed = 7
    L, _ = _field(sig_l, seed)
    R, _ = _field(sig_r, seed, peak=2000.0)
    G, _ = _field(sig_g, seed, peak=2400.0)
    B, _ = _field(sig_b, seed, peak=1500.0)
    return {"L": L, "R": R, "G": G, "B": B}


def test_half_widths_read_the_planes_own_gaussians():
    pl = _planes()
    widths, n = cp.star_half_widths(pl, "L")
    assert n >= cp.CHROMA_MATCH_MIN_STARS
    assert widths["R"] == pytest.approx(2.0 * HALF, abs=0.25)
    assert widths["B"] == pytest.approx(3.0 * HALF, abs=0.3)
    assert widths["L"] == pytest.approx(2.0 * HALF, abs=0.25)


def test_the_sharper_planes_are_blurred_to_the_broadest_and_l_is_untouched():
    pl = _planes()
    info = {}
    out = cp.match_colour_widths(pl, info=info)
    cm = info["chroma_match"]
    assert cm["reference"] == "L" and cm["stars"] >= cp.CHROMA_MATCH_MIN_STARS
    want = float(np.sqrt(3.0 ** 2 - 2.0 ** 2))        # 2.24 px takes a 2 px star to 3
    assert cm["sigmas"]["R"] == pytest.approx(want, abs=0.3)
    assert cm["sigmas"]["G"] == pytest.approx(want, abs=0.3)
    assert cm["sigmas"]["B"] == 0.0
    after, _ = cp.star_half_widths({c: out[c] for c in "RGB"}, "G")
    assert after["R"] == pytest.approx(after["B"], abs=0.2)
    assert after["G"] == pytest.approx(after["B"], abs=0.2)
    assert out["L"] is pl["L"] and out["B"] is pl["B"]                 # not blurred: the same arrays
    # flux is conserved by the blur: a star's total light is what it was
    assert float(out["R"].sum()) == pytest.approx(float(pl["R"].sum()), rel=1e-3)


def test_matched_planes_need_nothing_and_pinned_sigmas_reproduce():
    pl = _planes(sig_b=2.0)
    info = {}
    out = cp.match_colour_widths(pl, info=info)
    assert all(v == 0.0 for v in info["chroma_match"]["sigmas"].values())
    assert all(out[c] is pl[c] for c in "LRGB")
    pl2 = _planes()
    info2 = {}
    first = cp.match_colour_widths(pl2, info=info2)
    again = cp.match_colour_widths(pl2, sigmas=info2["chroma_match"]["sigmas"])
    assert np.array_equal(first["R"], again["R"]) and np.array_equal(first["B"], again["B"])


def test_a_wild_measurement_is_capped_and_a_short_set_is_left_alone():
    pl = _planes(sig_b=7.0)                                 # would ask sigma 6.7 of R and G
    info = {}
    cp.match_colour_widths(pl, info=info)
    cm = info["chroma_match"]
    assert cm["sigmas"]["R"] == cp.CHROMA_MATCH_MAX_SIGMA and "R" in cm["capped"]
    info = {}
    out = cp.match_colour_widths({"R": pl["R"], "G": pl["G"]}, info=info)
    assert info["chroma_match"]["skipped"].startswith("needs R, G and B") and out["R"] is pl["R"]
    flat = {c: np.random.default_rng(1).normal(0, 3, (300, 300)).astype(np.float32) for c in "LRGB"}
    info = {}
    cp.match_colour_widths(flat, info=info)
    assert "skipped" in info["chroma_match"]


def test_prepare_measures_on_request_pins_on_request_and_records_it():
    pl = _planes()
    info = {}
    subbed, _white = cp._prepare(pl, False, 4, 99.95, ha_gain=None, info=info, chroma_match=True)
    cm = info["chroma_match"]
    assert cm["sigmas"]["R"] > 1.5 and cm["sigmas"]["B"] == 0.0
    info2 = {}
    pinned, _ = cp._prepare(pl, False, 4, 99.95, ha_gain=None, info=info2, chroma_sigmas=cm["sigmas"])
    assert np.allclose(pinned["R"], subbed["R"]) and info2["chroma_match"] is None
    info3 = {}
    plain, _ = cp._prepare(pl, False, 4, 99.95, ha_gain=None, info=info3)
    assert info3["chroma_match"] is None
    assert not np.allclose(plain["R"], subbed["R"])


def test_shared_from_raw_carries_the_sigmas():
    from scripts.n2n_lrgb_render import shared_from_raw
    assert shared_from_raw({"chroma_match": {"sigmas": {"R": 1.9, "G": 1.9, "B": 0.0}}})["chroma_sigmas"] == \
        {"R": 1.9, "G": 1.9, "B": 0.0}
    assert shared_from_raw({})["chroma_sigmas"] is None
