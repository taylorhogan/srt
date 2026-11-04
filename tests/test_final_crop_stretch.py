"""The crop is cut on the routine stretch, never the final's (2026-11-04).

The first final render of Melotte 15 (auto_stretch_v2) set its white at
12 ADU, against the routine's 114, and the crop pushed beside it was a
saturated picture of a field the routine renders had shown right for a
week. n2n_lrgb_render.compose_rgb draws a set on whatever stretch it is
given, so the routine's and the final's come from the same planes; the
routine passes the routine stretch for the crop in every mode.
"""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("PIL")

from scripts import n2n_lrgb_render as R  # noqa: E402


def _set(seed=0, n=64):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:n, :n]
    neb = 30.0 * np.exp(-(((x - 32) ** 2 + (y - 32) ** 2) / (2 * 10.0 ** 2)))
    return {c: (neb * g + rng.normal(0, 0.5, (n, n))).astype(np.float32)
            for c, g in (("R", 0.6), ("G", 1.0), ("B", 0.4))}


def test_stretch_plane_is_linear_then_arcsinh_between_black_and_white():
    chan = np.array([-1.0, 0.0, 50.0, 100.0, 200.0], dtype=np.float32)
    out = R.stretch_plane(chan, 0.0, 100.0, 1.0)
    assert out[0] == 0.0 and out[1] == 0.0
    assert out[3] == pytest.approx(1.0, abs=1e-6) and out[4] == pytest.approx(1.0, abs=1e-6)
    assert 0.0 < out[2] < 1.0
    # softening pulls the middle up
    assert R.stretch_plane(chan, 0.0, 100.0, 0.05)[2] > out[2]


def test_compose_rgb_is_the_planes_on_the_stretch_it_is_given():
    s = _set()
    blacks = {"R": 0.0, "G": 0.0, "B": 0.0}
    rgb = R.compose_rgb(s, blacks, 30.0, 0.1)
    assert rgb.shape == (64, 64, 3) and rgb.min() >= 0.0 and rgb.max() <= 1.0
    np.testing.assert_allclose(rgb[..., 1], R.stretch_plane(s["G"], 0.0, 30.0, 0.1), atol=1e-6)


def test_a_white_far_below_the_routines_saturates_the_field():
    # the Melotte 15 final: white 12 ADU against the routine's 114
    s = _set()
    blacks = {"R": 0.0, "G": 0.0, "B": 0.0}
    routine = R.compose_rgb(s, blacks, 114.0, 0.025)
    final = R.compose_rgb(s, blacks, 12.0, 0.01)
    assert (final >= 0.999).mean() > 10 * max((routine >= 0.999).mean(), 1e-4)
    assert np.median(final) > 3 * np.median(routine)


def test_compose_rgb_is_lit_by_l_and_a_synthetic_pair_the_same_way():
    s = _set()
    blacks = {"R": 0.0, "G": 0.0, "B": 0.0, "L": 0.0}
    with_l = R.compose_rgb({**s, "L": s["G"] * 1.5}, blacks, 30.0, 0.1)
    plain = R.compose_rgb(s, blacks, 30.0, 0.1)
    assert with_l.mean() > plain.mean()
    syn = R.synthetic_black({"R": 1.0, "G": 2.0, "B": 3.0})
    assert syn["LSYN"] == syn["LSYN_REF"] == pytest.approx(0.299 * 1 + 0.587 * 2 + 0.114 * 3, rel=1e-6)
    ref = s["G"].copy()
    lit = R.compose_rgb({**s, "LSYN": ref * 1.2, "LSYN_REF": ref},
                        {**blacks, "LSYN": 0.0, "LSYN_REF": 0.0}, 30.0, 0.1)
    assert lit.mean() >= plain.mean()


def test_saturation_scales_chroma_about_luminance():
    s = _set()
    blacks = {"R": 0.0, "G": 0.0, "B": 0.0}
    plain = R.compose_rgb(s, blacks, 30.0, 0.1)
    vivid = R.compose_rgb(s, blacks, 30.0, 0.1, saturation=2.0)
    y = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    # the luminance holds wherever the scaled chroma did not clip
    inside = (vivid > 0.0).all(axis=2) & (vivid < 1.0).all(axis=2)
    assert inside.mean() > 0.3
    np.testing.assert_allclose((vivid @ y)[inside], (plain @ y)[inside], atol=1e-5)
    assert (vivid - (vivid @ y)[..., None]).std() > (plain - (plain @ y)[..., None]).std()


def test_the_routine_cuts_the_crop_on_the_routine_stretch_in_final_mode():
    """The source reads: in final mode the crop's source is compose_rgb on the
    routine stretch, and the meta's auto_crop[tag].stretch says 'routine'."""
    import inspect
    src = inspect.getsource(R.routine)
    i = src.index("if final_stretch is not None:\n", src.index("def emit("))
    block = src[i:i + 600]
    assert 'compose_rgb(subbed, routine["blacks"], routine["white"], routine["soft"]' in block
    assert 'rec["stretch"] = "routine"' in src
