"""The learned deconvolver's data path and head: nn/deblur_data, nn/deblur_model.

Phase 1 of docs/LEARNED_DEBLUR_PLAN.md. What these pin down:

- the scatter head conserves flux by construction and starts at identity;
- a prepared stack round-trips through its sidecar, and a training pair has
  the right shapes, a mask that drops plateaus and coverage fill, and a
  target that is the un-blurred patch;
- tiled inference on an identity model returns the input (plateaus and all),
  so the stitching cannot be the source of any later artefact.
"""
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("skimage")
pytest.importorskip("sep")
pytest.importorskip("torch")   # the Spark has it; iris-pc's .venv and CI do not

from nn import deblur_data as dd  # noqa: E402

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "scripts"))


def _kernel(fwhm=7.5, size=65):
    yy, xx = np.mgrid[:size, :size] - size // 2
    s = fwhm / 2.3548
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * s * s)) + 0.02 * np.exp(-np.hypot(xx, yy) / 12)
    return k / k.sum()


def _field(seed=0, size=900, sky=500.0, sig=3.0, sat=60000.0):
    """A synthetic stack with its sky in: stars over four decades, a galaxy,
    one clipped star, a strip of coverage fill (the stacker's exact-median
    signature)."""
    rng = np.random.default_rng(seed)
    k = _kernel()
    S = np.zeros((size, size))
    pos = [(450, 450)]
    # isolated, as measure_psf needs; attempt-limited, because a count target
    # at this spacing cannot always be met (random packing saturates at ~55%
    # of the ideal) and the loop then never ends
    for _ in range(4000):
        y, x = rng.integers(80, size - 80, 2)
        if all(np.hypot(y - py, x - px) > 95 for py, px in pos):
            pos.append((int(y), int(x)))
        if len(pos) >= 28:
            break
    for y, x in pos[1:]:
        S[y, x] += 10 ** rng.uniform(2.5, 5.5)
    S[450, 450] += 3e7                                  # clips
    yy, xx = np.mgrid[:size, :size]
    S += 40 * np.exp(-np.hypot(yy - 300, xx - 600) / 40)
    from scipy.signal import fftconvolve
    X = fftconvolve(S, k, mode="same") + sky
    X = np.minimum(X, 0.98 * sat) + rng.normal(0, sig, X.shape)
    X[:40, :] = np.median(X)                            # coverage fill
    return X.astype(np.float32), k


def test_scatter_conserves_flux_and_starts_at_identity():
    torch = pytest.importorskip("torch")
    from nn import deblur_model as dm
    m = dm.build()
    x = torch.randn(1, 2, 96, 96) * 0.5
    x[0, 0, 40, 40] = 9.0                               # asinh(4000): a bright star
    with torch.no_grad():
        y = m(x)
    lin_in, lin_out = torch.sinh(x[:, :1]), torch.sinh(y)
    # identity init: 98% of every pixel stays put
    assert float((lin_out - lin_in).abs().max()) < 0.03 * float(lin_in.abs().max())
    # random weights: flux conserved in the interior (the window reaches 21 px)
    torch.nn.init.normal_(m.net.head.weight, std=0.3)
    with torch.no_grad():
        y = m(x)
    a, b = torch.sinh(x[:, :1])[..., 24:-24, 24:-24].sum(), torch.sinh(y)[..., 24:-24, 24:-24].sum()
    assert abs(float(a) - float(b)) < 0.02 * abs(float(a)) + 1.0
    assert float(torch.sinh(y).min()) >= float(torch.sinh(x[:, :1]).min()) - 1e-3  # nothing new below the floor


def test_scale_kernel_keeps_unit_sum_and_changes_width():
    k = _kernel()
    for f in (0.8, 1.0, 1.25):
        s = dd.scale_kernel(k, f)
        assert abs(s.sum() - 1) < 1e-6
        assert s.shape == k.shape
    assert dd.scale_kernel(k, 1.25).max() < k.max() < dd.scale_kernel(k, 0.8).max()


def test_prepare_and_pair(tmp_path):
    X, k = _field()
    side = dd.prepare_stack(X, "t__L_train0_raw", out_dir=tmp_path, log=lambda s: None)
    assert side is not None and side["stars"] >= dd.MIN_STARS
    assert 5.5 < side["fwhm"] < 9.5
    assert side["flat"].sum() >= 4                      # the clipped star
    assert side["fill"][:40].all()
    man = {"stacks": {"t__L_train0_raw": {"dso": "t", "filter": "L", "domain": "raw"}}}
    import json
    json.dump(man, open(tmp_path / "manifest.json", "w"))
    items = dd.list_prepared(tmp_path)
    assert len(items) == 1 and items[0].filt == "L" and items[0].domain == "raw"
    ds = dd.DeblurDataset(items, patch_size=256, pairs_per_epoch=4, seed=1)
    it = ds.items[0]
    inp, tgt, mask = ds.make_pair(it, 300, 300, factor=1.0)
    assert inp.shape == (2, 256, 256) and tgt.shape == (1, 256, 256) and mask.shape == (1, 256, 256)
    # the target is the patch itself, in asinh units
    a = np.asarray(it.frame[300:556, 300:556], dtype=np.float64)
    assert np.allclose(tgt[0], np.arcsinh(a / it.scale), atol=1e-4)
    # the input is blurrier than the target and its kernel plane blurrier still
    def fine(z):
        from scipy.ndimage import gaussian_filter
        return float(np.std(z - gaussian_filter(z, 2)))
    assert fine(tgt[0]) > fine(inp[0]) > fine(inp[1])
    # the mask drops the plateau (at 450,450 -> patch 150,150) and its
    # SAT_DILATE surround, and keeps the rest
    assert mask[0, 150, 150] == 0 and mask[0, 150 + 30, 150] == 1 and mask.mean() > 0.9
    # a patch has to sit MARGIN inside the frame, so the fill strip at the top
    # can only be reached when MARGIN < 40; the mask rule itself is covered by
    # the sidecar's fill mask above
    # a dataset item comes out as tensors of the same shapes
    torch = pytest.importorskip("torch")
    xi, yi, mi = ds[0]
    assert tuple(xi.shape) == (2, 256, 256) and tuple(yi.shape) == (1, 256, 256) and tuple(mi.shape) == (1, 256, 256)
    assert torch.isfinite(xi).all() and torch.isfinite(yi).all()


def test_tiled_inference_identity_returns_input():
    torch = pytest.importorskip("torch")
    from nn import deblur_model as dm
    X, k = _field(seed=2, size=700)
    m = dm.build()
    out, info = dm.deblur_frame(X, m, k, device="cpu", tile=256, overlap=48)
    assert info["tiles"] > 1
    d = np.abs(out - X)
    # identity keeps 98% of each pixel's flux on itself and the rest lands
    # within the window, so the frame is unchanged but for a leak round the
    # brightest stars: the median deviation is far below a sigma
    assert np.median(d) < 0.5
    # the stitch itself: tiles must agree with a single tile over the frame
    whole, info1 = dm.deblur_frame(X, m, k, device="cpu", tile=800, overlap=48)
    assert info1["tiles"] == 1
    assert np.abs(out - whole)[48:-48, 48:-48].max() < 0.05
    # the plateau is the input exactly
    assert np.array_equal(out[445:456, 445:456], X[445:456, 445:456])


def test_deblur_channels_each_by_own_kernel_keeps_border_and_passes_thin_planes():
    torch = pytest.importorskip("torch")
    from nn import deblur_model as dm
    X, _k = _field(seed=3, size=700)
    X = X.copy()
    X[:20, :] = np.nan                                   # a registration border
    flat = np.full_like(X, 500.0) + np.random.default_rng(0).normal(0, 3.0, X.shape).astype(np.float32)
    m = dm.build()                                       # identity at init
    out, infos = dm.deblur_channels({"L": X, "Ha": flat}, m, device="cpu")
    assert set(out) == {"L", "Ha"}
    assert infos["L"]["why"] is None and infos["L"]["stars"] > 0
    assert np.isnan(out["L"][:20, :]).all() and np.isfinite(out["L"][20:, :]).all()
    assert np.median(np.abs(out["L"][20:] - X[20:])) < 0.5
    assert infos["Ha"]["why"] == "no kernel" and out["Ha"] is flat   # starless plane: untouched


def test_deblur_channels_planes_selects_the_luminance_only():
    torch = pytest.importorskip("torch")
    from nn import deblur_model as dm
    X, _k = _field(seed=3, size=700)
    R = X * 0.5
    m = dm.build()
    out, infos = dm.deblur_channels({"L": X, "R": R}, m, device="cpu", planes=("L",))
    assert set(out) == {"L", "R"}
    assert infos["L"]["why"] is None and infos["L"]["stars"] > 0
    assert infos["R"] == {"why": "not selected"} and out["R"] is R   # the colour plane as given


def test_stacking_deblur_method_learned_is_luminance_only_and_the_default():
    torch = pytest.importorskip("torch")
    from nn import deblur_model as dm
    from stacking import deblur as sd, color_process
    assert color_process.DEBLUR_METHOD == "learned"
    X, _k = _field(seed=3, size=700)
    R = X * 0.5
    out, infos = sd.deblur_channels({"L": X, "R": R, "G": R}, 40, method="learned", model=dm.build())
    assert infos["L"]["method"] == "learned" and infos["L"]["why"] is None and infos["L"]["stars"] > 0
    assert infos["R"].get("kept") and infos["R"]["why"] is None and out["R"] is R and out["G"] is R
    assert np.median(np.abs(out["L"] - X)) < 0.5                      # identity model: unchanged


def test_stacking_deblur_method_learned_falls_back_to_starless_without_a_model(monkeypatch):
    pytest.importorskip("torch")
    from nn import deblur_model as dm
    from stacking import deblur as sd
    monkeypatch.setattr(dm, "load", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no checkpoint")))
    rng = np.random.default_rng(1)
    flat = (300 + rng.normal(0, 2, (300, 300))).astype(np.float32)    # starless refuses: no stars
    out, infos = sd.deblur_channels({"L": flat}, 8, method="learned", log=None)
    assert out["L"] is flat and infos["L"]["fallback"].startswith("starless:")


def test_stacking_deblur_method_learned_without_an_l_makes_one():
    """An HOO set has no L. Per-plane deconvolution made coloured specks
    (nebula sweep 2026-10-21); no_l="synthetic" (2026-10-22) makes a
    luminance from the planes by the palette's weights, sharpens that, and
    hands back the sharpened and unsharpened pair (LSYN / LSYN_REF) with the
    planes kept as given, so compose() scales the picture by their ratio.
    The default rule stays starless (dark squares round bright stars).
    Named explicitly, the planes still go through."""
    torch = pytest.importorskip("torch")
    from nn import deblur_model as dm
    from stacking import deblur as sd
    X, _k = _field(seed=3, size=700)
    O3 = X * 0.7
    lines = []
    out, infos = sd.deblur_channels({"R": X, "G": O3, "B": O3}, 40, method="learned",
                                    model=dm.build(), planes=("L",), log=lines.append,
                                    no_l="synthetic")
    assert "L" not in out and out["LSYN"].shape == X.shape and "LSYN_REF" in out
    assert infos["LSYN"]["synthetic"] == "0.299 R + 0.587 G + 0.114 B"
    assert infos["LSYN"]["why"] is None and not infos["LSYN"].get("fallback")
    np.testing.assert_allclose(out["LSYN_REF"], 0.299 * X + 0.701 * O3, rtol=1e-5)
    assert np.nanmax(np.abs(out["LSYN"] - out["LSYN_REF"])) > 0        # the network moved something
    # the clipped star (R clips at 450,450; the sum does not) is kept as given:
    # the planes' plateau masks are inherited, so no moat is dug round it
    keep = sd.plateau_union({"R": X, "G": O3, "B": O3})
    assert keep[450, 450] and any("plateau px inherited" in m for m in lines)
    np.testing.assert_array_equal(out["LSYN"][keep], out["LSYN_REF"][keep])
    assert all(infos[c]["kept"] and out[c] is {"R": X, "G": O3, "B": O3}[c] for c in "RGB")
    assert any("made one, L = 0.299 R + 0.587 G + 0.114 B" in m for m in lines)
    # the synthetic plane is the weighted sum of the planes, with their sky in
    L, w = sd.synthetic_luminance({"R": X, "G": O3, "B": O3})
    assert abs(sum(w.values()) - 1.0) < 1e-6
    np.testing.assert_allclose(L, 0.299 * X + 0.701 * O3, rtol=1e-5)
    # a set whose planes give no kernel goes to starless, saying why
    rng = np.random.default_rng(1)
    flat = (300 + rng.normal(0, 2, (300, 300))).astype(np.float32)      # starless refuses: no stars
    lines = []
    out, infos = sd.deblur_channels({"R": flat, "G": flat, "B": flat}, 8, method="learned",
                                    model=dm.build(), planes=("L",), log=lines.append,
                                    no_l="synthetic")
    assert "LSYN" not in out and out["R"] is flat
    assert all(infos[c]["fallback"].startswith("starless: the synthetic luminance") for c in "RGB")
    assert any("RL8 on every distinct plane" in m for m in lines)
    # the default rule for a set with no L is still starless (NO_L_RULE), saying why
    assert sd.NO_L_RULE == "starless"
    lines = []
    out, infos = sd.deblur_channels({"R": flat, "G": flat, "B": flat}, 8, method="learned",
                                    model=dm.build(), planes=("L",), log=lines.append)
    assert "LSYN" not in out and out["R"] is flat
    assert all(infos[c]["fallback"].startswith("starless: no luminance plane") for c in "RGB")
    with pytest.raises(ValueError):
        sd.deblur_channels({"R": flat}, 8, method="learned", model=dm.build(), no_l="magic")
    # named explicitly, the planes go through the network
    out, infos = sd.deblur_channels({"R": X, "G": O3, "B": O3}, 40, method="learned", model=dm.build(),
                                    planes=("R", "G", "B"))
    assert all(infos[c]["why"] is None and not infos[c].get("fallback") for c in "RGB")
    assert out["G"] is out["B"]                                        # the shared plane, done once


def test_compose_synthetic_pair_is_a_no_op_when_nothing_moved_and_a_gain_where_it_did():
    """The composite lit by a synthetic pair differs from the plain composite
    only where the deconvolved frame differs from its reference: identical
    frames compose to the identical picture (a real L would not -- the
    per-plane black clip makes a summed L darker at the faint end, measured
    27 % on trunk), and a brighter spot in LSYN brightens that spot only."""
    from stacking import color_process as cp
    rng = np.random.default_rng(5)
    base = 50 + rng.normal(0, 2, (200, 200)).astype(np.float32)
    R, G, B = base * 1.0, base * 0.8 + 20, base * 0.6
    L = 0.299 * R + 0.587 * G + 0.114 * B
    plain = cp.compose({"R": R, "G": G, "B": B}, deblur=0, white_balance="none")
    same = cp.compose({"R": R, "G": G, "B": B, "LSYN": L.copy(), "LSYN_REF": L.copy()},
                      deblur=0, white_balance="none")
    np.testing.assert_allclose(same, plain, atol=1e-6)
    moved = L.copy(); moved[90:110, 90:110] *= 1.5
    lit = cp.compose({"R": R, "G": G, "B": B, "LSYN": moved, "LSYN_REF": L.copy()},
                     deblur=0, white_balance="none")
    inside = (slice(90, 110), slice(90, 110))
    assert np.all(lit[inside] >= plain[inside]) and lit[inside].mean() > plain[inside].mean() * 1.05
    outside = np.ones((200, 200), bool); outside[inside] = False
    np.testing.assert_allclose(lit[outside], plain[outside], atol=1e-6)


def test_taper_kernel_keeps_the_core_drops_the_wing_and_measures_fwhm():
    k = _kernel(fwhm=7.5)
    assert abs(dd.kernel_fwhm(k) - 7.5) < 0.6                 # the wing term broadens it a little
    t = dd.taper_kernel(k, (1.5, 3.0))
    assert abs(t.sum() - 1) < 1e-9 and t.shape == k.shape
    c = k.shape[0] // 2
    yy, xx = np.mgrid[:k.shape[0], :k.shape[0]] - c
    r = np.hypot(yy, xx)
    inner, outer = r <= 1.5 * 7.5 - 1, r >= 3.0 * 7.5 + 1
    # inside the inner radius the shape is the kernel's own (one factor), beyond the outer nothing
    ratio = t[inner] / k[inner]
    assert ratio.std() / ratio.mean() < 1e-6 and ratio.mean() > 1.0
    assert t[outer].max() == 0.0 and k[outer].max() > 0.0
    # None is the whole kernel, unit sum
    assert np.allclose(dd.taper_kernel(k, None), k / k.sum())
    with pytest.raises(ValueError):
        dd.taper_kernel(k, (3.0, 1.5))


def test_pair_is_made_with_the_tapered_kernel_and_the_checkpoint_carries_it(tmp_path):
    torch = pytest.importorskip("torch")
    from nn import deblur_model as dm
    X, k = _field(seed=3)
    dd.prepare_stack(X, "t__L_train0_raw", out_dir=tmp_path, log=lambda s: None)
    items = dd.list_prepared(tmp_path)
    whole = dd.DeblurDataset(items, patch_size=256, pairs_per_epoch=2, seed=1, kernel_taper=None)
    core = dd.DeblurDataset(items, patch_size=256, pairs_per_epoch=2, seed=1, kernel_taper=(1.5, 3.0))
    assert whole.kernel_taper is None and core.kernel_taper == (1.5, 3.0)
    iw, tw, _ = whole.make_pair(whole.items[0], 300, 300, factor=1.0)
    ic, tc, _ = core.make_pair(core.items[0], 300, 300, factor=1.0)
    assert np.allclose(tw, tc)                                  # same target
    assert not np.allclose(iw[1], ic[1])                        # the kernel plane differs
    # the core kernel blurs less, so the core input sits nearer the target
    assert np.abs(ic[0] - tc[0]).mean() < np.abs(iw[0] - tw[0]).mean()
    # the checkpoint stamps the taper and load attaches it to the model
    m = dm.build()
    dm.save(m, tmp_path / "core.pt", {"name": "core"}, kernel_taper=(1.5, 3.0))
    dm.save(m, tmp_path / "whole.pt", {"name": "whole"})
    mc, _ = dm.load(tmp_path / "core.pt", "cpu")
    mw, _ = dm.load(tmp_path / "whole.pt", "cpu")
    assert mc.kernel_taper == (1.5, 3.0) and mw.kernel_taper is None
    # inference tapers by the model's own setting
    seen = []
    orig = dd.taper_kernel
    try:
        dd.taper_kernel = lambda psf, taper=dd.KERNEL_TAPER: seen.append(taper) or orig(psf, taper)
        out_c, info_c = dm.deblur_frame(X, mc, k, device="cpu", tile=512, overlap=48)
        out_w, info_w = dm.deblur_frame(X, mw, k, device="cpu", tile=512, overlap=48)
    finally:
        dd.taper_kernel = orig
    assert seen == [(1.5, 3.0), None]
    assert info_c["kernel_taper"] == (1.5, 3.0) and info_w["kernel_taper"] is None
