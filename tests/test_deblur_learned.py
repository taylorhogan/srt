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
