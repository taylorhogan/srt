"""The learned star/nebula split's data recipe and tiler (2026-11-05,
docs/STARSPLIT_PLAN.md). The network itself is judged by `score` and by
eye; these pin the parts that must hold whatever it learns."""
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nn import starsplit_data as D  # noqa: E402


def _psf(r=20, fwhm=5.0):
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    s = fwhm / 2.3548
    p = np.exp(-(xx ** 2 + yy ** 2) / (2 * s * s)) + 0.002 / (1 + (xx ** 2 + yy ** 2) / 36.0)
    return (p / p.sum()).astype(np.float32)


def test_injected_stars_carry_their_peaks_and_flux():
    rng = np.random.default_rng(0)
    psf = _psf()
    stars, table = D.inject_stars((128, 128), psf, rng, 25, rms=1.0, sat=60000.0)
    assert stars.shape == (128, 128) and len(table) == 25 and np.isfinite(stars).all()
    assert (table[:, 2] >= D.AMP_SIGMA_MIN).all() and (table[:, 2] <= D.AMP_SAT_MAX * 60000).all()
    # one isolated star at an integer position: its peak is the asked amplitude
    one, t = D.inject_stars((256, 256), psf, np.random.default_rng(1), 1, rms=1.0, sat=60000.0)
    y, x, a = t[0]
    assert 0.4 * a <= one.max() <= 1.01 * a               # the bilinear sub-pixel shift spreads the peak
    assert 0.5 * a / psf.max() <= one.sum() <= 1.01 * a / psf.max()   # the flux is the stamp's, less an edge cut


def test_knots_and_fractals_are_background_sized():
    rng = np.random.default_rng(2)
    k = D.synthetic_knots((96, 96), rng, fwhm_px=5.0, n=5, rms=1.0)
    assert k.shape == (96, 96) and k.max() <= 40.0 and k.min() >= 0.0
    assert D.synthetic_knots((96, 96), rng, 5.0, 0, 1.0).max() == 0.0
    f = D.fractal_field((64, 64), rng, rms=2.0)
    assert f.shape == (64, 64) and f.min() == 0.0 and 10.0 <= f.max() <= 120.0


def test_dataset_pairs_are_background_plus_stars(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    bg = (rng.normal(0, 1.0, (700, 700)) + 20 * np.exp(-((np.mgrid[:700, :700][0] - 350) ** 2) / 8000.0)).astype(np.float32)
    np.save(tmp_path / "t__Ha_bg.npy", bg)
    np.savez_compressed(tmp_path / "t__Ha_psf.npz", psf=_psf(), spikes=np.zeros((41, 41), np.float32),
                        sat=60000.0, rms=1.0, density=30 / (512.0 * 512.0))
    (tmp_path / "manifest.json").write_text(json.dumps({"planes": {"t__Ha": {"fwhm_px": 5.0}}}))
    monkeypatch.setattr(D, "FLAT_FRACTION", 0.0)
    monkeypatch.setattr(D, "FRACTAL_FRACTION", 0.0)
    ds = D.StarSplitDataset(["t__Ha"], prepared=tmp_path, patch=256)
    x, t, scale, truth = ds.make_pair(np.random.default_rng(4), with_truth=True)
    assert x.shape == t.shape == (1, 256, 256) and x.dtype == np.float32 and scale > 0
    # input = target + stars, clipped at saturation, both asinh(./scale)
    np.testing.assert_allclose(np.sinh(x[0]) * scale, np.minimum(truth["target"] + truth["stars"], 60000.0), rtol=1e-4, atol=1e-2)
    np.testing.assert_allclose(np.sinh(t[0]) * scale, truth["target"], rtol=1e-4, atol=1e-2)
    assert (x >= t - 1e-5).all()                       # stars only ever add light
    assert len(truth["table"]) >= 1 and truth["stars"].max() > 5 * truth["rms"]


def test_split_frame_tiles_an_identity_model_back_to_the_frame():
    torch = pytest.importorskip("torch")
    from nn import starsplit

    class Ident(torch.nn.Module):
        def forward(self, x):
            return x

    rng = np.random.default_rng(5)
    frame = (rng.normal(200, 3, (300, 420)) + 50 * np.exp(-((np.mgrid[:300, :420][1] - 200) ** 2) / 5000.0)).astype(np.float32)
    frame[10:14, 10:14] = np.nan
    starless, stars = starsplit.split_frame(frame, Ident(), "cpu", tile=128, overlap=16, log=lambda s: None)
    ok = np.isfinite(frame)
    np.testing.assert_allclose(starless[ok], frame[ok], atol=0.05)
    assert np.abs(stars[ok]).max() < 0.05 and np.isnan(starless[~ok]).all()


def test_loss_is_asinh_l2_by_default():
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import starsplit_train as T
    o = torch.tensor([[1.0, 2.0]]); t = torch.tensor([[1.0, 4.0]])
    assert float(T._loss(o, t, 0.0)) == pytest.approx(2.0)
    assert float(T._loss(o, t, 0.5)) > 2.0


def test_bright_star_fill_removes_a_compact_nub_and_leaves_the_rest():
    pytest.importorskip("sep")
    from nn import starsplit
    rng = np.random.default_rng(7)
    h, w = 300, 300
    frame = (rng.normal(100, 1.0, (h, w))).astype(np.float32)
    yy, xx = np.mgrid[:h, :w]
    star = 5000.0 * np.exp(-((yy - 150) ** 2 + (xx - 150) ** 2) / (2 * 2.5 ** 2))
    frame += star.astype(np.float32)
    # the network's output: star gone but a nub of 300 ADU at the core inside a -40 ADU ring
    starless = frame - star.astype(np.float32)
    rr = np.hypot(yy - 150, xx - 150)
    starless += np.where(rr < 3, 300.0, 0.0).astype(np.float32) + np.where((rr >= 5) & (rr < 8), -40.0, 0.0).astype(np.float32)
    out, n = starsplit.fill_bright_residuals(frame, starless, log=lambda s: None)
    assert n == 1
    core = out[140:161, 140:161]
    assert abs(float(core.mean()) - 100.0) < 1.0 and float(core.std()) < 3.0
    far = rr > 40
    np.testing.assert_array_equal(out[far], starless[far])          # untouched away from the star
    # a faint star is not a candidate
    faint = (frame - star).astype(np.float32); faint[50, 50] += 400
    _, n2 = starsplit.fill_bright_residuals(faint, faint.copy(), log=lambda s: None)
    assert n2 == 0


def test_arm_fill_takes_the_far_arms_off_a_bright_star():
    pytest.importorskip("sep")
    from nn import starsplit
    rng = np.random.default_rng(11)
    h, w = 400, 400
    frame = rng.normal(100, 1.0, (h, w)).astype(np.float32)
    yy, xx = np.mgrid[:h, :w]
    star = 6000.0 * np.exp(-((yy - 200) ** 2 + (xx - 200) ** 2) / (2 * 2.5 ** 2))
    frame += star.astype(np.float32)
    # the network's output: the star gone, but two bars (the far arms) of 6 ADU along 45 deg
    starless = frame - star.astype(np.float32)
    u = (xx - 200) * np.cos(np.radians(45)) + (yy - 200) * np.sin(np.radians(45))
    v = -(xx - 200) * np.sin(np.radians(45)) + (yy - 200) * np.cos(np.radians(45))
    bars = ((np.abs(v) < 2) | (np.abs(u) < 2)) & (np.hypot(u, v) < 70) & (np.hypot(u, v) > 12)
    starless += np.where(bars, 6.0, 0.0).astype(np.float32)
    before = float(starless[bars].mean() - 100.0)
    out, n = starsplit.fill_bright_residuals(frame, starless, arms=[(200.0, 200.0, 45.0, 80.0)], log=lambda s: None)
    after = float(out[bars].mean() - 100.0)
    assert before > 5.0 and after < 1.0
    far = np.hypot(u, v) > 120
    np.testing.assert_array_equal(out[far], starless[far])


def test_starless_product_is_for_palette_recipes_only(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import n2n_lrgb_render as R
    msgs = []
    got = R.starless_product(None, tmp_path, {}, {}, {}, {}, {"L": "L", "R": "R", "G": "G", "B": "B"}, {}, msgs.append)
    assert got is None and "palette recipes only" in msgs[0]


def test_the_routine_writes_the_starless_beside_palette_renders_in_shadow():
    import inspect
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import n2n_lrgb_render as R
    src = inspect.getsource(R.routine)
    assert 'starless_product(args, out_dir, raw, raw_info, routine, opts, mapping, meta, log)' in src
    assert '!= "off" and not getattr(args, "final", False)' in src
    assert src.index("push_render(args, out_dir, written, meta, log, crops=crops)") < src.index("push_starless(args, out_dir, starless_path, log)")
    assert '"starless"' in inspect.getsource(R.starless_product)
