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
