"""The background removal's fourth stage: the residual mesh.

After the sky surface, the coarse mesh and the edge roll-off, the sky still
carried its 100-500 px structure (flat residuals, glow): on ngc2276 +/-1.2 ADU
against a black 1.5 ADU under the sky, patches clipped black and the rest
blotchy on the foot of the stretch (2026-11-16). The fine mesh takes it out
with the extended light masked, and stands aside on a field that is all
nebula."""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("photutils")
from stacking import color_process as cp  # noqa: E402


def _field(patches=True, seed=5):
    """The real geometry scaled by 2.7: a 2400 x 3600 frame, whose coarse mesh
    box is 600 px, with sky patches of 120 px sigma (the 300 px structure of
    ngc2276 against its 1600 px boxes) and a galaxy with a halo."""
    rng = np.random.default_rng(seed)
    h, w = 2400, 3600
    y, x = np.mgrid[:h, :w]
    sky = 350 + 20.0 * x / w
    if patches:
        for (cx, cy, amp) in ((500, 500, 3.0), (2600, 1700, -2.5), (1800, 2100, 2.5), (3100, 400, -2.0),
                              (900, 1600, -2.5), (2300, 700, 2.0)):
            sky = sky + amp * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * 120.0 ** 2)))
    galaxy = 60 * np.exp(-(((x - 1400) ** 2 + (y - 1000) ** 2) / (2 * 40.0 ** 2)))
    halo = 6 * np.exp(-(((x - 1400) ** 2 + (y - 1000) ** 2) / (2 * 160.0 ** 2)))
    frame = (sky + galaxy + halo + rng.normal(0, 4.0, (h, w))).astype(np.float32)
    return frame, galaxy + halo


def _mid_scale(sub, exclude=None):
    """Medians of 300 px cells, the galaxy's cells left out."""
    h, w = sub.shape
    cells = []
    for yy in range(0, h - 300 + 1, 300):
        for xx in range(0, w - 300 + 1, 300):
            if exclude and abs(xx + 150 - exclude[0]) < 600 and abs(yy + 150 - exclude[1]) < 600:
                continue
            cells.append(float(np.median(sub[yy:yy + 300, xx:xx + 300])))
    return np.array(cells)


def test_mid_scale_patches_are_taken_out_and_the_galaxy_is_kept():
    frame, obj = _field()
    coarse = cp._remove_gradient(frame, 4, fine=False)
    info: dict = {}
    sub = cp._remove_gradient(frame, 4, info=info, name="L")
    before = _mid_scale(coarse, exclude=(1400, 1000))
    after = _mid_scale(sub, exclude=(1400, 1000))
    span_before = before.max() - before.min()
    span_after = after.max() - after.min()
    assert span_before > 2.0                          # the patches survive the three older stages
    assert span_after < 0.5 * span_before and span_after < 1.2   # and not the fourth
    fi = info["fine_mesh"]["L"]
    assert "skipped" not in fi and fi["object_fraction"] < 0.5 and fi["box_px"] == 256
    # the galaxy and its halo are masked, not flattened
    assert sub[1000, 1400] > 0.9 * obj[1000, 1400]
    assert sub[1000, 1600: 1660].mean() > 0.6 * obj[1000, 1630]   # the skirt at half a sigma keeps most of itself


def test_a_field_that_is_mostly_object_skips_the_fine_stage(monkeypatch):
    # a residual plane whose structure IS the target: filaments over most of the field
    rng = np.random.default_rng(7)
    h, w = 1200, 1800
    y, x = np.mgrid[:h, :w]
    resid = rng.normal(0, 4.0, (h, w)).astype(np.float32)
    for cx in range(100, w, 180):
        for cy in range(100, h, 180):
            resid += (25 * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * 70.0 ** 2)))).astype(np.float32)
    model, info = cp.fine_residual_background(resid)
    assert model is None and info["skipped"].startswith("mostly object")
    # and through _remove_gradient a skipped stage changes nothing
    frame, _ = _field()
    monkeypatch.setattr(cp, "FINE_MESH_MAX_OBJECT", 0.0)
    info = {}
    sub = cp._remove_gradient(frame, 4, info=info, name="Ha")
    assert info["fine_mesh"]["Ha"]["skipped"].startswith("mostly object")
    assert np.array_equal(sub, cp._remove_gradient(frame, 4, fine=False))


def test_the_stage_never_raises_and_a_tiny_frame_is_left_alone():
    model, info = cp.fine_residual_background(np.zeros((40, 40), dtype=np.float32))
    assert model is None and info["skipped"] == "frame too small"
