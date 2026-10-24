"""The compose step's background removal reaches the frame's edges.

A sky slope (moonlight, haze) is a plane across the frame. The mesh model
alone holds its value in the outer half-box, so the slope survived there:
ngc2276 on 2026-10-22 came out with one edge clipped black and the other a
bright band. The plane goes first; the mesh handles the dome and leaves
galaxies and nebulae alone as before."""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
from stacking import color_process as cp  # noqa: E402


def _field(slope=30.0, dome=12.0, sigma=4.0, seed=3):
    rng = np.random.default_rng(seed)
    h, w = 800, 1200
    y, x = np.mgrid[:h, :w]
    sky = 350 + slope * x / w - dome * (((x - w / 2) / w) ** 2 + ((y - h / 2) / h) ** 2) * 4
    galaxy = 60 * np.exp(-(((x - 900) ** 2 + (y - 120) ** 2) / (2 * 25.0 ** 2)))
    frame = (sky + galaxy + rng.normal(0, sigma, (h, w))).astype(np.float32)
    return frame, galaxy, sigma, sky


def _edge_residual(sub):
    h, w = sub.shape
    mid = sub[h // 4: 3 * h // 4]
    return (float(np.median(mid[:, : w // 10])), float(np.median(mid[:, 2 * w // 5: 3 * w // 5])),
            float(np.median(mid[:, 9 * w // 10:])))


def test_sky_plane_finds_the_slope_and_ignores_the_galaxy():
    frame, galaxy, sigma, sky = _field()
    plane = cp.sky_plane(frame)
    h, w = frame.shape
    tilt = float(np.median(plane[:, 9 * w // 10:]) - np.median(plane[:, : w // 10]))
    assert abs(tilt - 30.0 * 0.9) < 2.0            # the outer tenths' centres are 0.9 of the width apart
    # the surface follows the sky to the edges and the galaxy does not pull it up under itself
    assert np.abs(plane - sky)[::50, ::50].max() < 0.5 * sigma
    assert abs(plane[120, 900] - sky[120, 900]) < 0.5 * sigma


def test_background_removal_reaches_the_edges_and_keeps_the_galaxy():
    pytest.importorskip("photutils")
    frame, galaxy, sigma, sky = _field()
    sub = cp._remove_gradient(frame, 4)
    left, centre, right = _edge_residual(sub)
    # what is left at the edges is the dome's curvature in the outer half-box, well under half a sigma
    assert abs(left) < 0.4 * sigma and abs(right) < 0.4 * sigma and abs(centre) < 0.3 * sigma
    # the galaxy's peak survives (the mesh is coarse on purpose)
    assert sub[120, 900] > 0.9 * galaxy[120, 900]
    # without the plane the mesh alone left the edges a sigma off: this is the case the fix is for
    from astropy.stats import SigmaClip
    from photutils.background import Background2D, SExtractorBackground
    box = max(64, min(frame.shape) // 4)
    bare = (frame - galaxy).astype(np.float32)                  # the sky alone: slope, dome, noise
    old = bare - Background2D(bare, box_size=box, filter_size=3, sigma_clip=SigmaClip(sigma=3.0),
                              bkg_estimator=SExtractorBackground()).background
    l0, _, r0 = _edge_residual(old)
    assert r0 - l0 > 1.0 * sigma and l0 < -0.6 * sigma


def test_a_flat_sky_is_left_flat():
    pytest.importorskip("photutils")
    frame, galaxy, sigma, sky = _field(slope=0.0, dome=0.0)
    sub = cp._remove_gradient(frame, 4)
    left, centre, right = _edge_residual(sub)
    assert max(abs(left), abs(centre), abs(right)) < 0.2 * sigma


def test_edge_rolloff_is_taken_out_and_the_interior_is_untouched():
    pytest.importorskip("photutils")
    frame, galaxy, sigma, sky = _field()
    h, w = frame.shape
    x = np.arange(w, dtype=np.float32)
    rise = 8.0 * np.clip((x - 0.95 * w) / (0.05 * w), 0, 1) ** 2              # the flat's edge: +8 ADU in the last 5 %
    frame = (frame + rise[None, :]).astype(np.float32)
    sub = cp._remove_gradient(frame, 4)
    mid = sub[h // 4: 3 * h // 4]
    assert abs(float(np.median(mid[:, 98 * w // 100:]))) < 0.3 * sigma
    assert sub[120, 900] > 0.9 * galaxy[120, 900]
    # a nebula band across the whole width keeps its interior: only the outer tenth is touched
    y, xx = np.mgrid[:h, :w]
    band = 40 * np.exp(-((y - h / 2) ** 2) / (2 * 130.0 ** 2))
    rng = np.random.default_rng(9)
    neb = (sky + band + rng.normal(0, sigma, (h, w))).astype(np.float32)
    mesh_only = cp._remove_gradient(neb, 4)
    with_edge = cp._edge_rolloff(mesh_only)
    inner = (slice(h // 2 - 10, h // 2 + 10), slice(w // 4, 3 * w // 4))
    assert abs(float(np.median(with_edge[inner]) - np.median(mesh_only[inner]))) < 0.1 * sigma
